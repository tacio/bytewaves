//! Platform-neutral ByteWaves protocol, modulation, and streaming decoder.

use std::collections::VecDeque;
use std::f32::consts::PI;
use std::fmt;

use crc32fast::Hasher;
use reed_solomon::{Decoder as RsDecoder, Encoder as RsEncoder};
use serde::Serialize;

pub const PROTOCOL_VERSION: u8 = 2;
pub const ECC_BYTES: usize = 16;
pub const MAX_BLOCK_SIZE: usize = 255;
pub const MAX_PAYLOAD_BYTES: usize = MAX_BLOCK_SIZE - ECC_BYTES - 4;
pub const TONE_DURATION: f32 = 0.020;
pub const GUARD_DURATION: f32 = 0.020;
pub const TRANSMISSION_PAUSE: f32 = 1.0;
pub const FREQUENCIES: [f32; 8] = [
    1200.0, 1600.0, 2100.0, 2750.0, 3600.0, 4700.0, 6100.0, 7800.0,
];
pub const TRAINING_PREAMBLE: [u8; 8] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80];

const HEADER_BYTES: usize = 3;
const RAMP_DURATION: f32 = 0.002;
const SEARCH_STEP_DURATION: f32 = 0.001;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModemError {
    EmptyPayload,
    PayloadTooLarge { actual: usize, maximum: usize },
    InvalidSampleRate(u32),
    InvalidHeader,
    ReedSolomon,
    CrcMismatch,
    InvalidUtf8,
}

impl fmt::Display for ModemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPayload => write!(f, "message is empty"),
            Self::PayloadTooLarge { actual, maximum } => {
                write!(f, "message is {actual} UTF-8 bytes; maximum is {maximum}")
            }
            Self::InvalidSampleRate(rate) => write!(f, "unsupported sample rate: {rate}"),
            Self::InvalidHeader => write!(f, "invalid protocol header"),
            Self::ReedSolomon => write!(f, "Reed-Solomon decoding failed"),
            Self::CrcMismatch => write!(f, "CRC mismatch"),
            Self::InvalidUtf8 => write!(f, "decoded payload is not valid UTF-8"),
        }
    }
}

impl std::error::Error for ModemError {}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum DecoderEvent {
    Synchronized,
    FrameProgress {
        received: usize,
        total: usize,
    },
    DecodeFailed {
        reason: String,
    },
    MessageDecoded {
        text: String,
        corrected_errors: usize,
    },
}

/// Encode text into the protocol bytes following the training preamble.
pub fn encode_packet(text: &str) -> Result<Vec<u8>, ModemError> {
    let payload = text.as_bytes();
    validate_payload(payload)?;

    let mut protected = Vec::with_capacity(payload.len() + 4);
    protected.extend_from_slice(payload);
    protected.extend_from_slice(&crc32(payload).to_be_bytes());
    let encoded = RsEncoder::new(ECC_BYTES).encode(&protected);
    let encoded_len = encoded.len();
    debug_assert!(encoded_len <= MAX_BLOCK_SIZE);

    let mut packet = Vec::with_capacity(HEADER_BYTES + encoded_len);
    packet.push(PROTOCOL_VERSION);
    packet.push(encoded_len as u8);
    packet.push(crc8(&packet));
    packet.extend_from_slice(&encoded);
    Ok(packet)
}

/// Decode protocol bytes following the training preamble.
pub fn decode_packet(packet: &[u8]) -> Result<(String, usize), ModemError> {
    let encoded_len = validate_header(packet)?;
    if packet.len() != HEADER_BYTES + encoded_len {
        return Err(ModemError::InvalidHeader);
    }
    let (corrected, corrected_errors) = RsDecoder::new(ECC_BYTES)
        .correct_err_count(&packet[HEADER_BYTES..], None)
        .map_err(|_| ModemError::ReedSolomon)?;
    let data = corrected.data();
    if data.len() < 4 {
        return Err(ModemError::InvalidHeader);
    }
    let payload_end = data.len() - 4;
    let expected_crc = u32::from_be_bytes(data[payload_end..].try_into().expect("four CRC bytes"));
    if crc32(&data[..payload_end]) != expected_crc {
        return Err(ModemError::CrcMismatch);
    }
    let text = std::str::from_utf8(&data[..payload_end]).map_err(|_| ModemError::InvalidUtf8)?;
    Ok((text.to_owned(), corrected_errors))
}

/// Generate one complete frame, including training preamble and a trailing transmission pause.
pub fn encode_text(text: &str, sample_rate: u32) -> Result<Vec<f32>, ModemError> {
    validate_sample_rate(sample_rate)?;
    let packet = encode_packet(text)?;
    let mut symbols = Vec::with_capacity(TRAINING_PREAMBLE.len() + packet.len());
    symbols.extend_from_slice(&TRAINING_PREAMBLE);
    symbols.extend_from_slice(&packet);
    let mut wave = modulate_symbols(&symbols, sample_rate);
    wave.resize(
        wave.len() + samples_for(TRANSMISSION_PAUSE, sample_rate),
        0.0,
    );
    Ok(wave)
}

pub fn modulate_symbols(symbols: &[u8], sample_rate: u32) -> Vec<f32> {
    let tone_samples = samples_for(TONE_DURATION, sample_rate);
    let guard_samples = samples_for(GUARD_DURATION, sample_rate);
    let ramp_samples = samples_for(RAMP_DURATION, sample_rate).max(1);
    let mut wave = Vec::with_capacity(symbols.len() * (tone_samples + guard_samples));

    for &symbol in symbols {
        let active = symbol.count_ones().max(1) as f32;
        for sample_idx in 0..tone_samples {
            let time = sample_idx as f32 / sample_rate as f32;
            let mut value = 0.0;
            for (bit, frequency) in FREQUENCIES.iter().enumerate() {
                if symbol & (1 << bit) != 0 {
                    value += (2.0 * PI * frequency * time).sin();
                }
            }
            let attack = (sample_idx as f32 / ramp_samples as f32).min(1.0);
            let release = ((tone_samples - sample_idx - 1) as f32 / ramp_samples as f32).min(1.0);
            let envelope = 0.5 - 0.5 * (PI * attack.min(release)).cos();
            wave.push(value * envelope / active);
        }
        wave.resize(wave.len() + guard_samples, 0.0);
    }
    wave
}

/// Incremental decoder accepting arbitrary microphone chunk sizes.
pub struct StreamingDecoder {
    sample_rate: u32,
    tone_samples: usize,
    symbol_samples: usize,
    search_step: usize,
    buffer: VecDeque<f32>,
    search_offset: usize,
    synchronized_at: Option<usize>,
    calibration: [f32; 8],
    expected_encoded_len: Option<usize>,
    accumulated_scores: Option<(usize, Vec<[f32; 8]>, usize)>,
}

impl StreamingDecoder {
    pub fn new(sample_rate: u32) -> Result<Self, ModemError> {
        validate_sample_rate(sample_rate)?;
        Ok(Self {
            sample_rate,
            tone_samples: samples_for(TONE_DURATION, sample_rate),
            symbol_samples: samples_for(TONE_DURATION + GUARD_DURATION, sample_rate),
            search_step: samples_for(SEARCH_STEP_DURATION, sample_rate).max(1),
            buffer: VecDeque::new(),
            search_offset: 0,
            synchronized_at: None,
            calibration: [1.0; 8],
            expected_encoded_len: None,
            accumulated_scores: None,
        })
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.search_offset = 0;
        self.synchronized_at = None;
        self.calibration = [1.0; 8];
        self.expected_encoded_len = None;
        self.accumulated_scores = None;
    }

    pub fn push_samples(&mut self, samples: &[f32]) -> Vec<DecoderEvent> {
        self.buffer.extend(
            samples
                .iter()
                .map(|sample| if sample.is_finite() { *sample } else { 0.0 }),
        );
        let mut events = Vec::new();

        loop {
            if self.synchronized_at.is_none() {
                let needed = TRAINING_PREAMBLE.len() * self.symbol_samples;
                if self.buffer.len() < self.search_offset + needed {
                    self.trim_hunt_buffer(needed);
                    break;
                }
                if let Some(calibration) = self.preamble_calibration(self.search_offset) {
                    self.calibration = calibration;
                    self.synchronized_at = Some(self.search_offset);
                    self.expected_encoded_len = None;
                    events.push(DecoderEvent::Synchronized);
                } else {
                    self.search_offset += self.search_step;
                    continue;
                }
            }

            let frame_start = self.synchronized_at.expect("synchronized frame start");
            let header_start = frame_start + TRAINING_PREAMBLE.len() * self.symbol_samples;
            let header_samples = HEADER_BYTES * self.symbol_samples;
            if self.buffer.len() < header_start + header_samples {
                break;
            }

            let mut header = [0_u8; HEADER_BYTES];
            let encoded_len = if let Some(length) = self.expected_encoded_len {
                header[0] = PROTOCOL_VERSION;
                header[1] = length as u8;
                header[2] = crc8(&header[..2]);
                length
            } else {
                for (index, byte) in header.iter_mut().enumerate() {
                    *byte = self.decode_symbol(header_start + index * self.symbol_samples);
                }
                match validate_header(&header) {
                    Ok(length) => {
                        self.expected_encoded_len = Some(length);
                        length
                    }
                    Err(error) => {
                        events.push(DecoderEvent::DecodeFailed {
                            reason: error.to_string(),
                        });
                        self.resume_hunting(frame_start + self.search_step);
                        continue;
                    }
                }
            };

            let frame_symbols = TRAINING_PREAMBLE.len() + HEADER_BYTES + encoded_len;
            let frame_samples = frame_symbols * self.symbol_samples;
            if self.buffer.len() < frame_start + frame_samples {
                let available = self
                    .buffer
                    .len()
                    .saturating_sub(header_start + header_samples)
                    / self.symbol_samples;
                events.push(DecoderEvent::FrameProgress {
                    received: available.min(encoded_len),
                    total: encoded_len,
                });
                break;
            }

            let payload_start = header_start + header_samples;
            let mut frame_scores = Vec::with_capacity(encoded_len);
            for index in 0..encoded_len {
                frame_scores
                    .push(self.normalized_scores(payload_start + index * self.symbol_samples));
            }
            match &mut self.accumulated_scores {
                Some((length, accumulated, transmissions)) if *length == encoded_len => {
                    for (total, scores) in accumulated.iter_mut().zip(frame_scores) {
                        for carrier in 0..8 {
                            total[carrier] += scores[carrier];
                        }
                    }
                    *transmissions += 1;
                }
                _ => self.accumulated_scores = Some((encoded_len, frame_scores, 1)),
            }
            let (_, accumulated, _) = self
                .accumulated_scores
                .as_ref()
                .expect("frame scores were stored");
            let mut packet = Vec::with_capacity(HEADER_BYTES + encoded_len);
            packet.extend_from_slice(&header);
            packet.extend(accumulated.iter().map(|scores| scores_to_symbol(*scores)));
            match decode_packet(&packet) {
                Ok((text, corrected_errors)) => {
                    self.accumulated_scores = None;
                    events.push(DecoderEvent::MessageDecoded {
                        text,
                        corrected_errors,
                    });
                }
                Err(error) => events.push(DecoderEvent::DecodeFailed {
                    reason: error.to_string(),
                }),
            }
            self.consume(frame_start + frame_samples);
        }
        events
    }

    fn preamble_calibration(&self, start: usize) -> Option<[f32; 8]> {
        let mut calibration = [0.0; 8];
        for expected in 0..8 {
            let powers = self.carrier_powers(start + expected * self.symbol_samples);
            let strongest = powers
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))?
                .0;
            let second = powers
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != strongest)
                .map(|(_, power)| *power)
                .fold(0.0_f32, f32::max);
            if strongest != expected || powers[expected] < 1e-7 || powers[expected] < second * 2.5 {
                return None;
            }
            calibration[expected] = powers[expected];
        }
        Some(calibration)
    }

    fn decode_symbol(&self, start: usize) -> u8 {
        scores_to_symbol(self.normalized_scores(start))
    }

    fn normalized_scores(&self, start: usize) -> [f32; 8] {
        let powers = self.carrier_powers(start);
        std::array::from_fn(|index| powers[index] / self.calibration[index].max(1e-12))
    }

    fn carrier_powers(&self, start: usize) -> [f32; 8] {
        std::array::from_fn(|carrier| {
            let omega = 2.0 * PI * FREQUENCIES[carrier] / self.sample_rate as f32;
            let coefficient = 2.0 * omega.cos();
            let (mut previous, mut previous_two) = (0.0_f32, 0.0_f32);
            for sample in self.buffer.iter().skip(start).take(self.tone_samples) {
                let current = sample + coefficient * previous - previous_two;
                previous_two = previous;
                previous = current;
            }
            let power = previous * previous + previous_two * previous_two
                - coefficient * previous * previous_two;
            power.max(0.0) / (self.tone_samples * self.tone_samples) as f32
        })
    }

    fn trim_hunt_buffer(&mut self, needed: usize) {
        if self.search_offset > needed {
            let consume = self.search_offset - needed;
            self.consume(consume);
        }
    }

    fn resume_hunting(&mut self, offset: usize) {
        self.synchronized_at = None;
        self.expected_encoded_len = None;
        self.search_offset = offset;
    }

    fn consume(&mut self, count: usize) {
        for _ in 0..count.min(self.buffer.len()) {
            self.buffer.pop_front();
        }
        self.search_offset = 0;
        self.synchronized_at = None;
        self.expected_encoded_len = None;
    }
}

fn validate_payload(payload: &[u8]) -> Result<(), ModemError> {
    if payload.is_empty() {
        return Err(ModemError::EmptyPayload);
    }
    if payload.len() > MAX_PAYLOAD_BYTES {
        return Err(ModemError::PayloadTooLarge {
            actual: payload.len(),
            maximum: MAX_PAYLOAD_BYTES,
        });
    }
    Ok(())
}

fn validate_sample_rate(sample_rate: u32) -> Result<(), ModemError> {
    if !(32_000..=192_000).contains(&sample_rate) || sample_rate as f32 <= FREQUENCIES[7] * 2.0 {
        return Err(ModemError::InvalidSampleRate(sample_rate));
    }
    Ok(())
}

fn validate_header(packet: &[u8]) -> Result<usize, ModemError> {
    if packet.len() < HEADER_BYTES
        || packet[0] != PROTOCOL_VERSION
        || packet[2] != crc8(&packet[..2])
    {
        return Err(ModemError::InvalidHeader);
    }
    let encoded_len = packet[1] as usize;
    if !(ECC_BYTES + 5..=MAX_BLOCK_SIZE).contains(&encoded_len) {
        return Err(ModemError::InvalidHeader);
    }
    Ok(encoded_len)
}

fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

fn crc8(data: &[u8]) -> u8 {
    data.iter().fold(0_u8, |mut crc, byte| {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
        crc
    })
}

fn samples_for(duration: f32, sample_rate: u32) -> usize {
    (duration * sample_rate as f32).round() as usize
}

fn scores_to_symbol(scores: [f32; 8]) -> u8 {
    let max_score = scores.iter().copied().fold(0.0_f32, f32::max);
    if max_score < 0.015 {
        return 0;
    }
    scores.iter().enumerate().fold(0_u8, |byte, (bit, score)| {
        if *score > max_score * 0.20 && *score > 0.015 {
            byte | (1 << bit)
        } else {
            byte
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_round_trip_and_error_correction() {
        let packet = encode_packet("Hello 世界 🌍").unwrap();
        assert_eq!(decode_packet(&packet).unwrap().0, "Hello 世界 🌍");

        let mut damaged = packet.clone();
        for index in 0..8 {
            damaged[HEADER_BYTES + index] ^= 0x5a;
        }
        let (decoded, corrected) = decode_packet(&damaged).unwrap();
        assert_eq!(decoded, "Hello 世界 🌍");
        assert_eq!(corrected, 8);
    }

    #[test]
    fn payload_limits_are_utf8_bytes() {
        assert!(encode_packet("").is_err());
        assert!(encode_packet(&"a".repeat(MAX_PAYLOAD_BYTES)).is_ok());
        assert!(matches!(
            encode_packet(&"é".repeat(MAX_PAYLOAD_BYTES)),
            Err(ModemError::PayloadTooLarge { .. })
        ));
    }

    #[test]
    fn modulation_is_finite_and_bounded() {
        let wave = encode_text("test", 48_000).unwrap();
        assert!(wave
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 1.01));
        assert!(wave.iter().any(|value| value.abs() > 0.01));
    }

    #[test]
    fn streaming_decoder_handles_sample_offsets_and_chunks() {
        for sample_rate in [44_100, 48_000] {
            let wave = encode_text("chunked audio", sample_rate).unwrap();
            let mut input = vec![0.0; (sample_rate / 137) as usize];
            input.extend_from_slice(&wave);
            let mut decoder = StreamingDecoder::new(sample_rate).unwrap();
            let mut events = Vec::new();
            for chunk in input.chunks(317) {
                events.extend(decoder.push_samples(chunk));
            }
            assert!(events.iter().any(|event| matches!(event, DecoderEvent::MessageDecoded { text, .. } if text == "chunked audio")), "events: {events:?}");
        }
    }

    #[test]
    fn streaming_decoder_tolerates_gain_and_noise() {
        let sample_rate = 48_000;
        let mut wave = encode_text("noisy channel", sample_rate).unwrap();
        let mut state = 0x1234_5678_u32;
        for sample in &mut wave {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((state >> 8) as f32 / 16_777_215.0 - 0.5) * 0.02;
            *sample = *sample * 0.45 + noise;
        }
        let mut decoder = StreamingDecoder::new(sample_rate).unwrap();
        let events = decoder.push_samples(&wave);
        assert!(events.iter().any(|event| matches!(event, DecoderEvent::MessageDecoded { text, .. } if text == "noisy channel")));
    }

    #[test]
    fn combines_matching_repeated_frames() {
        let sample_rate = 48_000;
        let symbol_samples = samples_for(TONE_DURATION + GUARD_DURATION, sample_rate);
        let first_payload_symbol = (TRAINING_PREAMBLE.len() + HEADER_BYTES) * symbol_samples;
        let mut first = encode_text("soft combining across repeated frames", sample_rate).unwrap();
        let mut second = first.clone();
        for symbol in 0..10 {
            first[first_payload_symbol + symbol * symbol_samples
                ..first_payload_symbol + (symbol + 1) * symbol_samples]
                .fill(0.0);
        }
        for symbol in 10..20 {
            second[first_payload_symbol + symbol * symbol_samples
                ..first_payload_symbol + (symbol + 1) * symbol_samples]
                .fill(0.0);
        }

        let mut decoder = StreamingDecoder::new(sample_rate).unwrap();
        let mut events = decoder.push_samples(&first);
        assert!(events
            .iter()
            .any(|event| matches!(event, DecoderEvent::DecodeFailed { .. })));
        events.extend(decoder.push_samples(&second));
        assert!(events.iter().any(|event| matches!(event, DecoderEvent::MessageDecoded { text, .. } if text == "soft combining across repeated frames")));
    }

    #[test]
    fn rejects_bad_header_and_crc() {
        let mut packet = encode_packet("crc").unwrap();
        packet[2] ^= 1;
        assert_eq!(decode_packet(&packet), Err(ModemError::InvalidHeader));
    }
}
