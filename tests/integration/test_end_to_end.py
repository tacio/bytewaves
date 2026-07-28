import numpy as np
import pytest

from bytewaves.modem import Decoder, text_to_sound


@pytest.mark.parametrize("sample_rate", [44_100, 48_000])
def test_complete_transmission_at_browser_sample_rates(sample_rate):
    wave = text_to_sound("Hello 世界 🌍", sample_rate)
    decoder = Decoder(sample_rate)
    events = []
    for start in range(0, len(wave), 317):
        events.extend(decoder.push_samples(wave[start : start + 317]))
    assert any(event["type"] == "messageDecoded" and event["text"] == "Hello 世界 🌍" for event in events)


def test_decoder_handles_noise_before_frame_and_arbitrary_chunks():
    sample_rate = 48_000
    wave = text_to_sound("offset", sample_rate)
    input_audio = np.concatenate([np.zeros(353, dtype=np.float32), wave])
    decoder = Decoder(sample_rate)
    events = []
    for start in range(0, len(input_audio), 509):
        events.extend(decoder.push_samples(input_audio[start : start + 509]))
    assert any(event["type"] == "synchronized" for event in events)
    assert any(event["type"] == "messageDecoded" and event["text"] == "offset" for event in events)


def test_decoder_reset_discards_partial_frame():
    sample_rate = 44_100
    wave = text_to_sound("reset", sample_rate)
    decoder = Decoder(sample_rate)
    decoder.push_samples(wave[: len(wave) // 2])
    decoder.reset()
    events = decoder.push_samples(wave)
    assert any(event["type"] == "messageDecoded" and event["text"] == "reset" for event in events)
