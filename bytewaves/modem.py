"""Desktop audio shell for the shared ByteWaves Rust modem core."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

import numpy as np
import sounddevice as sd

from ._core import MAX_PAYLOAD_BYTES
from ._core import Decoder as _CoreDecoder
from ._core import encode_text as _encode_text

SAMPLE_RATE = 44_100
BASE_FREQ = 1200.0
BYTE_DURATION = 0.020
PAUSE_DURATION = 0.020
TRANSMISSION_PAUSE = 1.0
MAX_BLOCK_SIZE = 255
N_ECC_BYTES = 16
PREAMBLE = b"\x01\x02\x04\x08\x10\x20\x40\x80"
FREQUENCIES = np.array([1200.0, 1600.0, 2100.0, 2750.0, 3600.0, 4700.0, 6100.0, 7800.0])

sending_thread: threading.Thread | None = None
receiving_thread: threading.Thread | None = None
stop_sending_flag = threading.Event()
stop_receiving_flag = threading.Event()


class _LmsFilter:
    """Small compatibility LMS filter; protocol-v2 decoding lives in Rust."""

    def __init__(self, length: int, mu: float) -> None:
        self.mu = mu
        self.w = np.zeros(length)

    def run(self, desired: np.ndarray, received: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        output = self.predict(received)
        error = desired[: len(output)] - output
        return output, error, self.w

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if not len(signal):
            return np.array([], dtype=float)
        kernel = self.w.copy()
        if not np.any(kernel):
            kernel[0] = 1.0
        return np.convolve(signal, kernel, mode="full")[: len(signal)]


class AdaptiveEqualizer:
    """Deprecated compatibility wrapper retained for callers of the 0.1 API."""

    def __init__(self, filter_len: int = 20, mu: float = 0.01) -> None:
        self.filter_len = filter_len
        self.filt = _LmsFilter(filter_len, mu)

    def train(self, received_signal: np.ndarray, desired_signal: np.ndarray) -> None:
        if len(received_signal) < len(desired_signal):
            print("Warning: Equalizer training signal is shorter than desired signal. Truncating.")
            desired_signal = desired_signal[: len(received_signal)]
        if len(received_signal) < self.filter_len:
            print("Warning: Signal too short for training the equalizer.")
            return
        self.filt.run(desired_signal, received_signal)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        return self.filt.predict(signal)


class Decoder:
    """Python-friendly streaming decoder returning dictionaries."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._decoder = _CoreDecoder(sample_rate)

    def reset(self) -> None:
        self._decoder.reset()

    def push_samples(self, samples: np.ndarray) -> list[dict[str, Any]]:
        contiguous = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        return json.loads(self._decoder.push_samples_json(contiguous))


def text_to_sound(text: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Encode UTF-8 text as one protocol-v2 frame."""
    try:
        return np.asarray(_encode_text(text, sample_rate), dtype=np.float32)
    except ValueError as error:
        print(f"Error: {error}")
        return np.array([], dtype=np.float32)


def _generate_sound_from_bytes(byte_sequence: bytes | list[int], include_pauses: bool = True) -> np.ndarray:
    """Legacy 0.1 modulation helper retained for API compatibility."""
    if not byte_sequence:
        return np.array([], dtype=np.float32)
    tone_samples = round(SAMPLE_RATE * BYTE_DURATION)
    guard_samples = round(SAMPLE_RATE * PAUSE_DURATION)
    time = np.arange(tone_samples) / SAMPLE_RATE
    parts: list[np.ndarray] = []
    for value in byte_sequence:
        wave = np.zeros(tone_samples)
        active = max(value.bit_count(), 1)
        for bit, frequency in enumerate(FREQUENCIES):
            if value & (1 << bit):
                wave += np.sin(2 * np.pi * frequency * time) / active
        parts.append(wave)
        if include_pauses:
            parts.append(np.zeros(guard_samples))
    return np.concatenate(parts).astype(np.float32)


def find_scores_from_audio_chunk(chunk: np.ndarray) -> np.ndarray:
    """Legacy carrier scoring helper; streaming receive uses ``Decoder``."""
    samples = np.nan_to_num(np.asarray(chunk, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if len(samples) < 2:
        return np.zeros(8)
    spectrum = np.abs(np.fft.rfft(samples))
    bins = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
    noise_floor = max(float(np.mean(spectrum)), 1e-9)
    return np.array([float(spectrum[np.argmin(np.abs(bins - frequency))] / noise_floor - 10.0) for frequency in FREQUENCIES])


def scores_to_byte(scores: np.ndarray) -> int:
    return sum(1 << bit for bit, score in enumerate(scores) if score > 0)


def send_loop(wave: np.ndarray) -> None:
    if wave.size == 0:
        print("Cannot send an empty message.")
        return
    print("\nBroadcasting sound... Press Enter to stop.")
    while not stop_sending_flag.is_set():
        sd.play(wave, SAMPLE_RATE)
        sd.wait()
    print("Broadcast stopped.")


def start_sending() -> None:
    global sending_thread
    text = input(f"Enter text to send (maximum {MAX_PAYLOAD_BYTES} UTF-8 bytes): ")
    wave = text_to_sound(text)
    if wave.size == 0:
        return
    stop_sending_flag.clear()
    sending_thread = threading.Thread(target=send_loop, args=(wave,))
    sending_thread.start()
    try:
        input()
    finally:
        stop_sending_flag.set()
        sending_thread.join()


def receive_loop() -> None:
    print("\nListening for data... Press Enter to stop.")
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    decoder = Decoder(SAMPLE_RATE)

    def audio_callback(indata: np.ndarray, frames: int, time_val: Any, status: Any) -> None:
        del frames, time_val
        if status:
            print(status)
        audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback, blocksize=2048):
        while not stop_receiving_flag.is_set():
            try:
                samples = audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            for event in decoder.push_samples(samples):
                event_type = event["type"]
                if event_type == "synchronized":
                    print("Signal synchronized; receiving frame...")
                elif event_type == "decodeFailed":
                    print(f"Decode failed: {event['reason']}; listening for the next repetition.")
                elif event_type == "messageDecoded":
                    print(f"\n--- SUCCESS! ---\nDecoded Text: {event['text']}\n------------------\n")
    print("Receiver stopped.")


def start_receiving() -> None:
    global receiving_thread
    stop_receiving_flag.clear()
    receiving_thread = threading.Thread(target=receive_loop)
    receiving_thread.start()
    try:
        input()
    finally:
        stop_receiving_flag.set()
        receiving_thread.join()


def main() -> None:
    print("--- ByteWaves Acoustic Data Modem (protocol v2) ---")
    while True:
        choice = input("\nChoose an option:\n1. Send text\n2. Receive data\n3. Exit\n> ").strip()
        if choice == "1":
            start_sending()
        elif choice == "2":
            start_receiving()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
