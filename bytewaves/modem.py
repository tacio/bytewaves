# acoustic_modem.py
#
# A Python script to encode and decode arbitrary data into sound waves.
# It uses a simple multi-frequency shift keying (MFSK) approach, where each
# bit of a byte corresponds to a specific frequency. If a bit is set, its
# corresponding frequency is included in the sound played for that byte.
#
# It also includes Reed-Solomon error correction and intelligently combines
# repeated transmissions to improve decoding accuracy in noisy environments.
#
# Dependencies:
# pip install sounddevice numpy reedsolo padasip

import sounddevice as sd
import numpy as np
import threading
import time
import queue
import collections
import zlib
import padasip as pa
from reedsolo import RSCodec, ReedSolomonError

# --- Configuration ---
SAMPLE_RATE = 44100  # Samples per second
BYTE_DURATION = 0.02  # Duration of each sound pulse in seconds (20ms)
PAUSE_DURATION = 0.02  # Pause between sound pulses (20ms)
TRANSMISSION_PAUSE = 1.0 # Pause between full message transmissions (1s)
MAX_BLOCK_SIZE = 255   # Max total bytes for Reed-Solomon
N_ECC_BYTES = 16       # Number of error correction bytes to add
PREAMBLE = b'\x16\x16\x88\x88' # A more robust 4-byte preamble for sync

# Frequencies for the 8 bits of a byte. Chosen to be distinct and audible.
# Using a logarithmic scale to space them out perceptually.
BASE_FREQ = 1000.0
FREQUENCIES = np.logspace(np.log10(BASE_FREQ), np.log10(BASE_FREQ * 16), 8)

# --- Global State ---
sending_thread = None
stop_sending_flag = threading.Event()
receiving_thread = None
stop_receiving_flag = threading.Event()


# --- Adaptive Equalizer ---
class AdaptiveEqualizer:
    """A wrapper for an adaptive filter to compensate for channel distortion."""
    def __init__(self, filter_len=20, mu=0.01):
        self.filter_len = filter_len
        self.filt = pa.filters.FilterLMS(n=filter_len, mu=mu, w="zeros")

    def train(self, received_signal, desired_signal):
        if len(received_signal) < len(desired_signal):
            print(f"Warning: Equalizer training signal is shorter than desired signal. Truncating.")
            desired_signal = desired_signal[:len(received_signal)]
        if len(received_signal) < self.filter_len:
            print("Warning: Signal too short for training the equalizer.")
            return
        self.filt.run(desired_signal, received_signal)
        print(f"Equalizer trained. Final weights: {self.filt.w}")

    def apply(self, signal):
        return self.filt.predict(signal)


def _generate_sound_from_bytes(byte_sequence, include_pauses=True):
    """Generates a sound wave from a sequence of bytes."""
    full_wave = []
    t_byte = np.linspace(0, BYTE_DURATION, int(SAMPLE_RATE * BYTE_DURATION), endpoint=False)
    silence_byte = np.zeros(int(SAMPLE_RATE * PAUSE_DURATION))

    for byte_val in byte_sequence:
        wave = np.zeros_like(t_byte)
        for i in range(8):
            if (byte_val >> i) & 1:
                wave += np.sin(2 * np.pi * FREQUENCIES[i] * t_byte)
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        full_wave.append(wave)
        if include_pauses:
            full_wave.append(silence_byte)

    return np.concatenate(full_wave).astype(np.float32)


# --- Encoding and Transmission ---

def text_to_sound(text):
    """Converts a string to a sound wave with a preamble, length, CRC, and error correction."""
    print("Encoding text to sound...")
    data_bytes = text.encode('utf-8')
    crc = zlib.crc32(data_bytes)
    data_with_crc = data_bytes + crc.to_bytes(4, 'big')
    n = len(data_with_crc) + N_ECC_BYTES

    if n > MAX_BLOCK_SIZE:
        max_data_len = MAX_BLOCK_SIZE - N_ECC_BYTES - 4
        print(f"Error: Data is too long ({len(data_bytes)} bytes). Maximum is {max_data_len} bytes.")
        return np.array([])

    try:
        rs_codec = RSCodec(N_ECC_BYTES, nsize=n)
        encoded_message = rs_codec.encode(data_with_crc)
    except Exception as e:
        print(f"Error during Reed-Solomon encoding: {e}")
        return np.array([])

    length_byte = bytes([n])
    packet = PREAMBLE + length_byte + encoded_message
    print(f"Full packet size: {len(packet)} bytes (Preamble+Length+Data+CRC+ECC)")
    return _generate_sound_from_bytes(packet)

def send_loop(wave):
    if wave.size == 0:
        print("Cannot send empty wave.")
        return
    print("\nBroadcasting sound... Press Enter to stop.")
    while not stop_sending_flag.is_set():
        sd.play(wave, SAMPLE_RATE)
        sd.wait()
        stop_sending_flag.wait(TRANSMISSION_PAUSE)
    print("Broadcast stopped.")

def start_sending():
    global sending_thread, stop_sending_flag
    try:
        text = input("Enter text to send: ")
        if not text:
            print("Input is empty.")
            return
        wave = text_to_sound(text)
        stop_sending_flag.clear()
        sending_thread = threading.Thread(target=send_loop, args=(wave,))
        sending_thread.start()
        input()
        stop_sending_flag.set()
        sending_thread.join()
    except KeyboardInterrupt:
        print("\nStopping sender.")
        if sending_thread:
            stop_sending_flag.set()
            sending_thread.join()

# --- Decoding and Receiving ---

def find_scores_from_audio_chunk(chunk):
    fft_result = np.fft.fft(chunk)
    fft_freqs = np.fft.fftfreq(len(chunk), 1/SAMPLE_RATE)
    magnitudes = np.abs(fft_result)
    positive_mask = fft_freqs > 0
    fft_freqs = fft_freqs[positive_mask]
    magnitudes = magnitudes[positive_mask]
    scores = np.zeros(8)
    noise_floor = np.mean(magnitudes)
    if noise_floor < 1e-9: noise_floor = 1e-9
    for i, freq in enumerate(FREQUENCIES):
        freq_index = np.argmin(np.abs(fft_freqs - freq))
        window = 5
        start = max(0, freq_index - window)
        end = min(len(magnitudes), freq_index + window + 1)
        peak_magnitude = np.max(magnitudes[start:end])
        threshold_factor = 10
        score = (peak_magnitude / noise_floor) - threshold_factor
        scores[i] = score
    return scores

def scores_to_byte(scores):
    byte_val = 0
    for i, score in enumerate(scores):
        if score > 0:
            byte_val |= (1 << i)
    return byte_val

def attempt_decode(accumulated_scores, n, transmissions_count):
    print(f"\nAttempting decode on {n}-byte packet after {transmissions_count} transmission(s)...")
    best_guess_bytes = bytearray(scores_to_byte(scores) for scores in accumulated_scores)
    try:
        rs_codec = RSCodec(N_ECC_BYTES, nsize=n)
        decoded_message, _ = rs_codec.decode(best_guess_bytes)
        received_crc_bytes = decoded_message[-4:]
        received_data = decoded_message[:-4]
        received_crc = int.from_bytes(received_crc_bytes, 'big')
        expected_crc = zlib.crc32(received_data)
        if received_crc != expected_crc:
            print("--- DECODING FAILED: CRC MISMATCH ---")
            return False
        text = received_data.decode('utf-8', errors='ignore')
        print(f"\n--- SUCCESS! ---\nDecoded Text: {text}\n------------------\n")
        return True
    except ReedSolomonError:
        print("--- DECODING FAILED: REED-SOLOMON ---")
        return False
    except Exception as e:
        print(f"\nAn error occurred during final decoding: {e}")
        return False

def receive_loop():
    print("\nListening for data... Press Enter to stop.")
    chunk_size = int(SAMPLE_RATE * (BYTE_DURATION + PAUSE_DURATION))
    q = queue.Queue()
    def audio_callback(indata, frames, time, status):
        if status: print(status)
        q.put(indata.copy())

    state = "HUNTING"
    preamble_byte_buffer = collections.deque(maxlen=len(PREAMBLE))
    preamble_audio_buffer = collections.deque(maxlen=len(PREAMBLE))
    byte_idx, transmissions_count = 0, 0

    equalizer = AdaptiveEqualizer(filter_len=40)
    ideal_preamble_signal = _generate_sound_from_bytes(PREAMBLE, include_pauses=False)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=chunk_size):
        while not stop_receiving_flag.is_set():
            try:
                chunk = q.get(timeout=0.5)
                raw_audio_signal = chunk[:, 0]
                data_chunk = raw_audio_signal[:int(SAMPLE_RATE * BYTE_DURATION)]

                if np.sum(raw_audio_signal**2) < 0.01:
                    continue

                if state == "HUNTING":
                    scores = find_scores_from_audio_chunk(data_chunk)
                    detected_byte = scores_to_byte(scores)
                    preamble_byte_buffer.append(detected_byte)
                    preamble_audio_buffer.append(data_chunk)
                    if bytes(preamble_byte_buffer) == PREAMBLE:
                        print("Preamble detected! Training equalizer...")
                        received_preamble_signal = np.concatenate(list(preamble_audio_buffer))
                        equalizer.train(received_preamble_signal, ideal_preamble_signal)
                        state = "AWAITING_LENGTH"
                        print("Equalizer trained. Awaiting packet length...")

                elif state in ["AWAITING_LENGTH", "COLLECTING"]:
                    equalized_chunk = equalizer.apply(data_chunk)
                    scores = find_scores_from_audio_chunk(equalized_chunk)
                    detected_byte = scores_to_byte(scores)

                    if state == "AWAITING_LENGTH":
                        expected_packet_size_n = detected_byte
                        if N_ECC_BYTES + 4 < expected_packet_size_n <= MAX_BLOCK_SIZE:
                            print(f"Packet length received: {expected_packet_size_n} bytes.")
                            accumulated_scores = [np.zeros(8) for _ in range(expected_packet_size_n)]
                            byte_idx, transmissions_count = 0, 0
                            state = "COLLECTING"
                        else:
                            print(f"Invalid length ({expected_packet_size_n}). Returning to hunt.")
                            state = "HUNTING"

                    elif state == "COLLECTING":
                        accumulated_scores[byte_idx] += scores
                        byte_idx += 1
                        if byte_idx >= expected_packet_size_n:
                            transmissions_count += 1
                            if attempt_decode(accumulated_scores, expected_packet_size_n, transmissions_count):
                                state = "HUNTING"
                            else:
                                byte_idx = 0
            except queue.Empty:
                continue
            except Exception as e:
                print(f"An error in the receive loop: {e}")
                state = "HUNTING"
    print("Receiver stopped.")

def start_receiving():
    global receiving_thread, stop_receiving_flag
    stop_receiving_flag.clear()
    receiving_thread = threading.Thread(target=receive_loop)
    receiving_thread.start()
    input()
    stop_receiving_flag.set()
    receiving_thread.join()

# --- Main Application Logic ---
def main():
    print("--- Acoustic Data Modem ---")
    while True:
        choice = input("\nChoose an option:\n1. Send text\n2. Receive data\n3. Exit\n> ").strip()
        if choice == '1':
            start_sending()
        elif choice == '2':
            start_receiving()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    print("Goodbye!")

if __name__ == '__main__':
    main()