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
# pip install sounddevice numpy reedsolo

import sounddevice as sd
import numpy as np
import threading
import time
import queue
import collections
from reedsolo import RSCodec, ReedSolomonError

# --- Configuration ---
SAMPLE_RATE = 44100  # Samples per second
BYTE_DURATION = 0.02  # Duration of each sound pulse in seconds (20ms)
PAUSE_DURATION = 0.02  # Pause between sound pulses (20ms)
TRANSMISSION_PAUSE = 1.0 # Pause between full message transmissions (1s)
BLOCK_SIZE = 255       # Max total bytes for Reed-Solomon
N_ECC_BYTES = 16       # Number of error correction bytes to add
PREAMBLE = b'\x16\x16\x16' # SYN character, repeated. A short preamble for sync.

# Frequencies for the 8 bits of a byte. Chosen to be distinct and audible.
# Using a logarithmic scale to space them out perceptually.
BASE_FREQ = 1000.0
FREQUENCIES = np.logspace(np.log10(BASE_FREQ), np.log10(BASE_FREQ * 16), 8)

# --- Global State ---
sending_thread = None
stop_sending_flag = threading.Event()
receiving_thread = None
stop_receiving_flag = threading.Event()

# Initialize Reed-Solomon Codec
# The codec can correct up to N_ECC_BYTES / 2 errors in a block.
rs = RSCodec(N_ECC_BYTES)

# --- Encoding and Transmission ---

def text_to_sound(text):
    """Converts a string to a sound wave with a preamble, length, and error correction."""
    print("Encoding text to sound...")
    data_bytes = text.encode('utf-8')

    max_len = BLOCK_SIZE - N_ECC_BYTES
    if len(data_bytes) > max_len:
        print(f"Error: Data is too long ({len(data_bytes)} bytes). Maximum is {max_len} bytes.")
        return np.array([])

    try:
        encoded_bytes_with_ecc = rs.encode(data_bytes)
        print(f"Original data: {len(data_bytes)} bytes. With ECC: {len(encoded_bytes_with_ecc)} bytes.")
    except Exception as e:
        print(f"Error during Reed-Solomon encoding: {e}")
        return np.array([])

    # Construct the full packet: Preamble + Length + Data + ECC
    length_byte = bytes([len(encoded_bytes_with_ecc)])
    packet = PREAMBLE + length_byte + encoded_bytes_with_ecc
    print(f"Full packet size: {len(packet)} bytes (Preamble+Length+Data+ECC)")

    # Generate sound for each byte in the packet
    full_wave = []
    t_byte = np.linspace(0, BYTE_DURATION, int(SAMPLE_RATE * BYTE_DURATION), endpoint=False)
    silence_byte = np.zeros(int(SAMPLE_RATE * PAUSE_DURATION))

    for byte_val in packet:
        wave = np.zeros_like(t_byte)
        for i in range(8):
            if (byte_val >> i) & 1:
                wave += np.sin(2 * np.pi * FREQUENCIES[i] * t_byte)
        
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))

        full_wave.append(wave)
        full_wave.append(silence_byte)

    return np.concatenate(full_wave).astype(np.float32)

def send_loop(wave):
    """Plays the generated sound wave on a loop until stopped."""
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
    """Gets user input and starts the sending process."""
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

        input()  # Wait for user to press Enter
        stop_sending_flag.set()
        sending_thread.join()

    except KeyboardInterrupt:
        print("\nStopping sender.")
        if sending_thread:
            stop_sending_flag.set()
            sending_thread.join()

# --- Decoding and Receiving ---

def find_scores_from_audio_chunk(chunk):
    """Analyzes an audio chunk with FFT to determine scores for each of the 8 bits."""
    fft_result = np.fft.fft(chunk)
    fft_freqs = np.fft.fftfreq(len(chunk), 1/SAMPLE_RATE)
    magnitudes = np.abs(fft_result)

    positive_mask = fft_freqs > 0
    fft_freqs = fft_freqs[positive_mask]
    magnitudes = magnitudes[positive_mask]

    scores = np.zeros(8)
    noise_floor = np.mean(magnitudes)
    if noise_floor < 1e-9: noise_floor = 1e-9 # Avoid division by zero

    for i, freq in enumerate(FREQUENCIES):
        freq_index = np.argmin(np.abs(fft_freqs - freq))
        window = 5 
        start = max(0, freq_index - window)
        end = min(len(magnitudes), freq_index + window + 1)
        peak_magnitude = np.max(magnitudes[start:end])
        
        threshold_factor = 10 # Higher means less sensitive to noise.
        score = (peak_magnitude / noise_floor) - threshold_factor
        scores[i] = score
            
    return scores

def scores_to_byte(scores):
    """Converts an array of 8 bit scores to a single byte (hard decision)."""
    byte_val = 0
    for i, score in enumerate(scores):
        if score > 0:
            byte_val |= (1 << i)
    return byte_val

def attempt_decode(accumulated_scores, transmissions_count):
    """Given accumulated scores, constructs the most likely byte sequence and tries to decode it."""
    print(f"\nAttempting decode after combining {transmissions_count} transmission(s)...")
    
    best_guess_bytes = bytearray(scores_to_byte(scores) for scores in accumulated_scores)

    try:
        decoded_data, decoded_ecc = rs.decode(best_guess_bytes)
        text = decoded_data.decode('utf-8', errors='ignore')
        print(f"\n--- SUCCESS! ---")
        print(f"Decoded Text: {text}")
        print(f"Corrected {len(decoded_ecc)} errors in the final combined block.")
        print("------------------\n")
        return True
        
    except ReedSolomonError:
        print("--- DECODING FAILED ---")
        print("Too many errors to correct. Waiting for more transmissions to improve data...")
        return False
    except Exception as e:
        print(f"\nAn error occurred during final decoding: {e}")
        return False

def receive_loop():
    """Records audio, processes it using a state machine, and prints the decoded text."""
    print("\nListening for data... Press Enter to stop.")
    
    chunk_size = int(SAMPLE_RATE * (BYTE_DURATION + PAUSE_DURATION))
    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status: print(status)
        q.put(indata.copy())

    state = "HUNTING"
    preamble_buffer = collections.deque(maxlen=len(PREAMBLE))
    expected_length, byte_idx, transmissions_count = 0, 0, 0
    accumulated_scores = []
    last_signal_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=chunk_size):
        while not stop_receiving_flag.is_set():
            try:
                chunk = q.get(timeout=0.5)
                audio_signal = chunk[:, 0]

                if np.sum(audio_signal**2) < 0.01:
                    if state == "COLLECTING" and time.time() - last_signal_time > 1.0:
                        print("Signal lost, returning to hunt for preamble...")
                        state = "HUNTING"
                    continue
                
                last_signal_time = time.time()
                data_chunk = audio_signal[:int(SAMPLE_RATE * BYTE_DURATION)]
                scores = find_scores_from_audio_chunk(data_chunk)
                detected_byte = scores_to_byte(scores)

                if state == "HUNTING":
                    preamble_buffer.append(detected_byte)
                    if bytes(preamble_buffer) == PREAMBLE:
                        print("Preamble detected! Awaiting packet length...")
                        state = "AWAITING_LENGTH"

                elif state == "AWAITING_LENGTH":
                    expected_length = detected_byte
                    if 0 < expected_length <= BLOCK_SIZE:
                        print(f"Packet length received: {expected_length} bytes. Starting collection.")
                        accumulated_scores = [np.zeros(8) for _ in range(expected_length)]
                        byte_idx, transmissions_count = 0, 0
                        state = "COLLECTING"
                    else:
                        print(f"Invalid length ({expected_length}). Returning to hunt.")
                        state = "HUNTING"
                
                elif state == "COLLECTING":
                    accumulated_scores[byte_idx] += scores
                    byte_idx += 1
                    
                    if byte_idx >= expected_length:
                        transmissions_count += 1
                        if attempt_decode(accumulated_scores, transmissions_count):
                            print("Successfully decoded. Hunting for new preamble...")
                            state = "HUNTING"
                        else:
                            byte_idx = 0 # Reset to accumulate next transmission

            except queue.Empty:
                continue
            except Exception as e:
                print(f"An error in the receive loop: {e}")
                state = "HUNTING"

    print("Receiver stopped.")

def start_receiving():
    """Starts the receiving thread."""
    global receiving_thread, stop_receiving_flag
    
    stop_receiving_flag.clear()
    receiving_thread = threading.Thread(target=receive_loop)
    receiving_thread.start()
    
    input() # Wait for user to press Enter
    stop_receiving_flag.set()
    receiving_thread.join()


# --- Main Application Logic ---
def main():
    """Main function to run the CLI."""
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

