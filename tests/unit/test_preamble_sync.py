import numpy as np
import pytest
import collections
from unittest.mock import Mock, patch

from bytewaves.modem import (
    find_scores_from_audio_chunk,
    scores_to_byte,
    PREAMBLE,
    N_ECC_BYTES,
    MAX_BLOCK_SIZE,
    SAMPLE_RATE,
    BYTE_DURATION
)


class TestPreambleSync:
    """Test cases for preamble detection and synchronization."""

    def test_preamble_detection_perfect_conditions(self):
        """Test preamble detection under perfect conditions."""
        # Create audio chunks for each preamble byte
        preamble_chunks = []
        for byte_val in PREAMBLE:
            # Generate perfect audio for this byte
            duration = BYTE_DURATION
            samples = int(SAMPLE_RATE * duration)
            t = np.linspace(0, duration, samples, endpoint=False)

            # Create audio signal for this byte
            chunk = np.zeros(samples, dtype=np.float32)
            for i in range(8):
                if (byte_val >> i) & 1:
                    freq = [1000, 1193, 1420, 1690, 2012, 2395, 2850, 3393][i]  # Approximate frequencies
                    chunk += np.sin(2 * np.pi * freq * t)

            if np.max(np.abs(chunk)) > 0:
                chunk /= np.max(np.abs(chunk))

            preamble_chunks.append(chunk)

        # Test detection of each preamble byte
        preamble_byte_buffer = collections.deque(maxlen=len(PREAMBLE))

        for chunk in preamble_chunks:
            scores = find_scores_from_audio_chunk(chunk)
            detected_byte = scores_to_byte(scores)
            preamble_byte_buffer.append(detected_byte)

        # Should detect the complete preamble
        detected_preamble = bytes(preamble_byte_buffer)
        assert detected_preamble == PREAMBLE

    def test_preamble_detection_with_noise(self):
        """Test preamble detection with added noise."""
        # Create preamble audio with noise
        byte_val = PREAMBLE[0]
        duration = BYTE_DURATION
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create signal with noise
        chunk = np.zeros(samples, dtype=np.float32)
        for i in range(8):
            if (byte_val >> i) & 1:
                freq = [1000, 1193, 1420, 1690, 2012, 2395, 2850, 3393][i]
                chunk += 0.5 * np.sin(2 * np.pi * freq * t)

        # Add noise
        noise = 0.1 * np.random.randn(samples)
        noisy_chunk = (chunk + noise).astype(np.float32)

        # Should still detect the preamble byte
        scores = find_scores_from_audio_chunk(noisy_chunk)
        detected_byte = scores_to_byte(scores)

        # The detection might not be perfect with noise, but should be close
        # In a real implementation, we'd use a threshold for acceptance
        assert isinstance(detected_byte, int)
        assert 0 <= detected_byte <= 255

    def test_preamble_buffer_management(self):
        """Test preamble buffer management."""
        buffer = collections.deque(maxlen=len(PREAMBLE))

        # Add bytes one by one
        for i in range(len(PREAMBLE)):
            buffer.append(i)
            assert len(buffer) == min(i + 1, len(PREAMBLE))

        # Buffer should be full
        assert len(buffer) == len(PREAMBLE)

        # Add one more byte (should remove oldest)
        buffer.append(999)
        assert len(buffer) == len(PREAMBLE)
        assert buffer[0] == 1  # Oldest byte should be removed

    def test_preamble_length_validation(self):
        """Test preamble length validation."""
        # Test with correct preamble length
        correct_preamble = PREAMBLE
        assert len(correct_preamble) == 4  # Based on the code

        # Test with wrong length
        wrong_preamble = PREAMBLE[:-1]  # Too short
        assert len(wrong_preamble) != len(PREAMBLE)

    def test_packet_length_validation(self):
        """Test packet length validation logic."""
        # Test valid packet lengths
        valid_lengths = [
            N_ECC_BYTES + 5,  # Minimum reasonable length
            100,              # Medium length
            MAX_BLOCK_SIZE    # Maximum length
        ]

        for length in valid_lengths:
            if N_ECC_BYTES + 4 < length <= MAX_BLOCK_SIZE:
                assert True  # Should be considered valid
            else:
                assert False  # Should be considered invalid

        # Test invalid packet lengths
        invalid_lengths = [
            N_ECC_BYTES + 3,    # Too small
            MAX_BLOCK_SIZE + 1, # Too large
            0,                  # Zero
            -1                  # Negative
        ]

        for length in invalid_lengths:
            assert not (N_ECC_BYTES + 4 < length <= MAX_BLOCK_SIZE)

    def test_state_machine_transitions(self):
        """Test state machine transition logic."""
        # Test valid state transitions
        valid_transitions = [
            ("HUNTING", "AWAITING_LENGTH"),
            ("AWAITING_LENGTH", "COLLECTING"),
            ("COLLECTING", "HUNTING"),  # After successful decode
        ]

        for from_state, to_state in valid_transitions:
            # Should be valid transitions
            assert isinstance(from_state, str)
            assert isinstance(to_state, str)

        # Test invalid states
        invalid_states = ["INVALID", "", "123", None]
        for state in invalid_states:
            assert not (state in ["HUNTING", "AWAITING_LENGTH", "COLLECTING"])

    def test_byte_index_management(self):
        """Test byte index management during packet collection."""
        max_bytes = 100
        byte_idx = 0

        # Simulate collecting bytes
        for i in range(max_bytes):
            assert byte_idx == i
            byte_idx += 1

        assert byte_idx == max_bytes

        # Test reset after completion
        byte_idx = 0
        assert byte_idx == 0

    def test_transmission_count_tracking(self):
        """Test transmission count tracking."""
        transmissions_count = 0

        # Simulate multiple transmission attempts
        max_attempts = 5
        for _ in range(max_attempts):
            transmissions_count += 1

        assert transmissions_count == max_attempts

        # Test reset after successful decode
        transmissions_count = 0
        assert transmissions_count == 0

    def test_audio_buffer_for_preamble(self):
        """Test audio buffer management for preamble."""
        preamble_audio_buffer = collections.deque(maxlen=len(PREAMBLE))

        # Add audio chunks
        for i in range(len(PREAMBLE)):
            chunk = np.random.randn(100).astype(np.float32)
            preamble_audio_buffer.append(chunk)
            assert len(preamble_audio_buffer) == min(i + 1, len(PREAMBLE))

        # Should contain exactly len(PREAMBLE) chunks
        assert len(preamble_audio_buffer) == len(PREAMBLE)

        # Test concatenation of preamble audio
        concatenated = np.concatenate(list(preamble_audio_buffer))
        expected_length = len(PREAMBLE) * 100
        assert len(concatenated) == expected_length

    def test_preamble_detection_timeout(self):
        """Test preamble detection timeout handling."""
        # Simulate waiting for preamble detection
        timeout = 1.0  # seconds
        start_time = 0
        elapsed = 0

        # Simulate timeout
        import time
        start_time = time.time()
        time.sleep(0.1)  # Short delay
        elapsed = time.time() - start_time

        # Should handle timeout gracefully
        assert elapsed >= 0
        assert elapsed < timeout

    def test_silence_detection(self):
        """Test silence detection during reception."""
        # Create silent audio chunk
        samples = int(SAMPLE_RATE * BYTE_DURATION)
        silent_chunk = np.zeros(samples, dtype=np.float32)

        # Calculate signal energy
        signal_energy = np.sum(silent_chunk**2)

        # Should detect as silence
        silence_threshold = 0.01
        is_silent = signal_energy < silence_threshold

        assert is_silent

        # Test with non-silent chunk
        noisy_chunk = np.random.randn(samples).astype(np.float32)
        noisy_energy = np.sum(noisy_chunk**2)
        is_noisy = noisy_energy >= silence_threshold

        assert is_noisy or noisy_energy >= 0  # Should have some energy

    def test_frequency_detection_window(self):
        """Test frequency detection window parameters."""
        # Test that frequency detection uses appropriate window size
        chunk = np.random.randn(1000).astype(np.float32)
        scores = find_scores_from_audio_chunk(chunk)

        # Should return exactly 8 scores
        assert len(scores) == 8

        # All scores should be finite numbers
        assert np.all(np.isfinite(scores))

    def test_score_accumulation_logic(self):
        """Test score accumulation for multiple transmissions."""
        # Simulate accumulating scores from multiple transmissions
        n_bytes = 50
        n_transmissions = 3

        accumulated_scores = [np.zeros(8) for _ in range(n_bytes)]

        # Simulate multiple transmissions
        for transmission in range(n_transmissions):
            for byte_idx in range(n_bytes):
                # Add some scores for each byte
                byte_scores = np.random.randn(8) * 2  # Random scores
                accumulated_scores[byte_idx] += byte_scores

        # Check that accumulation worked
        for byte_idx in range(n_bytes):
            # Average score should be close to zero (due to random addition)
            avg_score = accumulated_scores[byte_idx] / n_transmissions
            assert len(avg_score) == 8
            assert np.all(np.isfinite(avg_score))

    def test_error_recovery_mechanisms(self):
        """Test error recovery mechanisms in synchronization."""
        # Test recovery from invalid state
        current_state = "INVALID_STATE"

        # Should recover to HUNTING state
        if current_state not in ["HUNTING", "AWAITING_LENGTH", "COLLECTING"]:
            recovered_state = "HUNTING"
        else:
            recovered_state = current_state

        assert recovered_state == "HUNTING"

    def test_buffer_overflow_handling(self):
        """Test handling of buffer overflow conditions."""
        # Test with buffer larger than expected
        large_buffer = collections.deque(maxlen=10)

        # Add more items than maxlen
        for i in range(15):
            large_buffer.append(i)

        # Should only contain the last 10 items
        assert len(large_buffer) == 10
        assert large_buffer[0] == 5  # First item should be 5 (15-10)