import numpy as np
import pytest

from bytewaves.modem import (
    _generate_sound_from_bytes,
    SAMPLE_RATE,
    BYTE_DURATION,
    PAUSE_DURATION,
    FREQUENCIES,
    BASE_FREQ
)


class TestAudioGeneration:
    """Test cases for audio generation functions."""

    def test_generate_sound_from_single_byte(self):
        """Test sound generation from a single byte."""
        test_byte = 0x55  # 01010101 in binary
        result = _generate_sound_from_bytes([test_byte])

        # Check that result is not empty
        assert len(result) > 0
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

        # Check duration is correct (1 byte duration, no pause at end)
        expected_samples = int(SAMPLE_RATE * BYTE_DURATION)
        assert len(result) == expected_samples

    def test_generate_sound_from_multiple_bytes(self):
        """Test sound generation from multiple bytes."""
        test_bytes = [0x55, 0xAA, 0xFF]
        result = _generate_sound_from_bytes(test_bytes)

        # Should have 3 byte durations + 2 pauses
        expected_samples = (3 * int(SAMPLE_RATE * BYTE_DURATION) +
                           2 * int(SAMPLE_RATE * PAUSE_DURATION))
        assert len(result) == expected_samples

    def test_generate_sound_from_all_bits_set(self):
        """Test sound generation with byte 0xFF (all bits set)."""
        test_byte = 0xFF
        result = _generate_sound_from_bytes([test_byte])

        # All frequencies should be present
        expected_samples = int(SAMPLE_RATE * BYTE_DURATION)
        assert len(result) == expected_samples

        # Check that the signal has significant energy
        assert np.max(np.abs(result)) > 0

    def test_generate_sound_from_no_bits_set(self):
        """Test sound generation with byte 0x00 (no bits set)."""
        test_byte = 0x00
        result = _generate_sound_from_bytes([test_byte])

        # Should generate silence
        expected_samples = int(SAMPLE_RATE * BYTE_DURATION)
        assert len(result) == expected_samples

        # Check that result is all zeros (or very close)
        assert np.allclose(result, 0, atol=1e-10)

    def test_generate_sound_without_pauses(self):
        """Test sound generation without pauses between bytes."""
        test_bytes = [0x55, 0xAA]
        result = _generate_sound_from_bytes(test_bytes, include_pauses=False)

        # Should only have byte durations, no pauses
        expected_samples = 2 * int(SAMPLE_RATE * BYTE_DURATION)
        assert len(result) == expected_samples

    def test_frequency_generation_accuracy(self):
        """Test that correct frequencies are generated for each bit."""
        # Test each bit individually
        for bit in range(8):
            test_byte = 1 << bit  # Only one bit set
            result = _generate_sound_from_bytes([test_byte])

            # Check that we get a sinusoidal signal at the expected frequency
            # Use FFT to verify the frequency content
            fft_result = np.fft.fft(result)
            fft_freqs = np.fft.fftfreq(len(result), 1/SAMPLE_RATE)
            magnitudes = np.abs(fft_result)

            # Find the peak frequency (should be close to expected frequency)
            positive_freqs = fft_freqs[fft_freqs > 0]
            positive_magnitudes = magnitudes[fft_freqs > 0]

            if len(positive_magnitudes) > 0:
                peak_freq_idx = np.argmax(positive_magnitudes)
                detected_freq = positive_freqs[peak_freq_idx]

                # Should be close to the expected frequency for this bit
                expected_freq = FREQUENCIES[bit]
                freq_tolerance = 50  # Hz tolerance for frequency detection

                assert abs(detected_freq - expected_freq) < freq_tolerance

    def test_amplitude_normalization(self):
        """Test that generated audio is properly normalized."""
        # Test with multiple bits set (should sum and normalize)
        test_byte = 0xFF  # All bits set
        result = _generate_sound_from_bytes([test_byte])

        # Should be normalized to maximum amplitude of 1.0
        assert np.max(np.abs(result)) <= 1.0

        # Test with single bit (should also be normalized)
        test_byte = 0x01  # Only bit 0 set
        result = _generate_sound_from_bytes([test_byte])

        # Single frequency should also be normalized
        assert 0.5 <= np.max(np.abs(result)) <= 1.0

    def test_empty_byte_sequence(self):
        """Test handling of empty byte sequence."""
        result = _generate_sound_from_bytes([])
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_sound_generation_deterministic(self):
        """Test that sound generation is deterministic."""
        test_bytes = [0x55, 0xAA, 0xFF]

        result1 = _generate_sound_from_bytes(test_bytes)
        result2 = _generate_sound_from_bytes(test_bytes)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_byte_duration_calculation(self):
        """Test that byte duration is calculated correctly."""
        test_byte = 0x55
        result = _generate_sound_from_bytes([test_byte])

        expected_samples = int(SAMPLE_RATE * BYTE_DURATION)
        assert len(result) == expected_samples

    def test_pause_duration_calculation(self):
        """Test that pause duration between bytes is correct."""
        test_bytes = [0x55, 0xAA]
        result = _generate_sound_from_bytes(test_bytes)

        # Should have byte + pause + byte
        byte_samples = int(SAMPLE_RATE * BYTE_DURATION)
        pause_samples = int(SAMPLE_RATE * PAUSE_DURATION)
        expected_samples = 2 * byte_samples + pause_samples

        assert len(result) == expected_samples

    def test_frequency_spacing(self):
        """Test that frequencies are properly spaced logarithmically."""
        # Check that frequencies are in ascending order
        assert np.all(np.diff(FREQUENCIES) > 0)

        # Check that they follow logarithmic spacing
        ratios = FREQUENCIES[1:] / FREQUENCIES[:-1]
        # Should be approximately constant ratio
        mean_ratio = np.mean(ratios)
        assert 1.2 < mean_ratio < 2.0  # Reasonable range for logarithmic spacing

    def test_base_frequency_configuration(self):
        """Test that base frequency is correctly configured."""
        # The first frequency should be close to BASE_FREQ
        assert abs(FREQUENCIES[0] - BASE_FREQ) < 1.0

    def test_sound_wave_properties(self):
        """Test basic properties of generated sound waves."""
        test_byte = 0x01  # Single frequency
        result = _generate_sound_from_bytes([test_byte])

        # Should be a valid audio signal
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

        # Should have some energy
        assert np.std(result) > 0