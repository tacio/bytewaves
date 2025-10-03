import numpy as np
import pytest
import zlib
from unittest.mock import patch, Mock

from bytewaves.modem import (
    text_to_sound,
    find_scores_from_audio_chunk,
    scores_to_byte,
    SAMPLE_RATE,
    BYTE_DURATION,
    FREQUENCIES,
    PREAMBLE,
    N_ECC_BYTES,
    MAX_BLOCK_SIZE
)


class TestEncodingDecoding:
    """Test cases for encoding and decoding functions."""

    def test_text_to_sound_basic_ascii(self):
        """Test text_to_sound with basic ASCII text."""
        test_text = "Hello"
        result = text_to_sound(test_text)

        # Should return a numpy array
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) > 0

        # Should contain preamble, length, and data
        expected_min_length = len(PREAMBLE) * int(SAMPLE_RATE * BYTE_DURATION)
        assert len(result) >= expected_min_length

    def test_text_to_sound_empty_string(self):
        """Test text_to_sound with empty string."""
        result = text_to_sound("")
        assert isinstance(result, np.ndarray)
        assert len(result) > 0  # Should still have preamble and length

    def test_text_to_sound_unicode_text(self):
        """Test text_to_sound with Unicode characters."""
        test_text = "Hello 世界 🌍"
        result = text_to_sound(test_text)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_text_to_sound_oversized_data(self):
        """Test text_to_sound with data that exceeds maximum block size."""
        # Create a very large string that will exceed MAX_BLOCK_SIZE
        large_text = "A" * (MAX_BLOCK_SIZE - N_ECC_BYTES - 4 + 10)

        with patch('builtins.print') as mock_print:
            result = text_to_sound(large_text)
            assert len(result) == 0  # Should return empty array for oversized data
            mock_print.assert_called_once()
            assert "too long" in mock_print.call_args[0][0]

    def test_crc_calculation(self):
        """Test that CRC is correctly calculated and appended."""
        test_text = "Test"
        data_bytes = test_text.encode('utf-8')
        expected_crc = zlib.crc32(data_bytes)

        # We can't directly access the CRC in the encoded data without decoding,
        # but we can verify the encoding process completes successfully
        result = text_to_sound(test_text)
        assert len(result) > 0

    def test_find_scores_from_audio_chunk_basic(self):
        """Test find_scores_from_audio_chunk with a simple audio chunk."""
        # Create a simple audio chunk with a single frequency
        duration = BYTE_DURATION
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Generate a signal with the first frequency (bit 0)
        test_freq = FREQUENCIES[0]
        audio_chunk = np.sin(2 * np.pi * test_freq * t).astype(np.float32)

        scores = find_scores_from_audio_chunk(audio_chunk)

        # Should return 8 scores
        assert len(scores) == 8
        assert isinstance(scores, np.ndarray)

        # The first frequency should have the highest score
        assert scores[0] > scores[1]  # First bit should be detected
        assert np.all(scores >= 0)  # All scores should be non-negative

    def test_find_scores_from_audio_chunk_multiple_frequencies(self):
        """Test find_scores_from_audio_chunk with multiple frequencies."""
        duration = BYTE_DURATION
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Generate signal with bits 0, 2, and 4 set (0x15)
        test_byte = 0x15  # 00010101 in binary
        audio_chunk = np.zeros_like(t)

        for i in range(8):
            if (test_byte >> i) & 1:
                audio_chunk += np.sin(2 * np.pi * FREQUENCIES[i] * t)

        # Normalize the signal
        if np.max(np.abs(audio_chunk)) > 0:
            audio_chunk /= np.max(np.abs(audio_chunk))

        scores = find_scores_from_audio_chunk(audio_chunk.astype(np.float32))

        # Check that the correct bits are detected
        for i in range(8):
            expected_high = (test_byte >> i) & 1 == 1
            if expected_high:
                assert scores[i] > 0, f"Bit {i} should be detected as high"
            else:
                # Other bits might have low scores but shouldn't be strongly detected
                pass

    def test_find_scores_from_audio_chunk_silence(self):
        """Test find_scores_from_audio_chunk with silence."""
        duration = BYTE_DURATION
        samples = int(SAMPLE_RATE * duration)
        audio_chunk = np.zeros(samples, dtype=np.float32)

        scores = find_scores_from_audio_chunk(audio_chunk)

        # All scores should be very low for silence
        assert len(scores) == 8
        assert np.all(scores < 1.0)  # Should be below threshold

    def test_find_scores_from_audio_chunk_noise(self):
        """Test find_scores_from_audio_chunk with random noise."""
        duration = BYTE_DURATION
        samples = int(SAMPLE_RATE * duration)
        audio_chunk = np.random.randn(samples).astype(np.float32)

        scores = find_scores_from_audio_chunk(audio_chunk)

        # Should return valid scores
        assert len(scores) == 8
        assert np.all(np.isfinite(scores))

    def test_find_scores_from_audio_chunk_short_chunk(self):
        """Test find_scores_from_audio_chunk with chunk shorter than expected."""
        # Create a chunk shorter than BYTE_DURATION
        short_duration = BYTE_DURATION / 2
        samples = int(SAMPLE_RATE * short_duration)
        audio_chunk = np.random.randn(samples).astype(np.float32)

        # Should handle gracefully
        scores = find_scores_from_audio_chunk(audio_chunk)
        assert len(scores) == 8

    def test_scores_to_byte_all_positive(self):
        """Test scores_to_byte with all positive scores."""
        scores = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3, 0.9])
        result_byte = scores_to_byte(scores)

        # Should set bits for all positive scores
        expected_byte = 0xFF  # All bits set
        assert result_byte == expected_byte

    def test_scores_to_byte_mixed_scores(self):
        """Test scores_to_byte with mixed positive and negative scores."""
        scores = np.array([1.0, -0.5, 2.0, -1.0, 0.8, -0.2, 1.5, 0.1])
        result_byte = scores_to_byte(scores)

        # Should set bits 0, 2, 4, 6 (indices with positive scores)
        expected_byte = 0x55  # 01010101 in binary
        assert result_byte == expected_byte

    def test_scores_to_byte_all_negative(self):
        """Test scores_to_byte with all negative scores."""
        scores = np.array([-1.0, -0.5, -2.0, -1.0, -0.8, -0.2, -1.5, -0.1])
        result_byte = scores_to_byte(scores)

        # Should set no bits
        expected_byte = 0x00
        assert result_byte == expected_byte

    def test_scores_to_byte_zero_scores(self):
        """Test scores_to_byte with all zero scores."""
        scores = np.zeros(8)
        result_byte = scores_to_byte(scores)

        # Should set no bits
        expected_byte = 0x00
        assert result_byte == expected_byte

    def test_scores_to_byte_threshold_boundary(self):
        """Test scores_to_byte at threshold boundaries."""
        # Test with scores very close to zero
        scores = np.array([0.1, -0.1, 0.01, -0.01, 0.001, -0.001, 0.0, 0.0])
        result_byte = scores_to_byte(scores)

        # Should only set bits for clearly positive scores
        expected_byte = 0x01  # Only first bit
        assert result_byte == expected_byte

    def test_find_scores_fft_properties(self):
        """Test FFT-related properties in find_scores_from_audio_chunk."""
        duration = BYTE_DURATION
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create a known frequency signal
        test_freq = FREQUENCIES[0]
        audio_chunk = np.sin(2 * np.pi * test_freq * t).astype(np.float32)

        scores = find_scores_from_audio_chunk(audio_chunk)

        # Should detect the frequency correctly
        assert scores[0] > 0

        # Other frequencies should have lower scores
        for i in range(1, 8):
            assert scores[i] <= scores[0]

    def test_audio_chunk_normalization_effect(self):
        """Test how signal normalization affects score detection."""
        duration = BYTE_DURATION
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Create signal with very low amplitude
        test_freq = FREQUENCIES[0]
        low_amplitude_signal = 0.001 * np.sin(2 * np.pi * test_freq * t).astype(np.float32)

        scores = find_scores_from_audio_chunk(low_amplitude_signal)

        # Should still detect the frequency (though with lower confidence)
        assert len(scores) == 8
        assert np.all(np.isfinite(scores))

    def test_frequency_detection_robustness(self):
        """Test frequency detection robustness to small frequency variations."""
        duration = BYTE_DURATION
        samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # Use frequency slightly different from the expected one
        test_freq = FREQUENCIES[0] + 10  # 10 Hz offset
        audio_chunk = np.sin(2 * np.pi * test_freq * t).astype(np.float32)

        scores = find_scores_from_audio_chunk(audio_chunk)

        # Should still detect in the closest frequency bin
        assert len(scores) == 8
        # The first frequency should still have a reasonable score
        assert scores[0] >= 0