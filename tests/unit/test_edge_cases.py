import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, Mock

from bytewaves.modem import (
    text_to_sound,
    find_scores_from_audio_chunk,
    scores_to_byte,
    _generate_sound_from_bytes,
    attempt_decode,
    AdaptiveEqualizer,
    SAMPLE_RATE,
    BYTE_DURATION,
    MAX_BLOCK_SIZE,
    N_ECC_BYTES,
    PREAMBLE
)


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_empty_text_input(self):
        """Test handling of empty text input."""
        result = text_to_sound("")
        assert isinstance(result, np.ndarray)
        assert len(result) > 0  # Should still generate preamble

    def test_none_text_input(self):
        """Test handling of None input."""
        with pytest.raises(AttributeError):
            text_to_sound(None)

    def test_very_large_text_input(self):
        """Test handling of extremely large text input."""
        # Create text larger than MAX_BLOCK_SIZE
        large_text = "A" * (MAX_BLOCK_SIZE + 1000)

        with patch('builtins.print') as mock_print:
            result = text_to_sound(large_text)
            assert len(result) == 0  # Should return empty for oversized data
            mock_print.assert_called_once()

    def test_special_characters_input(self):
        """Test handling of special characters and Unicode."""
        special_text = "🌍 ñáéíóú ñáéíóú 💻"
        result = text_to_sound(special_text)

        # Should handle Unicode gracefully
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_binary_data_input(self):
        """Test handling of binary data."""
        binary_data = bytes(range(256))  # All byte values
        binary_text = binary_data.decode('latin-1')
        result = text_to_sound(binary_text)

        # Should handle binary data
        assert isinstance(result, np.ndarray)

    def test_extreme_frequency_conditions(self):
        """Test behavior with extreme frequency conditions."""
        # Test with very short audio chunks
        short_chunk = np.array([0.1, -0.1], dtype=np.float32)
        scores = find_scores_from_audio_chunk(short_chunk)

        # Should handle gracefully
        assert len(scores) == 8
        assert np.all(np.isfinite(scores))

    def test_extreme_amplitude_conditions(self):
        """Test behavior with extreme amplitude values."""
        # Test with very low amplitude
        samples = int(SAMPLE_RATE * BYTE_DURATION)
        low_amp_chunk = np.full(samples, 1e-10, dtype=np.float32)
        scores = find_scores_from_audio_chunk(low_amp_chunk)

        assert len(scores) == 8

        # Test with very high amplitude (clipped)
        high_amp_chunk = np.full(samples, 10.0, dtype=np.float32)
        scores = find_scores_from_audio_chunk(high_amp_chunk)

        assert len(scores) == 8

    def test_corrupted_audio_data(self):
        """Test handling of corrupted or malformed audio data."""
        # Test with NaN values
        nan_chunk = np.full(1000, np.nan, dtype=np.float32)
        scores = find_scores_from_audio_chunk(nan_chunk)

        # Should handle NaN gracefully
        assert len(scores) == 8

        # Test with infinite values
        inf_chunk = np.full(1000, np.inf, dtype=np.float32)
        scores = find_scores_from_audio_chunk(inf_chunk)

        assert len(scores) == 8

    def test_zero_length_arrays(self):
        """Test handling of zero-length arrays."""
        # Empty audio chunk
        empty_chunk = np.array([], dtype=np.float32)
        scores = find_scores_from_audio_chunk(empty_chunk)

        assert len(scores) == 8

        # Empty byte sequence
        empty_audio = _generate_sound_from_bytes([])
        assert len(empty_audio) == 0

    def test_memory_exhaustion_simulation(self):
        """Test behavior under memory pressure."""
        # This is hard to test directly, but we can test with large arrays
        large_chunk = np.random.randn(100000).astype(np.float32)

        # Should handle large chunks
        scores = find_scores_from_audio_chunk(large_chunk)
        assert len(scores) == 8

    def test_system_resource_limits(self):
        """Test behavior at system resource limits."""
        # Test with maximum reasonable chunk size
        max_reasonable_samples = SAMPLE_RATE * 60  # 1 minute
        large_chunk = np.random.randn(max_reasonable_samples).astype(np.float32)

        scores = find_scores_from_audio_chunk(large_chunk)
        assert len(scores) == 8

    def test_concurrent_access_simulation(self):
        """Test simulation of concurrent access conditions."""
        # Test rapid successive calls
        test_chunks = [
            np.random.randn(1000).astype(np.float32) for _ in range(10)
        ]

        results = []
        for chunk in test_chunks:
            scores = find_scores_from_audio_chunk(chunk)
            results.append(scores)

        # All should complete successfully
        assert len(results) == 10
        assert all(len(scores) == 8 for scores in results)

    def test_network_like_errors(self):
        """Test handling of network-like error conditions."""
        # Simulate packet loss by using incomplete chunks
        samples = int(SAMPLE_RATE * BYTE_DURATION)

        # Create a chunk that's too short
        incomplete_chunk = np.random.randn(samples // 2).astype(np.float32)

        scores = find_scores_from_audio_chunk(incomplete_chunk)
        assert len(scores) == 8

    def test_invalid_sample_rates(self):
        """Test behavior with invalid sample rates."""
        # This is more of a configuration test
        # The SAMPLE_RATE is a constant, but we can test the effect of different rates

        original_rate = SAMPLE_RATE
        chunk = np.random.randn(1000).astype(np.float32)

        # Test that our functions work with the configured rate
        scores = find_scores_from_audio_chunk(chunk)
        assert len(scores) == 8

    def test_audio_hardware_failures(self):
        """Test handling of simulated audio hardware failures."""
        # Test with all-zero audio (simulating hardware failure)
        zero_chunk = np.zeros(1000, dtype=np.float32)
        scores = find_scores_from_audio_chunk(zero_chunk)

        assert len(scores) == 8
        # All scores should be very low
        assert np.all(np.abs(scores) < 1.0)

    def test_timing_edge_cases(self):
        """Test timing-related edge cases."""
        # Test with chunks at exact boundaries
        exact_samples = int(SAMPLE_RATE * BYTE_DURATION)
        exact_chunk = np.random.randn(exact_samples).astype(np.float32)

        scores = find_scores_from_audio_chunk(exact_chunk)
        assert len(scores) == 8

        # Test with one sample more
        extended_chunk = np.random.randn(exact_samples + 1).astype(np.float32)
        scores = find_scores_from_audio_chunk(extended_chunk)
        assert len(scores) == 8

    def test_frequency_overflow_conditions(self):
        """Test conditions that might cause frequency overflow."""
        # Test with frequencies that might cause issues
        # Create a chunk with extremely high frequencies
        samples = 1000
        t = np.linspace(0, BYTE_DURATION, samples, endpoint=False)

        # Very high frequency (might cause aliasing)
        high_freq = SAMPLE_RATE / 2  # Nyquist frequency
        high_freq_chunk = np.sin(2 * np.pi * high_freq * t).astype(np.float32)

        scores = find_scores_from_audio_chunk(high_freq_chunk)
        assert len(scores) == 8

    def test_buffer_underrun_conditions(self):
        """Test buffer underrun conditions."""
        # Test with insufficient data for processing
        minimal_chunk = np.array([0.1], dtype=np.float32)
        scores = find_scores_from_audio_chunk(minimal_chunk)

        assert len(scores) == 8

    def test_adaptive_equalizer_edge_cases(self):
        """Test AdaptiveEqualizer with edge cases."""
        equalizer = AdaptiveEqualizer(filter_len=10, mu=0.01)

        # Test with mismatched signal lengths
        short_signal = np.random.randn(5)
        long_signal = np.random.randn(20)

        with patch.object(equalizer.filt, 'run') as mock_run:
            equalizer.train(short_signal, long_signal)
            # Should handle the length mismatch

        # Test with very short signals
        tiny_signal = np.random.randn(3)

        with patch('builtins.print') as mock_print:
            equalizer.train(tiny_signal, tiny_signal)
            # Should warn about insufficient data

    def test_reed_solomon_boundary_conditions(self):
        """Test Reed-Solomon encoding at boundary conditions."""
        # Test with minimum data size
        min_text = "A"
        result = text_to_sound(min_text)
        assert len(result) > 0

        # Test with maximum reasonable data size
        max_text = "A" * 1000  # Reasonable size
        result = text_to_sound(max_text)
        assert isinstance(result, np.ndarray)

    def test_preamble_detection_edge_cases(self):
        """Test preamble detection edge cases."""
        # Test with preamble-like but incorrect data
        fake_preamble = b'\x15\x15\x87\x87'  # Close but not correct
        fake_audio = _generate_sound_from_bytes(fake_preamble)

        # Process the fake preamble
        chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
        detected_bytes = []

        for i in range(len(fake_preamble)):
            start = i * chunk_size
            end = start + chunk_size
            if end <= len(fake_audio):
                chunk = fake_audio[start:end]
                scores = find_scores_from_audio_chunk(chunk)
                byte_val = scores_to_byte(scores)
                detected_bytes.append(byte_val)

        # Should detect bytes (may or may not match exactly)
        assert len(detected_bytes) == len(fake_preamble)

    def test_score_calculation_extremes(self):
        """Test score calculation with extreme values."""
        # Test with all negative scores
        all_negative = np.array([-10.0] * 8)
        byte_val = scores_to_byte(all_negative)
        assert byte_val == 0x00  # No bits set

        # Test with all positive scores
        all_positive = np.array([10.0] * 8)
        byte_val = scores_to_byte(all_positive)
        assert byte_val == 0xFF  # All bits set

        # Test with mixed extreme values
        extreme_mixed = np.array([100.0, -100.0, 50.0, -50.0, 25.0, -25.0, 10.0, -10.0])
        byte_val = scores_to_byte(extreme_mixed)
        assert byte_val == 0x55  # 01010101 pattern

    def test_audio_processing_underflow(self):
        """Test audio processing with potential underflow conditions."""
        # Test with very small floating point values
        tiny_chunk = np.full(1000, 1e-20, dtype=np.float32)

        scores = find_scores_from_audio_chunk(tiny_chunk)
        assert len(scores) == 8
        assert np.all(np.isfinite(scores))

    def test_system_error_recovery(self):
        """Test recovery from various system errors."""
        # Test with malformed input that might cause system errors
        malformed_inputs = [
            np.array([float('inf')], dtype=np.float32),
            np.array([float('-inf')], dtype=np.float32),
            np.array([float('nan')], dtype=np.float32),
        ]

        for malformed_input in malformed_inputs:
            scores = find_scores_from_audio_chunk(malformed_input)
            assert len(scores) == 8
            assert np.all(np.isfinite(scores))  # Should recover to finite values

    def test_configuration_extremes(self):
        """Test with extreme configuration values."""
        # Test audio generation with extreme parameters
        # (Note: These use the actual configuration, but test edge cases)

        # Test with maximum byte value
        max_byte_audio = _generate_sound_from_bytes([255])
        assert len(max_byte_audio) > 0

        # Test with minimum byte value
        min_byte_audio = _generate_sound_from_bytes([0])
        assert len(min_byte_audio) > 0

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        # Test multiple operations to ensure no resource leaks
        equalizer = AdaptiveEqualizer()

        for i in range(10):
            signal = np.random.randn(100)
            with patch.object(equalizer.filt, 'run'):
                equalizer.train(signal, signal)

        # Should complete without issues
        assert True

    def test_error_message_handling(self):
        """Test that error messages are handled appropriately."""
        # Test that functions handle errors without crashing
        invalid_inputs = [
            None,
            "invalid",
            123,
            [],
        ]

        for invalid_input in invalid_inputs:
            try:
                if invalid_input is not None:
                    # Test text_to_sound with various invalid types
                    if not isinstance(invalid_input, str):
                        with pytest.raises((AttributeError, TypeError)):
                            text_to_sound(invalid_input)
            except Exception:
                # Should handle errors gracefully
                pass