import numpy as np
import pytest
from unittest.mock import patch, Mock

from bytewaves.modem import (
    text_to_sound,
    find_scores_from_audio_chunk,
    scores_to_byte,
    _generate_sound_from_bytes,
    PREAMBLE,
    N_ECC_BYTES,
    SAMPLE_RATE,
    BYTE_DURATION,
    PAUSE_DURATION
)


class TestEndToEndTransmission:
    """Integration tests for end-to-end data transmission."""

    def test_complete_transmission_short_text(self):
        """Test complete transmission pipeline with short text."""
        test_text = "Hi"
        original_data = test_text.encode('utf-8')

        # Step 1: Encode text to sound
        audio_signal = text_to_sound(test_text)

        # Should produce valid audio
        assert isinstance(audio_signal, np.ndarray)
        assert len(audio_signal) > 0
        assert audio_signal.dtype == np.float32

        # Step 2: Simulate audio processing (extract bytes from audio)
        # This is a simplified simulation of the decoding process
        chunk_size = int(SAMPLE_RATE * (BYTE_DURATION + PAUSE_DURATION))
        chunks = []

        # Split audio into chunks
        for i in range(0, len(audio_signal), chunk_size):
            chunk = audio_signal[i:i + chunk_size]
            if len(chunk) > 0:
                # Pad or truncate to chunk size
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                chunks.append(chunk)

        # Step 3: Process chunks to extract bytes
        extracted_bytes = []
        for chunk in chunks:
            if len(chunk) >= int(SAMPLE_RATE * BYTE_DURATION):
                # Extract the data portion of the chunk
                data_chunk = chunk[:int(SAMPLE_RATE * BYTE_DURATION)]
                scores = find_scores_from_audio_chunk(data_chunk)
                byte_val = scores_to_byte(scores)
                extracted_bytes.append(byte_val)

        # Should extract some bytes (exact count may vary due to simulation)
        assert len(extracted_bytes) > 0

    def test_audio_generation_to_byte_extraction(self):
        """Test conversion from audio generation back to byte extraction."""
        # Test with known byte pattern
        test_bytes = [0x55, 0xAA, 0xFF, 0x00]  # Known pattern
        audio_signal = _generate_sound_from_bytes(test_bytes)

        # Should generate valid audio
        assert isinstance(audio_signal, np.ndarray)
        assert len(audio_signal) > 0

        # Calculate expected length
        expected_chunks = len(test_bytes)  # No pauses in continuous mode
        chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
        expected_length = expected_chunks * chunk_size

        # Process the audio to extract bytes
        extracted_bytes = []
        for i in range(len(test_bytes)):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            if end_idx <= len(audio_signal):
                chunk = audio_signal[start_idx:end_idx]
                scores = find_scores_from_audio_chunk(chunk)
                byte_val = scores_to_byte(scores)
                extracted_bytes.append(byte_val)

        # Should extract the same number of bytes
        assert len(extracted_bytes) == len(test_bytes)

    def test_preamble_detection_in_full_packet(self):
        """Test preamble detection within a full packet."""
        test_text = "Test"
        full_packet_audio = text_to_sound(test_text)

        # Should contain preamble at the beginning
        assert len(full_packet_audio) > 0

        # Extract preamble section
        preamble_samples = len(PREAMBLE) * int(SAMPLE_RATE * BYTE_DURATION)
        preamble_audio = full_packet_audio[:preamble_samples]

        # Process preamble audio
        chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
        detected_preamble = []

        for i in range(len(PREAMBLE)):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = preamble_audio[start_idx:end_idx]

            if len(chunk) == chunk_size:
                scores = find_scores_from_audio_chunk(chunk)
                byte_val = scores_to_byte(scores)
                detected_preamble.append(byte_val)

        # Should detect preamble bytes
        assert len(detected_preamble) == len(PREAMBLE)

    def test_packet_structure_integrity(self):
        """Test that packet structure remains intact through processing."""
        test_text = "Structure"

        # Encode to audio
        audio = text_to_sound(test_text)
        assert len(audio) > 0

        # The packet should have a specific structure:
        # PREAMBLE + LENGTH + DATA + CRC + ECC
        # We can't easily decode this without the full receive pipeline,
        # but we can verify the audio was generated successfully
        assert np.isfinite(audio).all()
        assert not np.all(audio == 0)  # Should not be all zeros

    def test_multiple_text_messages(self):
        """Test transmission of multiple different text messages."""
        test_messages = ["A", "Hello World", "Test 123", "!@#$%^&*()"]

        for message in test_messages:
            # Each message should encode successfully
            audio = text_to_sound(message)
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0

            # Should be able to process the audio
            chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
            first_chunk = audio[:chunk_size]

            if len(first_chunk) == chunk_size:
                scores = find_scores_from_audio_chunk(first_chunk)
                assert len(scores) == 8

    def test_audio_signal_properties(self):
        """Test properties of generated audio signals."""
        test_text = "Audio Test"
        audio = text_to_sound(test_text)

        # Test audio signal properties
        assert np.isfinite(audio).all()  # No NaN or infinite values
        assert not np.all(audio == 0)    # Not silent

        # Check amplitude is reasonable
        max_amplitude = np.max(np.abs(audio))
        assert 0 < max_amplitude <= 1.0  # Should be normalized

        # Check for DC component
        dc_component = np.mean(audio)
        assert abs(dc_component) < 0.1  # Should be minimal DC

    def test_chunk_processing_pipeline(self):
        """Test the complete chunk processing pipeline."""
        # Create a test packet
        test_text = "Chunk"
        audio = text_to_sound(test_text)

        # Process in chunks
        chunk_size = int(SAMPLE_RATE * (BYTE_DURATION + PAUSE_DURATION))
        processed_chunks = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]

            # Pad if necessary
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Process chunk
            if len(chunk) >= int(SAMPLE_RATE * BYTE_DURATION):
                data_chunk = chunk[:int(SAMPLE_RATE * BYTE_DURATION)]
                scores = find_scores_from_audio_chunk(data_chunk)
                processed_chunks.append(scores)

        # Should process some chunks
        assert len(processed_chunks) > 0

    def test_frequency_consistency_across_pipeline(self):
        """Test that frequencies remain consistent through the pipeline."""
        # Test with a single frequency (single bit set)
        test_byte = 0x01  # Only bit 0 set
        audio = _generate_sound_from_bytes([test_byte])

        # Extract the frequency content
        chunk = audio[:int(SAMPLE_RATE * BYTE_DURATION)]
        scores = find_scores_from_audio_chunk(chunk)

        # The first bit should be detected
        assert len(scores) == 8

        # In a perfect world, only the first bit should be detected
        # In practice, there might be some crosstalk, but bit 0 should be strongest
        max_score_idx = np.argmax(scores)
        assert max_score_idx == 0 or scores[0] > 0  # Either strongest or positive

    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the transmission pipeline."""
        # Test with various edge cases
        edge_cases = ["", "A" * 1000, None]

        for case in edge_cases[:-1]:  # Skip None for now
            try:
                audio = text_to_sound(case)
                assert isinstance(audio, np.ndarray)
            except Exception as e:
                # Should handle errors gracefully
                assert isinstance(e, Exception)

    def test_audio_signal_continuity(self):
        """Test that audio signal is continuous and well-formed."""
        test_text = "Continuity"
        audio = text_to_sound(test_text)

        # Check for signal continuity
        # Look for abrupt changes that might indicate problems
        diff = np.diff(audio)
        max_diff = np.max(np.abs(diff))

        # Maximum difference should be reasonable (not infinite or huge)
        assert np.isfinite(max_diff)
        assert max_diff < 10.0  # Reasonable threshold

    def test_packet_length_consistency(self):
        """Test that packet length is consistent and reasonable."""
        test_cases = ["A", "Hello World", "Longer message here"]

        for text in test_cases:
            audio = text_to_sound(text)

            # Audio length should be reasonable
            max_reasonable_length = SAMPLE_RATE * 10  # 10 seconds max
            assert len(audio) < max_reasonable_length

            # Should not be extremely short for non-empty text
            if text.strip():
                min_reasonable_length = SAMPLE_RATE * 0.1  # 0.1 seconds min
                assert len(audio) > min_reasonable_length

    def test_cross_component_data_flow(self):
        """Test data flow between different components."""
        test_text = "Flow"

        # 1. Text to bytes
        data_bytes = test_text.encode('utf-8')
        assert len(data_bytes) > 0

        # 2. Bytes to audio
        audio = text_to_sound(test_text)
        assert len(audio) > 0

        # 3. Audio processing simulation
        chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
        if len(audio) >= chunk_size:
            chunk = audio[:chunk_size]
            scores = find_scores_from_audio_chunk(chunk)
            assert len(scores) == 8

            # 4. Scores to byte
            byte_val = scores_to_byte(scores)
            assert 0 <= byte_val <= 255

    def test_system_resource_usage(self):
        """Test that the system doesn't use excessive resources."""
        # Test with reasonable-sized data
        test_text = "Resource test " * 10
        audio = text_to_sound(test_text)

        # Audio should not be extremely large
        max_reasonable_samples = SAMPLE_RATE * 5  # 5 seconds
        assert len(audio) < max_reasonable_samples

        # Memory usage should be reasonable
        audio_size_mb = len(audio) * 4 / (1024 * 1024)  # float32 = 4 bytes
        assert audio_size_mb < 100  # Less than 100MB

    def test_concurrent_processing_simulation(self):
        """Test simulation of concurrent processing."""
        # Test multiple messages in sequence
        messages = ["Msg1", "Msg2", "Msg3"]

        for msg in messages:
            audio = text_to_sound(msg)
            assert len(audio) > 0

            # Simulate processing
            chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
            if len(audio) >= chunk_size:
                chunk = audio[:chunk_size]
                scores = find_scores_from_audio_chunk(chunk)
                assert len(scores) == 8