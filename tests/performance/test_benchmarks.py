import time
import psutil
import numpy as np
import pytest
import os
from unittest.mock import patch

from bytewaves.modem import (
    text_to_sound,
    find_scores_from_audio_chunk,
    _generate_sound_from_bytes,
    AdaptiveEqualizer,
    SAMPLE_RATE,
    BYTE_DURATION,
    MAX_BLOCK_SIZE
)


class TestPerformance:
    """Performance and reliability test cases."""

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_audio_generation_performance(self):
        """Test audio generation performance."""
        test_cases = [
            ("Short", "Hi"),
            ("Medium", "Hello World!"),
            ("Long", "A" * 1000),
        ]

        for name, text in test_cases:
            initial_memory = self.get_memory_usage()

            start_time = time.time()
            audio = text_to_sound(text)
            end_time = time.time()

            final_memory = self.get_memory_usage()
            memory_used = final_memory - initial_memory

            # Performance assertions
            generation_time = end_time - start_time
            assert generation_time < 1.0  # Should complete within 1 second

            # Memory usage should be reasonable
            assert memory_used < 100  # Less than 100MB

            # Audio should be valid
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0

    def test_audio_processing_performance(self):
        """Test audio processing performance."""
        # Create test audio
        samples = int(SAMPLE_RATE * BYTE_DURATION)
        test_chunk = np.random.randn(samples).astype(np.float32)

        # Test multiple processing iterations
        num_iterations = 100
        start_time = time.time()

        for _ in range(num_iterations):
            scores = find_scores_from_audio_chunk(test_chunk)

        end_time = time.time()
        total_time = end_time - start_time

        # Should process quickly
        avg_time_per_chunk = total_time / num_iterations
        assert avg_time_per_chunk < 0.1  # Less than 100ms per chunk

    def test_equalizer_training_performance(self):
        """Test adaptive equalizer training performance."""
        equalizer = AdaptiveEqualizer(filter_len=32, mu=0.01)

        # Create training data
        signal_length = 1000
        received_signal = np.random.randn(signal_length)
        desired_signal = np.random.randn(signal_length)

        start_time = time.time()

        with patch.object(equalizer.filt, 'run'):
            equalizer.train(received_signal, desired_signal)

        end_time = time.time()
        training_time = end_time - start_time

        # Training should be reasonably fast
        assert training_time < 1.0  # Less than 1 second

    def test_memory_efficiency(self):
        """Test memory efficiency of operations."""
        initial_memory = self.get_memory_usage()

        # Perform multiple operations
        for i in range(10):
            text = f"Test message {i}"
            audio = text_to_sound(text)

            # Process the audio
            chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
            if len(audio) >= chunk_size:
                chunk = audio[:chunk_size]
                scores = find_scores_from_audio_chunk(chunk)

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 50  # Less than 50MB increase

    def test_large_data_handling(self):
        """Test handling of large data efficiently."""
        # Test with reasonably large data
        large_text = "A" * 5000

        start_time = time.time()
        audio = text_to_sound(large_text)
        end_time = time.time()

        processing_time = end_time - start_time

        # Should handle large data efficiently
        assert processing_time < 5.0  # Less than 5 seconds
        assert len(audio) > 0

    def test_concurrent_operations_performance(self):
        """Test performance with concurrent-like operations."""
        start_time = time.time()

        # Simulate multiple concurrent operations
        texts = [f"Message {i}" for i in range(20)]

        for text in texts:
            audio = text_to_sound(text)
            # Simulate processing
            if len(audio) > 0:
                chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                chunk = audio[:chunk_size]
                scores = find_scores_from_audio_chunk(chunk)

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle concurrent-like operations efficiently
        assert total_time < 10.0  # Less than 10 seconds for 20 operations

    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency."""
        # Get initial CPU times
        process = psutil.Process(os.getpid())
        initial_cpu_times = process.cpu_times()

        start_time = time.time()

        # Perform CPU-intensive operations
        for i in range(50):
            chunk = np.random.randn(1000).astype(np.float32)
            scores = find_scores_from_audio_chunk(chunk)

        end_time = time.time()

        final_cpu_times = process.cpu_times()
        cpu_time_used = (final_cpu_times.user + final_cpu_times.system) - \
                       (initial_cpu_times.user + initial_cpu_times.system)

        execution_time = end_time - start_time

        # CPU efficiency check
        if execution_time > 0:
            cpu_efficiency = cpu_time_used / execution_time
            assert cpu_efficiency < 50  # Reasonable CPU usage

    def test_scalability_with_data_size(self):
        """Test scalability as data size increases."""
        data_sizes = [10, 100, 1000, 5000]

        times = []

        for size in data_sizes:
            text = "A" * size

            start_time = time.time()
            audio = text_to_sound(text)
            end_time = time.time()

            times.append(end_time - start_time)

            # Each operation should complete
            assert len(audio) > 0

        # Check that time scales reasonably (not exponentially)
        # Time should not increase dramatically with data size
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1] if times[i-1] > 0 else 1
            size_ratio = data_sizes[i] / data_sizes[i-1]

            # Time ratio should not be much larger than size ratio
            assert ratio < size_ratio * 10  # Allow 10x time for complexity

    def test_repeated_operations_stability(self):
        """Test stability with repeated operations."""
        # Perform many repeated operations
        num_operations = 100
        errors = []

        for i in range(num_operations):
            try:
                text = f"Test {i}"
                audio = text_to_sound(text)

                if len(audio) > 0:
                    chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                    chunk = audio[:chunk_size]
                    scores = find_scores_from_audio_chunk(chunk)

            except Exception as e:
                errors.append(e)

        # Should have very few errors
        error_rate = len(errors) / num_operations
        assert error_rate < 0.1  # Less than 10% error rate

    def test_peak_memory_usage(self):
        """Test peak memory usage during intensive operations."""
        initial_memory = self.get_memory_usage()

        # Perform memory-intensive operations
        peak_memory = initial_memory

        for i in range(20):
            # Create larger and larger audio
            text = "A" * (100 * (i + 1))
            audio = text_to_sound(text)

            current_memory = self.get_memory_usage()
            peak_memory = max(peak_memory, current_memory)

            # Process the audio
            if len(audio) > 0:
                chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                for j in range(0, min(len(audio), chunk_size * 5), chunk_size):
                    chunk = audio[j:j + chunk_size]
                    if len(chunk) == chunk_size:
                        scores = find_scores_from_audio_chunk(chunk)

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        # Peak memory should be reasonable
        assert peak_memory < 200  # Less than 200MB peak
        assert memory_increase < 100  # Less than 100MB net increase

    def test_timing_consistency(self):
        """Test timing consistency across operations."""
        # Perform the same operation multiple times and check timing consistency
        text = "Consistency test"
        times = []

        for _ in range(20):
            start_time = time.time()
            audio = text_to_sound(text)
            end_time = time.time()

            times.append(end_time - start_time)

            # Process audio
            if len(audio) > 0:
                chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                chunk = audio[:chunk_size]
                scores = find_scores_from_audio_chunk(chunk)

        # Calculate timing statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        coefficient_of_variation = std_time / mean_time if mean_time > 0 else 0

        # Timing should be reasonably consistent
        assert coefficient_of_variation < 0.5  # Less than 50% variation
        assert mean_time < 1.0  # Average should be reasonable

    def test_resource_cleanup_effectiveness(self):
        """Test that resources are effectively cleaned up."""
        initial_memory = self.get_memory_usage()

        # Perform operations that allocate memory
        for i in range(10):
            large_text = "A" * 1000
            audio = text_to_sound(large_text)

            # Process in chunks
            chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
            for j in range(0, min(len(audio), chunk_size * 3), chunk_size):
                chunk = audio[j:j + chunk_size]
                if len(chunk) == chunk_size:
                    scores = find_scores_from_audio_chunk(chunk)

        # Force garbage collection
        import gc
        gc.collect()

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory

        # Memory should not grow unbounded
        assert memory_increase < 50  # Less than 50MB increase after cleanup

    def test_throughput_under_load(self):
        """Test throughput under continuous load."""
        start_time = time.time()
        operations_count = 0

        # Simulate continuous operation for a period
        test_duration = 2.0  # 2 seconds
        end_time = start_time + test_duration

        while time.time() < end_time:
            text = f"Load test {operations_count}"
            audio = text_to_sound(text)

            if len(audio) > 0:
                chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                chunk = audio[:chunk_size]
                scores = find_scores_from_audio_chunk(chunk)

            operations_count += 1

        actual_duration = time.time() - start_time
        throughput = operations_count / actual_duration

        # Should maintain reasonable throughput
        assert throughput > 5  # At least 5 operations per second
        assert operations_count > 0

    def test_battery_efficiency_simulation(self):
        """Test simulation of battery efficiency (CPU efficiency)."""
        # This is a proxy for battery efficiency through CPU usage
        process = psutil.Process(os.getpid())

        start_time = time.time()
        initial_cpu_times = process.cpu_times()

        # Perform operations
        for i in range(30):
            chunk = np.random.randn(2000).astype(np.float32)
            scores = find_scores_from_audio_chunk(chunk)

        end_time = time.time()
        final_cpu_times = process.cpu_times()

        cpu_time_used = (final_cpu_times.user + final_cpu_times.system) - \
                       (initial_cpu_times.user + initial_cpu_times.system)
        wall_time = end_time - start_time

        if wall_time > 0:
            cpu_efficiency_ratio = cpu_time_used / wall_time
            # CPU time should not be much larger than wall time
            assert cpu_efficiency_ratio < 10  # Reasonable ratio

    def test_stress_test_short_duration(self):
        """Test system under short-duration stress."""
        # Perform intensive operations for a short period
        start_time = time.time()

        for i in range(100):
            # Create various types of audio
            text = f"Stress {i}" * (i % 5 + 1)  # Varying lengths
            audio = text_to_sound(text)

            # Process multiple chunks
            chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
            for j in range(min(5, len(audio) // chunk_size)):
                start_idx = j * chunk_size
                chunk = audio[start_idx:start_idx + chunk_size]
                if len(chunk) == chunk_size:
                    scores = find_scores_from_audio_chunk(chunk)

        end_time = time.time()
        stress_duration = end_time - start_time

        # Should complete stress test in reasonable time
        assert stress_duration < 10.0  # Less than 10 seconds

    def test_long_running_stability(self):
        """Test stability during long-running operations."""
        # Simulate long-running operation
        start_time = time.time()

        for i in range(50):
            # Vary the operation to simulate real usage
            if i % 2 == 0:
                text = "Long running test"
                audio = text_to_sound(text)
            else:
                chunk = np.random.randn(1000).astype(np.float32)
                scores = find_scores_from_audio_chunk(chunk)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete without issues
        assert duration < 20.0  # Less than 20 seconds
        assert duration > 1.0   # Should take some time

    def test_memory_leak_detection(self):
        """Test for potential memory leaks."""
        # This is a basic test - in practice, you'd use more sophisticated tools
        initial_memory = self.get_memory_usage()

        # Perform operations in a loop
        for iteration in range(5):
            # Each iteration should not accumulate significant memory
            for i in range(20):
                text = f"Leak test {iteration}-{i}"
                audio = text_to_sound(text)

                if len(audio) > 0:
                    chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                    chunk = audio[:chunk_size]
                    scores = find_scores_from_audio_chunk(chunk)

            # Check memory after each iteration
            current_memory = self.get_memory_usage()
            iteration_memory_increase = current_memory - initial_memory

            # Memory increase per iteration should be bounded
            assert iteration_memory_increase < 30  # Less than 30MB per iteration

    def test_performance_regression_detection(self):
        """Test for performance regression detection."""
        # Establish baseline performance
        text = "Performance regression test"
        baseline_times = []

        # Measure baseline
        for _ in range(10):
            start_time = time.time()
            audio = text_to_sound(text)
            if len(audio) > 0:
                chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                chunk = audio[:chunk_size]
                scores = find_scores_from_audio_chunk(chunk)
            end_time = time.time()
            baseline_times.append(end_time - start_time)

        baseline_mean = np.mean(baseline_times)
        baseline_std = np.std(baseline_times)

        # Test current performance
        current_times = []
        for _ in range(10):
            start_time = time.time()
            audio = text_to_sound(text)
            if len(audio) > 0:
                chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                chunk = audio[:chunk_size]
                scores = find_scores_from_audio_chunk(chunk)
            end_time = time.time()
            current_times.append(end_time - start_time)

        current_mean = np.mean(current_times)

        # Performance should not have regressed significantly
        # Current performance should be within 3 standard deviations
        performance_threshold = baseline_mean + 3 * baseline_std
        assert current_mean < performance_threshold

    def test_resource_utilization_optimization(self):
        """Test that resource utilization is optimized."""
        # Test memory efficiency
        initial_memory = self.get_memory_usage()

        # Perform optimized operations
        texts = [f"Optimization {i}" for i in range(50)]

        for text in texts:
            audio = text_to_sound(text)
            # Only process a small portion to save resources
            if len(audio) > 0:
                chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                chunk = audio[:chunk_size]
                scores = find_scores_from_audio_chunk(chunk)

        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory

        # Memory usage should be optimized
        assert memory_used < 75  # Less than 75MB for 50 operations

    def test_reliability_under_varying_load(self):
        """Test reliability under varying load conditions."""
        load_patterns = [
            ("Light", 10, 0.1),
            ("Medium", 50, 0.05),
            ("Heavy", 100, 0.01),
        ]

        for pattern_name, num_operations, delay in load_patterns:
            start_time = time.time()

            for i in range(num_operations):
                text = f"{pattern_name} {i}"
                audio = text_to_sound(text)

                if len(audio) > 0:
                    chunk_size = int(SAMPLE_RATE * BYTE_DURATION)
                    chunk = audio[:chunk_size]
                    scores = find_scores_from_audio_chunk(chunk)

                # Simulate delay between operations
                time.sleep(delay)

            end_time = time.time()
            duration = end_time - start_time

            # Should complete all operations
            assert duration > 0
            # Duration should be roughly proportional to load
            expected_min_duration = num_operations * delay
            assert duration >= expected_min_duration * 0.8  # At least 80% of expected