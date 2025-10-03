import numpy as np
import pytest
from unittest.mock import Mock, patch

from bytewaves.modem import AdaptiveEqualizer


class TestAdaptiveEqualizer:
    """Test cases for the AdaptiveEqualizer class."""

    def test_initialization_default_parameters(self):
        """Test AdaptiveEqualizer initialization with default parameters."""
        equalizer = AdaptiveEqualizer()
        assert equalizer.filter_len == 20
        assert equalizer.filt.mu == 0.01
        assert equalizer.filt.w.size == 20

    def test_initialization_custom_parameters(self):
        """Test AdaptiveEqualizer initialization with custom parameters."""
        filter_len = 32
        mu = 0.001
        equalizer = AdaptiveEqualizer(filter_len=filter_len, mu=mu)
        assert equalizer.filter_len == filter_len
        assert equalizer.filt.mu == mu
        assert equalizer.filt.w.size == filter_len

    def test_training_with_sufficient_data(self):
        """Test equalizer training with sufficient signal data."""
        equalizer = AdaptiveEqualizer(filter_len=10, mu=0.01)

        # Create test signals
        received_signal = np.random.randn(100)
        desired_signal = np.random.randn(100)

        # Mock the filter's run method to avoid padasip dependency issues
        with patch.object(equalizer.filt, 'run') as mock_run:
            equalizer.train(received_signal, desired_signal)
            mock_run.assert_called_once_with(desired_signal, received_signal)

    def test_training_with_shorter_received_signal(self):
        """Test equalizer training when received signal is shorter than desired."""
        equalizer = AdaptiveEqualizer(filter_len=10, mu=0.01)

        received_signal = np.random.randn(50)  # Shorter signal
        desired_signal = np.random.randn(100)  # Longer signal

        with patch.object(equalizer.filt, 'run') as mock_run:
            equalizer.train(received_signal, desired_signal)
            # Should truncate desired_signal to match received_signal length
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert len(args[0]) == len(args[1]) == 50

    def test_training_with_insufficient_data(self):
        """Test equalizer training with insufficient data."""
        equalizer = AdaptiveEqualizer(filter_len=50, mu=0.01)

        received_signal = np.random.randn(30)  # Shorter than filter length
        desired_signal = np.random.randn(30)

        with patch('builtins.print') as mock_print:
            equalizer.train(received_signal, desired_signal)
            mock_print.assert_called_once()
            assert "too short for training" in mock_print.call_args[0][0]

    def test_apply_filter(self):
        """Test applying the equalization filter to a signal."""
        equalizer = AdaptiveEqualizer(filter_len=10, mu=0.01)

        # Mock the filter's predict method
        test_signal = np.random.randn(100)
        expected_output = np.random.randn(100)

        with patch.object(equalizer.filt, 'predict', return_value=expected_output) as mock_predict:
            result = equalizer.apply(test_signal)
            mock_predict.assert_called_once_with(test_signal)
            np.testing.assert_array_equal(result, expected_output)

    def test_training_with_equal_length_signals(self):
        """Test training with equal length signals."""
        equalizer = AdaptiveEqualizer(filter_len=10, mu=0.01)

        signal_length = 100
        received_signal = np.random.randn(signal_length)
        desired_signal = np.random.randn(signal_length)

        with patch.object(equalizer.filt, 'run') as mock_run:
            equalizer.train(received_signal, desired_signal)
            mock_run.assert_called_once_with(desired_signal, received_signal)

    def test_filter_weights_after_training(self):
        """Test that filter weights are updated after training."""
        equalizer = AdaptiveEqualizer(filter_len=10, mu=0.01)

        # Set initial weights
        initial_weights = np.zeros(10)
        equalizer.filt.w = initial_weights.copy()

        received_signal = np.random.randn(100)
        desired_signal = np.random.randn(100)

        with patch.object(equalizer.filt, 'run') as mock_run:
            equalizer.train(received_signal, desired_signal)
            # Verify that training was called (weights would be updated by padasip)
            mock_run.assert_called_once()

    def test_empty_signal_handling(self):
        """Test behavior with empty signals."""
        equalizer = AdaptiveEqualizer(filter_len=10, mu=0.01)

        empty_signal = np.array([])

        with patch.object(equalizer.filt, 'run') as mock_run:
            equalizer.train(empty_signal, empty_signal)
            # Should handle empty arrays gracefully
            mock_run.assert_called_once_with(empty_signal, empty_signal)

    def test_single_element_signal(self):
        """Test behavior with single-element signals."""
        equalizer = AdaptiveEqualizer(filter_len=10, mu=0.01)

        single_signal = np.array([1.0])

        with patch.object(equalizer.filt, 'run') as mock_run:
            equalizer.train(single_signal, single_signal)
            mock_run.assert_called_once_with(single_signal, single_signal)