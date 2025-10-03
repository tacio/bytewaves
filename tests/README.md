# ByteWaves Testing Framework

This document describes the comprehensive testing strategy and framework for the ByteWaves acoustic modem project.

## Overview

The testing framework is designed to ensure the reliability, performance, and correctness of the acoustic modem through:

- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing component interactions
- **Performance Tests**: Ensuring system meets performance requirements
- **Edge Case Tests**: Robustness under unusual conditions

## Test Structure

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── test_equalizer.py         # AdaptiveEqualizer tests
│   ├── test_audio_generation.py  # Audio generation tests
│   ├── test_encoding_decoding.py # Encoding/decoding tests
│   ├── test_reed_solomon.py      # Error correction tests
│   ├── test_preamble_sync.py     # Synchronization tests
│   └── test_edge_cases.py        # Edge cases and error conditions
├── integration/                   # Integration tests
│   └── test_end_to_end.py        # End-to-end transmission tests
├── performance/                   # Performance and reliability tests
│   └── test_benchmarks.py        # Performance benchmarks
└── fixtures/                      # Test data and fixtures
    ├── audio_samples/            # Golden audio samples
    └── test_data/                # Test datasets
```

## Running Tests

### Quick Start

```bash
# Run all tests
./scripts/test.sh

# Or using pytest directly
pytest tests/
```

### Running Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Performance tests only
pytest tests/performance/

# Specific test file
pytest tests/unit/test_equalizer.py

# Specific test class or method
pytest tests/unit/test_equalizer.py::TestAdaptiveEqualizer::test_initialization_default_parameters
```

### Running with Coverage

```bash
# Generate coverage report
pytest tests/ --cov=bytewaves --cov-report=html

# View coverage in terminal
pytest tests/ --cov=bytewaves --cov-report=term-missing

# Generate XML coverage for CI
pytest tests/ --cov=bytewaves --cov-report=xml
```

## Test Categories

### Unit Tests

#### AdaptiveEqualizer Tests (`test_equalizer.py`)
- Initialization with different parameters
- Training with various signal conditions
- Filter application and adaptation
- Edge cases: insufficient data, mismatched signals

#### Audio Generation Tests (`test_audio_generation.py`)
- Sound generation from byte sequences
- Frequency generation accuracy
- Amplitude normalization
- Pause insertion between bytes
- Empty input handling

#### Encoding/Decoding Tests (`test_encoding_decoding.py`)
- Text to sound conversion
- CRC calculation and validation
- Frequency detection from audio chunks
- Score to byte conversion
- FFT processing and frequency analysis

#### Reed-Solomon Tests (`test_reed_solomon.py`)
- Reed-Solomon encoding with different data sizes
- Error correction capabilities
- CRC validation after decoding
- Error handling for corrupted data

#### Preamble Synchronization Tests (`test_preamble_sync.py`)
- Preamble detection in various conditions
- State machine transitions
- Buffer management
- Packet length validation

#### Edge Cases Tests (`test_edge_cases.py`)
- Boundary conditions (empty inputs, maximum sizes)
- Invalid inputs and malformed data
- Resource constraints
- System error recovery

### Integration Tests

#### End-to-End Tests (`test_end_to_end.py`)
- Complete transmission pipeline simulation
- Audio generation to byte extraction
- Preamble detection in full packets
- Cross-component data flow
- System resource usage

### Performance Tests

#### Benchmark Tests (`test_benchmarks.py`)
- Audio processing performance benchmarks
- Memory usage profiling
- Timing consistency measurements
- Throughput under load
- Resource utilization optimization

## Continuous Integration

The project uses GitHub Actions for automated testing:

- **Multi-Python Version Testing**: Tests run on Python 3.8-3.11
- **Code Coverage**: Automatic coverage reporting to Codecov
- **Linting**: Code style and quality checks
- **Security**: Dependency vulnerability scanning
- **Build Verification**: Package building and artifact upload

## Writing New Tests

### Test File Naming Convention
- Test files: `test_<component>.py`
- Test classes: `Test<Component>`
- Test methods: `test_<functionality>`

### Test Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Use mocks for external dependencies
3. **Fixtures**: Use pytest fixtures for reusable test data
4. **Assertions**: Clear, descriptive assertions
5. **Coverage**: Aim for high test coverage of critical paths

### Example Test Structure

```python
import pytest
import numpy as np
from bytewaves.modem import Component

class TestComponent:
    def test_basic_functionality(self):
        """Test basic component functionality."""
        component = Component()
        result = component.process(data)
        assert result == expected

    def test_edge_case_handling(self):
        """Test edge case handling."""
        component = Component()
        result = component.process(edge_case_data)
        assert result is not None

    @pytest.mark.parametrize("input,expected", test_cases)
    def test_parameterized_cases(self, input, expected):
        """Test multiple cases using parameterization."""
        component = Component()
        result = component.process(input)
        assert result == expected
```

## Performance Benchmarks

The performance tests establish baselines for:

- **Audio Generation**: < 1 second for typical messages
- **Audio Processing**: < 100ms per chunk
- **Memory Usage**: < 100MB for typical operations
- **Throughput**: > 5 operations per second
- **CPU Efficiency**: Reasonable CPU utilization

## Debugging Tests

### Running Tests in Verbose Mode
```bash
pytest tests/ -v -s
```

### Debugging Failed Tests
```bash
pytest tests/ -v --pdb  # Drop into debugger on failure
pytest tests/ --lf      # Stop on first failure
```

### Profiling Test Performance
```bash
pytest tests/performance/ --profile
```

## Contributing

When adding new features:

1. Write unit tests for new functionality
2. Add integration tests for component interactions
3. Include performance tests for critical paths
4. Update this documentation
5. Ensure all tests pass in CI

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Audio Device Issues**: Tests mock audio I/O, but real audio tests need hardware
3. **Memory Issues**: Performance tests monitor memory usage
4. **Timing Issues**: Some tests may be flaky on slower systems

### Getting Help

- Check the CI logs for detailed error information
- Review test output for assertion details
- Use `--verbose` flag for more detailed output
- Check coverage reports for untested code paths

## Test Data

Test fixtures and golden samples are stored in `tests/fixtures/`:

- **Audio Samples**: Reference audio for regression testing
- **Test Data**: Various datasets for different scenarios
- **Noise Profiles**: Different noise conditions for testing

This testing framework ensures the ByteWaves acoustic modem is reliable, performant, and robust across all operating conditions.