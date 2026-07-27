# ByteWaves - Acoustic Data Modem

ByteWaves is a Python-based acoustic modem that encodes and decodes data into sound waves using a multi-frequency shift keying (MFSK) technique. It leverages Reed-Solomon error correction to ensure data integrity even in noisy environments. This project provides a simple yet effective way to transmit data between devices using only a speaker and a microphone.

## Features

- **Multi-Frequency Shift Keying (MFSK)**: Robust modulation scheme using 8 distinct frequencies
- **Reed-Solomon Error Correction**: Advanced error correction with configurable ECC bytes
- **Adaptive Equalization**: Real-time channel compensation for audio distortion
- **CRC Validation**: Cyclic redundancy checks for data integrity verification
- **Comprehensive Testing**: Extensive test suite with unit, integration, and performance tests
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Installation

You can install ByteWaves directly from PyPI using pip:

```sh
pip install bytewaves
```

### Development Installation

For development and testing, install with development dependencies:

```sh
# Using uv (recommended)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

## Usage

Once installed, you can run the modem from your terminal:

```sh
bytewaves
```

This will launch the interactive command-line interface, where you can choose to send or receive data.

### Testing

The project includes a comprehensive testing framework:

```sh
# Run all tests
./scripts/test.sh

# Run specific test categories
just test-unit # Unit tests only
just test-int  # Integration tests only
just test-perf # Performance tests only

# Run with coverage report
just test-all
```

For detailed testing documentation, see the
[testing guide](https://github.com/tacio/bytewaves/blob/main/tests/README.md).

## Technical Specifications

### Modulation Scheme
- **Type**: Multi-Frequency Shift Keying (MFSK)
- **Frequencies**: 8 logarithmically-spaced frequencies (1000Hz base)
- **Symbol Rate**: 50 baud (20ms per byte + 20ms pause)
- **Frequency Range**: 1000Hz - 3393Hz (audible range)

### Error Correction
- **Algorithm**: Reed-Solomon error correction
- **ECC Bytes**: 16 bytes (configurable)
- **Maximum Block Size**: 255 bytes
- **Error Correction Capacity**: Up to 8 bytes per block

### Audio Parameters
- **Sample Rate**: 44,100 Hz
- **Bit Depth**: 32-bit float
- **Channels**: Mono
- **Normalization**: Automatic amplitude normalization

### Protocol Features
- **Preamble**: 4-byte synchronization pattern (`0x16168888`)
- **Packet Structure**: Preamble + Length + Data + CRC + ECC
- **CRC**: CRC32 for data integrity verification
- **Adaptive Equalization**: LMS filter for channel compensation

## Roadmap

This roadmap outlines the future direction of the ByteWaves project, focusing on key areas of improvement to enhance its functionality, robustness, and user experience.

### Protocol Improvements ✅

The current protocol implementation includes several advanced features for reliability and efficiency.

- **✅ Variable-Length Packets**: Supports variable-length packets with intelligent size optimization
- **✅ Robust Synchronization**: Preamble-based synchronization with unique 4-byte pattern for reliable lock-in
- **✅ Data Integrity**: CRC32 checksum validation for early corruption detection before error correction

### Algorithmic Enhancements ✅

Advanced signal processing algorithms are implemented to boost performance in challenging acoustic environments.

- **🟡 Soft-Decision Decoding**: Current implementation uses hard-decision decoding (planned enhancement for leveraging confidence scores)
- **✅ Adaptive Equalization**: Real-time adaptive filter using LMS algorithm to compensate for frequency-dependent distortion from speakers, microphones, and environmental factors

### Testing Framework ✅

A comprehensive testing suite has been implemented to ensure the reliability and stability of the modem.

- **✅ Unit Tests**: Comprehensive unit tests for all critical components, including encoding, decoding, and signal processing functions
- **✅ Integration Tests**: End-to-end transmission simulation under various conditions
- **✅ Performance Tests**: Benchmarking and resource usage monitoring
- **✅ Edge Case Testing**: Robustness testing for error conditions and boundary cases
- **✅ CI/CD Pipeline**: Automated testing with GitHub Actions across multiple Python versions

### Documentation

Clear and comprehensive documentation is crucial for making the project accessible and maintainable.

- **Protocol Specification**: Write a detailed document specifying the communication protocol, including packet structure, modulation scheme, and error correction methods.
- **API Reference**: Generate a complete API reference for the codebase to assist developers in extending the project.
- **User Guide**: Create a user-friendly guide with examples and tutorials on how to use the modem for sending and receiving data.

### User Interface (UI)

Improving the user interface will make the modem more intuitive and easier to use.

- **Graphical User Interface (GUI)**: Develop a GUI that provides real-time feedback, including:
  - A frequency spectrum visualizer.
  - A constellation diagram to monitor signal quality.
  - Progress indicators for data transmission.
- **Command-Line Interface (CLI) Enhancements**: Improve the existing CLI with more informative output and better command parsing.

### Packaging and Installation

To simplify distribution and installation, the project will be packaged for standard repositories.

- **✅ PyPI Distribution**: The project includes standard wheel/source builds and automated trusted publishing to PyPI.
- **Dependency Management**: Streamline dependency management to ensure a smooth installation process across different platforms.

## Development

### Setting Up Development Environment

1. **Clone the repository**:
   ```sh
   git clone https://github.com/tacio/bytewaves.git
   cd bytewaves
   ```

2. **Install development dependencies**:
   ```sh
   # Using uv (recommended)
   uv sync --dev

   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Run tests**:
   ```sh
   # Quick test run
   just quick-test

   # Full test suite
   just test-all

   # Run specific test categories
   just test-unit # Unit tests
   just test-int  # Integration tests
   just test-perf # Performance tests
   ```

### Code Quality

The project maintains high code quality standards:

```sh
# Format code
just format

# Lint code
just lint

# Run type checks
just typecheck
```

### Testing Framework

The comprehensive testing framework includes:

- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end transmission simulation
- **Performance Tests**: Benchmarking and resource monitoring
- **Edge Case Tests**: Robustness under error conditions

See the [testing guide](https://github.com/tacio/bytewaves/blob/main/tests/README.md) for detailed testing documentation.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `just ci`
6. Submit a pull request

### Releasing to PyPI

Releases use PyPI Trusted Publishing; no long-lived API token is stored in GitHub.

1. In PyPI's publishing settings, add a pending trusted publisher for project `bytewaves`, owner `tacio`, repository `bytewaves`, workflow
   `publish.yml`, and environment `pypi`.
2. Update `project.version` in `pyproject.toml` and commit the change.
3. Create and publish a GitHub release tagged `v<version>` (for example, `v0.1.0`).

The release workflow verifies that the tag matches the package version, builds both the wheel and source distribution, validates their
metadata, publishes them to PyPI, and records build provenance.

### Project Structure

```
bytewaves/
├── __init__.py           # Package initialization
├── modem.py             # Main modem implementation
└── ...

tests/
├── unit/                # Unit tests
├── integration/         # Integration tests
├── performance/         # Performance tests
└── README.md           # Testing documentation

scripts/
└── test.sh             # Test execution script

.github/workflows/
└── ci.yml              # CI/CD pipeline
```
