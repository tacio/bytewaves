# ByteWaves - Acoustic Data Modem

ByteWaves is a Python-based acoustic modem that encodes and decodes data into sound waves using a multi-frequency shift keying (MFSK) technique. It leverages Reed-Solomon error correction to ensure data integrity even in noisy environments. This project provides a simple yet effective way to transmit data between devices using only a speaker and a microphone.

## Installation

You can install ByteWaves directly from PyPI using pip:

```sh
pip install bytewaves
```

## Usage

Once installed, you can run the modem from your terminal:

```sh
bytewaves
```

This will launch the interactive command-line interface, where you can choose to send or receive data.

## Roadmap

This roadmap outlines the future direction of the ByteWaves project, focusing on key areas of improvement to enhance its functionality, robustness, and user experience.

### Protocol Improvements

The current protocol is functional but has room for enhancement to improve its reliability and efficiency.

- **Variable-Length Packets**: Transition from fixed-length to variable-length packets to optimize data transmission for different message sizes.
- **Enhanced Synchronization**: Implement a more robust synchronization mechanism, such as a preamble with a unique pattern, to improve lock-in time and reduce false positives.
- **Data Integrity**: Add a checksum (e.g., CRC32) to each packet to verify data integrity before error correction, allowing for faster rejection of corrupted packets.

### Algorithmic Enhancements

Advancing the signal processing algorithms will significantly boost performance, especially in challenging acoustic environments.

- **Soft-Decision Decoding**: Move from the current hard-decision decoding to a soft-decision approach. This will leverage the confidence scores of each bit to make more informed error corrections, improving accuracy in noisy conditions.
- **Adaptive Equalization**: Introduce an adaptive equalization filter to compensate for frequency-dependent distortion caused by speakers, microphones, and the environment.

### Testing Framework

To ensure the reliability and stability of the modem, a comprehensive testing suite is essential.

- **Unit Tests**: Develop unit tests for all critical components, including encoding, decoding, and signal processing functions.
- **Integration Tests**: Create integration tests that simulate end-to-end data transmission under various noise levels and channel conditions.

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

- **PyPI Distribution**: Package the project for distribution on the Python Package Index (PyPI), allowing for easy installation via `pip`.
- **Dependency Management**: Streamline dependency management to ensure a smooth installation process across different platforms.