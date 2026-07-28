# ByteWaves

ByteWaves is a private, serverless acoustic text modem. It turns short UTF-8 messages into audible multi-carrier sound and decodes them through a microphone using Reed–Solomon error correction and CRC validation.

The same Rust modem core powers:

- An installable browser app compiled to WebAssembly.
- The existing Python desktop CLI through a PyO3 extension.

No message or microphone data leaves the device. The web build is a static Progressive Web App and can run offline after its first load.

## Web app

Requirements: a current Rust toolchain with the `wasm32-unknown-unknown` target, `wasm-pack`, and Node.js 20 or newer.

```sh
just web-install
just web-build
cd web && npm run dev
```

The production site is generated in `web/dist/`. GitHub Actions deploys this directory to GitHub Pages. Microphone capture requires HTTPS or localhost and is requested only after the user presses **Start listening**.

## Python CLI

Python 3.11 or newer and a Rust compiler are required when installing from source. Published releases provide platform wheels.

```sh
uv sync --dev
uv run bytewaves
```

The CLI requires a speaker and microphone. Automated tests do not require audio hardware.

## Protocol v2

Protocol v2 intentionally replaces the original 0.1 wire format. Browser and Python 0.2 clients interoperate with one another but not with 0.1 clients.

```text
TRAINING PREAMBLE | VERSION | ENCODED LENGTH | HEADER CRC8 | RS(DATA | CRC32 | ECC16)
```

- Eight audible carriers: 1.2, 1.6, 2.1, 2.75, 3.6, 4.7, 6.1, and 7.8 kHz.
- Each byte is represented by simultaneous carriers for its set bits.
- Symbols use a 20 ms tone and 20 ms guard with a short anti-click ramp.
- The physical timing is sample-rate independent; 44.1 and 48 kHz are tested.
- The training sequence establishes symbol timing and calibrates each carrier.
- Messages are limited to 235 UTF-8 bytes, followed by CRC32 and 16 Reed–Solomon bytes.
- A complete frame plus a one-second pause is repeated until sending stops.

## Architecture

```text
crates/bytewaves-core    Protocol, ECC, modulation, streaming decoder
crates/bytewaves-wasm    wasm-bindgen browser API
crates/bytewaves-python  PyO3 extension exposed as bytewaves._core
bytewaves/modem.py       sounddevice CLI and Python compatibility layer
web/                     Vanilla TypeScript, Web Audio, Worker, and PWA shell
```

The browser pre-renders outgoing samples into an `AudioBufferSourceNode`. Incoming microphone PCM is collected by an `AudioWorklet` and transferred to a Web Worker containing the WASM decoder. It does not use `SharedArrayBuffer`, server APIs, or cross-origin-isolation headers.

## Development

```sh
just rust-check       # rustfmt, clippy, and Rust tests
just test-unit        # Python unit tests
just test-int         # Python integration tests
just lint             # Python lint
just typecheck        # Python type check
just web-test         # TypeScript unit tests and type check
just web-e2e          # Playwright browser tests (installed browsers required)
just web-build        # WASM + production PWA
just ci               # everything
```

Releases use Maturin to build the Python extension and platform wheels. The browser bundle is independently deployed by the Pages workflow.

## Python API migration

Version 0.2 retains `bytewaves.modem.text_to_sound()` and the `bytewaves` CLI. The supported streaming interface is now `bytewaves.modem.Decoder`. Low-level 0.1 FFT and equalizer helpers remain only as deprecated compatibility utilities; new code should use the shared decoder.

## License

MIT
