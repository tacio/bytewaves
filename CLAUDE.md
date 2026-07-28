# CLAUDE.md

Read `README.md` and `AGENTS.md` first.

## Overview

ByteWaves is an acoustic text modem with one shared Rust protocol/DSP core. It is exposed to browsers through WebAssembly and to the Python desktop CLI through PyO3.

## Architecture

- `crates/bytewaves-core`: protocol v2 framing, CRC32, Reed–Solomon, synthesis, carrier calibration, and incremental decoding.
- `crates/bytewaves-wasm`: `wasm-bindgen` API used by the Vite web app.
- `crates/bytewaves-python`: Maturin/PyO3 extension installed as `bytewaves._core`.
- `bytewaves/modem.py`: `sounddevice` playback/capture, terminal UI, and compatibility wrappers.
- `web/`: Vanilla TypeScript PWA. An AudioWorklet captures PCM and a Web Worker runs the decoder.

Protocol changes belong in `bytewaves-core`. Keep the v2 header, timings, carrier list, 235-byte UTF-8 maximum, CRC32, and 16 ECC bytes consistent by using the core rather than reimplementing them in a binding.

## Commands

```sh
uv sync --dev
just rust-check
just test-unit
just test-int
just lint
just typecheck
just web-install
just web-test
just web-build
just ci
```

The web build requires `wasm-pack` and the `wasm32-unknown-unknown` Rust target. The CLI needs real audio hardware; tests do not.

## Conventions

- Python line length is 127 and uses Ruff plus `ty`.
- Rust must pass rustfmt and Clippy with warnings denied.
- TypeScript must pass `tsc --noEmit` and Vitest.
- Do not introduce `SharedArrayBuffer`: GitHub Pages deployment intentionally works without cross-origin-isolation headers.
- Microphone permission must only be requested from the explicit Receive action.
