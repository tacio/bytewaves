# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Read the `README.md` file first.

## Overview

ByteWaves is an acoustic data modem: it encodes arbitrary text into audible sound (played through a speaker) and decodes it back from a microphone. Modulation is Multi-Frequency Shift Keying (MFSK) with Reed-Solomon error correction, CRC32 integrity checking, and an adaptive LMS equalizer for channel compensation.

The entire implementation lives in a single module: `bytewaves/modem.py` (~290 lines). `bytewaves/__init__.py` is empty. The CLI entry point is `bytewaves.modem:main`.

## Commands

Use `uv` (the project's package manager). The `justfile` wraps everything and auto-falls back to `pip` if `uv` is absent.

```sh
uv sync --dev               # install with dev dependencies (or: just install)
uv run bytewaves            # launch the interactive send/receive CLI (needs real audio I/O)

just test                   # all tests, verbose
just test-unit              # unit tests only (justfile uses test-unit/test-int/test-perf)
just test-all               # all tests with HTML + term coverage
uv run pytest tests/unit/test_equalizer.py::TestAdaptiveEqualizer::test_initialization_default_parameters  # single test
uv run pytest tests/ --lf   # rerun last failures; -x stop on first; -s show prints

just lint                   # flake8 + black --check + isort --check (bytewaves/ + tests/)
just format                 # black + isort (line length 127)
just ci                     # lint + test-all (test-all already emits coverage, so no separate coverage run)
```

`./scripts/test.sh` runs unit → integration → performance with cumulative coverage and generates `htmlcov/` + `coverage.xml`. CI (`.github/workflows/ci.yml`) tests across Python 3.11–3.13, matching `pyproject.toml` (`requires-python = ">=3.11"`).

Running the modem end-to-end requires speaker/microphone hardware (`sounddevice`); the test suite exercises the codec logic without real audio.

## Conventions

- All tuning constants (sample rate, durations, frequency table, ECC byte count, preamble) are module-level globals at the top of `modem.py`. Changing the protocol means changing these on both paths consistently.
- Line length is 127 (black, isort, flake8 all configured for it); flake8 ignores E203/W503.
- Tests are split into `tests/unit/`, `tests/integration/`, `tests/performance/`; classes are `Test*`, functions `test_*`. See `tests/README.md` for fixture/parametrize conventions.


## Architecture

The protocol is a layered packet pipeline. Understanding the wire format is key — both encode and decode sides must agree on it:

```
PREAMBLE (4 bytes: 0x16 0x16 0x88 0x88) | LENGTH (1 byte = n) | [ data + CRC32(4) + ECC(16) ]
                                                                  └─ Reed-Solomon block of size n ─┘
```

**Encode path** (`text_to_sound` → `_generate_sound_from_bytes`):
1. UTF-8 encode text, append 4-byte big-endian CRC32.
2. Reed-Solomon encode (`N_ECC_BYTES=16`) → block of size `n` (must be ≤ `MAX_BLOCK_SIZE=255`).
3. Prepend `PREAMBLE` + a single length byte `n`.
4. Each byte becomes a 20ms tone: bit `i` set → add a sine at `FREQUENCIES[i]`. The 8 frequencies are logarithmically spaced from 1000Hz to 16000Hz. Tones are separated by 20ms silence.

**Decode path** is a state machine inside `receive_loop` (`HUNTING → AWAITING_LENGTH → COLLECTING`):
1. `HUNTING`: FFT each audio chunk (`find_scores_from_audio_chunk`), threshold per-frequency magnitude against the noise floor, reconstruct candidate bytes, and slide a buffer until it matches `PREAMBLE`.
2. On preamble match, train the `AdaptiveEqualizer` (padasip LMS filter) using the captured preamble audio vs. the regenerated ideal preamble signal.
3. `AWAITING_LENGTH`: read `n`, validate range, allocate per-byte score accumulators.
4. `COLLECTING`: equalize each chunk, accumulate FFT scores per byte position **across repeated transmissions** (the sender broadcasts in a loop), then `attempt_decode` runs Reed-Solomon + CRC check. Accumulating scores over multiple transmissions is the core noise-robustness mechanism — a failed decode keeps collecting more transmissions rather than discarding.

**Key invariant**: `RSCodec(N_ECC_BYTES, nsize=n)` must use the same `n` on encode and decode. Decode derives `n` from the transmitted length byte.

Send/receive each run on a background thread (`sending_thread`/`receiving_thread`) controlled by module-level `threading.Event` stop flags; the main thread blocks on `input()` to stop them.

