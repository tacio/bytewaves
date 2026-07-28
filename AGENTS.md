# AGENTS.md

Guidance for AI agents working in this repo. Read `README.md` and `CLAUDE.md` for full architecture; this file captures only what is easy to get wrong.

## Project shape

- `crates/bytewaves-core` is the protocol-v2 source of truth. `crates/bytewaves-wasm` and `crates/bytewaves-python` expose it to the browser and Python respectively.
- `bytewaves/modem.py` owns only desktop audio I/O, CLI state, and deprecated 0.1 compatibility helpers. The CLI entry point remains `bytewaves.modem:main` (`uv run bytewaves`).
- `web/` is a Vanilla TypeScript/Vite PWA. Capture runs in an AudioWorklet and decoding runs in a Web Worker.
- Wire format: per-carrier training preamble, version, encoded length, header CRC8, then `RS(DATA | CRC32 | ECC16)`. Protocol timings and frequencies live in `bytewaves-core`; never duplicate new protocol behavior in Python or TypeScript.

## Commands

Use `uv` via the `justfile` (requires `just`; recipes auto-fall back to `pip` if `uv` is absent).

```sh
uv sync --dev                  # install with dev deps (just install)
just test                      # all tests
just test-unit | test-int | test-perf   # one suite each (recipe names use these)
just lint                      # Ruff checks (bytewaves/ + tests/)
just typecheck                 # ty type checking (bytewaves/)
just format                    # Ruff formatting (applies fixes)
just ci                        # Python + Rust + web verification
just rust-check                # rustfmt + clippy + workspace tests
just web-test | web-build      # frontend verification / production PWA
```

## Testing quirks

- Python tests cover the desktop binding and compatibility layer; Rust tests are authoritative for packet, ECC, modulation, and streaming decoding behavior. Web tests use Vitest.
- Tests do **not** need real audio hardware: they import `bytewaves.modem` (which imports `sounddevice` at module load, but only playback/recording calls touch hardware) and exercise the codec logic with mocked audio I/O. Running the actual CLI (`uv run bytewaves`) does require speaker/mic.
- No real audio hardware is required; all automated acoustic tests use generated PCM.
- Tests are split into `tests/unit/`, `tests/integration/`, `tests/performance/`. Classes are `Test*`, functions `test_*` (enforced by pytest config).

## Style

- Python formatting and linting use Ruff; type checking uses `ty`. Run `just format` before `just lint`.
- `pyproject.toml` requires Python 3.11 or newer; `.python-version` pins 3.13 and CI tests 3.11–3.13.

## Verification before finishing a task

Run `just lint`, `just typecheck`, `just rust-check`, `just test-unit`, and `just web-test` (or `just ci` for full).


<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, **always prefix with `rtk`**. This reduces context
usage by 60-90% with zero behavior change. If rtk has no filter for a command,
it passes through unchanged — so it is always safe to use.

## Key Commands
```bash
# Git (59-80% savings)
rtk git status          rtk git diff            rtk git log

# Files & Search (60-75% savings)
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk find <pattern>      rtk diff <file>

# Test (90-99% savings) — shows failures only
rtk pytest tests/       rtk cargo test          rtk test <cmd>

# Build & Lint (80-90% savings) — shows errors only
rtk tsc                 rtk lint                rtk cargo build
rtk prettier --check    rtk mypy                rtk ruff check

# Analysis (70-90% savings)
rtk err <cmd>           rtk log <file>          rtk json <file>
rtk summary <cmd>       rtk deps                rtk env

# GitHub (26-87% savings)
rtk gh pr view <n>      rtk gh run list         rtk gh issue list

# Infrastructure (85% savings)
rtk docker ps           rtk kubectl get         rtk docker logs <c>

# Package managers (70-90% savings)
rtk pip list            rtk pnpm install        rtk npm run <script>
```

## Rules
- In command chains, prefix each segment: `rtk git add . && rtk git commit -m "msg"`
- For debugging, use raw command without rtk prefix
- `rtk proxy <cmd>` runs command without filtering but tracks usage
<!-- /headroom:rtk-instructions -->
