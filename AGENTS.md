# AGENTS.md

Guidance for AI agents working in this repo. Read `README.md` and `CLAUDE.md` for full architecture; this file captures only what is easy to get wrong.

## Project shape

- The entire implementation is one file: `bytewaves/modem.py` (~290 lines). `bytewaves/__init__.py` is empty. CLI entry point is `bytewaves.modem:main` (`uv run bytewaves`).
- All protocol tuning constants (sample rate, durations, `FREQUENCIES`, `N_ECC_BYTES=16`, `MAX_BLOCK_SIZE=255`, `PREAMBLE=b'\x16\x16\x88\x88'`) are module-level globals at the top of `modem.py`. Any protocol change must keep encode and decode paths consistent.
- Wire format: `PREAMBLE(4) | LENGTH(1 byte = n) | Reed-Solomon block of size n (data + CRC32(4) + ECC(16))`. Decode derives `n` from the length byte and must build `RSCodec(N_ECC_BYTES, nsize=n)` with the same `n` used on encode. Exceeding `MAX_BLOCK_SIZE=255` on either side breaks decoding silently.

## Commands

Use `uv` via the `justfile` (requires `just`; recipes auto-fall back to `pip` if `uv` is absent).

```sh
uv sync --dev                  # install with dev deps (just install)
just test                      # all tests
just test-unit | test-int | test-perf   # one suite each (recipe names use these)
just lint                      # flake8 + black --check + isort --check (bytewaves/ + tests/)
just typecheck                 # mypy type checking (bytewaves/ + tests/)
just format                    # black + isort (applies fixes)
just ci                        # lint -> test-all (test-all already emits coverage)
```

## Testing quirks

- `pyproject.toml` `[tool.pytest.ini_options].addopts` already injects `--cov=bytewaves --cov-report=term-missing --strict-markers`. So **any** `pytest` run computes coverage; there is no separate "coverage" invocation needed. `just test-all` / `scripts/test.sh` additionally emit HTML (+ XML via `scripts/test.sh`). `just ci` deliberately does **not** re-run coverage after `test-all`.
- Tests do **not** need real audio hardware: they import `bytewaves.modem` (which imports `sounddevice` at module load, but only playback/recording calls touch hardware) and exercise the codec logic with mocked audio I/O. Running the actual CLI (`uv run bytewaves`) does require speaker/mic.
- No `conftest.py` exists anywhere; `tests/fixtures/{audio_samples,test_data}` are empty. Do not assume shared fixtures — each test file is self-contained.
- Tests are split into `tests/unit/`, `tests/integration/`, `tests/performance/`. Classes are `Test*`, functions `test_*` (enforced by pytest config).

## Style

- Line length **127** across black, isort, and flake8 (not the default 88). flake8 ignores `E203`/`W503`. Run `just format` before `just lint`.
- Python: `pyproject.toml` requires `>=3.11`; `.python-version` pins 3.13. The CI matrix in `.github/workflows/ci.yml` lists 3.8–3.11 and is stale (deps like `numpy>=2.3.3` need 3.11+) — trust `pyproject.toml` for local dev.

## Verification before finishing a task

Run `just lint`, `just typecheck`, and `just test-unit` (or `just ci` for full). If you add a test, confirm it passes under `uv run pytest` and does not regress coverage.


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
