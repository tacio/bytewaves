# ByteWaves task runner (requires just: https://github.com/casey/just)
# Use `uv` when available; fall back to pip otherwise.

# Default target
default: help

# Show available recipes
help:
    @echo "ByteWaves (just)"
    @echo "================"
    @echo ""
    @echo "Recipes:"
    @echo "  install      - Install the package with development dependencies"
    @echo "  test         - Run all tests"
    @echo "  test-unit    - Run unit tests only"
    @echo "  test-int     - Run integration tests only"
    @echo "  test-perf    - Run performance tests only"
    @echo "  test-all     - Run all tests with coverage"
    @echo "  quick-test   - Run unit tests, stop on first failure"
    @echo "  benchmark    - Run performance benchmarks"
    @echo "  lint         - Run linting checks (bytewaves/ + tests/)"
    @echo "  format       - Format code with black and isort"
    @echo "  clean        - Clean up generated files"
    @echo "  coverage     - Generate HTML coverage report"
    @echo "  ci           - Run full CI pipeline locally (lint -> test-all)"
    @echo "  rust-check   - Format, lint, and test the shared Rust core"
    @echo "  wasm-build   - Generate browser bindings with wasm-pack"
    @echo "  web-install  - Install locked frontend dependencies"
    @echo "  web-test     - Run frontend unit tests"
    @echo "  web-e2e      - Run Playwright browser tests"
    @echo "  web-build    - Build the production PWA"
    @echo "  health-check - Verify import and compile"

# Install the package with development dependencies
install:
    @echo "Installing ByteWaves with development dependencies..."
    @if command -v uv &> /dev/null; then \
        uv sync --dev; \
    else \
        pip install -e ".[dev]"; \
    fi

# Run all tests
test:
    @echo "Running all tests..."
    @if command -v uv &> /dev/null; then \
        uv run pytest tests/ -v; \
    else \
        pytest tests/ -v; \
    fi

# Run unit tests only
test-unit:
    @echo "Running unit tests..."
    @if command -v uv &> /dev/null; then \
        uv run pytest tests/unit/ -v; \
    else \
        pytest tests/unit/ -v; \
    fi

# Run integration tests only
test-int:
    @echo "Running integration tests..."
    @if command -v uv &> /dev/null; then \
        uv run pytest tests/integration/ -v; \
    else \
        pytest tests/integration/ -v; \
    fi

# Run performance tests only
test-perf:
    @echo "Running performance tests..."
    @if command -v uv &> /dev/null; then \
        uv run pytest tests/performance/ -v; \
    else \
        pytest tests/performance/ -v; \
    fi

# Run all tests with coverage (HTML + terminal-missing)
test-all:
    @echo "Running all tests with coverage..."
    @if command -v uv &> /dev/null; then \
        uv run pytest tests/ --cov=bytewaves --cov-report=html --cov-report=term-missing; \
    else \
        pytest tests/ --cov=bytewaves --cov-report=html --cov-report=term-missing; \
    fi

# Run unit tests, stop on first failure (fast feedback)
quick-test:
    @echo "Running quick test suite..."
    @if command -v uv &> /dev/null; then \
        uv run pytest tests/unit/ -x; \
    else \
        pytest tests/unit/ -x; \
    fi

# Run performance benchmarks
benchmark:
    @echo "Running benchmarks..."
    @if command -v uv &> /dev/null; then \
        uv run pytest tests/performance/test_benchmarks.py -v; \
    else \
        pytest tests/performance/test_benchmarks.py -v; \
    fi

# Run linting checks (bytewaves/ + tests/)
lint:
    @echo "Running linting checks..."
    @if command -v uv &> /dev/null; then \
        uv run ruff check bytewaves/ tests/; \
    else \
        ruff check bytewaves/ tests/; \
    fi

# Run type checking (bytewaves/ + tests/)
typecheck:
    @echo "Running type checks..."
    @if command -v uv &> /dev/null; then \
        uv run ty check bytewaves/; \
    else \
        ty check bytewaves/; \
    fi

# Run linting and type checking
check: lint typecheck

# Format code with ruff
format:
    @echo "Formatting code..."
    @if command -v uv &> /dev/null; then \
        uv run ruff format bytewaves/ tests/; \
    else \
        ruff format bytewaves/ tests/; \
    fi

# Clean up generated files
clean:
    @echo "Cleaning up generated files..."
    @rm -rf build/ dist/ *.egg-info/ .coverage coverage.xml htmlcov/ .pytest_cache/ __pycache__/
    @find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    @find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Generate HTML coverage report
coverage:
    @echo "Generating coverage report..."
    @if command -v uv &> /dev/null; then \
        uv run pytest tests/ --cov=bytewaves --cov-report=html; \
    else \
        pytest tests/ --cov=bytewaves --cov-report=html; \
    fi
    @echo "Coverage report generated in htmlcov/index.html"

# Verify import and compile
health-check:
    @echo "Running health checks..."
    @if command -v uv &> /dev/null; then \
        uv run python -c "import bytewaves; print('OK Import successful')"; \
        uv run python -m py_compile bytewaves/*.py; \
    else \
        python -c "import bytewaves; print('OK Import successful')"; \
        python -m py_compile bytewaves/*.py; \
    fi
    @echo "Health check completed"

# Local CI simulation (lint -> test-all; test-all already emits coverage, so no separate coverage run)
rust-check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace

wasm-build:
    wasm-pack build crates/bytewaves-wasm --target web --out-dir ../../web/src/wasm --out-name bytewaves_wasm

web-install:
    cd web && npm ci

web-test: wasm-build
    cd web && npm test
    cd web && npm run typecheck

web-e2e: web-build
    cd web && npm run test:e2e

web-build:
    cd web && npm run build

ci: lint typecheck rust-check test-all web-test web-build web-e2e
    @echo "CI pipeline completed successfully!"
