.PHONY: help install test test-unit test-integration test-performance test-all lint format clean coverage docs

# Default target
help:
	@echo "ByteWaves Testing Framework"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install the package with development dependencies"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-int      - Run integration tests only"
	@echo "  test-perf     - Run performance tests only"
	@echo "  test-all      - Run all tests with coverage"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black and isort"
	@echo "  clean         - Clean up generated files"
	@echo "  coverage      - Generate coverage report"
	@echo "  docs          - Generate documentation"
	@echo "  ci            - Run full CI pipeline locally"

# Installation
install:
	@echo "Installing ByteWaves with development dependencies..."
	@if command -v uv &> /dev/null; then \
		uv sync --dev; \
	else \
		pip install -e ".[dev]"; \
	fi

# Testing targets
test:
	@echo "Running all tests..."
	@if command -v uv &> /dev/null; then \
		uv run pytest tests/ -v; \
	else \
		pytest tests/ -v; \
	fi

test-unit:
	@echo "Running unit tests..."
	@if command -v uv &> /dev/null; then \
		uv run pytest tests/unit/ -v; \
	else \
		pytest tests/unit/ -v; \
	fi

test-int:
	@echo "Running integration tests..."
	@if command -v uv &> /dev/null; then \
		uv run pytest tests/integration/ -v; \
	else \
		pytest tests/integration/ -v; \
	fi

test-perf:
	@echo "Running performance tests..."
	@if command -v uv &> /dev/null; then \
		uv run pytest tests/performance/ -v; \
	else \
		pytest tests/performance/ -v; \
	fi

test-all:
	@echo "Running all tests with coverage..."
	@if command -v uv &> /dev/null; then \
		uv run pytest tests/ --cov=bytewaves --cov-report=html --cov-report=term-missing; \
	else \
		pytest tests/ --cov=bytewaves --cov-report=html --cov-report=term-missing; \
	fi

# Code quality
lint:
	@echo "Running linting checks..."
	@if command -v uv &> /dev/null; then \
		uv run flake8 bytewaves/; \
		uv run black --check --diff bytewaves/; \
		uv run isort --check-only --diff bytewaves/; \
	else \
		flake8 bytewaves/; \
		black --check --diff bytewaves/; \
		isort --check-only --diff bytewaves/; \
	fi

format:
	@echo "Formatting code..."
	@if command -v uv &> /dev/null; then \
		uv run black bytewaves/; \
		uv run isort bytewaves/; \
	else \
		black bytewaves/; \
		isort bytewaves/; \
	fi

# Cleanup
clean:
	@echo "Cleaning up generated files..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf htmlcov/
	@rm -rf .pytest_cache/
	@rm -rf __pycache__/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Coverage
coverage:
	@echo "Generating coverage report..."
	@if command -v uv &> /dev/null; then \
		uv run pytest tests/ --cov=bytewaves --cov-report=html; \
	else \
		pytest tests/ --cov=bytewaves --cov-report=html; \
	fi
	@echo "Coverage report generated in htmlcov/index.html"

# Documentation
docs:
	@echo "Generating documentation..."
	@if command -v uv &> /dev/null; then \
		uv run sphinx-build docs/ docs/_build/; \
	else \
		sphinx-build docs/ docs/_build/; \
	fi

# Local CI simulation
ci: lint test-all coverage
	@echo "CI pipeline completed successfully!"

# Quick test (fast feedback)
quick-test:
	@echo "Running quick test suite..."
	@if command -v uv &> /dev/null; then \
		uv run pytest tests/unit/ -x; \
	else \
		pytest tests/unit/ -x; \
	fi

# Benchmark specific components
benchmark:
	@echo "Running benchmarks..."
	@if command -v uv &> /dev/null; then \
		uv run pytest tests/performance/test_benchmarks.py -v; \
	else \
		pytest tests/performance/test_benchmarks.py -v; \
	fi

# Security scan
security:
	@echo "Running security checks..."
	@if command -v uv &> /dev/null; then \
		uv run safety check; \
		uv run bandit -r bytewaves/; \
	else \
		safety check; \
		bandit -r bytewaves/; \
	fi

# Development workflow
dev-setup: install
	@echo "Setting up development environment..."
	@cp -n requirements-dev.txt requirements-dev.txt.backup 2>/dev/null || true
	@echo "Installing pre-commit hooks..."
	@if command -v uv &> /dev/null; then \
		uv run pre-commit install; \
	else \
		pre-commit install; \
	fi

# Quick check for common issues
health-check:
	@echo "Running health checks..."
	@if command -v uv &> /dev/null; then \
		uv run python -c "import bytewaves; print('✅ Import successful')"; \
		uv run python -m py_compile bytewaves/*.py; \
	else \
		python -c "import bytewaves; print('✅ Import successful')"; \
		python -m py_compile bytewaves/*.py; \
	fi
	@echo "Health check completed"