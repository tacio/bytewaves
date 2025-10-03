#!/bin/bash

# Test script for ByteWaves acoustic modem

set -e

echo "🧪 Running ByteWaves Test Suite"
echo "================================"

# Check if uv is available
if command -v uv &> /dev/null; then
    PACKAGE_MANAGER="uv"
elif command -v pip &> /dev/null; then
    PACKAGE_MANAGER="pip"
else
    echo "❌ Error: Neither uv nor pip found"
    exit 1
fi

echo "📦 Using package manager: $PACKAGE_MANAGER"

# Install dependencies if needed
if [ "$PACKAGE_MANAGER" = "uv" ]; then
    echo "🔧 Installing dependencies with uv..."
    uv sync --dev
else
    echo "🔧 Installing dependencies with pip..."
    pip install -e ".[dev]"
fi

# Run different test categories
echo ""
echo "🔍 Running unit tests..."
if [ "$PACKAGE_MANAGER" = "uv" ]; then
    uv run pytest tests/unit/ -v --cov=bytewaves --cov-report=term-missing
else
    python -m pytest tests/unit/ -v --cov=bytewaves --cov-report=term-missing
fi

echo ""
echo "🔗 Running integration tests..."
if [ "$PACKAGE_MANAGER" = "uv" ]; then
    uv run pytest tests/integration/ -v --cov=bytewaves --cov-report=term-missing --cov-append
else
    python -m pytest tests/integration/ -v --cov=bytewaves --cov-report=term-missing --cov-append
fi

echo ""
echo "⚡ Running performance tests..."
if [ "$PACKAGE_MANAGER" = "uv" ]; then
    uv run pytest tests/performance/ -v --cov=bytewaves --cov-report=term-missing --cov-append
else
    python -m pytest tests/performance/ -v --cov=bytewaves --cov-report=term-missing --cov-append
fi

echo ""
echo "📊 Generating coverage report..."
if [ "$PACKAGE_MANAGER" = "uv" ]; then
    uv run pytest tests/ --cov=bytewaves --cov-report=html --cov-report=xml
else
    python -m pytest tests/ --cov=bytewaves --cov-report=html --cov-report=xml
fi

echo ""
echo "✅ All tests completed successfully!"
echo ""
echo "📈 Coverage reports generated:"
echo "   - HTML: htmlcov/index.html"
echo "   - XML: coverage.xml"
echo ""
echo "🎯 To run specific test categories:"
echo "   - Unit tests: pytest tests/unit/"
echo "   - Integration tests: pytest tests/integration/"
echo "   - Performance tests: pytest tests/performance/"
echo "   - All tests: pytest tests/"