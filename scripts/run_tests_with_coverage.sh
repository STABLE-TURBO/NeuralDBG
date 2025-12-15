#!/bin/bash
# Run tests with coverage and generate TEST_COVERAGE_SUMMARY.md

echo "Neural DSL - Test Coverage Runner"
echo "==================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python not found. Please activate your virtual environment first."
    echo "Example: source .venv/bin/activate"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Run the test coverage script
$PYTHON_CMD scripts/generate_test_coverage_summary.py
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "==================================="
    echo "Test coverage report generated successfully!"
    echo ""
    echo "View the report:"
    echo "  - TEST_COVERAGE_SUMMARY.md"
    echo "  - htmlcov/index.html"
    echo ""
    echo "To open HTML report in browser:"
    echo "  - macOS: open htmlcov/index.html"
    echo "  - Linux: xdg-open htmlcov/index.html"
else
    echo "==================================="
    echo "Test coverage run completed with errors."
    echo "Check output above for details."
fi

echo ""
exit $EXIT_CODE
