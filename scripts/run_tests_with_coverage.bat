@echo off
REM Run tests with coverage and generate TEST_COVERAGE_SUMMARY.md

echo Neural DSL - Test Coverage Runner
echo ===================================
echo.

REM Check if virtual environment is activated
where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please activate your virtual environment first.
    echo Example: .\.venv\Scripts\Activate
    exit /b 1
)

REM Run the test coverage script
python scripts\generate_test_coverage_summary.py
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% equ 0 (
    echo ===================================
    echo Test coverage report generated successfully!
    echo.
    echo View the report:
    echo   - TEST_COVERAGE_SUMMARY.md
    echo   - htmlcov\index.html
    echo.
    echo To open HTML report in browser:
    echo   start htmlcov\index.html
) else (
    echo ===================================
    echo Test coverage run completed with errors.
    echo Check output above for details.
)

echo.
exit /b %EXIT_CODE%
