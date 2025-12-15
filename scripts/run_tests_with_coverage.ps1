# PowerShell script to run tests with coverage and generate TEST_COVERAGE_SUMMARY.md

Write-Host "Neural DSL - Test Coverage Runner" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Error: Python not found. Please activate your virtual environment first." -ForegroundColor Red
    Write-Host "Example: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

# Display Python version
$pythonVersion = & python --version 2>&1
Write-Host "Using: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Run the test coverage script
Write-Host "Running test coverage script..." -ForegroundColor Yellow
& python scripts/generate_test_coverage_summary.py
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "===================================" -ForegroundColor Green
    Write-Host "Test coverage report generated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "View the report:" -ForegroundColor Cyan
    Write-Host "  - TEST_COVERAGE_SUMMARY.md" -ForegroundColor White
    Write-Host "  - htmlcov\index.html" -ForegroundColor White
    Write-Host ""
    Write-Host "To open HTML report in browser:" -ForegroundColor Cyan
    Write-Host "  Invoke-Item htmlcov\index.html" -ForegroundColor White
    Write-Host "  # or simply:" -ForegroundColor Gray
    Write-Host "  start htmlcov\index.html" -ForegroundColor White
} else {
    Write-Host "===================================" -ForegroundColor Red
    Write-Host "Test coverage run completed with errors." -ForegroundColor Red
    Write-Host "Check output above for details." -ForegroundColor Yellow
}

Write-Host ""
exit $exitCode
