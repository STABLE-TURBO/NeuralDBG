@echo off
REM Batch script to remove cache directories and test artifacts
REM Run this script to clean up development artifacts

echo ======================================================================
echo Cache and Artifacts Cleanup Script
echo ======================================================================
echo.

echo Removing cache directories...
echo ----------------------------------------------------------------------
for /d /r %%i in (__pycache__) do @if exist "%%i" (
    echo Removing: %%i
    rmdir /s /q "%%i" 2>nul
)

for /d /r %%i in (.pytest_cache) do @if exist "%%i" (
    echo Removing: %%i
    rmdir /s /q "%%i" 2>nul
)

for /d /r %%i in (.hypothesis) do @if exist "%%i" (
    echo Removing: %%i
    rmdir /s /q "%%i" 2>nul
)

for /d /r %%i in (.mypy_cache) do @if exist "%%i" (
    echo Removing: %%i
    rmdir /s /q "%%i" 2>nul
)

for /d /r %%i in (.ruff_cache) do @if exist "%%i" (
    echo Removing: %%i
    rmdir /s /q "%%i" 2>nul
)

echo.
echo Removing virtual environment directories...
echo ----------------------------------------------------------------------
if exist .venv (
    echo Removing: .venv
    rmdir /s /q .venv 2>nul
)

if exist .venv312 (
    echo Removing: .venv312
    rmdir /s /q .venv312 2>nul
)

if exist venv (
    echo Removing: venv
    rmdir /s /q venv 2>nul
)

echo.
echo Removing test artifacts...
echo ----------------------------------------------------------------------
del /s /q test_*.html 2>nul
del /s /q test_*.png 2>nul

echo.
echo Removing temporary Python scripts...
echo ----------------------------------------------------------------------
if exist sample_tensorflow.py (
    echo Removing: sample_tensorflow.py
    del /q sample_tensorflow.py 2>nul
)

if exist sample_pytorch.py (
    echo Removing: sample_pytorch.py
    del /q sample_pytorch.py 2>nul
)

echo.
echo ======================================================================
echo Cleanup Complete!
echo ======================================================================
echo All cache directories and test artifacts have been processed.
echo All patterns are already in .gitignore and will not be tracked by git.
echo.
pause
