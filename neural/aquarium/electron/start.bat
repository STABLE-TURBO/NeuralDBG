@echo off
REM Neural Aquarium IDE - Electron Startup Script

echo Starting Neural Aquarium IDE...

REM Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed. Please install Node.js 16 or higher.
    exit /b 1
)

REM Check if npm is installed
where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: npm is not installed. Please install npm.
    exit /b 1
)

REM Navigate to electron directory
cd /d "%~dp0"

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

REM Start the application
echo Launching Aquarium IDE...
call npm start
