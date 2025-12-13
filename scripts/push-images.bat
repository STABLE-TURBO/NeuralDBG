@echo off
REM Push all Docker images for Neural DSL

setlocal

REM Default values
if "%REGISTRY%"=="" set REGISTRY=docker.io
if "%REPO%"=="" set REPO=neural-dsl
if "%TAG%"=="" set TAG=latest

echo Pushing Neural DSL Docker images...
echo Registry: %REGISTRY%
echo Repository: %REPO%
echo Tag: %TAG%
echo.

REM Check if should login
if not "%REGISTRY%"=="docker.io" if not "%REGISTRY%"=="localhost" (
    echo Make sure you are logged in to %REGISTRY%
    echo Run: docker login %REGISTRY%
    echo.
)

REM Push images
echo Pushing API image...
docker push %REGISTRY%/%REPO%/api:%TAG%
if errorlevel 1 (
    echo Failed to push API image
    exit /b 1
)

echo Pushing Worker image...
docker push %REGISTRY%/%REPO%/worker:%TAG%
if errorlevel 1 (
    echo Failed to push Worker image
    exit /b 1
)

echo Pushing Dashboard image...
docker push %REGISTRY%/%REPO%/dashboard:%TAG%
if errorlevel 1 (
    echo Failed to push Dashboard image
    exit /b 1
)

echo Pushing No-Code image...
docker push %REGISTRY%/%REPO%/nocode:%TAG%
if errorlevel 1 (
    echo Failed to push No-Code image
    exit /b 1
)

echo Pushing Aquarium IDE image...
docker push %REGISTRY%/%REPO%/aquarium:%TAG%
if errorlevel 1 (
    echo Failed to push Aquarium IDE image
    exit /b 1
)

echo.
echo All images pushed successfully!
echo.
echo Images available at:
echo   - %REGISTRY%/%REPO%/api:%TAG%
echo   - %REGISTRY%/%REPO%/worker:%TAG%
echo   - %REGISTRY%/%REPO%/dashboard:%TAG%
echo   - %REGISTRY%/%REPO%/nocode:%TAG%
echo   - %REGISTRY%/%REPO%/aquarium:%TAG%

endlocal
