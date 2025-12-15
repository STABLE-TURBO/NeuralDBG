@echo off
REM Build all Docker images for Neural DSL

setlocal

REM Default values
if "%REGISTRY%"=="" set REGISTRY=docker.io
if "%REPO%"=="" set REPO=neural-dsl
if "%TAG%"=="" set TAG=latest

echo Building Neural DSL Docker images...
echo Registry: %REGISTRY%
echo Repository: %REPO%
echo Tag: %TAG%
echo.

REM Build API image
echo Building API image...
docker build -f deployment/docker/Dockerfile.api -t %REGISTRY%/%REPO%/api:%TAG% .
if errorlevel 1 (
    echo Failed to build API image
    exit /b 1
)

REM Build Worker image
echo Building Worker image...
docker build -f deployment/docker/Dockerfile.worker -t %REGISTRY%/%REPO%/worker:%TAG% .
if errorlevel 1 (
    echo Failed to build Worker image
    exit /b 1
)

REM Build Dashboard image
echo Building Dashboard image...
docker build -f deployment/docker/Dockerfile.dashboard -t %REGISTRY%/%REPO%/dashboard:%TAG% .
if errorlevel 1 (
    echo Failed to build Dashboard image
    exit /b 1
)

REM Build No-Code image
echo Building No-Code image...
docker build -f deployment/docker/Dockerfile.nocode -t %REGISTRY%/%REPO%/nocode:%TAG% .
if errorlevel 1 (
    echo Failed to build No-Code image
    exit /b 1
)

REM Build Aquarium IDE image
echo Building Aquarium IDE image...
docker build -f deployment/docker/Dockerfile.aquarium -t %REGISTRY%/%REPO%/aquarium:%TAG% .
if errorlevel 1 (
    echo Failed to build Aquarium IDE image
    exit /b 1
)

echo.
echo All images built successfully!
echo.
echo Images:
echo   - %REGISTRY%/%REPO%/api:%TAG%
echo   - %REGISTRY%/%REPO%/worker:%TAG%
echo   - %REGISTRY%/%REPO%/dashboard:%TAG%
echo   - %REGISTRY%/%REPO%/nocode:%TAG%
echo   - %REGISTRY%/%REPO%/aquarium:%TAG%
echo.
echo To push images, run:
echo   docker push %REGISTRY%/%REPO%/api:%TAG%
echo   docker push %REGISTRY%/%REPO%/worker:%TAG%
echo   docker push %REGISTRY%/%REPO%/dashboard:%TAG%
echo   docker push %REGISTRY%/%REPO%/nocode:%TAG%
echo   docker push %REGISTRY%/%REPO%/aquarium:%TAG%

endlocal
