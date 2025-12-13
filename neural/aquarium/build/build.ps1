param(
    [string]$Platform = "",
    [string]$Arch = "",
    [switch]$Sign = $false,
    [switch]$Release = $false,
    [switch]$Clean = $false,
    [switch]$Help = $false
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Set-Location $ProjectDir

function Show-Help {
    Write-Host @"
Neural Aquarium Build Script for Windows

Usage: .\build\build.ps1 [OPTIONS]

Options:
    -Platform <PLATFORM>    Target platform (win, mac, linux, all)
    -Arch <ARCH>           Target architecture (x64, arm64, all)
    -Sign                  Enable code signing
    -Release               Build for release (includes auto-update)
    -Clean                 Clean build artifacts before building
    -Help                  Show this help message

Examples:
    .\build\build.ps1 -Platform win -Arch x64
    .\build\build.ps1 -Platform all -Sign -Release
    .\build\build.ps1 -Clean -Platform win

Environment Variables:
    GH_TOKEN                  GitHub token for release publishing
    WIN_CSC_LINK             Windows certificate file path
    WIN_CSC_KEY_PASSWORD     Windows certificate password

"@
}

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host ""
}

if ($Help) {
    Show-Help
    exit 0
}

Write-Header "Neural Aquarium Build Script"

if ([string]::IsNullOrEmpty($Platform)) {
    $Platform = "win"
    Write-Host "Auto-detected platform: $Platform"
}

if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    Remove-Item -Path "dist" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "node_modules\.cache" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Clean complete." -ForegroundColor Green
    Write-Host ""
}

Write-Host "Installing dependencies..." -ForegroundColor Yellow
npm ci
if ($LASTEXITCODE -ne 0) {
    Write-Error "npm ci failed"
    exit 1
}

Write-Host ""
Write-Host "Building React application..." -ForegroundColor Yellow
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Error "React build failed"
    exit 1
}

Write-Host ""
Write-Host "Building Electron application for $Platform..." -ForegroundColor Yellow

$BuildCmd = "npm run electron:build"

switch ($Platform.ToLower()) {
    "win" { $BuildCmd = "$BuildCmd:win" }
    "windows" { $BuildCmd = "$BuildCmd:win" }
    "mac" { $BuildCmd = "$BuildCmd:mac" }
    "macos" { $BuildCmd = "$BuildCmd:mac" }
    "darwin" { $BuildCmd = "$BuildCmd:mac" }
    "linux" { $BuildCmd = "$BuildCmd:linux" }
    "all" { $BuildCmd = "$BuildCmd:all" }
    default {
        Write-Error "Unknown platform: $Platform. Valid platforms: win, mac, linux, all"
        exit 1
    }
}

if ($Release) {
    if ([string]::IsNullOrEmpty($env:GH_TOKEN)) {
        Write-Warning "GH_TOKEN not set. Publishing will fail."
        Write-Warning "Set GH_TOKEN environment variable for release builds."
    }
    $env:PUBLISH = "always"
} else {
    $env:PUBLISH = "never"
}

if (-not $Sign) {
    $env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
}

Invoke-Expression $BuildCmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "Electron build failed"
    exit 1
}

Write-Header "Build complete!"

Write-Host "Output directory: $ProjectDir\dist"
Write-Host ""

if (Test-Path "$ProjectDir\dist") {
    Write-Host "Built files:" -ForegroundColor Green
    Get-ChildItem "$ProjectDir\dist" -File | ForEach-Object {
        $size = if ($_.Length -gt 1MB) {
            "{0:N2} MB" -f ($_.Length / 1MB)
        } else {
            "{0:N2} KB" -f ($_.Length / 1KB)
        }
        Write-Host "  $($_.Name) ($size)"
    }
    
    $checksumFiles = Get-ChildItem "$ProjectDir\dist\checksums-*.txt" -ErrorAction SilentlyContinue
    if ($checksumFiles) {
        Write-Host ""
        Write-Host "Checksums generated:" -ForegroundColor Green
        $checksumFiles | ForEach-Object {
            Get-Content $_.FullName
        }
    }
}

Write-Host ""
Write-Host "To test the application:"
Write-Host "  npm run electron"
Write-Host ""
Write-Host "To create a release:"
Write-Host "  git tag aquarium-v0.3.0"
Write-Host "  git push origin aquarium-v0.3.0"
Write-Host ""
