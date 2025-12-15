# PowerShell script to remove cache directories and test artifacts
# Run this script to clean up development artifacts

Write-Host "======================================================================"
Write-Host "Cache and Artifacts Cleanup Script"
Write-Host "======================================================================"
Write-Host ""

$ErrorActionPreference = "SilentlyContinue"

# Function to remove directory with progress
function Remove-DirectoryRecursive {
    param([string]$Path, [string]$Name)
    
    if (Test-Path $Path) {
        Write-Host "Removing $Name : $Path" -ForegroundColor Yellow
        try {
            # First remove all files
            Get-ChildItem -Path $Path -Recurse -Force -File | Remove-Item -Force -ErrorAction SilentlyContinue
            # Then remove directories from deepest to shallowest
            Get-ChildItem -Path $Path -Recurse -Force -Directory | 
                Sort-Object -Property FullName -Descending | 
                Remove-Item -Force -ErrorAction SilentlyContinue
            # Finally remove the root directory
            Remove-Item -Path $Path -Force -ErrorAction SilentlyContinue
            
            if (-not (Test-Path $Path)) {
                Write-Host "  SUCCESS: Removed $Name" -ForegroundColor Green
                return $true
            } else {
                Write-Host "  WARNING: Directory may still exist, check manually" -ForegroundColor Yellow
                return $false
            }
        } catch {
            Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    }
    return $false
}

# Function to find and remove pattern recursively
function Remove-Pattern {
    param([string]$Pattern, [string]$Description, [switch]$IsDirectory)
    
    Write-Host "Searching for $Description..." -ForegroundColor Cyan
    $items = Get-ChildItem -Path . -Recurse -Force -Filter $Pattern -ErrorAction SilentlyContinue
    $count = 0
    
    foreach ($item in $items) {
        if ($IsDirectory -and $item.PSIsContainer) {
            if (Remove-DirectoryRecursive -Path $item.FullName -Name $item.Name) {
                $count++
            }
        } elseif (-not $IsDirectory -and -not $item.PSIsContainer) {
            Write-Host "Removing file: $($item.FullName)" -ForegroundColor Yellow
            Remove-Item -Path $item.FullName -Force -ErrorAction SilentlyContinue
            if (-not (Test-Path $item.FullName)) {
                Write-Host "  SUCCESS: Removed $($item.Name)" -ForegroundColor Green
                $count++
            }
        }
    }
    
    Write-Host "Removed $count $Description`n" -ForegroundColor Green
    return $count
}

Write-Host "Step 1: Removing cache directories..." -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------"
$cacheTotal = 0
$cacheTotal += Remove-Pattern -Pattern "__pycache__" -Description "__pycache__ directories" -IsDirectory
$cacheTotal += Remove-Pattern -Pattern ".pytest_cache" -Description ".pytest_cache directories" -IsDirectory
$cacheTotal += Remove-Pattern -Pattern ".hypothesis" -Description ".hypothesis directories" -IsDirectory
$cacheTotal += Remove-Pattern -Pattern ".mypy_cache" -Description ".mypy_cache directories" -IsDirectory
$cacheTotal += Remove-Pattern -Pattern ".ruff_cache" -Description ".ruff_cache directories" -IsDirectory
Write-Host "Total cache directories removed: $cacheTotal`n" -ForegroundColor Green

Write-Host "Step 2: Removing virtual environment directories..." -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------"
$venvTotal = 0
$venvDirs = @(".venv", ".venv312", "venv")
foreach ($venvDir in $venvDirs) {
    if (Remove-DirectoryRecursive -Path $venvDir -Name $venvDir) {
        $venvTotal++
    }
}
Write-Host "Total virtual environments removed: $venvTotal`n" -ForegroundColor Green

Write-Host "Step 3: Removing test artifacts..." -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------"
$artifactTotal = 0
$artifactTotal += Remove-Pattern -Pattern "test_*.html" -Description "test HTML files"
$artifactTotal += Remove-Pattern -Pattern "test_*.png" -Description "test PNG files"
Write-Host "Total test artifacts removed: $artifactTotal`n" -ForegroundColor Green

Write-Host "Step 4: Removing temporary Python scripts..." -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------------"
$scriptTotal = 0
$tempScripts = @("sample_tensorflow.py", "sample_pytorch.py")
foreach ($script in $tempScripts) {
    if (Test-Path $script) {
        Remove-Item -Path $script -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path $script)) {
            Write-Host "Removed: $script" -ForegroundColor Green
            $scriptTotal++
        }
    }
}
Write-Host "Total temporary scripts removed: $scriptTotal`n" -ForegroundColor Green

Write-Host "======================================================================"
Write-Host "Cleanup Complete!" -ForegroundColor Green
Write-Host "======================================================================"
Write-Host "Summary:"
Write-Host "  - Cache directories: $cacheTotal"
Write-Host "  - Virtual environments: $venvTotal"
Write-Host "  - Test artifacts: $artifactTotal"
Write-Host "  - Temporary scripts: $scriptTotal"
Write-Host ""
Write-Host "Note: Large directories like .venv may take time to remove."
Write-Host "If cleanup is incomplete, you can safely delete these directories manually."
Write-Host "All patterns are already in .gitignore and will not be tracked by git."
Write-Host ""
