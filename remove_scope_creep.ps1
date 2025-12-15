# PowerShell script to remove scope creep modules for v0.4.0 refocusing
# Run this script to complete the directory removal

Write-Host "Removing scope creep directories for v0.4.0..." -ForegroundColor Cyan

# Already removed:
# - neural/cost/
# - neural/monitoring/
# - neural/profiling/
# - neural/docgen/

# Remove remaining tracked directories
Write-Host "Removing tracked directories..." -ForegroundColor Yellow
git rm -rf neural/teams
git rm -rf neural/mlops  
git rm -rf neural/data
git rm -rf neural/config
git rm -rf neural/education
git rm -rf neural/plugins
git rm -rf neural/explainability

# Check for untracked directories and remove them
Write-Host "Checking for untracked directories..." -ForegroundColor Yellow
$untrackedDirs = @('neural/marketplace', 'neural/collaboration', 'neural/federated')
foreach ($dir in $untrackedDirs) {
    if (Test-Path $dir) {
        Write-Host "Removing untracked directory: $dir" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $dir
    }
}

Write-Host "Scope creep removal complete!" -ForegroundColor Green
Write-Host "Run 'git status' to see the changes." -ForegroundColor Cyan
