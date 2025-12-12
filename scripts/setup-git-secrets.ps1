# Setup script for git-secrets on Windows
# This script configures git-secrets patterns to prevent committing secrets

Write-Host "Setting up git-secrets patterns..." -ForegroundColor Green

# Check if git-secrets is installed
try {
    $gitSecretsVersion = git secrets --version 2>$null
    Write-Host "git-secrets is installed: $gitSecretsVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: git-secrets is not installed" -ForegroundColor Red
    Write-Host "Please install git-secrets manually:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://github.com/awslabs/git-secrets" -ForegroundColor Yellow
    Write-Host "  2. Add to PATH or use pre-commit hooks instead" -ForegroundColor Yellow
    exit 1
}

# Initialize git-secrets in the repository
Write-Host "Initializing git-secrets in repository..." -ForegroundColor Cyan
git secrets --install -f

# Register AWS patterns
Write-Host "Registering AWS secret patterns..." -ForegroundColor Cyan
git secrets --register-aws

# Add custom patterns for common secrets
Write-Host "Adding custom secret patterns..." -ForegroundColor Cyan

# API keys and tokens
git secrets --add 'password\s*=\s*["\x27][^\x27"]*["\x27]'
git secrets --add 'api[_-]?key\s*=\s*["\x27][^\x27"]*["\x27]'
git secrets --add 'secret[_-]?key\s*=\s*["\x27][^\x27"]*["\x27]'
git secrets --add 'token\s*=\s*["\x27][^\x27"]*["\x27]'
git secrets --add '[a-zA-Z0-9_-]*[pP]assword[a-zA-Z0-9_-]*\s*=\s*["\x27][^\x27"]*["\x27]'

# Private keys
git secrets --add 'private[_-]?key'
git secrets --add 'BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY'

# Add allowed patterns for examples
git secrets --add --allowed 'example.*password'
git secrets --add --allowed 'test.*password'
git secrets --add --allowed 'your_secure_password'

Write-Host "git-secrets setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  - git-secrets will automatically scan commits" -ForegroundColor White
Write-Host "  - To scan the entire repository: git secrets --scan" -ForegroundColor White
Write-Host "  - To scan history: git secrets --scan-history" -ForegroundColor White
Write-Host "  - To list patterns: git secrets --list" -ForegroundColor White
