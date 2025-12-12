#!/bin/bash
# Setup script for git-secrets
# This script installs and configures git-secrets to prevent committing secrets

set -e

echo "Setting up git-secrets..."

# Check if git-secrets is already installed
if command -v git-secrets &> /dev/null; then
    echo "git-secrets is already installed"
else
    echo "Installing git-secrets..."
    
    # Clone and install git-secrets
    if [ ! -d "/tmp/git-secrets" ]; then
        git clone https://github.com/awslabs/git-secrets.git /tmp/git-secrets
    fi
    
    cd /tmp/git-secrets
    sudo make install || make install
    cd -
    
    echo "git-secrets installed successfully"
fi

# Initialize git-secrets in the repository
echo "Initializing git-secrets in repository..."
git secrets --install -f

# Register AWS patterns
echo "Registering AWS secret patterns..."
git secrets --register-aws

# Add custom patterns for common secrets
echo "Adding custom secret patterns..."

# API keys and tokens
git secrets --add 'password\s*=\s*["'"'"'][^'"'"'"]*["'"'"']'
git secrets --add 'api[_-]?key\s*=\s*["'"'"'][^'"'"'"]*["'"'"']'
git secrets --add 'secret[_-]?key\s*=\s*["'"'"'][^'"'"'"]*["'"'"']'
git secrets --add 'token\s*=\s*["'"'"'][^'"'"'"]*["'"'"']'
git secrets --add '[a-zA-Z0-9_-]*[pP]assword[a-zA-Z0-9_-]*\s*=\s*["'"'"'][^'"'"'"]*["'"'"']'

# Private keys
git secrets --add 'private[_-]?key'
git secrets --add 'BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY'

# Generic high-entropy strings (potential secrets)
git secrets --add --allowed 'example.*password'
git secrets --add --allowed 'test.*password'
git secrets --add --allowed 'your_secure_password'

echo "git-secrets setup complete!"
echo ""
echo "Usage:"
echo "  - git-secrets will automatically scan commits"
echo "  - To scan the entire repository: git secrets --scan"
echo "  - To scan history: git secrets --scan-history"
echo "  - To list patterns: git secrets --list"
