# Security Setup Guide

This guide explains how to set up and use the security tools integrated into the Neural DSL CI/CD pipeline.

## Overview

Neural DSL uses multiple security scanning tools to ensure code and dependency security:

- **Bandit**: Python security linter for code vulnerabilities
- **Safety**: Dependency vulnerability scanner
- **pip-audit**: Python package vulnerability auditor
- **git-secrets**: Prevents committing secrets to the repository
- **TruffleHog**: Scans git history for leaked secrets
- **detect-secrets**: Pre-commit hook for secret detection

## Automated Scanning

### CI Pipeline (On Push/PR)

The `security.yml` workflow runs on every push and pull request:

1. **Bandit** - Scans Python code for security issues
2. **Safety** - Checks dependencies for known vulnerabilities
3. **git-secrets** - Scans for committed secrets
4. **TruffleHog** - Scans git history for verified secrets

Results are uploaded as workflow artifacts.

### Scheduled Audit (Weekly)

The `security-audit.yml` workflow runs weekly on Mondays at 03:00 UTC:

1. **pip-audit** - Comprehensive dependency audit
2. **Full Audit** - Combined bandit, safety, and pip-audit scan
3. **Auto Issue Creation** - Creates GitHub issues if vulnerabilities are found

You can also trigger this workflow manually via the Actions tab.

## Local Setup

### 1. Install Security Tools

```bash
# Install via pip
pip install bandit[toml] safety pip-audit detect-secrets

# Or install with project
pip install -e ".[full]"
```

### 2. Setup git-secrets

#### Linux/macOS
```bash
# Run the setup script
chmod +x scripts/setup-git-secrets.sh
./scripts/setup-git-secrets.sh
```

#### Windows
```powershell
# Install git-secrets first (manually or via chocolatey)
choco install git-secrets

# Run the setup script
.\scripts\setup-git-secrets.ps1
```

### 3. Setup Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run against all files (optional)
pre-commit run --all-files
```

### 4. Initialize detect-secrets Baseline

```bash
# Create initial baseline
detect-secrets scan > .secrets.baseline

# Update baseline when you add legitimate secrets (like test fixtures)
detect-secrets scan --baseline .secrets.baseline
```

## Running Security Scans Locally

### Bandit (Code Security)

```bash
# Scan with default settings
python -m bandit -r neural/

# Scan with medium/high severity only
python -m bandit -r neural/ -ll

# Generate JSON report
python -m bandit -r neural/ -f json -o bandit-report.json

# Use project configuration
python -m bandit -c pyproject.toml -r neural/
```

### Safety (Dependency Vulnerabilities)

```bash
# Check installed packages
python -m safety check

# Generate JSON report
python -m safety check --json --output safety-report.json

# Check with full policy
python -m safety check --full-report
```

### pip-audit (Package Audit)

```bash
# Audit installed packages
python -m pip_audit

# Detailed output with descriptions
python -m pip_audit --desc

# Generate JSON report
python -m pip_audit --format json --output pip-audit-report.json

# Skip optional dependencies
python -m pip_audit --skip-editable
```

### git-secrets (Secret Scanning)

```bash
# Scan current files
git secrets --scan

# Scan entire history
git secrets --scan-history

# List registered patterns
git secrets --list

# Add custom pattern
git secrets --add 'pattern-regex'

# Add allowed pattern (whitelist)
git secrets --add --allowed 'safe-pattern'
```

### detect-secrets (Pre-commit)

```bash
# Scan repository
detect-secrets scan

# Audit findings (interactive)
detect-secrets audit .secrets.baseline

# Compare against baseline
detect-secrets scan --baseline .secrets.baseline
```

## Configuration

### Bandit Configuration

Edit `pyproject.toml`:

```toml
[tool.bandit]
exclude_dirs = [".venv", ".venv312", "venv", "tests", "docs"]
skips = ["B101", "B601"]

[tool.bandit.assert_used]
skips = ["*/test_*.py", "*/tests/*.py"]
```

### git-secrets Patterns

Custom patterns are configured in the setup scripts:
- `scripts/setup-git-secrets.sh` (Linux/macOS)
- `scripts/setup-git-secrets.ps1` (Windows)

Common patterns included:
- AWS credentials
- API keys and tokens
- Passwords
- Private keys

### Pre-commit Configuration

Edit `.pre-commit-config.yaml` to customize hook behavior.

## CI/CD Integration

### GitHub Actions Workflows

1. **`.github/workflows/security.yml`**
   - Runs on: push, pull_request, manual
   - Jobs: bandit, safety, git-secrets, trufflehog

2. **`.github/workflows/security-audit.yml`**
   - Runs on: schedule (weekly), manual
   - Jobs: pip-audit, full-audit
   - Creates issues on failures

3. **`.github/workflows/ci.yml`**
   - Includes: bandit, safety, pip-audit
   - continue-on-error: true (informational)

### Artifacts

Security reports are uploaded as workflow artifacts:
- `bandit-report.json`
- `safety-report.json`
- `pip-audit-report.json`
- `security-audit-reports/` (combined)

Download from the Actions tab → Workflow run → Artifacts section.

## Handling Findings

### False Positives

#### Bandit
```python
# Suppress specific check
value = eval(user_input)  # nosec B307

# Suppress with comment
# nosec: B101 - assert is safe in this context
assert condition
```

#### detect-secrets
```bash
# Add to baseline
detect-secrets scan --baseline .secrets.baseline

# Mark as false positive in audit
detect-secrets audit .secrets.baseline
```

### True Positives

1. **Critical/High Severity**
   - Fix immediately
   - Do not merge until resolved

2. **Medium Severity**
   - Create issue for tracking
   - Fix in next sprint

3. **Low Severity**
   - Consider fixing or documenting
   - May be accepted risk

### Dependency Vulnerabilities

1. **Update dependency**
   ```bash
   pip install --upgrade package-name
   ```

2. **Pin to safe version**
   ```toml
   # setup.py or requirements.txt
   package-name>=safe.version
   ```

3. **Check for patches**
   ```bash
   pip-audit --fix
   ```

## Best Practices

1. **Run locally before committing**
   - Use pre-commit hooks
   - Test changes with security tools

2. **Keep dependencies updated**
   - Review Dependabot PRs
   - Run audits regularly

3. **Never commit secrets**
   - Use environment variables
   - Use `.env.example` templates
   - Configure git-secrets

4. **Review security reports**
   - Check workflow artifacts
   - Address issues promptly

5. **Document exceptions**
   - Comment nosec directives
   - Update security docs

## Troubleshooting

### git-secrets not found
```bash
# Install manually
git clone https://github.com/awslabs/git-secrets.git
cd git-secrets
make install
```

### Pre-commit hooks failing
```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean

# Reinstall
pre-commit uninstall
pre-commit install
```

### Bandit configuration not loading
```bash
# Verify config
python -m bandit --version
python -m bandit -c pyproject.toml --help

# Test config
python -m bandit -c pyproject.toml -r neural/ -v
```

### Safety API rate limit
```bash
# Use cached database
python -m safety check --cache

# Wait and retry
sleep 60 && python -m safety check
```

## Additional Resources

- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Safety Documentation](https://pyup.io/safety/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
- [git-secrets Repository](https://github.com/awslabs/git-secrets)
- [detect-secrets Documentation](https://github.com/Yelp/detect-secrets)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

## Support

For security concerns, see [SECURITY.md](../SECURITY.md) for reporting guidelines.
