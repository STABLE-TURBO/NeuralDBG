# Security Quick Reference

Quick commands for running security scans locally and in CI.

## Local Scans

### Quick Security Check (All Tools)
```bash
# Run all security scans
python -m bandit -r neural/ -ll
python -m safety check
python -m pip_audit --desc
git secrets --scan
```

### Bandit (Code Security)
```bash
python -m bandit -r neural/                    # Full scan
python -m bandit -r neural/ -ll                # Medium/High only
python -m bandit -c pyproject.toml -r neural/  # Use config
```

### Safety (Dependencies)
```bash
python -m safety check                         # Check installed packages
python -m safety check --full-report           # Detailed report
```

### pip-audit (Package Vulnerabilities)
```bash
python -m pip_audit                            # Basic audit
python -m pip_audit --desc                     # With descriptions
python -m pip_audit --fix                      # Auto-fix (experimental)
```

### git-secrets (Secret Scanning)
```bash
git secrets --scan                             # Scan current files
git secrets --scan-history                     # Scan entire history
git secrets --list                             # List patterns
```

### detect-secrets (Pre-commit)
```bash
detect-secrets scan                            # Scan repository
detect-secrets audit .secrets.baseline         # Review findings
```

## CI/CD Workflows

### Security Scanning (On Push/PR)
- **Workflow**: `.github/workflows/security.yml`
- **Triggers**: push, pull_request, manual
- **Jobs**: bandit, safety, git-secrets, trufflehog

### Security Audit (Scheduled)
- **Workflow**: `.github/workflows/security-audit.yml`
- **Schedule**: Weekly (Monday 03:00 UTC)
- **Triggers**: schedule, manual
- **Jobs**: pip-audit, full-audit

### Main CI Pipeline
- **Workflow**: `.github/workflows/ci.yml`
- **Includes**: bandit, safety, pip-audit (continue-on-error)

## Setup Commands

### Install Security Tools
```bash
pip install bandit[toml] safety pip-audit detect-secrets
```

### Setup git-secrets
```bash
# Linux/macOS
chmod +x scripts/setup-git-secrets.sh
./scripts/setup-git-secrets.sh

# Windows
.\scripts\setup-git-secrets.ps1
```

### Setup Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Initialize detect-secrets
```bash
detect-secrets scan > .secrets.baseline
```

## Suppressing False Positives

### Bandit
```python
value = eval(expr)  # nosec B307
```

### detect-secrets
```bash
detect-secrets audit .secrets.baseline  # Interactive audit
```

## Report Locations

### Local Reports
- `bandit-report.json`
- `safety-report.json`
- `pip-audit-report.json`

### CI Artifacts
- Workflow run → Artifacts section
- `bandit-report`, `safety-report`, `pip-audit-report`
- `security-audit-reports` (combined)

## Common Issues

### Tool Not Found
```bash
pip install <tool-name>
which bandit  # or: where bandit (Windows)
```

### Permission Denied (git-secrets)
```bash
chmod +x scripts/setup-git-secrets.sh
```

### Pre-commit Hook Failure
```bash
pre-commit clean
pre-commit install --install-hooks
```

## Documentation

- **Full Guide**: [docs/SECURITY_SETUP.md](../docs/SECURITY_SETUP.md)
- **Security Policy**: [SECURITY.md](../SECURITY.md)
- **Report Vulnerability**: Lemniscate_zero@proton.me

## Quick Workflow

```bash
# Before committing
pre-commit run --all-files

# Full local security scan
python -m bandit -r neural/ -ll && \
python -m safety check && \
python -m pip_audit --desc && \
git secrets --scan

# View reports
cat bandit-report.json | python -m json.tool
```
