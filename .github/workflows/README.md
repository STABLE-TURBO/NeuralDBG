# GitHub Actions Workflows

This directory contains essential CI/CD workflows for the Neural DSL project.

## Active Workflows

### ci.yml - Continuous Integration
**Triggers:** Push, Pull Request, Nightly Schedule

**Jobs:**
- **Lint & Type Check** - Ruff linting, Mypy type checking, Flake8
- **Unit Tests** - Python 3.8-3.12 on Ubuntu & Windows
- **Integration Tests** - Cross-platform integration testing
- **E2E Tests** - End-to-end testing
- **UI Tests** - Dashboard UI testing
- **Supply Chain Audit** - Security scanning (Bandit, Safety, pip-audit)
- **Coverage Report** - Code coverage aggregation

**Purpose:** Comprehensive testing and quality assurance for all changes.

### release.yml - Release Automation
**Triggers:** Version tags (v*)

**Jobs:**
- **Build** - Build source and wheel distributions
- **Publish to PyPI** - Automated PyPI publishing via trusted publishing
- **GitHub Release** - Create GitHub release with artifacts

**Purpose:** Automated release pipeline from git tag to published package.

### codeql.yml - Security Analysis
**Triggers:** Push to main, Pull Requests, Weekly Schedule

**Jobs:**
- **Analyze** - CodeQL security analysis for Python and JavaScript/TypeScript

**Purpose:** Deep security scanning for vulnerabilities and code quality issues.

### validate-examples.yml - Example Validation
**Triggers:** Changes to examples/ or neural/, Daily Schedule

**Jobs:**
- **Validate DSL Examples** - Validate syntax and compilation (TensorFlow & PyTorch)
- **Validate Notebooks** - Check notebook format and structure
- **End-to-End Test** - Full compilation and visualization workflow

**Purpose:** Ensure example files remain valid and functional.

## Configuration Requirements

### Secrets
| Secret | Required For | Purpose |
|--------|--------------|---------|
| `CODECOV_TOKEN` | ci.yml | Code coverage reporting |
| `GITHUB_TOKEN` | All | Automatic (provided by GitHub) |

### Environments
- **pypi** - Required for release.yml with write permissions via trusted publishing

## Local Development

Run the same checks locally before pushing:

```bash
# Lint
python -m ruff check .

# Type check
python -m mypy neural/ --ignore-missing-imports

# Tests
python -m pytest tests/ -v

# Security
python -m bandit -r neural/ -ll
```

## Workflow Matrix

**CI Test Matrix:**
- Python: 3.8, 3.9, 3.10, 3.11, 3.12
- OS: Ubuntu, Windows
- Total: 10 combinations per test suite

**CodeQL Analysis:**
- Languages: Python, JavaScript/TypeScript

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python CI Best Practices](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Trusted Publishing for PyPI](https://docs.pypi.org/trusted-publishers/)
