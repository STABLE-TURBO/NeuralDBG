# Linting and Code Quality Scripts

This directory contains scripts for maintaining code quality in the Neural DSL project.

## Scripts Overview

### 1. `fix_imports.py`

Automatically fixes import ordering in Python files according to PEP 8 and isort conventions.

**Usage:**
```bash
# Fix imports in all neural/ files
python scripts/fix_imports.py

# Preview changes without modifying files
python scripts/fix_imports.py --dry-run
```

**What it does:**
- Classifies imports into: future, stdlib, thirdparty, firstparty
- Sorts imports alphabetically within each group
- Adds proper spacing between groups
- Preserves docstrings and comments
- Handles multiline imports correctly

**Import Order:**
1. Future imports (`from __future__ import annotations`)
2. Standard library imports
3. Third-party imports (numpy, torch, etc.)
4. First-party imports (neural.*)
5. Relative imports (from .*)

### 2. `lint_and_fix.py`

Comprehensive linting workflow that runs multiple tools in sequence.

**Usage:**
```bash
python scripts/lint_and_fix.py
```

**What it does:**
1. Installs required tools (ruff, mypy, isort) if needed
2. Runs isort to fix import ordering
3. Runs ruff check with auto-fix
4. Runs ruff format for code formatting
5. Performs final linting check
6. Runs mypy type checking

**Output:**
- Shows which tools are available
- Reports success/failure for each step
- Displays linting and type errors if any

### 3. `pre_commit_check.py`

Pre-commit check script for manual or automated verification before commits.

**Usage:**
```bash
# Run all checks
python scripts/pre_commit_check.py

# Skip slower checks (type checking)
python scripts/pre_commit_check.py --fast

# Auto-fix issues when possible
python scripts/pre_commit_check.py --fix
```

**What it does:**
- Checks which tools are available
- Runs import ordering check
- Runs ruff linting
- Runs code formatting check
- Runs type checking (unless --fast)
- Provides summary and recommendations

**Exit codes:**
- `0`: All checks passed
- `1`: Some checks failed

## Quick Reference

### Before Committing
```bash
# Fix all issues automatically
python scripts/pre_commit_check.py --fix

# Verify all checks pass
python scripts/pre_commit_check.py
```

### Daily Development
```bash
# Fix imports
python scripts/fix_imports.py

# Run linting
python -m ruff check . --fix

# Format code
python -m ruff format .
```

### CI/CD Pipeline
```bash
# Check everything (no fixes)
python scripts/pre_commit_check.py
```

## Tool Configuration

### Ruff
- Configuration: `pyproject.toml` → `[tool.ruff]`
- Line length: 100 characters
- Enabled rules: E (pycodestyle), F (pyflakes), I (isort)
- Target: Python 3.11

### Mypy
- Configuration: `mypy.ini`
- Tiered strictness by module
- Ignore missing imports: True

### Pre-commit
- Configuration: `.pre-commit-config.yaml`
- Includes: ruff, mypy, bandit, security checks
- Install: `pre-commit install`

## Common Workflows

### Initial Setup
```bash
# Install tools
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Fix existing code
python scripts/fix_imports.py
python scripts/lint_and_fix.py
```

### Before Committing
```bash
# Auto-fix issues
python scripts/pre_commit_check.py --fix

# Verify changes
git diff

# Commit
git add .
git commit -m "Your commit message"
```

### CI/CD Integration
```yaml
# .github/workflows/ci.yml
- name: Install tools
  run: pip install ruff mypy

- name: Run pre-commit checks
  run: python scripts/pre_commit_check.py
```

## Troubleshooting

### "Module not found" errors
```bash
pip install ruff mypy isort
# or
pip install -r requirements-dev.txt
```

### Import errors during linting
Some imports may fail if optional dependencies aren't installed. This is expected for modules that handle missing dependencies gracefully.

### Type errors
Check `mypy.ini` for the strictness level of the module you're working on. Some modules have relaxed type checking.

### Pre-commit hooks not running
```bash
# Reinstall hooks
pre-commit install

# Update hooks
pre-commit autoupdate
```

## Best Practices

1. **Run checks before committing**
   ```bash
   python scripts/pre_commit_check.py --fix
   ```

2. **Use fast mode during development**
   ```bash
   python scripts/pre_commit_check.py --fast
   ```

3. **Fix imports regularly**
   ```bash
   python scripts/fix_imports.py
   ```

4. **Let ruff auto-fix**
   ```bash
   python -m ruff check . --fix
   ```

5. **Review changes before committing**
   ```bash
   git diff
   ```

## Documentation

For more detailed information, see:
- `LINTING_GUIDE.md` - Comprehensive linting guide
- `LINTING_FIXES_SUMMARY.md` - Summary of fixes applied
- `pyproject.toml` - Ruff configuration
- `mypy.ini` - Mypy configuration

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the documentation files
3. Run with `--help` flag (if supported)
4. Check the CI/CD logs for examples

## Contributing

When adding new scripts:
1. Add documentation here
2. Follow existing script patterns
3. Include usage examples
4. Add error handling
5. Provide helpful output messages
