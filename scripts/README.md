# Neural DSL - Development Scripts

This directory contains utility scripts for development, testing, deployment, and repository maintenance.

## Directory Structure

```
scripts/
├── automation/          # Automation scripts for releases and workflows
├── cleanup/            # Repository cleanup utilities
├── README.md          # This file
├── README_LINTING.md  # Detailed linting documentation
├── CLEANUP_README.md  # Detailed cleanup documentation
└── [various scripts]  # Individual utility scripts
```

## Test Coverage Scripts (Moved from Root)

These essential development scripts were moved from the root directory to keep the repository organized.

### `generate_test_coverage_summary.py`
Runs pytest with coverage and generates a comprehensive `TEST_COVERAGE_SUMMARY.md` report.

**Usage:**
```bash
python scripts/generate_test_coverage_summary.py
```

**Output:**
- `TEST_COVERAGE_SUMMARY.md` - Detailed coverage report with statistics
- `htmlcov/index.html` - Interactive HTML coverage report
- `coverage.json` - Machine-readable coverage data

### Test Coverage Runners

Convenience wrappers for running test coverage:

- **Bash:** `./scripts/run_tests_with_coverage.sh`
- **PowerShell:** `.\scripts\run_tests_with_coverage.ps1`
- **Batch:** `.\scripts\run_tests_with_coverage.bat`

All three scripts check for Python, activate the virtual environment if needed, and generate the coverage report.

## Linting & Code Quality Scripts

See [README_LINTING.md](README_LINTING.md) for detailed documentation.

### Quick Reference

- **`fix_imports.py`** - Automatically fix import ordering per PEP 8
- **`lint_and_fix.py`** - Comprehensive linting workflow (ruff, mypy, isort)
- **`pre_commit_check.py`** - Pre-commit verification script

**Common commands:**
```bash
# Fix all linting issues
python scripts/lint_and_fix.py

# Pre-commit checks
python scripts/pre_commit_check.py --fix

# Fix imports
python scripts/fix_imports.py
```

## Repository Cleanup Scripts

See [CLEANUP_README.md](CLEANUP_README.md) for detailed documentation.

### Quick Reference

- **`cleanup_repository.py`** - Clean up redundant documentation and artifacts
- **`cleanup_repository.ps1`** - PowerShell version with dry-run support
- **`cleanup_repository.sh`** - Bash version

**Common commands:**
```bash
# Python
python scripts/cleanup_repository.py

# PowerShell (dry-run)
.\scripts\cleanup_repository.ps1 -DryRun

# PowerShell (actual cleanup)
.\scripts\cleanup_repository.ps1

# Bash
./scripts/cleanup_repository.sh
```

## Deployment Scripts

### Container Images

- **`build-images.sh`** / **`build-images.bat`** - Build Docker images
- **`push-images.sh`** / **`push-images.bat`** - Push images to registry

### Kubernetes

- **`deploy-k8s.sh`** - Deploy to Kubernetes cluster
- **`deploy-helm.sh`** - Deploy using Helm charts

### API Servers

- **`run_api.sh`** / **`run_api.ps1`** - Start API server for development

## GitHub & Release Automation

- **`github_publish_simple.ps1`** - Simplified GitHub publishing workflow
- **`publish_to_github.ps1`** - Publish releases to GitHub
- **`create_issues.py`** - Create GitHub issues from test failures
- **`close_fixed_issues.py`** - Close fixed issues automatically

## Security Scripts

- **`setup-git-secrets.sh`** / **`setup-git-secrets.ps1`** - Configure git-secrets to prevent committing sensitive data

**Usage:**
```bash
# Bash
./scripts/setup-git-secrets.sh

# PowerShell
.\scripts\setup-git-secrets.ps1
```

## Development Utilities

- **`complexity.py`** - Analyze code complexity metrics
- **`quick_prototype.py`** - Quick prototyping utilities

## Migration Notes

The following scripts were **removed from the root directory** during cleanup:

### Deleted (Obsolete)
- `cleanup.bat`, `cleanup_docs.py`, `cleanup_redundant_files.py`, `cleanup_workflows.py`
- `preview_cleanup.py`, `run_cleanup.py` - Replaced by `scripts/cleanup_repository.*`
- `setup_complete.bat`, `setup_complete.ps1`, `setup_packages.py` - No longer needed
- `_create_education_dir.py`, `_setup_repo.py` - One-time setup scripts
- `batch_replace_prints.py`, `check_lint.py`, `debug_imports.py` - Temporary dev scripts

### Moved to `scripts/`
- `generate_test_coverage_summary.py` → `scripts/generate_test_coverage_summary.py`
- `run_tests_with_coverage.{sh,ps1,bat}` → `scripts/run_tests_with_coverage.{sh,ps1,bat}`

**Update your workflows:** If you have scripts or documentation referencing the old locations, update them:

```bash
# Old
python generate_test_coverage_summary.py

# New
python scripts/generate_test_coverage_summary.py
```

## Best Practices

### Running Scripts

1. **Always run from repository root:**
   ```bash
   # Good
   python scripts/generate_test_coverage_summary.py
   
   # Avoid
   cd scripts && python generate_test_coverage_summary.py
   ```

2. **Use virtual environment:**
   ```bash
   # Windows
   .\.venv\Scripts\Activate.ps1
   
   # Unix
   source .venv/bin/activate
   ```

3. **Check script help:**
   ```bash
   python scripts/<script_name>.py --help
   ```

### Adding New Scripts

When adding new utility scripts:

1. **Place in appropriate location:**
   - Linting/formatting → `scripts/`
   - Cleanup → `scripts/cleanup/`
   - Automation → `scripts/automation/`

2. **Add documentation:**
   - Update this README.md
   - Add docstring to the script
   - Include usage examples

3. **Follow conventions:**
   - Use argparse for CLI arguments
   - Provide `--help` flag
   - Include error handling
   - Print helpful progress messages

4. **Test before committing:**
   ```bash
   # Test your script
   python scripts/your_script.py
   
   # Run linting
   python -m ruff check scripts/your_script.py
   ```

## Documentation References

- **[AGENTS.md](../AGENTS.md)** - Repository setup and commands
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
- **[README_LINTING.md](README_LINTING.md)** - Detailed linting guide
- **[CLEANUP_README.md](CLEANUP_README.md)** - Detailed cleanup guide

## Troubleshooting

### Script not found
```bash
# Ensure you're in the repository root
pwd  # Should show the Neural DSL root directory

# List scripts
ls scripts/
```

### Import errors
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Or install package in editable mode
pip install -e ".[full]"
```

### Permission errors (Unix)
```bash
# Make script executable
chmod +x scripts/script_name.sh

# Run script
./scripts/script_name.sh
```

### Virtual environment not activated
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Unix (bash/zsh)
source .venv/bin/activate
```

## Support

For issues or questions:
1. Check this README and related documentation
2. Review script source code for inline documentation
3. Run script with `--help` flag if available
4. Check AGENTS.md for repository-specific guidance
5. Open an issue on GitHub with script output and error messages
