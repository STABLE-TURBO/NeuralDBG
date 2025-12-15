# Cache and Artifacts Cleanup Guide

This guide explains how to remove cache directories, virtual environments, and test artifacts from the Neural DSL repository.

## What Gets Cleaned Up

### Cache Directories (recursive throughout the repository)
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - pytest test cache
- `.hypothesis/` - Hypothesis testing cache
- `.mypy_cache/` - MyPy type checker cache
- `.ruff_cache/` - Ruff linter cache

### Virtual Environments (root directory only)
- `.venv/`, `.venv*/` - Python virtual environments
- `venv/`, `venv*/` - Alternative virtual environment names

### Test Artifacts
- `test_*.html` - Test HTML output files
- `test_*.png` - Test PNG image files
- `htmlcov/` - Coverage HTML reports

### Temporary Scripts
- `sample_tensorflow.py` - Temporary TensorFlow sample scripts
- `sample_pytorch.py` - Temporary PyTorch sample scripts

## Cleanup Methods

Choose the appropriate cleanup script for your operating system:

### Windows (PowerShell)
```powershell
.\cleanup_cache_and_artifacts.ps1
```

### Windows (Command Prompt)
```cmd
cleanup_cache_and_artifacts.bat
```

### Unix/Linux/macOS
```bash
chmod +x cleanup_cache_and_artifacts.sh
./cleanup_cache_and_artifacts.sh
```

### Python (Cross-platform)
```bash
python cleanup_cache_and_artifacts.py
```

## Manual Cleanup

If automated cleanup fails, you can manually remove directories:

### Windows PowerShell
```powershell
# Remove cache directories recursively
Get-ChildItem -Path . -Recurse -Force -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -Force -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -Force -Directory -Filter ".hypothesis" | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -Force -Directory -Filter ".mypy_cache" | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -Force -Directory -Filter ".ruff_cache" | Remove-Item -Recurse -Force

# Remove virtual environments
Remove-Item -Path ".venv" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "venv" -Recurse -Force -ErrorAction SilentlyContinue
```

### Unix/Linux/macOS
```bash
# Remove cache directories recursively
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name ".hypothesis" -exec rm -rf {} +
find . -type d -name ".mypy_cache" -exec rm -rf {} +
find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Remove virtual environments
rm -rf .venv venv

# Remove test artifacts
find . -name "test_*.html" -type f -delete
find . -name "test_*.png" -type f -delete

# Remove temporary scripts
rm -f sample_tensorflow.py sample_pytorch.py
```

## Git Ignore Coverage

All cleanup patterns are already included in `.gitignore`, ensuring they won't be tracked by Git:

```gitignore
# Python cache and compiled files
__pycache__/

# Virtual environments
.venv/
venv/
.venv*/
venv*/

# Testing artifacts
.hypothesis/
.pytest_cache/
htmlcov/

# Linting and type checking
.ruff_cache/
.mypy_cache/

# Generated files
*.html
*.png

# Development scripts
/test_*.py
/sample_*.py
```

## Notes

- **Large Directories**: Virtual environments (`.venv/`) can be very large (100s of MB to GBs). Cleanup may take several minutes.
- **Safe to Delete**: All these directories can be safely deleted and will be regenerated when needed.
- **Git Status**: After cleanup, run `git status` to verify no tracked files were affected.
- **Rebuild Environment**: After removing virtual environments, recreate them with:
  ```bash
  python -m venv .venv
  .\.venv\Scripts\Activate  # Windows
  source .venv/bin/activate  # Unix/Linux/macOS
  pip install -e ".[full]"
  pip install -r requirements-dev.txt
  ```

## Troubleshooting

### Permission Errors
If you encounter permission errors on Windows:
1. Close any IDEs or terminals using the virtual environment
2. Run PowerShell or Command Prompt as Administrator
3. Try the cleanup script again

### Incomplete Deletion
If directories remain after cleanup:
1. Check for open file handles (close IDEs, terminals)
2. Manually navigate and delete via File Explorer/Finder
3. On Windows, try using `rmdir /s /q .venv` from Command Prompt

### Performance
For faster cleanup of large directories on Windows:
- Use Command Prompt (`rmdir /s /q`) instead of PowerShell
- Use third-party tools like `rimraf` (Node.js) for faster deletion
