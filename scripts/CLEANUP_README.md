# Repository Cleanup Scripts

This directory contains scripts to clean up redundant documentation and artifacts from the Neural DSL repository.

## Problem Statement

The repository has accumulated 200+ redundant files including:
- 50+ implementation summary documents
- 30+ quick reference duplicates
- Multiple workflow files
- Test artifacts and cache directories
- Temporary fix scripts
- Duplicate install scripts and requirements files

This creates noise and makes the repo feel unmaintained despite active development.

## Cleanup Scripts

### Python Script: `cleanup_repository.py`

**Usage:**
```bash
python scripts/cleanup_repository.py
```

This script removes all redundant files and provides a summary of what was deleted.

### PowerShell Script: `cleanup_repository.ps1`

**Usage:**
```powershell
# Dry run (preview what would be deleted)
.\scripts\cleanup_repository.ps1 -DryRun

# Actually delete files
.\scripts\cleanup_repository.ps1
```

The PowerShell script includes a dry-run mode to preview changes before executing them.

## What Gets Removed

### 1. Implementation Summary Documents (50+ files)
- `*IMPLEMENTATION_SUMMARY.md`
- `*IMPLEMENTATION_COMPLETE.md`
- `*IMPLEMENTATION_CHECKLIST.md`
- Files scattered across root, neural/, docs/, tests/, examples/

### 2. Quick Reference Duplicates (30+ files)
- `*QUICK_REFERENCE.md`
- `*QUICK_REF.md`
- `*QUICK_START*.md`
- Files in neural/, docs/, tests/, examples/

### 3. Redundant Status Documents
- `SETUP_STATUS.md`
- `CHANGES_SUMMARY.md`
- `BUG_FIXES.md`
- `CLEANUP_PLAN.md`
- `DISTRIBUTION_JOURNAL.md`
- `EXTRACTED_PROJECTS.md`

### 4. Duplicate Release Documents
- `GITHUB_RELEASE_v0.3.0.md`
- `RELEASE_NOTES_v0.3.0.md`
- `RELEASE_VERIFICATION_v0.3.0.md`
- `MIGRATION_v0.3.0.md`

### 5. Redundant Guide Documents
- `AUTOMATION_GUIDE.md`
- `DEPENDENCY_GUIDE.md`
- `ERROR_MESSAGES_GUIDE.md`
- `GITHUB_PUBLISHING_GUIDE.md`
- Content should be in docs/ instead

### 6. Duplicate .github Documentation
- `MARKETING_AUTOMATION_*.md`
- `SECURITY_BADGES.md`
- `SECURITY_CHECKLIST.md`
- `WORKFLOW_CONSOLIDATION.md`

### 7. Redundant Workflow Files
- `automated_release.yml`
- `post_release.yml`
- `periodic_tasks.yml`
- `pytest-to-issues.yml`
- `close-fixed-issues.yml`

### 8. Duplicate Install Scripts
- `_install_dev.py`
- `_setup_repo.py`
- `install_deps.py`
- `install.bat`
- `install_dev.bat`

### 9. Temporary Test Scripts
- `repro_parser.py`
- `reproduce_issue.py`

### 10. Redundant Requirements Files
- `requirements-minimal.txt`
- `requirements-backends.txt`
- `requirements-viz.txt`
- `requirements-api.txt`

### 11. Python Cache Directories
- `__pycache__/` directories throughout the repo
- `.pyc` and `.pyo` files

### 12. Build Artifacts
- `neural/neural_dsl.egg-info/`

### 13. Temporary Fix Scripts (20+ files)
- `fix_*.py` scripts in scripts/
- `skip_*.py` scripts
- One-off repair scripts that are no longer needed

### 14. Redundant Documentation
- Duplicate docs in docs/ that repeat info from root
- Automation and release documentation scattered across multiple locations

### 15. Duplicate Virtual Environments
- `.venv312/` (redundant venv)

## What Is Preserved

The following important files are kept:
- `README.md` - Main project documentation
- `CHANGELOG.md` - Version history
- `CONTRIBUTING.md` - Contribution guidelines
- `SECURITY.md` - Security policy
- `LICENSE.md` - License information
- `REPOSITORY_STRUCTURE.md` - Repo organization
- `GETTING_STARTED.md` - Quick start guide
- `INSTALL.md` - Installation instructions
- `AGENTS.md` - Agent/automation context
- `DEPENDENCY_CHANGES.md` - Dependency changelog

## Updated .gitignore

The `.gitignore` file has been updated to prevent these redundant files from being re-added:

```gitignore
# Repository hygiene - implementation summaries and duplicate docs
*IMPLEMENTATION_SUMMARY.md
*IMPLEMENTATION_COMPLETE.md
*IMPLEMENTATION_CHECKLIST.md
*IMPLEMENTATION.md
*_SUMMARY.md
*QUICK_REFERENCE.md
*QUICK_REF.md
*_QUICK_START.md
PROJECT_SUMMARY.md
REFACTORING_SUMMARY.md
PACKAGING_SUMMARY.md

# Temporary scripts and reproductions
repro_*.py
reproduce_*.py
fix_*.py
skip_*.py
temp_*.py

# Duplicate venvs
.venv*/
venv*/
!.venv/
!venv/
```

## After Running Cleanup

1. **Review Changes:**
   ```bash
   git status
   ```

2. **Verify Repository Functionality:**
   ```bash
   # Run tests to ensure nothing broke
   pytest tests/ -v
   
   # Try importing the package
   python -c "import neural; print('OK')"
   ```

3. **Commit Changes:**
   ```bash
   git add -A
   git commit -m "chore: clean up 200+ redundant documentation and artifact files

   - Remove 50+ implementation summary documents
   - Remove 30+ quick reference duplicates
   - Remove duplicate workflows and .github docs
   - Remove temporary fix scripts and test artifacts
   - Remove Python cache directories and .pyc files
   - Remove duplicate install scripts and requirements files
   - Update .gitignore to prevent re-accumulation"
   ```

## Best Practices Going Forward

1. **Documentation Location:**
   - User-facing docs go in `docs/`
   - Developer setup info in `AGENTS.md` or `CONTRIBUTING.md`
   - Don't create `*_SUMMARY.md` or `*_IMPLEMENTATION.md` files

2. **Scripts:**
   - Keep utility scripts in `scripts/`
   - Remove temporary fix scripts after use
   - Name scripts descriptively (avoid `fix_*.py`)

3. **Workflows:**
   - Keep workflows consolidated
   - Don't duplicate functionality across multiple workflow files
   - Remove obsolete workflows promptly

4. **Testing:**
   - Test artifacts should never be committed
   - Use `.gitignore` patterns to exclude generated files

5. **Requirements:**
   - Keep one main `requirements.txt`
   - Use `requirements-dev.txt` for dev dependencies
   - Use `pyproject.toml` extras for optional features
   - Don't create feature-specific requirements files

## Automation

Consider adding a pre-commit hook or CI check to prevent accumulation of these files:

```yaml
# .github/workflows/hygiene.yml
name: Repository Hygiene

on: [push, pull_request]

jobs:
  check-redundant-files:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check for redundant files
        run: |
          # Fail if any implementation summaries exist
          if find . -name "*IMPLEMENTATION_SUMMARY.md" | grep -q .; then
            echo "Found redundant IMPLEMENTATION_SUMMARY.md files"
            exit 1
          fi
          # Add more checks as needed
```

## Questions?

If you have questions about the cleanup or need to preserve specific files, please:
1. Open an issue describing why the file should be kept
2. Update this README to document the exception
3. Update the cleanup scripts to skip the file
