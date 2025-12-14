# Repository Hygiene Guidelines

This document outlines the repository hygiene strategy for Neural DSL, including what was cleaned up, what to avoid going forward, and automation to prevent re-accumulation.

## Problem

The repository accumulated 200+ redundant files over time:
- 50+ implementation summary documents (`*IMPLEMENTATION_SUMMARY.md`)
- 30+ quick reference duplicates (`*QUICK_REFERENCE.md`, `*QUICK_REF.md`)
- Multiple duplicate workflow files
- Temporary fix scripts that were never removed
- Test artifacts and Python cache files
- Duplicate install scripts and requirements files
- Redundant documentation scattered across multiple locations

This created significant noise and made the repository feel unmaintained despite active development.

## Solution

### 1. Cleanup Scripts

Three cleanup scripts are provided to remove all redundant files:

```bash
# Python (cross-platform)
python scripts/cleanup_repository.py

# PowerShell (Windows)
.\scripts\cleanup_repository.ps1          # Execute cleanup
.\scripts\cleanup_repository.ps1 -DryRun  # Preview without deleting

# Bash (Linux/macOS)
bash scripts/cleanup_repository.sh         # Execute cleanup
bash scripts/cleanup_repository.sh --dry-run  # Preview without deleting
```

See `scripts/CLEANUP_README.md` for detailed documentation.

### 2. Updated .gitignore

The `.gitignore` file now includes patterns to prevent re-accumulation:

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
_temp_*.py

# Duplicate venvs
.venv*/
venv*/
!.venv/
!venv/
```

### 3. Pre-commit Hooks

Pre-commit hooks automatically check for problematic files before commits:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: check-redundant-docs
      name: Check for redundant documentation files
    - id: check-temp-scripts
      name: Check for temporary scripts in root
    - id: check-pyc-files
      name: Check for Python cache files
```

Install with:
```bash
pip install pre-commit
pre-commit install
```

### 4. CI/CD Checks

The `repository-hygiene.yml` workflow runs on every PR and push to ensure:
- No implementation summary files
- No quick reference duplicates
- No summary files
- No temporary scripts in root
- No Python cache files
- No duplicate virtual environments

## Best Practices

### Documentation

**DO:**
- Put user-facing documentation in `docs/`
- Keep developer info in `AGENTS.md` or `CONTRIBUTING.md`
- Use `CHANGELOG.md` for version history
- Use `README.md` for project overview

**DON'T:**
- Create `*_IMPLEMENTATION_SUMMARY.md` files
- Create `*_QUICK_REFERENCE.md` files
- Create `PROJECT_SUMMARY.md` or similar status docs
- Duplicate documentation across multiple locations

### Scripts

**DO:**
- Keep utility scripts in `scripts/`
- Name scripts descriptively (e.g., `cleanup_repository.py`)
- Remove temporary scripts after use
- Document scripts in `scripts/README.md`

**DON'T:**
- Leave temporary scripts in root (e.g., `fix_*.py`, `repro_*.py`)
- Create one-off fix scripts without cleaning them up
- Commit test or reproduction scripts

### Workflows

**DO:**
- Keep workflows consolidated and well-organized
- Remove obsolete workflows promptly
- Document workflow purposes in comments

**DON'T:**
- Duplicate functionality across multiple workflows
- Leave disabled or commented-out workflows
- Create temporary workflows without cleaning them up

### Testing

**DO:**
- Keep tests in `tests/` directory
- Use `.gitignore` for test artifacts
- Clean up test data after runs

**DON'T:**
- Commit test artifacts (`.pytest_cache`, `*.prof`, etc.)
- Commit Python cache files (`__pycache__`, `*.pyc`)
- Leave temporary test files in root

### Dependencies

**DO:**
- Use `pyproject.toml` for package metadata and extras
- Use `requirements.txt` for main dependencies
- Use `requirements-dev.txt` for dev dependencies

**DON'T:**
- Create feature-specific requirements files
- Duplicate dependencies across multiple files
- Commit lock files unless necessary

### Virtual Environments

**DO:**
- Use `.venv` or `venv` as the standard name
- Ensure venvs are in `.gitignore`

**DON'T:**
- Create multiple venvs with version numbers (`.venv312`, etc.)
- Commit virtual environment directories
- Use non-standard venv names

## File Categories

### Keep (Essential Files)

- `README.md` - Project overview
- `CHANGELOG.md` - Version history
- `CONTRIBUTING.md` - Contribution guidelines
- `SECURITY.md` - Security policy
- `LICENSE.md` - License information
- `REPOSITORY_STRUCTURE.md` - Repository organization
- `GETTING_STARTED.md` - Quick start guide
- `INSTALL.md` - Installation instructions
- `AGENTS.md` - AI agent context
- `DEPENDENCY_CHANGES.md` - Dependency changelog

### Remove (Redundant Files)

- Implementation summaries (50+ files)
- Quick reference duplicates (30+ files)
- Status/journal documents
- Duplicate release documents
- Redundant guides
- Temporary fix scripts
- Test artifacts
- Python cache files
- Duplicate venvs

## Cleanup Checklist

When performing cleanup:

- [ ] Run cleanup script with dry-run mode first
- [ ] Review what will be deleted
- [ ] Execute cleanup
- [ ] Verify repository still functions (`pytest`, `python -c "import neural"`)
- [ ] Check git status for unintended changes
- [ ] Update `.gitignore` if needed
- [ ] Install pre-commit hooks
- [ ] Commit with descriptive message

## Monitoring

### Manual Checks

Periodically check for accumulation:
```bash
# Check for implementation summaries
find . -name "*IMPLEMENTATION_SUMMARY.md"

# Check for quick references
find . -name "*QUICK_REFERENCE.md"

# Check for temporary scripts
find . -maxdepth 1 -name "fix_*.py" -o -name "repro_*.py"

# Check for Python cache
find . -name "__pycache__" -o -name "*.pyc" | grep -v ".venv"
```

### Automated Checks

- Pre-commit hooks run on every commit
- CI workflow runs on every PR/push
- Both will fail if problematic files are detected

## Recovery

If you accidentally delete important files:

```bash
# View deleted files
git log --diff-filter=D --summary

# Restore a specific file
git checkout HEAD~1 -- path/to/file

# Restore from a specific commit
git checkout <commit-hash> -- path/to/file
```

## Questions?

If you're unsure whether a file should be kept:

1. Check if it's in the "Keep" list above
2. Ask: "Is this file essential for users or developers?"
3. Ask: "Is this information available elsewhere?"
4. When in doubt, move to `docs/archive/` instead of deleting
5. Open an issue for discussion

## References

- [Cleanup Scripts Documentation](scripts/CLEANUP_README.md)
- [Pre-commit Configuration](.pre-commit-config.yaml)
- [Repository Hygiene Workflow](.github/workflows/repository-hygiene.yml)
- [.gitignore Patterns](.gitignore)
