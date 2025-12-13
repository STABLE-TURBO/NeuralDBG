# Release Quick Start Guide

Quick reference for creating Neural DSL releases.

## Prerequisites

### One-Time Setup

1. **Configure PyPI Trusted Publishing**
   - Go to https://pypi.org/manage/project/neural-dsl/settings/publishing/
   - Add publisher:
     - Owner: `Lemniscate-world`
     - Repository: `Neural`
     - Workflow: `automated_release.yml`
     - Environment: `pypi`

2. **Create GitHub Environment**
   - Go to repository Settings → Environments
   - Create environment `pypi`
   - (Optional) Add protection rules

## Making a Release

### Via GitHub Actions (Recommended)

1. Navigate to **Actions** tab
2. Select **Automated Release** workflow
3. Click **Run workflow**
4. Configure:
   - **Version bump**: Choose `patch`, `minor`, or `major`
   - **Skip tests**: Leave unchecked
   - **Skip lint**: Leave unchecked
   - **Draft release**: Check for review before publishing
   - **Test PyPI**: Check to test on TestPyPI first
5. Click **Run workflow**

### Via Command Line

```bash
# Patch release (0.3.0 → 0.3.1)
python scripts/automation/release_automation.py --version-type patch

# Minor release (0.3.0 → 0.4.0)
python scripts/automation/release_automation.py --version-type minor

# Major release (0.3.0 → 1.0.0)
python scripts/automation/release_automation.py --version-type major

# Test on TestPyPI first
python scripts/automation/release_automation.py --test-pypi

# Create draft release
python scripts/automation/release_automation.py --draft

# Skip tests (not recommended)
python scripts/automation/release_automation.py --skip-tests

# Use trusted publishing (for GitHub Actions)
python scripts/automation/release_automation.py --use-trusted-publishing
```

## Pre-Release Checklist

- [ ] Update `CHANGELOG.md` with new version section
- [ ] All tests passing locally: `pytest tests/ -v`
- [ ] Linting clean: `ruff check .`
- [ ] Type checking clean: `mypy neural/code_generation neural/utils`
- [ ] Version number correct in changelog
- [ ] All features documented
- [ ] Breaking changes clearly marked

## What Happens During Release

1. **Validation** (5-10 minutes)
   - Linting with Ruff
   - Type checking with mypy
   - Full test suite
   - Example validation

2. **Version Update**
   - Bump version in `setup.py` and `neural/__init__.py`
   - Update README badges
   - Commit changes
   - Create and push git tag

3. **Build & Publish**
   - Build source distribution and wheel
   - Verify package integrity
   - Parse CHANGELOG for release notes
   - Create GitHub release
   - Publish to PyPI via trusted publishing

4. **Post-Release**
   - Generate blog posts
   - Prepare social media updates

## After Release

### Verify

```bash
# Check GitHub release
https://github.com/Lemniscate-world/Neural/releases/tag/vX.Y.Z

# Verify PyPI
https://pypi.org/project/neural-dsl/X.Y.Z/

# Test installation
pip install neural-dsl==X.Y.Z
python -c "import neural; print(neural.__version__)"
```

### Announce

1. Discord: https://discord.gg/KFku4KvS
2. Twitter: @NLang4438
3. GitHub Discussions

## Common Scenarios

### Testing a Release

```bash
# Create draft release
Actions → Run workflow → Draft: ✓

# Or test on TestPyPI
Actions → Run workflow → Test PyPI: ✓

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ neural-dsl==X.Y.Z
```

### Emergency Hotfix

```bash
# Quick patch release
python scripts/automation/release_automation.py --version-type patch --skip-lint
```

### Multiple Releases in One Day

```bash
# First release: 0.3.0
python scripts/automation/release_automation.py --version-type minor

# Second release (hotfix): 0.3.1
python scripts/automation/release_automation.py --version-type patch
```

## Troubleshooting

### Tests Fail

```bash
# Run locally to debug
pytest tests/ -v -x

# Fix issues, then re-run workflow
```

### PyPI Publishing Fails

1. Check environment name is `pypi` (exact match)
2. Verify trusted publisher configuration on PyPI
3. Ensure workflow has `id-token: write` permission
4. Check workflow logs for specific error

### Version Already Exists

PyPI doesn't allow re-uploading. Increment version and try again:

```bash
python scripts/automation/release_automation.py --version-type patch
```

### Badge Updates Don't Show

```bash
# Test locally
python scripts/automation/update_badges.py X.Y.Z

# Check and commit changes
git add README.md
git commit -m "Update badges for vX.Y.Z"
git push
```

## Advanced Options

### Custom Version

Edit `setup.py` and `neural/__init__.py` manually before running:

```python
version = "0.3.1"  # setup.py
__version__ = "0.3.1"  # neural/__init__.py
```

Then push tag:

```bash
git tag v0.3.1
git push origin v0.3.1
```

### Rollback Release

GitHub release can be deleted, but PyPI release cannot:

```bash
# Delete GitHub release (via web UI)
# Delete tag locally and remotely
git tag -d v0.3.1
git push origin :refs/tags/v0.3.1
```

For PyPI, publish new version with fixes.

## Development Workflow

```bash
# Normal development
version = "0.3.0.dev0"  # Development version

# Before release
version = "0.3.0"  # Remove .dev suffix

# After release
version = "0.3.1.dev0"  # Next dev version
```

## Support

- Full documentation: [docs/RELEASE_WORKFLOW.md](RELEASE_WORKFLOW.md)
- Issues: https://github.com/Lemniscate-world/Neural/issues
- Discord: https://discord.gg/KFku4KvS
- Email: Lemniscate_zero@proton.me
