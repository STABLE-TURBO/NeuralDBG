# Neural DSL Release Workflow

This document describes the automated release workflow for Neural DSL, including version management, testing, publishing, and PyPI trusted publishing setup.

## Overview

The automated release workflow provides a comprehensive, production-ready release pipeline that includes:

- **Pre-release validation**: Tests, linting, and type checking
- **Automated versioning**: Automatic version bumping in `setup.py` and `neural/__init__.py`
- **README badge updates**: Automatic version and release date badge updates
- **Changelog parsing**: Automatic release notes generation from `CHANGELOG.md`
- **GitHub releases**: Automated GitHub release creation with artifacts
- **PyPI publishing**: Secure publishing using OIDC trusted publishing (no API tokens needed)
- **Post-release tasks**: Blog generation and social media updates

## Workflow Triggers

### Manual Trigger (Workflow Dispatch)

Trigger a release manually via GitHub Actions:

1. Go to **Actions** tab in GitHub
2. Select **Automated Release** workflow
3. Click **Run workflow**
4. Configure options:
   - **Version bump type**: `patch`, `minor`, or `major`
   - **Skip tests**: Skip test execution (not recommended)
   - **Skip lint**: Skip linting and type checking (not recommended)
   - **Draft release**: Create a draft release (for review before publishing)
   - **Test PyPI**: Publish to TestPyPI instead of production PyPI

### Automatic Trigger (Tag Push)

The workflow automatically runs when a version tag is pushed:

```bash
git tag v0.3.1
git push origin v0.3.1
```

## Pre-Release Validation

Before creating a release, the workflow performs comprehensive validation:

### 1. Linting (Ruff)

```bash
python -m ruff check . --output-format=github
```

Validates code style and catches common errors. Uses GitHub output format for annotations.

### 2. Type Checking (mypy)

```bash
python -m mypy neural/code_generation neural/utils --ignore-missing-imports
```

Performs static type checking on critical modules. Non-blocking to allow gradual typing improvements.

### 3. Test Suite

```bash
python -m pytest tests/ -v --tb=short --maxfail=5
```

Runs the full test suite with verbose output and stops after 5 failures.

### 4. Example Validation

```bash
python scripts/automation/example_validator.py
```

Validates all example DSL files to ensure they parse correctly.

## Version Management

### Automatic Version Bumping

The workflow automatically updates version numbers in:

1. **`setup.py`**: Package version
2. **`neural/__init__.py`**: `__version__` constant

Version format: `MAJOR.MINOR.PATCH` (e.g., `0.3.1`)

### Version Bump Types

- **patch**: `0.3.0` → `0.3.1` (bug fixes)
- **minor**: `0.3.0` → `0.4.0` (new features, backward compatible)
- **major**: `0.3.0` → `1.0.0` (breaking changes)

The workflow strips `-dev` suffixes before bumping.

## Release Notes Generation

Release notes are automatically generated from `CHANGELOG.md`:

### Changelog Format

```markdown
## [0.3.1] - 2025-01-15

### Added
- New feature description

### Fixed
- Bug fix description

### Improved
- Enhancement description
```

### Release Notes Script

```bash
python scripts/automation/parse_changelog.py 0.3.1
```

The script:
1. Extracts the section for the specified version
2. Formats it with installation instructions
3. Adds links to documentation and community resources

## README Badge Updates

Automatically updates badges in `README.md`:

```bash
python scripts/automation/update_badges.py 0.3.1
```

Updates:
- Version badge: `![Version](https://img.shields.io/badge/version-0.3.1-blue.svg)`
- Release date badge: `![Release Date](https://img.shields.io/badge/release_date-2025-01-15-green.svg)`
- Installation examples with specific version
- Beta status warning version references

## PyPI Publishing with Trusted Publishing

### What is Trusted Publishing?

Trusted Publishing (OIDC) is a secure method for publishing to PyPI without using API tokens. It uses GitHub's identity to authenticate directly with PyPI.

### Benefits

- ✅ **No API tokens to manage**: No secrets to store or rotate
- ✅ **Enhanced security**: Authentication tied to specific workflow and repository
- ✅ **Audit trail**: Clear provenance of published packages
- ✅ **Zero configuration**: No credentials needed in GitHub secrets

### Setup Instructions

#### 1. Configure PyPI Trusted Publisher

1. Log in to [PyPI](https://pypi.org/)
2. Go to your project: https://pypi.org/project/neural-dsl/
3. Navigate to **Manage** → **Publishing**
4. Click **Add a new publisher**
5. Configure the publisher:
   - **Owner**: `Lemniscate-world`
   - **Repository**: `Neural`
   - **Workflow name**: `automated_release.yml`
   - **Environment name**: `pypi`

#### 2. Create GitHub Environment

1. Go to repository **Settings** → **Environments**
2. Create environment named `pypi`
3. Add environment protection rules (optional but recommended):
   - Required reviewers
   - Wait timer
   - Deployment branches (e.g., only `main`)

#### 3. Workflow Configuration

The workflow is already configured with:

```yaml
permissions:
  id-token: write  # Required for trusted publishing
  contents: write  # Required for creating releases

environment:
  name: pypi
  url: https://pypi.org/project/neural-dsl/
```

### Publishing Process

#### Production PyPI

```yaml
- name: Publish to PyPI (Trusted Publishing)
  if: ${{ !inputs.test_pypi && !inputs.draft }}
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    packages-dir: dist/
    verbose: true
    print-hash: true
```

#### TestPyPI

For testing releases:

```yaml
- name: Publish to TestPyPI (Trusted Publishing)
  if: ${{ inputs.test_pypi }}
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://test.pypi.org/legacy/
    packages-dir: dist/
    verbose: true
```

## Complete Release Process

### Step-by-Step

1. **Prepare changelog**: Update `CHANGELOG.md` with changes for new version
2. **Trigger workflow**: Manual dispatch or push tag
3. **Pre-release validation**: 
   - Linting (Ruff)
   - Type checking (mypy)
   - Tests (pytest)
   - Example validation
4. **Version update**:
   - Calculate new version
   - Update `setup.py`
   - Update `neural/__init__.py`
   - Update README badges
5. **Commit and tag**:
   - Commit version changes
   - Create and push git tag
6. **Build package**:
   - Clean old builds
   - Build source distribution and wheel
   - Verify package integrity
7. **Parse changelog**:
   - Extract release notes
   - Format with installation instructions
8. **GitHub release**:
   - Create release with tag
   - Attach build artifacts
   - Include release notes
9. **PyPI publishing**:
   - Authenticate via OIDC
   - Publish to PyPI or TestPyPI
10. **Post-release**:
    - Generate blog posts
    - Update documentation
    - Social media announcements

### Workflow Jobs

```
pre-release-validation (validates code)
    ↓
release (creates release and publishes)
    ↓
post-release (announcements and updates)
```

## Usage Examples

### Release a Patch Version

```bash
# Via GitHub UI (recommended)
Actions → Automated Release → Run workflow
  - Version bump: patch
  - Keep other defaults

# Or via command line
python scripts/automation/release_automation.py --version-type patch
```

### Release a Minor Version

```bash
# Via GitHub UI
Actions → Automated Release → Run workflow
  - Version bump: minor

# Or via command line
python scripts/automation/release_automation.py --version-type minor
```

### Create Draft Release

```bash
# Via GitHub UI
Actions → Automated Release → Run workflow
  - Draft release: ✓

# Or via command line
python scripts/automation/release_automation.py --draft
```

### Test on TestPyPI

```bash
# Via GitHub UI
Actions → Automated Release → Run workflow
  - Test PyPI: ✓

# Or via command line
python scripts/automation/release_automation.py --test-pypi
```

## Local Testing

### Test Version Bumping

```bash
python scripts/automation/release_automation.py --version-type patch --skip-tests --use-trusted-publishing
```

### Test Changelog Parsing

```bash
python scripts/automation/parse_changelog.py 0.3.0
```

### Test Badge Updates

```bash
python scripts/automation/update_badges.py 0.3.1
```

### Test Build Only

```bash
python -m build
python -m twine check dist/*
```

## Troubleshooting

### Build Failures

**Issue**: Package build fails

**Solution**:
```bash
# Clean build artifacts
rm -rf dist/ build/ *.egg-info

# Rebuild
python -m build
```

### PyPI Publishing Fails

**Issue**: Trusted publishing authentication fails

**Solution**:
1. Verify PyPI publisher configuration matches workflow
2. Check environment name is exactly `pypi`
3. Ensure workflow has `id-token: write` permission
4. Verify repository owner and name are correct

### Version Already Exists

**Issue**: PyPI rejects upload because version exists

**Solution**:
- PyPI does not allow re-uploading same version
- Bump version and try again
- For testing, use TestPyPI instead

### Tests Fail

**Issue**: Pre-release validation fails

**Solution**:
```bash
# Run tests locally
python -m pytest tests/ -v

# Fix failing tests
# Re-run workflow
```

### Badge Updates Don't Work

**Issue**: README badges not updated

**Solution**:
```bash
# Test locally
python scripts/automation/update_badges.py 0.3.1

# Check regex patterns match your badge format
# Commit and push changes
```

## Security Best Practices

### Trusted Publishing

- ✅ Use trusted publishing instead of API tokens
- ✅ Configure environment protection rules
- ✅ Limit workflow to specific branches
- ✅ Review releases before non-draft publishing

### Secrets Management

- ❌ Never commit API tokens
- ❌ Never use personal access tokens for publishing
- ✅ Use OIDC trusted publishing
- ✅ Use GitHub environments for additional security

### Review Process

1. Create draft release first
2. Review release notes and artifacts
3. Test installation from TestPyPI
4. Publish to production PyPI only when ready

## Monitoring and Verification

### After Release

1. **Check GitHub Release**: 
   - https://github.com/Lemniscate-world/Neural/releases/tag/vX.Y.Z
   
2. **Verify PyPI Package**: 
   - https://pypi.org/project/neural-dsl/X.Y.Z/
   
3. **Test Installation**:
   ```bash
   pip install neural-dsl==X.Y.Z
   python -c "import neural; print(neural.__version__)"
   ```

4. **Check Badges**: 
   - README.md should show new version
   
5. **Review Artifacts**: 
   - GitHub Actions artifacts should include release assets

## Migration from Token-Based Publishing

### Old Method (Deprecated)

```yaml
env:
  TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
  TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```

### New Method (Current)

```yaml
permissions:
  id-token: write
environment:
  name: pypi
uses: pypa/gh-action-pypi-publish@release/v1
```

### Migration Steps

1. Configure PyPI trusted publisher (see setup above)
2. Remove old secrets from GitHub (optional)
3. Update workflow to use trusted publishing
4. Test with TestPyPI first
5. Publish to production PyPI

## References

- [PyPI Trusted Publishers Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OIDC](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [Python Build Documentation](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Semantic Versioning](https://semver.org/)

## Support

For questions or issues with the release workflow:

- Open an issue: https://github.com/Lemniscate-world/Neural/issues
- Discord: https://discord.gg/KFku4KvS
- Email: Lemniscate_zero@proton.me
