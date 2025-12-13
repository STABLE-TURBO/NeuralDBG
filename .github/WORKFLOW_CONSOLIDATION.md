# GitHub Actions Workflow Consolidation

## Summary of Changes

This document describes the consolidation and modernization of GitHub Actions workflows completed on this date.

## Workflows Consolidated

### 1. **python-package.yml** - DEPRECATED
- **Status**: Marked as deprecated, manual trigger only
- **Reason**: Functionality merged into `ci.yml`
- **Original Purpose**: Run tests with Python 3.9-3.11 on Ubuntu with flake8 linting
- **Migration**: All testing functionality now handled by `ci.yml` with broader coverage

### 2. **pytest.yml** - DEPRECATED
- **Status**: Marked as deprecated, manual trigger only
- **Reason**: Functionality merged into `ci.yml`
- **Original Purpose**: Run pytest with coverage, Selenium UI tests, and Discord notifications
- **Migration**: 
  - Unit/integration tests → `ci.yml` unit-tests, integration-tests, e2e-tests jobs
  - UI tests → `ci.yml` ui-tests job
  - Coverage reporting → `ci.yml` coverage-report job
  - Discord notifications preserved in ui-tests job

### 3. **ci.yml** - ENHANCED
- **Changes**:
  - Added flake8 linting from python-package.yml
  - Added UI tests with Selenium from pytest.yml
  - Added Discord webhook notifications on failure
  - All actions updated to v4/v5
  - Python setup consistently uses @v5 with pip caching
  - Checkout consistently uses @v4

## Workflows Updated to Modern Versions

### Action Version Updates

All workflows updated to use:
- `actions/checkout@v4` (was v3 in several files)
- `actions/setup-python@v5` (was v3/v4 in several files)
- `actions/cache@v4` (was v3 in pre-commit.yml)
- `actions/upload-artifact@v4` (already modern)
- `actions/download-artifact@v4` (already modern)
- `codecov/codecov-action@v4` (already modern)
- `github/codeql-action/*@v3` (already modern)

### Specific File Updates

#### **jekyll.yml**
- Updated `ruby/setup-ruby` from pinned SHA to `@v1`
- Already using modern actions (v4/v5)

#### **codacy.yml**
- Updated Codacy action from pinned SHA to `@master`
- Already using `actions/checkout@v4`
- Already using `github/codeql-action/upload-sarif@v3`

#### **complexity.yml**
- Updated `actions/checkout` from v3 → v4
- Updated `actions/setup-python` from v4 → v5
- Updated Python version from 3.9 → 3.11
- Added pip caching

#### **metrics.yml**
- Converted from minimal snippet to full workflow
- Added proper workflow structure with schedule trigger
- Already using `lowlighter/metrics@v3.34`

#### **pylint.yml**
- Updated `actions/setup-python` from v3 → v5
- Added Python 3.11 and 3.12 to matrix
- Added pip caching
- Updated to use `python -m pylint` for consistency

#### **post_release.yml**
- Updated `actions/checkout` from v3 → v4
- Updated `actions/setup-python` from v4 → v5
- Updated Python version from 3.10 → 3.11
- Added pip caching
- Improved command formatting

#### **pre-commit.yml**
- Updated `actions/cache` from v3 → v4
- Added pip caching to setup-python
- Already using v4/v5 for checkout/setup-python

#### **pypi.yml**
- Updated `actions/checkout` from v3 → v4
- Updated `actions/setup-python` from v4 → v5
- Updated Python version from '3.x' → '3.11'
- Added pip caching
- Improved pip install commands

#### **release.yml**
- Updated `actions/checkout` from v3 → v4
- Updated `actions/setup-python` from v4 → v5
- Updated `softprops/action-gh-release` from v1 → v2
- Added pip caching

## Workflows Already Modern (No Changes Needed)

The following workflows were already using modern action versions:
- `automated_release.yml` - v4/v5 actions
- `close-fixed-issues.yml` - v4/v5 actions
- `codeql.yml` - v4 checkout, v3 codeql
- `periodic_tasks.yml` - v4/v5 actions
- `pytest-to-issues.yml` - v4/v5 actions
- `python-publish.yml` - v4/v5 actions
- `security-audit.yml` - v4/v5 actions, v7 github-script
- `security.yml` - v4/v5 actions
- `validate_examples.yml` - v4/v5 actions

## Benefits of Consolidation

1. **Reduced Duplication**: Eliminated redundant test workflows
2. **Easier Maintenance**: Single source of truth for CI/CD in `ci.yml`
3. **Modern Actions**: All workflows using latest stable action versions
4. **Improved Caching**: Consistent pip caching across all Python setups
5. **Better Coverage**: Consolidated workflow has more comprehensive testing
6. **Clear Deprecation Path**: Old workflows clearly marked and disabled

## Migration Guide for Developers

### For Contributors
- Use `ci.yml` for all CI/CD checks
- Ignore deprecated workflows (they won't run automatically)
- All tests, linting, and coverage now in single workflow

### For Maintainers
- Monitor `ci.yml` for all test results
- Deprecated workflows can be deleted after transition period
- Update any external references to old workflow names

## Action Version Reference

| Action | Old Version | New Version |
|--------|-------------|-------------|
| actions/checkout | v3 | v4 |
| actions/setup-python | v3/v4 | v5 |
| actions/cache | v3 | v4 |
| actions/upload-artifact | v4 | v4 ✓ |
| actions/download-artifact | v4 | v4 ✓ |
| codecov/codecov-action | v3/v4 | v4 ✓ |
| github/codeql-action/* | v3 | v3 ✓ |
| softprops/action-gh-release | v1 | v2 |
| ruby/setup-ruby | SHA | v1 |
| codacy/codacy-analysis-cli-action | SHA | master |

## Python Version Updates

Standardized Python versions across workflows:
- Minimum: 3.8 (for compatibility)
- Default single-version jobs: 3.11
- Latest: 3.12
- Matrix tests: ["3.8", "3.9", "3.10", "3.11", "3.12"]

## Next Steps

1. Monitor consolidated CI workflow for any issues
2. After 30 days, delete deprecated workflow files
3. Update any documentation referencing old workflows
4. Consider further consolidation opportunities
