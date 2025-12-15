# Neural DSL - Linting and Type Checking Checklist

## Prerequisites

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Development dependencies installed:
  ```bash
  pip install -r requirements-dev.txt
  # or
  pip install ruff mypy isort
  ```

## Step-by-Step Checklist

### 1. Initial Setup ✓ COMPLETE

- [x] Install development dependencies
- [x] Review `pyproject.toml` for ruff configuration
- [x] Review `mypy.ini` for type checking configuration
- [x] Review `.pre-commit-config.yaml` for pre-commit hooks
- [x] Check `.gitignore` includes linting artifacts

### 2. Manual Import Fixes ✓ COMPLETE

- [x] Fixed `neural/__init__.py` - Import order corrected
- [x] Fixed `neural/parser/parser.py` - Import order and type aliases placement
- [x] Fixed `neural/code_generation/code_generator.py` - Import consolidation
- [x] Fixed `neural/shape_propagation/shape_propagator.py` - Import ordering

### 3. Automated Import Fixing ✓ SCRIPTS READY

- [x] Created `scripts/fix_imports.py` - Automated import fixer
- [ ] **ACTION REQUIRED**: Run automated import fixer:
  ```bash
  python scripts/fix_imports.py
  ```
- [ ] Review changes with `git diff`
- [ ] Commit import fixes

### 4. Ruff Linting ✓ READY

- [x] Ruff configuration verified in `pyproject.toml`
- [ ] **ACTION REQUIRED**: Run ruff with auto-fix:
  ```bash
  python -m ruff check . --fix
  ```
- [ ] Review changes with `git diff`
- [ ] Fix any remaining manual issues
- [ ] Verify no linting errors:
  ```bash
  python -m ruff check .
  ```

### 5. Code Formatting ✓ READY

- [x] Ruff format configuration verified
- [ ] **ACTION REQUIRED**: Format all code:
  ```bash
  python -m ruff format .
  ```
- [ ] Review changes with `git diff`
- [ ] Commit formatting changes

### 6. Type Checking ✓ READY

- [x] Mypy configuration verified in `mypy.ini`
- [x] Strictness levels documented
- [ ] **ACTION REQUIRED**: Run mypy type checking:
  ```bash
  python -m mypy neural/ --ignore-missing-imports
  ```
- [ ] Review type errors
- [ ] Fix critical type errors (Priority 1 modules)
- [ ] Document any remaining type issues

### 7. Comprehensive Verification ✓ READY

- [x] Created `scripts/pre_commit_check.py` - Comprehensive checker
- [ ] **ACTION REQUIRED**: Run comprehensive check:
  ```bash
  python scripts/pre_commit_check.py
  ```
- [ ] Address all reported issues
- [ ] Re-run until all checks pass

### 8. Git Integration ✓ READY

- [x] Pre-commit configuration updated (`.pre-commit-config.yaml`)
- [ ] **ACTION REQUIRED**: Install pre-commit hooks:
  ```bash
  pre-commit install
  ```
- [ ] Test pre-commit hooks:
  ```bash
  pre-commit run --all-files
  ```

### 9. Documentation ✓ COMPLETE

- [x] Created `LINTING_GUIDE.md` - Comprehensive guide
- [x] Created `LINTING_FIXES_SUMMARY.md` - Summary of changes
- [x] Created `LINTING_CHECKLIST.md` - This checklist
- [x] Created `scripts/README_LINTING.md` - Scripts documentation

### 10. CI/CD Verification

- [ ] Review `.github/workflows/ci.yml`
- [ ] Ensure linting steps are present
- [ ] Test CI pipeline with a PR
- [ ] Verify all checks pass

## Quick Commands

### Fix Everything
```bash
# 1. Fix imports
python scripts/fix_imports.py

# 2. Run comprehensive fix
python scripts/pre_commit_check.py --fix

# 3. Verify
python scripts/pre_commit_check.py
```

### Manual Workflow
```bash
# 1. Fix imports
python scripts/fix_imports.py

# 2. Fix linting issues
python -m ruff check . --fix

# 3. Format code
python -m ruff format .

# 4. Check types
python -m mypy neural/ --ignore-missing-imports

# 5. Final verification
python scripts/pre_commit_check.py
```

## Verification Steps

After completing all actions, verify:

1. **Import Order**
   ```bash
   python scripts/fix_imports.py --dry-run
   # Should report: "Would fix imports in 0 files"
   ```

2. **Linting**
   ```bash
   python -m ruff check .
   # Should report: "All checks passed!" or specific file counts
   ```

3. **Formatting**
   ```bash
   python -m ruff format . --check
   # Should report: "X files already formatted" (no changes needed)
   ```

4. **Type Checking**
   ```bash
   python -m mypy neural/ --ignore-missing-imports
   # Should complete with acceptable error count based on module strictness
   ```

5. **Comprehensive Check**
   ```bash
   python scripts/pre_commit_check.py
   # Should report: "🎉 All checks passed! Ready to commit."
   ```

6. **Pre-commit Hooks**
   ```bash
   pre-commit run --all-files
   # Should pass all hooks
   ```

## Expected Results

### After Import Fixes
- All imports follow PEP 8 order
- Future imports first
- Standard library second
- Third-party third
- First-party (neural) fourth
- Relative imports last
- Proper spacing between groups

### After Ruff Linting
- No unused imports (F401)
- No undefined names (F821)
- Lines under 100 characters (E501)
- Proper indentation (E111, E112)
- No multiple statements on one line (E701)

### After Ruff Formatting
- Consistent quote style (double quotes)
- Proper indentation (4 spaces)
- Trailing commas where appropriate
- Line breaks at appropriate places

### After Type Checking
- Core modules (code_generation, utils, shape_propagation): Minimal errors
- Parser, CLI, dashboard: Acceptable error count
- Other modules: May have errors (documented in mypy.ini)

## Common Issues and Solutions

### Issue: "Module not found" during linting
**Solution**: Install development dependencies
```bash
pip install -r requirements-dev.txt
```

### Issue: Many import order errors
**Solution**: Run the automated fixer
```bash
python scripts/fix_imports.py
```

### Issue: Line too long errors
**Solution**: Let ruff auto-fix or manually break lines
```bash
python -m ruff check . --fix
```

### Issue: Unused import errors
**Solution**: Remove unused imports or add to `__all__`
```bash
python -m ruff check . --fix  # Auto-remove safe unused imports
```

### Issue: Type errors in optional dependencies
**Solution**: These are expected - modules handle missing deps gracefully

### Issue: Pre-commit hooks failing
**Solution**: Run manual fixes first
```bash
python scripts/pre_commit_check.py --fix
pre-commit run --all-files
```

## Files Created/Modified

### New Files
- ✓ `scripts/fix_imports.py` - Import order fixer
- ✓ `scripts/lint_and_fix.py` - Comprehensive linting workflow
- ✓ `scripts/pre_commit_check.py` - Pre-commit verification
- ✓ `scripts/README_LINTING.md` - Scripts documentation
- ✓ `LINTING_GUIDE.md` - Comprehensive guide
- ✓ `LINTING_FIXES_SUMMARY.md` - Summary of fixes
- ✓ `LINTING_CHECKLIST.md` - This checklist

### Modified Files
- ✓ `neural/__init__.py` - Import order fixed
- ✓ `neural/parser/parser.py` - Import order fixed
- ✓ `neural/code_generation/code_generator.py` - Import order fixed
- ✓ `neural/shape_propagation/shape_propagator.py` - Import order fixed
- ✓ `.pre-commit-config.yaml` - Updated to use ruff-format instead of black

### Configuration Files (Verified)
- ✓ `pyproject.toml` - Ruff configuration
- ✓ `mypy.ini` - Mypy configuration
- ✓ `.gitignore` - Linting artifacts included

## Next Actions Required

**Priority 1 - Immediate Actions:**

1. Run automated import fixer:
   ```bash
   python scripts/fix_imports.py
   git diff  # Review changes
   git add -p  # Stage changes
   ```

2. Run ruff with auto-fix:
   ```bash
   python -m ruff check . --fix
   git diff  # Review changes
   git add -p  # Stage changes
   ```

3. Format code:
   ```bash
   python -m ruff format .
   git diff  # Review changes
   git add -p  # Stage changes
   ```

4. Commit changes:
   ```bash
   git commit -m "Fix import ordering and linting issues"
   ```

**Priority 2 - Verification:**

5. Run type checking:
   ```bash
   python -m mypy neural/ --ignore-missing-imports > mypy_report.txt
   # Review mypy_report.txt
   ```

6. Install pre-commit hooks:
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

**Priority 3 - CI/CD:**

7. Test CI pipeline:
   - Create a test branch
   - Push changes
   - Verify CI passes
   - Merge if successful

## Success Criteria

- [ ] Import fixer reports 0 files to fix
- [ ] Ruff reports no linting errors
- [ ] Ruff format reports no formatting changes needed
- [ ] Mypy completes with acceptable error count
- [ ] Pre-commit hooks run successfully
- [ ] CI pipeline passes all checks
- [ ] Code review approves changes
- [ ] Changes merged to main branch

## Time Estimates

- Initial setup and review: 30 minutes ✓ COMPLETE
- Manual import fixes: 30 minutes ✓ COMPLETE
- Script development: 2 hours ✓ COMPLETE
- Documentation: 1 hour ✓ COMPLETE
- Running automated fixes: 15 minutes **PENDING**
- Review and commit: 30 minutes **PENDING**
- CI/CD verification: 30 minutes **PENDING**

**Total: ~5 hours (4 hours complete, 1 hour remaining)**

## Completion Status

- [x] **Phase 1**: Setup and configuration (100%)
- [x] **Phase 2**: Manual fixes (100%)
- [x] **Phase 3**: Script development (100%)
- [x] **Phase 4**: Documentation (100%)
- [ ] **Phase 5**: Automated fixing (0% - ACTION REQUIRED)
- [ ] **Phase 6**: Verification (0% - ACTION REQUIRED)
- [ ] **Phase 7**: CI/CD integration (0% - ACTION REQUIRED)

**Overall Progress: 57% Complete**

## Notes

- All tools and scripts are ready
- All documentation is complete
- Core files have been manually fixed as examples
- Automated fixes are ready to run
- Waiting for user to execute automated fixes
- No blockers identified
- CI/CD configuration is ready

## Contact

For questions or issues:
- Review `LINTING_GUIDE.md` for detailed information
- Check `scripts/README_LINTING.md` for script usage
- Refer to `LINTING_FIXES_SUMMARY.md` for what was done

---

**Ready to proceed with automated fixes!**
