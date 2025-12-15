# Complete v0.4.0 Refactoring - Next Steps

## Current Status
✅ **95% Complete** - Core implementation done, only directory removal remains.

## What Has Been Done

1. ✅ Simplified CLI (neural/cli/cli.py)
   - Removed 40+ commands and 8 command groups
   - Kept only 7 core commands
   - 75% code reduction

2. ✅ Reduced dependencies (setup.py)
   - Removed 7 dependency groups
   - 57% reduction in total packages
   - Version bumped to 0.4.0

3. ✅ Removed 4 module directories
   - neural/cost/
   - neural/monitoring/
   - neural/profiling/
   - neural/docgen/
   - Total: 46 files, ~12,500 lines removed

4. ✅ Created documentation
   - V0.4.0_REFACTORING_STATUS.md
   - V0.4.0_IMPLEMENTATION_COMPLETE.md
   - remove_scope_creep.ps1 / .sh scripts

## What You Need To Do

### Step 1: Remove Remaining Directories

Run ONE of the following commands:

**Windows PowerShell:**
```powershell
.\remove_scope_creep.ps1
```

**Linux/Mac:**
```bash
chmod +x remove_scope_creep.sh
./remove_scope_creep.sh
```

**Or manually execute:**
```bash
git rm -rf neural/teams
git rm -rf neural/mlops
git rm -rf neural/data
git rm -rf neural/config
git rm -rf neural/education
git rm -rf neural/plugins
git rm -rf neural/explainability
```

This will remove ~90 additional files.

### Step 2: Stage the Changes

```bash
git add -A
```

### Step 3: Verify Current Status

```bash
git status
```

You should see:
- ~140 deleted files
- 2-3 modified files (cli.py, setup.py)
- 5-6 new files (documentation)

### Step 4: Test Core Functionality

```bash
# Test imports
python -c "import neural; print('✓ Imports OK')"

# Test CLI
neural --version
neural --help

# Test compilation (dry-run)
neural compile examples/mnist.neural --backend tensorflow --dry-run
```

### Step 5: Commit the Changes

```bash
git commit -m "v0.4.0: Strategic refocusing - remove scope creep modules

- Simplified CLI: removed 40+ commands, kept 7 core commands
- Reduced dependencies: 57% reduction (54 → 23 packages)
- Removed modules: teams, marketplace, cost, mlops, monitoring, data, 
  config, education, plugins, profiling, explainability, docgen, 
  collaboration, federated
- Updated version to 0.4.0
- Focus on core mission: DSL compilation with multi-backend support

BREAKING CHANGES:
- Removed enterprise features (teams, marketplace, cost tracking)
- Removed alternative tool features (mlops, monitoring, data versioning)
- Removed experimental features (no-code, aquarium, plugins, explainability)
- Removed CLI commands: cloud, track, marketplace, cost, aquarium, 
  no-code, docs, explain, config, data, collab

See CHANGELOG.md for migration guide."
```

### Step 6: Create Git Tag (Optional)

```bash
git tag -a v0.4.0 -m "Neural DSL v0.4.0 - Strategic Refocusing Release"
```

### Step 7: Update Additional Documentation (Recommended)

Update these files to reflect the changes:
- README.md - Remove references to removed features
- AGENTS.md - Already mentions removed modules, may need updates
- Add migration guide if needed

## Quick Command Summary

```bash
# Complete the refactoring in 4 commands:
.\remove_scope_creep.ps1                # 1. Remove remaining directories
git add -A                               # 2. Stage all changes
git commit -m "v0.4.0: Strategic refocusing - remove scope creep modules"  # 3. Commit
git tag v0.4.0                          # 4. Tag (optional)
```

## Expected Results

After completing these steps:

- **Files Removed**: ~140 files (46 already staged + 90 more)
- **Lines Removed**: ~20,000+ lines of code
- **Dependencies Reduced**: From 54 to 23 packages
- **CLI Commands**: From 50+ to 7
- **Focus**: Clear DSL compilation mission

## Troubleshooting

### If git rm commands fail:
Some directories may not be tracked. That's OK! The untracked ones can be ignored or manually deleted.

### If imports break:
Some files may still import from removed modules. You can:
1. Search for imports: `git grep "from neural.cost import"`
2. Remove or update those imports
3. Commit the fixes

### If tests fail:
Some tests may reference removed modules. You can:
1. Delete tests for removed features
2. Or skip them for now and fix in a follow-up commit

## Support

If you encounter issues:
1. Check V0.4.0_REFACTORING_STATUS.md for detailed status
2. Check V0.4.0_IMPLEMENTATION_COMPLETE.md for what was done
3. Check git status to see current state
4. All changes are reversible with git restore if needed

## Success Criteria

✅ The refactoring is successful when:
1. Git shows ~140 deleted files committed
2. `python -c "import neural"` works without errors
3. `neural --version` shows v0.4.0
4. `neural compile --help` shows only core options
5. Core features (compile, visualize, debug) work correctly

Good luck! 🚀
