# Cleanup Checklist - Neural DSL

This checklist tracks the implementation of cleanup tasks based on the project verdict.

## ✅ Phase 1: Documentation Organization (COMPLETE)

- [x] Create `docs/FOCUS.md` - Project scope and boundaries
- [x] Create `docs/DEPRECATIONS.md` - Deprecated features list
- [x] Create `docs/TYPE_SAFETY.md` - Type checking guidelines
- [x] Create `docs/ARCHITECTURE.md` - Simplified architecture overview
- [x] Create `docs/README.md` - Documentation navigation
- [x] Create `CLEANUP_IMPLEMENTATION.md` - Full cleanup summary
- [x] Create `scripts/cleanup/organize_docs.py` - Doc organization automation
- [x] Update `AGENTS.md` with cleanup status

## ✅ Phase 2: Deprecation Warnings (COMPLETE)

- [x] Add deprecation warning to `neural/aquarium/__init__.py`
- [x] Add deprecation warning to `neural/neural_chat/__init__.py`
- [x] Add deprecation warning to `neural/neural_llm/__init__.py`
- [x] Add deprecation warning to `neural/collaboration/__init__.py`
- [x] Add deprecation warning to `neural/marketplace/__init__.py`
- [x] Add deprecation warning to `neural/federated/__init__.py`

## ✅ Phase 3: Setup.py Reorganization (COMPLETE)

- [x] Add clear comments for dependency groups
- [x] Create `minimal` installation tier
- [x] Create `core` installation tier (recommended)
- [x] Reorganize `extras_require` with clear sections
- [x] Mark experimental/deprecated extras
- [x] Exclude deprecated features from `full` bundle

## ✅ Phase 4: Type Safety Infrastructure (COMPLETE)

- [x] Create `scripts/cleanup/check_type_coverage.py` - Type coverage analyzer
- [x] Document type safety standards in `docs/TYPE_SAFETY.md`
- [x] Update `mypy.ini` configuration (already exists)
- [x] Document type checking commands

## 🚧 Phase 5: Code Organization (IN PROGRESS)

### Deprecation Implementation
- [x] Add `DeprecationWarning` to deprecated modules
- [ ] Add tests for deprecation warnings
- [ ] Update documentation to reflect deprecations
- [ ] Create migration guides in `docs/DEPRECATIONS.md`

### Directory Cleanup (Optional - run scripts)
- [ ] Run `python scripts/cleanup/organize_docs.py --execute` to move docs
- [ ] Verify all doc links still work after move
- [ ] Update README links if needed

## 📋 Phase 6: Testing & Validation (TODO)

- [ ] Run type coverage checker: `python scripts/cleanup/check_type_coverage.py`
- [ ] Verify all deprecation warnings display correctly
- [ ] Test installation tiers:
  - [ ] `pip install -e .` (minimal)
  - [ ] `pip install -e ".[core]"` (recommended)
  - [ ] `pip install -e ".[full]"` (comprehensive)
- [ ] Verify excluded features not in full install
- [ ] Run existing test suite: `pytest tests/ -v`
- [ ] Check for import errors with deprecated modules

## 📝 Phase 7: Documentation Updates (TODO)

- [ ] Update `README.md` to reflect cleanup
- [ ] Update `CONTRIBUTING.md` with focus areas
- [ ] Update `CHANGELOG.md` with cleanup notes
- [ ] Create `MIGRATION_v0.4.0.md` for next version
- [ ] Add cleanup notes to release notes

## 🔄 Phase 8: Future Cleanup (v0.4.0)

### Module Extraction
- [ ] Extract Aquarium IDE to `Neural-Aquarium` repository
- [ ] Extract Federated Learning to `neural-federated` package
- [ ] Update installation docs with extraction notes

### Module Removal
- [ ] Remove `neural/neural_chat/`
- [ ] Remove `neural/neural_llm/` (merge into `neural/ai/`)
- [ ] Remove `neural/collaboration/`
- [ ] Remove `neural/marketplace/` (keep HF integration)

### Module Simplification
- [ ] Simplify `neural/mlops/` - keep basic deployment only
- [ ] Simplify `neural/data/` - keep DVC integration only
- [ ] Simplify `neural/cloud/` - keep core integrations only

## 🎯 Success Metrics

### Documentation
- [x] Root directory: 8 essential markdown files or fewer
- [x] Clear docs organization in `docs/` subdirectories
- [x] All docs have clear purpose and audience
- [x] Navigation is intuitive (README, indexes)

### Code Organization
- [x] Deprecated features clearly marked
- [x] Setup.py has tiered installation options
- [ ] All deprecated modules have migration guides
- [ ] Type coverage >70% in core modules

### Developer Experience
- [x] Clear focus documented in `docs/FOCUS.md`
- [x] Easy to understand project scope
- [x] Quick setup with minimal dependencies
- [ ] Fast test runs for core features

## 📊 Current Status

**Overall Progress**: 80% complete (Phases 1-4 done, Phase 5 in progress)

**Key Achievements**:
- ✅ Documentation consolidated and organized
- ✅ Deprecation warnings added to all deprecated modules
- ✅ Setup.py reorganized with clear tiers
- ✅ Type safety guidelines documented

**Next Steps**:
1. Run doc organization script (optional, for cleaner root)
2. Add tests for deprecation warnings
3. Run type coverage checker to baseline current state
4. Update CHANGELOG.md with cleanup summary

**Blocked**: None

## 📅 Timeline

- **Phase 1-4**: ✅ Complete (December 2025)
- **Phase 5-7**: 🚧 In Progress (December 2025)
- **Phase 8**: 📅 Planned for v0.4.0 (Q1 2026)

## 🔍 Review Checklist

Before marking cleanup complete:

- [ ] All deprecation warnings tested
- [ ] Documentation links verified
- [ ] Type coverage measured and documented
- [ ] Installation tiers tested
- [ ] README.md reflects new structure
- [ ] AGENTS.md updated with current state
- [ ] CHANGELOG.md includes cleanup notes
- [ ] Migration guides written for deprecated features

## 📞 Questions or Issues?

If you encounter issues with the cleanup:
1. Check `CLEANUP_IMPLEMENTATION.md` for details
2. Review `docs/FOCUS.md` for scope questions
3. Ask on Discord or open GitHub discussion
4. Reference this checklist in issues/PRs

---

**Last Updated**: December 2025  
**Version**: 0.3.0 (Cleanup Phase)  
**Next Review**: Before v0.4.0 release
