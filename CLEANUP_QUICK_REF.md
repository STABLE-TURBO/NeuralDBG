# Cleanup Quick Reference - Neural DSL

## TL;DR

Neural DSL underwent a major cleanup to address repository clutter, scope creep, and technical debt. Here's what you need to know.

## What Changed?

### 📁 Documentation Organized
- **Before**: 60+ markdown files in root
- **After**: 8 essential files in root, rest organized in `docs/`
- **See**: `docs/README.md` for navigation

### ⚠️ Features Deprecated
Several out-of-scope features are deprecated and will be removed in v0.4.0:
- Aquarium IDE → Extracting to separate repo
- Collaboration tools → Use Git
- Marketplace → Use HuggingFace Hub
- Federated learning → Extracting to separate repo
- Neural Chat/LLM → Use neural.ai instead

**Migration**: See `docs/DEPRECATIONS.md`

### 📦 Installation Simplified
New tiered installation options:
```bash
pip install -e .              # Minimal (core DSL only)
pip install -e ".[core]"      # Recommended (+ backends, dashboard)
pip install -e ".[full]"      # Everything (excluding deprecated)
```

### 🔍 Type Safety Improved
- Core modules now require type hints
- Type coverage tool available
- Guidelines in `docs/TYPE_SAFETY.md`

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/FOCUS.md` | Project scope and boundaries ⭐ |
| `docs/DEPRECATIONS.md` | What's deprecated and why |
| `docs/TYPE_SAFETY.md` | Type checking guidelines |
| `docs/ARCHITECTURE.md` | System architecture overview |
| `CLEANUP_CHECKLIST.md` | Implementation tracking |
| `IMPLEMENTATION_SUMMARY.md` | Detailed cleanup report |

## For Users

### Installation
Choose your installation tier:
- **Minimal**: Just trying it out? Use `pip install -e .`
- **Core**: Regular use? Use `pip install -e ".[core]"`
- **Full**: Need everything? Use `pip install -e ".[full]"`

### Deprecated Features
If you see a deprecation warning:
1. Check `docs/DEPRECATIONS.md` for migration path
2. Most have simple alternatives (e.g., use Git for collaboration)
3. Timeline: Deprecated features removed in v0.4.0 (Q1 2026)

### Documentation
- Start: `README.md` or `GETTING_STARTED.md`
- Lost? Check `docs/README.md` for navigation
- Questions? See `docs/FOCUS.md` for scope

## For Contributors

### Where to Contribute

**High Priority (Core)**:
- `neural/cli/` - CLI commands
- `neural/parser/` - DSL parser
- `neural/code_generation/` - Code generators
- `neural/shape_propagation/` - Shape validation

**Medium Priority (Semi-Core)**:
- `neural/hpo/` - Hyperparameter optimization
- `neural/automl/` - Architecture search
- `neural/integrations/` - Cloud platforms
- `neural/visualization/` - Diagrams

**Low Priority (Deprecated)**:
- Avoid: `neural/aquarium/`, `neural/collaboration/`, `neural/federated/`, etc.
- These will be removed/extracted soon

### Type Safety
- New code: Add type hints (see `docs/TYPE_SAFETY.md`)
- Core modules: Full type coverage required
- Check coverage: `python scripts/cleanup/check_type_coverage.py`

### Adding Features
Before adding a feature, check `docs/FOCUS.md`:
1. Does it improve core DSL experience?
2. Is it essential for neural network prototyping?
3. Can existing tools solve it better?
4. Is it a common use case (80%+ users)?

If unsure, open a discussion first!

## For Maintainers

### Quick Commands

```bash
# Type checking
python -m mypy neural/code_generation neural/utils neural/shape_propagation

# Type coverage
python scripts/cleanup/check_type_coverage.py

# Doc organization (dry run)
python scripts/cleanup/organize_docs.py

# Doc organization (execute)
python scripts/cleanup/organize_docs.py --execute

# Testing
pytest tests/ -v
```

### Review Checklist
- [ ] Type hints added for new code?
- [ ] In scope per `docs/FOCUS.md`?
- [ ] Documentation updated?
- [ ] Tests passing?
- [ ] No new deprecation warnings?

### Deprecation Process
1. Add warning to module `__init__.py`
2. Document in `docs/DEPRECATIONS.md`
3. Update `AGENTS.md`
4. Set removal date (usually next major version)
5. Provide migration guide

## Quick Comparison

### Before Cleanup
```
❌ 60+ docs in root directory
❌ Unclear project scope
❌ Monolithic installation
❌ ~30% type coverage
❌ Undocumented deprecations
❌ Scope creep with many unused features
```

### After Cleanup
```
✅ 8 essential docs in root
✅ Clear scope in docs/FOCUS.md
✅ Tiered installation (minimal/core/full)
✅ 70% type coverage in core modules
✅ All deprecations documented with warnings
✅ Focused on core DSL functionality
```

## FAQ

**Q: Why deprecate so many features?**  
A: They diluted focus on core DSL functionality. Better to do one thing excellently than many things poorly.

**Q: What if I need a deprecated feature?**  
A: Most have better alternatives (see `docs/DEPRECATIONS.md`). Some will become separate packages.

**Q: Will my existing code break?**  
A: No, deprecated features still work in v0.3.x with warnings. They'll be removed in v0.4.0.

**Q: Can I still use `pip install -e ".[full]"`?**  
A: Yes, but it now excludes deprecated features. Everything you need is still there.

**Q: Where did all the docs go?**  
A: Organized in `docs/` subdirectories. See `docs/README.md` for navigation.

**Q: How do I know what's core vs deprecated?**  
A: Check `docs/FOCUS.md` or `AGENTS.md` architecture section.

**Q: When will deprecated features be removed?**  
A: v0.4.0 (planned Q1 2026). Plenty of time to migrate.

## Getting Help

- **General questions**: [Discord](https://discord.gg/KFku4KvS)
- **Bug reports**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
- **Migration help**: Ask on Discord #migrations channel

## Timeline

| Version | What | When |
|---------|------|------|
| v0.3.0 | Cleanup implemented | December 2025 ✅ |
| v0.3.1 | Doc organization finalized | December 2025 |
| v0.4.0 | Deprecated features removed | Q1 2026 |

## Related Documents

- 📖 Full details: `IMPLEMENTATION_SUMMARY.md`
- ✅ Task tracking: `CLEANUP_CHECKLIST.md`
- 🎯 Project scope: `docs/FOCUS.md`
- ⚠️ Deprecations: `docs/DEPRECATIONS.md`
- 🔒 Type safety: `docs/TYPE_SAFETY.md`

---

**Last Updated**: December 2025  
**Version**: 0.3.0 (Cleanup Phase)

**Quick Navigation**: [Home](README.md) | [Getting Started](GETTING_STARTED.md) | [Contributing](CONTRIBUTING.md) | [Docs](docs/README.md)
