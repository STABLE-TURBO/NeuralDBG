# Neural DSL - Repository Documentation Index

**Last Updated**: January 2025 | **Version**: 0.4.0

Welcome to the Neural DSL documentation index. This guide helps you navigate all repository documentation.

---

## 🚀 Quick Start

### New Users
1. [README.md](README.md) - Project overview and quick example
2. [INSTALL.md](INSTALL.md) - Installation instructions
3. [docs/quickstart.md](docs/quickstart.md) - Getting started tutorial

### Contributors
1. [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
2. [AGENTS.md](AGENTS.md) - Developer guide and repository structure
3. [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview

### Current Release
1. [CHANGELOG.md](CHANGELOG.md) - Version history and release notes
2. [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - v0.4.0 cleanup documentation
3. [CLEANUP_QUICK_REFERENCE.md](CLEANUP_QUICK_REFERENCE.md) - Cleanup quick reference

---

## 📚 Core Documentation

### Essential Files

| Document | Description | Audience |
|----------|-------------|----------|
| [README.md](README.md) | Project overview, features, quick examples | All users |
| [INSTALL.md](INSTALL.md) | Installation and setup guide | New users |
| [CHANGELOG.md](CHANGELOG.md) | Version history and release notes | All users |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute to the project | Contributors |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community guidelines | All users |
| [LICENSE.md](LICENSE.md) | MIT License | All users |
| [SECURITY.md](SECURITY.md) | Security policy and reporting | All users |

### Technical Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [AGENTS.md](AGENTS.md) | Repository structure, commands, workflow | Contributors |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design | Contributors |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guides and strategies | Advanced users |

---

## 🧹 Cleanup Documentation (v0.4.0)

### Cleanup Documents

| Document | Description | Details |
|----------|-------------|---------|
| [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) | **Complete cleanup documentation** | Executive summary, detailed removal breakdown, rationale, benefits, migration guide |
| [CLEANUP_QUICK_REFERENCE.md](CLEANUP_QUICK_REFERENCE.md) | **Quick reference guide** | At-a-glance metrics, what was removed/kept, quick commands |

### What Changed in v0.4.0

- **200+ files** removed/archived
- **80% reduction** in GitHub Actions workflows (20+ → 4)
- **60% reduction** in root documentation files (50+ → 20)
- **70% reduction** in dependencies (50+ → 15 core packages)
- **~5-10 MB** disk space saved

**Read More**: [CHANGELOG.md - v0.4.0](CHANGELOG.md#040---2025-01-xx-refocusing-release)

---

## 📖 Feature Documentation

### Core Features

- **DSL Syntax**: [docs/dsl.md](docs/dsl.md)
- **CLI Reference**: [docs/cli.md](docs/cli_reference.md)
- **Shape Propagation**: [ARCHITECTURE.md](ARCHITECTURE.md) → Shape Validation section
- **Multi-Backend Code Generation**: [docs/](docs/)

### Optional Features

- **Hyperparameter Optimization (HPO)**: [docs/](docs/)
- **AutoML & NAS**: [docs/](docs/)
- **Debugging Dashboard**: [docs/](docs/)
- **Export & Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 🗂️ Consolidated Documentation

The archive directory has been removed. All documentation is now consolidated:

- **Aquarium IDE**: All docs in `docs/aquarium/AQUARIUM_IDE_COMPLETE_GUIDE.md`
- **Main Documentation**: Organized in `docs/` with clear structure
- **No Archive**: Redundant files removed, content consolidated

### What Was Removed

- 22 files from old `docs/archive/` (implementation summaries, planning docs, old HTML)
- Scattered Aquarium documentation (10+ files consolidated into AQUARIUM_IDE_COMPLETE_GUIDE.md)
- Redundant implementation summaries across the repository
- Feature implementation documents
- Development journals and planning docs

---

## 📂 Repository Structure

```
neural/
├── .github/
│   └── workflows/              # 4 essential CI/CD workflows
│       ├── essential-ci.yml    # Lint, test, security, coverage
│       ├── release.yml         # PyPI & GitHub releases
│       ├── codeql.yml          # Security analysis
│       └── validate-examples.yml # Example validation
├── docs/                       # Documentation
│   ├── archive/               # Archived implementation docs (50+ files)
│   ├── api/                   # API documentation
│   ├── tutorials/             # User tutorials
│   └── ...                    # Feature guides
├── examples/                   # DSL example files
├── neural/                     # Core source code
│   ├── cli/                   # CLI commands
│   ├── parser/                # DSL parser and grammar
│   ├── code_generation/       # Multi-backend code generators
│   ├── shape_propagation/     # Shape validation
│   ├── dashboard/             # NeuralDbg debugger
│   ├── hpo/                   # Hyperparameter optimization
│   └── automl/                # AutoML and NAS
├── tests/                      # Test suite
├── AGENTS.md                   # Agent development guide ⭐
├── ARCHITECTURE.md             # System architecture
├── CHANGELOG.md                # Version history ⭐
├── CLEANUP_SUMMARY.md          # v0.4.0 cleanup docs ⭐
├── CLEANUP_QUICK_REFERENCE.md  # Cleanup quick ref ⭐
├── CODE_OF_CONDUCT.md          # Community guidelines
├── CONTRIBUTING.md             # Contribution guide
├── DEPLOYMENT.md               # Deployment documentation
├── INSTALL.md                  # Installation guide
├── LICENSE.md                  # MIT License
├── README.md                   # Project overview ⭐
├── REPOSITORY_INDEX.md         # This file
├── SECURITY.md                 # Security policy
├── pyproject.toml              # Build configuration
├── setup.py                    # Package setup
└── requirements*.txt           # Dependencies

⭐ = Most important for new users/contributors
```

---

## 🔍 Find Documentation By Topic

### Installation & Setup
- [INSTALL.md](INSTALL.md) - Installation guide
- [README.md](README.md) - Quick setup
- [docs/quickstart.md](docs/quickstart.md) - Getting started

### Development
- [AGENTS.md](AGENTS.md) - Developer guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution workflow
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [docs/](docs/) - Feature guides

### Release & Migration
- [CHANGELOG.md](CHANGELOG.md) - All releases and changes
- [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - v0.4.0 cleanup
- Migration guides for removed features (in CHANGELOG.md)

### Community & Support
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community guidelines
- [SECURITY.md](SECURITY.md) - Security reporting
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to help

---

## 🛠️ Cleanup Scripts

Automated scripts for repository maintenance:

```bash
# Preview what cleanup would do
python preview_cleanup.py

# Execute full cleanup (archives docs, removes workflows)
python run_cleanup.py

# Individual cleanup operations
python cleanup_redundant_files.py   # Archive documentation
python cleanup_workflows.py         # Remove redundant workflows
```

**Documentation**: See [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)

---

## 📊 Repository Metrics (v0.4.0)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Root docs** | 50+ files | 20 files | -60% |
| **Workflows** | 20+ files | 4 files | -80% |
| **Dependencies** | 50+ packages | 15 packages | -70% |
| **Disk space** | - | ~5-10 MB saved | - |
| **Focus** | Feature-rich | Core DSL | Refocused |

---

## 🎯 Reading Paths

### Path 1: New User (30 minutes)
1. [README.md](README.md) - Overview and quick example
2. [INSTALL.md](INSTALL.md) - Installation
3. [docs/quickstart.md](docs/quickstart.md) - First steps
4. [docs/dsl.md](docs/dsl.md) - DSL syntax

**Outcome**: Ready to write and compile Neural DSL models

### Path 2: Contributor (60 minutes)
1. [README.md](README.md) - Project overview
2. [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution workflow
3. [AGENTS.md](AGENTS.md) - Repository structure and commands
4. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
5. [CHANGELOG.md](CHANGELOG.md) - Recent changes

**Outcome**: Ready to contribute code or documentation

### Path 3: Understanding v0.4.0 Cleanup (20 minutes)
1. [CHANGELOG.md - v0.4.0](CHANGELOG.md#040---2025-01-xx-refocusing-release) - Summary
2. [CLEANUP_QUICK_REFERENCE.md](CLEANUP_QUICK_REFERENCE.md) - Quick overview
3. [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - Full details (if needed)

**Outcome**: Understand cleanup rationale and changes

### Path 4: Advanced User (90 minutes)
1. [README.md](README.md) - Overview
2. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
3. [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment strategies
4. [docs/](docs/) - Feature-specific guides
5. [examples/](examples/) - Advanced examples

**Outcome**: Ready for production deployment

---

## 🔗 External Resources

- **GitHub Repository**: [https://github.com/Lemniscate-world/Neural](https://github.com/Lemniscate-world/Neural)
- **Discord Community**: [https://discord.gg/KFku4KvS](https://discord.gg/KFku4KvS)
- **Documentation Site**: [Coming soon]
- **PyPI Package**: [Coming soon]

---

## 📝 Document Maintenance

This index should be updated:
- **After major releases** - Update version numbers and links
- **When adding documentation** - Add entries to appropriate sections
- **When restructuring** - Update paths and organization
- **Quarterly** - Review and remove outdated references

---

## ✨ Quick Links

### Most Important Documents
- 📖 [README.md](README.md) - Start here!
- 📦 [INSTALL.md](INSTALL.md) - Installation
- 📋 [CHANGELOG.md](CHANGELOG.md) - What's new
- 🧹 [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - v0.4.0 cleanup
- 🔧 [AGENTS.md](AGENTS.md) - Developer guide
- 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) - System design

### Get Help
- 💬 [Discord Community](https://discord.gg/KFku4KvS)
- 🐛 [Security Issues](SECURITY.md)
- 🤝 [Contributing](CONTRIBUTING.md)

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Status**: ✅ Current
