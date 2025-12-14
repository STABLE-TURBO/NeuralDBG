# Neural DSL Documentation

Welcome to the Neural DSL documentation directory. This guide helps you navigate the organized documentation structure.

## Quick Navigation

### Essential Reading
- [Getting Started](../GETTING_STARTED.md) - Installation and first steps
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Changelog](../CHANGELOG.md) - Version history and changes

### Project Guidance
- [**FOCUS.md**](FOCUS.md) - **Start here!** Project scope, boundaries, and philosophy
- [DEPRECATIONS.md](DEPRECATIONS.md) - Deprecated features and migration paths
- [TYPE_SAFETY.md](TYPE_SAFETY.md) - Type checking guidelines and standards

### Feature Documentation
- [DSL Language Reference](dsl.md) - Complete syntax guide
- [Deployment Guide](deployment.md) - Production export options
- [AI Integration Guide](ai_integration_guide.md) - Natural language model generation

## Directory Structure

```
docs/
├── README.md                    # This file
├── FOCUS.md                     # Project scope and boundaries ⭐
├── DEPRECATIONS.md              # Deprecated features
├── TYPE_SAFETY.md               # Type checking guidelines
│
├── archive/                     # Historical documents
│   ├── *_IMPLEMENTATION*.md     # Implementation summaries
│   ├── RELEASE_*.md            # Old release notes
│   └── MIGRATION_*.md          # Historical migrations
│
├── automation/                  # Automation guides
│   └── AUTOMATION_GUIDE.md     # CI/CD and automation
│
├── dependencies/                # Dependency management
│   ├── DEPENDENCY_GUIDE.md     # Dependency documentation
│   └── DEPENDENCY_*.md         # Specific dependency docs
│
├── distribution/                # Distribution and publishing
│   ├── DISTRIBUTION_PLAN.md    # Release planning
│   └── GITHUB_PUBLISHING_GUIDE.md
│
├── setup/                       # Installation and setup
│   ├── INSTALL.md              # Installation guide
│   └── ERROR_MESSAGES_GUIDE.md # Troubleshooting
│
└── features/                    # Feature-specific docs
    ├── DEPLOYMENT_FEATURES.md  # Deployment options
    └── TRANSFORMER_*.md        # Transformer docs
```

## Documentation Philosophy

Our documentation follows these principles:

1. **Clarity First**: Clear, concise explanations over comprehensiveness
2. **Examples Driven**: Show, don't just tell
3. **Up-to-Date**: If it's documented, it should work
4. **Organized**: Easy to find what you need
5. **Honest**: Clear about limitations and deprecated features

## Core vs Peripheral

### Core Documentation (Priority 1)
These docs cover essential, actively maintained features:
- DSL syntax and parser
- Code generation (TensorFlow, PyTorch, ONNX)
- Shape propagation and validation
- CLI commands
- NeuralDbg dashboard

### Semi-Core Documentation (Priority 2)
Supported but not the primary focus:
- HPO (hyperparameter optimization)
- AutoML (simplified architecture search)
- Cloud integrations (AWS, GCP, Azure)
- Experiment tracking

### Deprecated/Experimental (Priority 3)
Features being phased out or experimental:
- See [DEPRECATIONS.md](DEPRECATIONS.md) for full list
- Aquarium IDE (extracting to separate repo)
- Collaboration tools (use Git instead)
- Marketplace (use HuggingFace Hub)
- Federated learning (extracting to separate repo)

## Finding What You Need

### "I want to..."

**...get started quickly**
→ [GETTING_STARTED.md](../GETTING_STARTED.md)

**...understand the DSL syntax**
→ [dsl.md](dsl.md)

**...compile my model to PyTorch/TensorFlow**
→ [CLI Commands](#) or [deployment.md](deployment.md)

**...debug my model**
→ NeuralDbg section in main README

**...deploy to production**
→ [deployment.md](deployment.md)

**...contribute code**
→ [CONTRIBUTING.md](../CONTRIBUTING.md) + [TYPE_SAFETY.md](TYPE_SAFETY.md)

**...understand project scope**
→ [FOCUS.md](FOCUS.md) ⭐

**...migrate from deprecated feature**
→ [DEPRECATIONS.md](DEPRECATIONS.md)

**...optimize hyperparameters**
→ HPO documentation (coming soon)

**...integrate with cloud platforms**
→ Cloud integration guides (coming soon)

## Contributing to Documentation

### Guidelines

1. **Location**: Put docs in the appropriate subdirectory
2. **Format**: Use Markdown with clear headers
3. **Examples**: Include working code examples
4. **Links**: Use relative links within docs
5. **Updates**: Update this README when adding major docs

### Documentation PRs

When submitting documentation:
- Test all code examples
- Check links work
- Add entry to this README if appropriate
- Follow the existing style and tone
- Keep it concise

## Archive Policy

Documents are moved to `archive/` when:
- They describe deprecated features
- They're superseded by newer docs
- They're historical implementation notes
- They're old release notes (>2 versions old)

Archived docs are kept for reference but not actively maintained.

## Getting Help

If you can't find what you need:

1. **Search the docs**: Use GitHub's search or grep
2. **Check examples**: Look in `examples/` directory
3. **Ask on Discord**: [Join our Discord](https://discord.gg/KFku4KvS)
4. **Open a discussion**: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
5. **Report missing docs**: Open an issue with "documentation" label

## Documentation Roadmap

### Short Term
- [ ] Complete DSL language reference
- [ ] Add HPO tutorial
- [ ] Expand deployment guide
- [ ] Create video tutorials

### Medium Term
- [ ] Interactive documentation site
- [ ] API reference (auto-generated)
- [ ] Best practices guide
- [ ] Performance tuning guide

### Long Term
- [ ] Comprehensive examples library
- [ ] Architecture patterns catalog
- [ ] Integration cookbook
- [ ] Educational curriculum

## Maintenance

This documentation is maintained by the Neural DSL team and community contributors. 

**Last major reorganization**: December 2025 (v0.3.0 cleanup)

**Next review scheduled**: Q1 2026

---

**Quick Links**:
[Home](../README.md) |
[Focus](FOCUS.md) |
[Getting Started](../GETTING_STARTED.md) |
[Contributing](../CONTRIBUTING.md) |
[Discord](https://discord.gg/KFku4KvS)
