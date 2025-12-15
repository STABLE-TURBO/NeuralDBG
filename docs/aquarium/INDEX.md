# Aquarium IDE - Documentation Hub

**Complete Resource Center** | **Version**: 1.0.0 | **Last Updated**: December 2024

<div align="center">

![Aquarium IDE](../images/aquarium/aquarium-banner.png)

**The Modern Web-Based IDE for Neural DSL**

[Quick Start](#quick-start) • [Documentation](#documentation) • [Support](#support) • [Community](#community)

</div>

---

## 🎯 Quick Navigation

### New Users Start Here
1. **[Installation Guide](installation.md)** - Get up and running in 5 minutes
2. **[Quick Start Tutorial](#quick-start)** - Your first model in 30 seconds
3. **[Quick Reference](QUICK_REFERENCE.md)** - One-page cheat sheet
4. **[Video Tutorials](video-tutorials.md)** - Learn by watching

### Experienced Users
- **[Complete Manual](AQUARIUM_IDE_MANUAL.md)** - Comprehensive user guide
- **[API Reference](API_REFERENCE.md)** - Python, REST, and Plugin APIs
- **[Keyboard Shortcuts](keyboard-shortcuts.md)** - Boost your productivity
- **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions

### Developers
- **[Architecture Overview](architecture.md)** - System design and components
- **[Plugin Development](plugin-development.md)** - Extend Aquarium with plugins
- **[Contributing Guide](../../CONTRIBUTING.md)** - Join the development

---

## 🚀 Quick Start

### Installation (30 Seconds)

```bash
# Install Aquarium
pip install neural-dsl[dashboard]

# Launch
python -m neural.aquarium.aquarium

# Open browser
http://localhost:8052
```

### Your First Model (2 Minutes)

**Option 1: Use Example (Easiest)**
1. Click **"Load Example"** button (left sidebar)
2. Click **"Parse DSL"** to validate
3. In Runner tab: Select **TensorFlow** backend, **MNIST** dataset
4. Click **"Compile"** then **"Run"**
5. Watch your model train! 🎉

**Option 2: Write Custom Model**
```neural
network SimpleNet {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

Then follow steps 2-5 above.

---

## 📚 Documentation

### Core Documentation

#### 1. [Complete User Manual](AQUARIUM_IDE_MANUAL.md)
**20 chapters covering everything**
- Getting Started
- DSL Editor
- Model Compilation
- Training Execution
- Debugging
- Export & Integration
- Welcome Screen & Tutorials
- Plugin System
- Hyperparameter Optimization
- Keyboard Shortcuts
- API Reference
- Troubleshooting
- Performance Optimization
- FAQ
- Architecture Overview
- Plugin Development
- Contributing

**Audience:** All users  
**Length:** ~15,000 words  
**Format:** Complete reference with examples

#### 2. [API Reference](API_REFERENCE.md)
**Complete API documentation**
- Python API (ExecutionManager, PluginManager, etc.)
- REST API endpoints
- Plugin API
- Component API (Parser, CodeGen, ShapeProp)
- TypeScript API
- CLI API
- Error codes and rate limits

**Audience:** Developers, integrators  
**Length:** ~8,000 words  
**Format:** Technical reference

#### 3. [Quick Reference](QUICK_REFERENCE.md)
**One-page cheat sheet**
- Essential commands
- DSL syntax reference
- Common patterns
- Configuration options
- Troubleshooting quick fixes
- Keyboard shortcuts
- Console output guide
- Pro tips

**Audience:** All users  
**Length:** ~2,000 words  
**Format:** Quick reference card

### Additional Guides

#### [Installation Guide](installation.md)
- System requirements
- Installation methods (pip, source)
- Backend setup (TensorFlow, PyTorch, ONNX)
- Configuration
- Verification
- Platform-specific instructions

**Audience:** New users  
**Estimated Time:** 5-10 minutes

#### [Keyboard Shortcuts](keyboard-shortcuts.md)
- Essential shortcuts (Parse, Compile, Run)
- Editor shortcuts (Copy, Paste, Undo)
- View shortcuts (Tabs, Zoom)
- Debugging shortcuts
- Browser shortcuts
- Custom keybindings
- Printable cheat sheet

**Audience:** All users  
**Use:** Productivity boost

#### [Troubleshooting Guide](troubleshooting.md)
- Installation issues
- Launch problems
- Connection issues
- Compilation errors
- Execution problems
- Performance issues
- UI/display issues
- Export/integration issues
- Platform-specific issues
- Debug mode
- Getting help

**Audience:** Users encountering issues  
**Format:** Problem → Solution

### Developer Documentation

#### [Architecture Overview](architecture.md)
- System design
- Component structure
- Data flow
- Technology stack
- Security architecture
- Scalability considerations

**Audience:** Contributors, developers  
**Purpose:** Understanding internals

#### [Plugin Development](plugin-development.md)
- Plugin architecture
- Creating plugins
- Plugin types (Panel, Theme, Command, etc.)
- Plugin API
- Example plugins
- Testing plugins
- Publishing plugins
- Best practices

**Audience:** Plugin developers  
**Prerequisites:** Python/TypeScript knowledge

### Specialized Guides

#### [Video Tutorials](video-tutorials.md)
- Introduction to Aquarium (10:30)
- DSL Syntax Basics (15:45)
- Building Your First Model (20:15)
- Using the AI Assistant (12:00)
- Debugging Neural Networks (18:30)
- Advanced Features (various)

**Audience:** Visual learners  
**Format:** Video with transcripts

---

## 📖 Documentation by Topic

### Getting Started
- ✅ [Installation Guide](installation.md) - Setup instructions
- ✅ [Quick Start](#quick-start) - First model in minutes
- ✅ [User Manual: Part I](AQUARIUM_IDE_MANUAL.md#part-i-getting-started) - Detailed introduction
- ✅ [Video: Introduction](video-tutorials.md) - Overview walkthrough

### Using the IDE
- ✅ [DSL Editor](AQUARIUM_IDE_MANUAL.md#5-dsl-editor) - Writing models
- ✅ [Model Compilation](AQUARIUM_IDE_MANUAL.md#6-model-compilation) - Building code
- ✅ [Training Execution](AQUARIUM_IDE_MANUAL.md#7-training-execution) - Running models
- ✅ [Quick Reference](QUICK_REFERENCE.md) - Syntax cheat sheet

### Advanced Features
- ✅ [Debugging](AQUARIUM_IDE_MANUAL.md#8-real-time-debugging) - NeuralDbg integration
- ✅ [Welcome Screen](AQUARIUM_IDE_MANUAL.md#10-welcome-screen--tutorials) - Tutorials and examples
- ✅ [Plugin System](AQUARIUM_IDE_MANUAL.md#11-plugin-system) - Extending Aquarium
- ✅ [HPO](AQUARIUM_IDE_MANUAL.md#12-hyperparameter-optimization) - Automated tuning

### Reference
- ✅ [Keyboard Shortcuts](keyboard-shortcuts.md) - Complete shortcut list
- ✅ [API Reference](API_REFERENCE.md) - Python, REST, Plugin APIs
- ✅ [DSL Syntax](QUICK_REFERENCE.md#-dsl-syntax-cheat-sheet) - Language reference
- ✅ [Error Codes](API_REFERENCE.md#error-codes) - Error reference

### Problem Solving
- ✅ [Troubleshooting](troubleshooting.md) - Common issues
- ✅ [FAQ](AQUARIUM_IDE_MANUAL.md#17-faq) - Frequently asked questions
- ✅ [Performance](AQUARIUM_IDE_MANUAL.md#16-performance-optimization) - Speed optimization
- ✅ [Debugging Tips](AQUARIUM_IDE_MANUAL.md#8-real-time-debugging) - Debug workflow

### Development
- ✅ [Architecture](architecture.md) - System design
- ✅ [Plugin Development](plugin-development.md) - Create plugins
- ✅ [Contributing](../../CONTRIBUTING.md) - Join development
- ✅ [API Reference](API_REFERENCE.md) - Developer APIs

---

## 🎓 Learning Paths

### Path 1: Complete Beginner (2-3 hours)

**Goal:** Train your first neural network

1. **Install** (10 min)
   - Follow [Installation Guide](installation.md)
   - Verify setup

2. **Quick Start** (15 min)
   - Launch Aquarium
   - Load example model
   - Run training
   - Export script

3. **Learn DSL Syntax** (30 min)
   - Read [DSL Syntax](QUICK_REFERENCE.md#-dsl-syntax-cheat-sheet)
   - Watch [DSL Basics Video](video-tutorials.md)
   - Try modifying examples

4. **Build Custom Model** (45 min)
   - Write simple classifier
   - Parse and debug
   - Train and evaluate
   - Export for reuse

5. **Explore Features** (30 min)
   - Try different backends
   - Use debugger
   - Install plugin
   - Review shortcuts

**Resources:**
- [Quick Reference](QUICK_REFERENCE.md)
- [Video Tutorials](video-tutorials.md)
- [User Manual Part I](AQUARIUM_IDE_MANUAL.md#part-i-getting-started)

### Path 2: Intermediate User (4-6 hours)

**Goal:** Master Aquarium features

**Prerequisites:** Completed Path 1

1. **Advanced DSL** (1 hour)
   - Complex architectures
   - Custom layers
   - Multiple inputs/outputs
   - Read [DSL Editor](AQUARIUM_IDE_MANUAL.md#5-dsl-editor)

2. **Multi-Backend Workflow** (1 hour)
   - TensorFlow → PyTorch conversion
   - ONNX export
   - Backend comparison
   - Read [Model Compilation](AQUARIUM_IDE_MANUAL.md#6-model-compilation)

3. **Debugging Mastery** (1.5 hours)
   - NeuralDbg deep dive
   - Layer inspection
   - Gradient analysis
   - Read [Debugging Guide](AQUARIUM_IDE_MANUAL.md#8-real-time-debugging)

4. **HPO Experiments** (1 hour)
   - Setup Optuna
   - Define search space
   - Run optimization
   - Read [HPO Guide](AQUARIUM_IDE_MANUAL.md#12-hyperparameter-optimization)

5. **Performance Optimization** (30 min)
   - GPU configuration
   - Memory management
   - Training speedup
   - Read [Performance Guide](AQUARIUM_IDE_MANUAL.md#16-performance-optimization)

**Resources:**
- [Complete Manual](AQUARIUM_IDE_MANUAL.md)
- [API Reference](API_REFERENCE.md)
- [Advanced Videos](video-tutorials.md)

### Path 3: Plugin Developer (6-8 hours)

**Goal:** Create and publish custom plugin

**Prerequisites:** Python/TypeScript knowledge

1. **Architecture Study** (1 hour)
   - Read [Architecture Overview](architecture.md)
   - Understand component structure
   - Review data flow

2. **Plugin System** (1.5 hours)
   - Read [Plugin Development](plugin-development.md)
   - Study example plugins
   - Understand plugin types
   - Review [Plugin API](API_REFERENCE.md#3-plugin-api)

3. **Build Simple Plugin** (2 hours)
   - Create panel plugin
   - Implement manifest
   - Test locally
   - Debug issues

4. **Advanced Plugin** (2 hours)
   - Multi-capability plugin
   - External API integration
   - Configuration handling
   - State management

5. **Publish Plugin** (30 min)
   - Package plugin
   - Publish to npm/PyPI
   - Document usage
   - Promote plugin

**Resources:**
- [Plugin Development Guide](plugin-development.md)
- [API Reference](API_REFERENCE.md)
- [Architecture Doc](architecture.md)

### Path 4: Contributor (Ongoing)

**Goal:** Contribute to Aquarium development

1. **Understand Codebase** (2-3 hours)
   - Clone repository
   - Setup dev environment
   - Read architecture
   - Run tests

2. **Pick First Issue** (varies)
   - Browse GitHub Issues
   - Pick "good first issue"
   - Comment on issue
   - Fork repository

3. **Make Contribution** (varies)
   - Create feature branch
   - Implement change
   - Add tests
   - Submit PR

4. **Review Process** (varies)
   - Respond to feedback
   - Make revisions
   - Get approval
   - Merge!

**Resources:**
- [Contributing Guide](../../CONTRIBUTING.md)
- [Architecture Overview](architecture.md)
- [GitHub Repository](https://github.com/Lemniscate-world/Neural)

---

## 🔍 Search by Use Case

### I want to...

**...train my first neural network**
→ [Quick Start](#quick-start) → [Installation Guide](installation.md)

**...understand DSL syntax**
→ [Quick Reference](QUICK_REFERENCE.md) → [DSL Editor Guide](AQUARIUM_IDE_MANUAL.md#5-dsl-editor)

**...switch between TensorFlow and PyTorch**
→ [Backend Selection](AQUARIUM_IDE_MANUAL.md#61-backend-selection)

**...debug why my model isn't learning**
→ [Debugging Guide](AQUARIUM_IDE_MANUAL.md#8-real-time-debugging) → [Troubleshooting](troubleshooting.md)

**...use custom datasets**
→ [Dataset Configuration](AQUARIUM_IDE_MANUAL.md#62-dataset-configuration)

**...optimize hyperparameters**
→ [HPO Guide](AQUARIUM_IDE_MANUAL.md#12-hyperparameter-optimization)

**...create a plugin**
→ [Plugin Development](plugin-development.md) → [Plugin API](API_REFERENCE.md#3-plugin-api)

**...integrate Aquarium into my workflow**
→ [Export & Integration](AQUARIUM_IDE_MANUAL.md#9-export--integration) → [API Reference](API_REFERENCE.md)

**...speed up training**
→ [Performance Optimization](AQUARIUM_IDE_MANUAL.md#16-performance-optimization)

**...fix installation issues**
→ [Troubleshooting](troubleshooting.md#installation-issues)

**...learn keyboard shortcuts**
→ [Keyboard Shortcuts](keyboard-shortcuts.md) → [Quick Reference](QUICK_REFERENCE.md)

**...contribute to development**
→ [Contributing Guide](../../CONTRIBUTING.md) → [Architecture](architecture.md)

---

## 📊 Documentation Statistics

### Coverage
- **User Guide**: ✅ Complete (15,000+ words)
- **API Reference**: ✅ Complete (8,000+ words)
- **Quick Reference**: ✅ Complete (2,000+ words)
- **Installation**: ✅ Complete
- **Troubleshooting**: ✅ Complete
- **Keyboard Shortcuts**: ✅ Complete
- **Architecture**: ✅ Complete
- **Plugin Development**: ✅ Complete
- **Video Tutorials**: ✅ Complete

### Formats
- Markdown documents: 10+
- Code examples: 200+
- Screenshots: 15+
- Video tutorials: 9
- API endpoints: 20+
- Keyboard shortcuts: 100+

### Languages
- English: ✅ Complete
- Other languages: 🔄 Coming soon

---

## 🆘 Support

### Getting Help

**Quick Questions:**
- Check [FAQ](AQUARIUM_IDE_MANUAL.md#17-faq)
- Search [Troubleshooting Guide](troubleshooting.md)
- Review [Quick Reference](QUICK_REFERENCE.md)

**Technical Issues:**
- [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues) - Bug reports
- [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions) - Q&A
- [Discord Server](https://discord.gg/KFku4KvS) - Real-time chat

**Security Issues:**
- Email: Lemniscate_zero@proton.me (private disclosure)

### Reporting Issues

**Good Bug Report Includes:**
1. Aquarium version
2. Python version
3. Operating system
4. Exact error message
5. Steps to reproduce
6. What you expected vs what happened

**Example:**
```
**Environment:**
- Aquarium: 1.0.0
- Python: 3.9.7
- OS: Windows 11

**Issue:**
Export fails with permission error

**Steps:**
1. Compile MNIST model
2. Click Export
3. Select location: C:\exports\
4. Error appears

**Expected:** File exported successfully
**Actual:** PermissionError

**What I tried:**
- Checked folder permissions
- Tried different location
- Restarted Aquarium
```

---

## 🌟 Community

### Contribute

**Ways to Help:**
1. **Report Bugs** - Found an issue? Let us know
2. **Suggest Features** - Have an idea? Share it
3. **Write Documentation** - Help others learn
4. **Create Plugins** - Extend Aquarium
5. **Answer Questions** - Help in Discord/Discussions
6. **Submit Code** - Fix bugs or add features

**Getting Started:**
- Read [Contributing Guide](../../CONTRIBUTING.md)
- Join [Discord](https://discord.gg/KFku4KvS)
- Browse [Good First Issues](https://github.com/Lemniscate-world/Neural/labels/good%20first%20issue)

### Stay Updated

- **Star on GitHub**: Get notified of releases
- **Follow on Twitter**: [@NLang4438](https://x.com/NLang4438)
- **Join Discord**: [Community server](https://discord.gg/KFku4KvS)
- **Watch Releases**: [GitHub Releases](https://github.com/Lemniscate-world/Neural/releases)

---

## 📅 Documentation Roadmap

### Completed ✅
- Complete User Manual
- API Reference
- Quick Reference
- Installation Guide
- Troubleshooting Guide
- Keyboard Shortcuts
- Architecture Overview
- Plugin Development Guide

### In Progress 🔄
- Video tutorial transcripts
- Interactive examples
- Code playground
- Multi-language translations

### Planned 📋
- Advanced tutorials (custom datasets, deployment)
- Best practices guide
- Performance tuning guide
- Integration examples
- Case studies
- Community cookbook

---

## 📄 License & Credits

### License
Aquarium IDE and its documentation are released under the **MIT License**.

See [LICENSE.md](../../LICENSE.md) for details.

### Credits

**Core Team:**
- Neural DSL Development Team

**Contributors:**
- [All Contributors](https://github.com/Lemniscate-world/Neural/graphs/contributors)

**Built With:**
- [Dash](https://dash.plotly.com/) - Web framework
- [Plotly](https://plotly.com/) - Visualization
- [Bootstrap](https://getbootstrap.com/) - UI components
- [Font Awesome](https://fontawesome.com/) - Icons

**Special Thanks:**
- Community contributors
- Beta testers
- Documentation reviewers

---

## 🔗 Quick Links

### Documentation
- [Complete Manual](AQUARIUM_IDE_MANUAL.md)
- [API Reference](API_REFERENCE.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Installation Guide](installation.md)
- [Troubleshooting](troubleshooting.md)

### Community
- [GitHub Repository](https://github.com/Lemniscate-world/Neural)
- [Discord Server](https://discord.gg/KFku4KvS)
- [Twitter](https://x.com/NLang4438)
- [Product Hunt](https://www.producthunt.com/posts/neural-2)

### Resources
- [Neural DSL Docs](../../docs/dsl.md)
- [Examples](../../examples/README.md)
- [Blog](../../docs/blog/README.md)
- [PyPI Package](https://pypi.org/project/neural-dsl/)

---

<div align="center">

## 🎉 Start Building!

**Ready to create amazing neural networks?**

[Install Aquarium](installation.md) • 
[Quick Start](#quick-start) • 
[Read Manual](AQUARIUM_IDE_MANUAL.md)

---

**Made with ❤️ by the Neural DSL Team**

[⭐ Star on GitHub](https://github.com/Lemniscate-world/Neural) • 
[📚 Documentation](https://github.com/Lemniscate-world/Neural/tree/main/docs) • 
[💬 Discord](https://discord.gg/KFku4KvS) • 
[🐦 Twitter](https://x.com/NLang4438)

</div>

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready  
**License**: MIT
