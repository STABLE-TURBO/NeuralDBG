# Aquarium IDE Documentation

<div align="center">

![Aquarium IDE](../images/aquarium/aquarium-banner.png)

**A Modern Web-Based IDE for Neural DSL**

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Support](#support)

</div>

## Overview

Aquarium IDE is a comprehensive, web-based Integrated Development Environment for Neural DSL. It provides an intuitive interface for writing, compiling, executing, and debugging neural network models across multiple backends.

### Key Features

✨ **Intuitive DSL Editor** - Syntax-highlighted editor with real-time validation  
🔧 **Multi-Backend Support** - TensorFlow, PyTorch, and ONNX backends  
🚀 **One-Click Compilation** - Generate executable Python code instantly  
📊 **Real-Time Training** - Execute and monitor training directly in the IDE  
🐛 **Integrated Debugging** - NeuralDbg integration for advanced debugging  
📦 **Easy Export** - Save and share models effortlessly  
🎨 **Dark Theme** - Professional, eye-friendly interface  
📚 **Built-in Examples** - Learn from 8+ pre-built model templates

## Installation

### Quick Install

```bash
# Install with dashboard support
pip install neural-dsl[dashboard]

# Or install full package
pip install neural-dsl[full]
```

### From Source

```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -e ".[dashboard]"
```

### Verify Installation

```bash
python -m neural.aquarium.aquarium
# Open http://localhost:8052
```

**Full installation guide**: [installation.md](installation.md)

## Quick Start

### Launch Aquarium

```bash
# Default launch
python -m neural.aquarium.aquarium

# Custom port
python -m neural.aquarium.aquarium --port 8053

# Debug mode
python -m neural.aquarium.aquarium --debug
```

### 5-Minute Tutorial

1. **Load Example**: Click "Load Example" button
2. **Parse Model**: Click "Parse DSL" to validate
3. **Configure**: Select backend (TensorFlow), dataset (MNIST), epochs (5)
4. **Compile**: Click "Compile" to generate code
5. **Run**: Click "Run" to start training
6. **Export**: Click "Export" to save your model

**Detailed guide**: [user-manual.md](user-manual.md#getting-started)

## Documentation

### Core Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[Installation Guide](installation.md)** | Complete installation instructions | Beginners |
| **[User Manual](user-manual.md)** | Comprehensive usage guide with screenshots | All Users |
| **[Keyboard Shortcuts](keyboard-shortcuts.md)** | Complete shortcut reference | All Users |
| **[Troubleshooting](troubleshooting.md)** | Common issues and solutions | All Users |

### Advanced Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[Architecture Overview](architecture.md)** | System design and components | Developers |
| **[Plugin Development](plugin-development.md)** | Create custom plugins | Developers |
| **[Video Tutorials](video-tutorials.md)** | Video tutorial library | All Users |

### Quick References

- **5-Minute Quick Start**: [user-manual.md#getting-started](user-manual.md#getting-started)
- **Common Issues**: [troubleshooting.md#common-issues](troubleshooting.md#common-issues)
- **Keyboard Shortcuts**: [keyboard-shortcuts.md#quick-reference](keyboard-shortcuts.md#quick-reference)
- **Plugin API**: [plugin-development.md#api-reference](plugin-development.md#api-reference)

## Features

### DSL Editor

- Syntax-highlighted text editor
- Real-time parse validation
- Model information panel
- 8+ built-in examples
- Dark theme for comfort

**Learn more**: [user-manual.md#dsl-editor](user-manual.md#dsl-editor)

### Model Compilation

- **Backend Selection**: TensorFlow, PyTorch, ONNX
- **Dataset Support**: MNIST, CIFAR10, CIFAR100, ImageNet, Custom
- **Training Config**: Epochs, batch size, validation split
- **Options**: Auto-flatten, HPO, verbose, save weights

**Learn more**: [user-manual.md#model-compilation--execution](user-manual.md#model-compilation--execution)

### Execution

- One-click training execution
- Real-time console output
- Process control (start/stop)
- Training metrics visualization
- Live progress monitoring

**Learn more**: [user-manual.md#execution-process](user-manual.md#execution-process)

### Debugging

- NeuralDbg integration
- Layer-by-layer inspection
- Gradient flow visualization
- Dead neuron detection
- Memory & FLOP profiling
- Anomaly detection

**Learn more**: [user-manual.md#debugging](user-manual.md#debugging)

### Export & Integration

- Export compiled scripts
- Open in external IDE
- File organization
- Version control ready
- Metadata support

**Learn more**: [user-manual.md#export--integration](user-manual.md#export--integration)

## Usage Examples

### Example 1: MNIST Classification

```neural
network MNISTClassifier {
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

**Steps**:
1. Paste DSL code in editor
2. Click "Parse DSL"
3. Select TensorFlow backend, MNIST dataset
4. Set epochs to 10
5. Click "Compile" then "Run"

### Example 2: Transfer Learning

```neural
network TransferLearning {
    input: (None, 224, 224, 3)
    layers:
        # Pretrained base (frozen)
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        # Custom top layers
        Flatten()
        Dense(units=256, activation=relu)
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.0001)
}
```

**More examples**: [user-manual.md#advanced-features](user-manual.md#advanced-features)

## Architecture

### System Overview

```
┌────────────────────────────────────────┐
│         Web Browser (UI)               │
│  Editor | Runner | Debugger | Viz      │
└────────────────┬───────────────────────┘
                 │ HTTP/WebSocket
┌────────────────▼───────────────────────┐
│         Dash Application               │
│  Callbacks | State | Layout            │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│      Business Logic Layer              │
│  ExecutionManager | ScriptGenerator    │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│      Neural DSL Core                   │
│  Parser | CodeGen | ShapeProp          │
└────────────────────────────────────────┘
```

**Detailed architecture**: [architecture.md](architecture.md)

### Component Structure

```
neural/aquarium/
├── aquarium.py              # Main application
├── config.py                # Configuration
├── examples.py              # Example models
├── src/
│   └── components/
│       ├── runner/          # Compilation & execution
│       │   ├── runner_panel.py
│       │   ├── execution_manager.py
│       │   └── script_generator.py
│       ├── settings/        # Configuration UI
│       └── project/         # Project management
└── backend/                 # Backend API (future)
```

## Development

### Running in Development Mode

```bash
# Clone repository
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate on Windows

# Install in editable mode
pip install -e ".[dashboard]"

# Run with debug enabled
python -m neural.aquarium.aquarium --debug
```

### Plugin Development

Extend Aquarium with custom plugins:

```python
# my_plugin/__init__.py
from neural.aquarium.src.components.plugin_base import PluginBase

class MyPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.name = "my-plugin"
        
    def activate(self):
        # Register components
        pass
```

**Complete guide**: [plugin-development.md](plugin-development.md)

## Troubleshooting

### Common Issues

**Port already in use**:
```bash
python -m neural.aquarium.aquarium --port 8053
```

**Module not found**:
```bash
pip install neural-dsl[dashboard]
```

**Browser can't connect**:
- Verify server is running
- Try `http://127.0.0.1:8052`
- Check firewall settings

**More solutions**: [troubleshooting.md](troubleshooting.md)

## Performance

### Optimization Tips

1. **Reduce batch size** if running out of memory
2. **Clear console** regularly to free DOM
3. **Use GPU** for faster training (if available)
4. **Start small** - test with few epochs first
5. **Monitor resources** - CPU, memory, GPU usage

**Detailed guide**: [user-manual.md#performance-optimization](user-manual.md#performance-optimization)

## Security

### Best Practices

- Run on localhost for development
- Use firewall for production deployment
- Validate all DSL inputs
- Sanitize file paths
- Don't run as root/administrator

**Security architecture**: [architecture.md#security-architecture](architecture.md#security-architecture)

## Roadmap

### Current Version (v0.1.0)

- ✅ DSL Editor with parsing
- ✅ Multi-backend support (TF/PyTorch/ONNX)
- ✅ Compilation and execution
- ✅ Real-time console output
- ✅ Export functionality
- ✅ Example models library

### Upcoming Features (v0.2.0)

- 🔄 Syntax highlighting in editor
- 🔄 Code completion
- 🔄 Real-time metrics visualization
- 🔄 NeuralDbg deep integration
- 🔄 Model comparison tools
- 🔄 Experiment tracking

### Future Vision

- 📅 Collaborative editing
- 📅 Cloud execution support
- 📅 Model marketplace
- 📅 Auto-architecture search
- 📅 Production deployment pipelines

**Full roadmap**: [../../ROADMAP.md](../../ROADMAP.md)

## Contributing

We welcome contributions! Here's how:

### Ways to Contribute

1. **Report Bugs**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
2. **Suggest Features**: [Discussions](https://github.com/Lemniscate-world/Neural/discussions)
3. **Submit Pull Requests**: Bug fixes, features, documentation
4. **Write Tutorials**: Share your knowledge
5. **Help Others**: Answer questions in Discord/Discussions

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Neural.git
cd Neural

# Install development dependencies
pip install -r requirements-dev.txt
pre-commit install

# Run tests
pytest tests/ -v

# Make changes and submit PR
```

**Contributing guide**: [../../CONTRIBUTING.md](../../CONTRIBUTING.md)

## Community & Support

### Get Help

- **Documentation**: You're reading it!
- **GitHub Issues**: [Report bugs](https://github.com/Lemniscate-world/Neural/issues)
- **Discussions**: [Ask questions](https://github.com/Lemniscate-world/Neural/discussions)
- **Discord**: [Join chat](https://discord.gg/KFku4KvS)
- **Email**: Lemniscate_zero@proton.me

### Stay Updated

- **Star the repo**: Get notified of releases
- **Follow on Twitter**: [@NLang4438](https://x.com/NLang4438)
- **Join Discord**: [Community server](https://discord.gg/KFku4KvS)
- **Watch releases**: [GitHub Releases](https://github.com/Lemniscate-world/Neural/releases)

## Resources

### Documentation

- [Neural DSL Documentation](../../docs/dsl.md)
- [API Reference](../../docs/api/README.md)
- [Examples](../../examples/README.md)
- [Blog & Tutorials](../../docs/blog/README.md)

### External Resources

- [GitHub Repository](https://github.com/Lemniscate-world/Neural)
- [PyPI Package](https://pypi.org/project/neural-dsl/)
- [Product Hunt](https://www.producthunt.com/posts/neural-2)
- [Discord Community](https://discord.gg/KFku4KvS)

## License

Aquarium IDE is part of Neural DSL and is released under the MIT License. See [LICENSE.md](../../LICENSE.md) for details.

## Acknowledgments

### Credits

- **Core Team**: Neural DSL Development Team
- **Contributors**: [All Contributors](https://github.com/Lemniscate-world/Neural/graphs/contributors)
- **Frameworks**: Dash, Plotly, Bootstrap
- **Community**: Thank you for your support!

### Powered By

- [Dash](https://dash.plotly.com/) - Web framework
- [Plotly](https://plotly.com/) - Data visualization
- [Bootstrap](https://getbootstrap.com/) - UI components
- [Font Awesome](https://fontawesome.com/) - Icons

## FAQ

### General

**Q: What is Aquarium IDE?**  
A: A web-based IDE for Neural DSL that provides model editing, compilation, execution, and debugging.

**Q: Is it free?**  
A: Yes, completely free and open source (MIT License).

**Q: What browsers are supported?**  
A: Chrome, Firefox, Safari, Edge (modern versions).

### Technical

**Q: Which Python version?**  
A: Python 3.8 or higher.

**Q: Can I use custom datasets?**  
A: Yes, select "Custom" and provide the path.

**Q: Does it work offline?**  
A: Yes, runs locally. Only icons require internet (CDN).

**Q: Can I extend it?**  
A: Yes, through the plugin system. See [plugin-development.md](plugin-development.md).

**More FAQs**: [user-manual.md#faq](user-manual.md#faq)

## Version History

### v0.1.0 (Current)
- Initial release
- DSL Editor with parsing
- Multi-backend compilation
- Training execution
- Export functionality
- 8 example models

**Full changelog**: [../../CHANGELOG.md](../../CHANGELOG.md)

## Citation

If you use Aquarium IDE in your research, please cite:

```bibtex
@software{neural_aquarium,
  title = {Aquarium IDE: A Web-Based IDE for Neural DSL},
  author = {Neural DSL Development Team},
  year = {2024},
  url = {https://github.com/Lemniscate-world/Neural},
  version = {0.1.0}
}
```

---

<div align="center">

**Made with ❤️ by the Neural DSL Team**

[⭐ Star on GitHub](https://github.com/Lemniscate-world/Neural) • 
[📚 Documentation](installation.md) • 
[💬 Discord](https://discord.gg/KFku4KvS) • 
[🐦 Twitter](https://x.com/NLang4438)

</div>

---

**Version**: 1.0  
**Last Updated**: December 2024  
**License**: MIT
