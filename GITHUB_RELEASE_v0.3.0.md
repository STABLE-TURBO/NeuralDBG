# Neural DSL v0.3.0 - AI-Powered Development & Production Deployment

We're excited to announce Neural DSL v0.3.0, a major milestone that transforms Neural into a complete AI-powered development platform with production-ready deployment capabilities!

## 🎯 Release Highlights

### 🤖 AI-Powered Development
Build neural networks using **natural language**! Describe your model in plain English (or 12+ languages), and Neural generates the DSL code automatically.

```python
from neural.ai.nl_processor import NaturalLanguageProcessor

nl = NaturalLanguageProcessor()
dsl_code = nl.generate_dsl("Create a CNN for MNIST with 32 and 64 filters")
```

**Features:**
- Multi-LLM support (OpenAI, Anthropic, Ollama)
- 12+ language support
- Rule-based fallback (no LLM required)
- Conversational model building

📖 [AI Integration Guide](docs/ai_integration_guide.md)

---

### 🚀 Production Deployment
Export and serve models in production with a single command!

```bash
# ONNX with optimization
neural export mnist.neural --format onnx --optimize

# TensorFlow Lite with quantization
neural export mnist.neural --backend tensorflow --format tflite \
  --quantize --quantization-type int8

# Deploy with TensorFlow Serving
neural export mnist.neural --backend tensorflow --format savedmodel \
  --deployment tfserving --model-name mnist_model
```

**Export Formats:**
- ✅ ONNX (10+ optimization passes, 3x speedup)
- ✅ TensorFlow Lite (4x size reduction with quantization)
- ✅ TorchScript (PyTorch production)
- ✅ SavedModel (TensorFlow Serving)

**Serving Platforms:**
- ✅ TensorFlow Serving (auto config, Docker Compose, test clients)
- ✅ TorchServe (MAR prep, config generation, management API)

📖 [Deployment Guide](docs/deployment.md) | [Quick Start](docs/DEPLOYMENT_QUICK_START.md)

---

### 🔄 Full Automation
Zero-touch releases, blog posts, and maintenance!

```bash
# Run all automation tasks
python scripts/automation/master_automation.py --all
```

**Automated:**
- ✅ Releases (version bumping, GitHub, PyPI)
- ✅ Blog posts (Medium, Dev.to, GitHub)
- ✅ Testing & validation
- ✅ Social media updates
- ✅ Daily maintenance (GitHub Actions)

📖 [Automation Guide](AUTOMATION_GUIDE.md)

---

## 📚 What's New

### Added
- **AI Integration**: Natural language to DSL conversion with multi-LLM support
- **ONNX Export**: Cross-framework with 10+ optimization passes
- **TFLite Export**: Mobile/edge deployment with quantization
- **TorchScript Export**: PyTorch production deployment
- **TF Serving Integration**: Config generation, Docker deployment
- **TorchServe Integration**: MAR preparation, management API
- **Export CLI**: `neural export` with comprehensive options
- **Automation System**: Releases, blog posts, tests, social media
- **Enhanced Documentation**: AI guide, deployment guide, automation guide
- **Example Scripts**: `deployment_example.py`, `edge_deployment_example.py`, `ai_examples.py`

### Improved
- **Enhanced Error Messages**: Context-aware with suggestions
- **Optional Dependencies**: Modular installation (install only what you need)
- **Parser**: HPO log_range, device placement, layer validation
- **Code Quality**: Repository cleanup, better test coverage

### Fixed
- HPO log_range parameter naming (min/max consistency)
- Device placement parsing in grammar
- TRACE_DATA attribute in dashboard
- Layer validation (MaxPooling2D, BatchNormalization, Dropout, Conv2D)
- Flaky tests in CI pipeline

---

## 📦 Installation

### PyPI (Recommended)
```bash
# Minimal
pip install neural-dsl

# Full features
pip install neural-dsl[full]

# Specific features
pip install neural-dsl[backends,hpo,visualization]
```

### From Source
```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -e ".[full]"
```

---

## 🚀 Quick Start

### 1. AI-Powered Development
```python
from neural.ai.nl_processor import NaturalLanguageProcessor

nl = NaturalLanguageProcessor()
code = nl.generate_dsl("Create a ResNet-style CNN for image classification")
print(code)
```

### 2. Export for Production
```bash
neural export examples/mnist.neural --format onnx --optimize
```

### 3. Deploy with Serving Platform
```bash
neural export examples/mnist.neural --backend tensorflow \
  --format savedmodel --deployment tfserving --model-name mnist
```

---

## 📖 Documentation

- [Release Notes](RELEASE_NOTES_v0.3.0.md) - Detailed release notes
- [Getting Started](GETTING_STARTED.md) - Quick introduction
- [AI Guide](docs/ai_integration_guide.md) - Natural language to DSL
- [Deployment Guide](docs/deployment.md) - Production deployment
- [Automation Guide](AUTOMATION_GUIDE.md) - Automated workflows
- [Contributing](CONTRIBUTING.md) - How to contribute

---

## 🔄 Migration from v0.2.x

✅ **No breaking changes** - All existing code works unchanged

```bash
# Update
pip install --upgrade neural-dsl

# Optional dependencies now in feature groups
pip install neural-dsl[full]  # For same behavior as v0.2.x
```

See [Migration Guide](MIGRATION_GUIDE_DEPENDENCIES.md) for details.

---

## 🐛 Known Issues

- Some complex nested HPO configurations may require additional validation
- Edge cases in TensorFlow backend HPO integration need further testing
- Certain advanced layer configurations may not be fully supported in PyTorch backend

Report issues: https://github.com/Lemniscate-world/Neural/issues

---

## 🙏 Acknowledgments

Thanks to everyone who contributed feedback, bug reports, and support!

Special thanks to:
- The open-source community
- Beta testers
- Everyone who starred, forked, or shared Neural

---

## 📞 Community

- **Discord**: https://discord.gg/KFku4KvS
- **Twitter**: https://x.com/NLang4438
- **Discussions**: https://github.com/Lemniscate-world/Neural/discussions

---

## ⭐ Support the Project

If you find Neural DSL useful, please:
- ⭐ Star the repository
- 🔄 Share with others
- 🐛 Report issues
- 🤝 Contribute code

---

**Full Changelog**: https://github.com/Lemniscate-world/Neural/blob/main/CHANGELOG.md

**Happy model building! 🚀**
