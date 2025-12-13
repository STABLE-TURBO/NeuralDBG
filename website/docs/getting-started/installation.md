---
sidebar_position: 1
---

# Installation

Get Neural DSL up and running in minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

## Quick Install

### From PyPI (Recommended)

Install the minimal version:

```bash
pip install neural-dsl
```

Or install with all features:

```bash
pip install neural-dsl[full]
```

### From Source

Clone and install:

```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -e ".[full]"
```

## Optional Dependencies

Neural DSL uses a modular dependency structure. Install only what you need:

### Backend Support

```bash
# All backends (TensorFlow, PyTorch, ONNX)
pip install neural-dsl[backends]

# Or individual backends
pip install tensorflow>=2.6
pip install torch>=1.10.0
pip install onnx>=1.10
```

### Additional Features

```bash
# Hyperparameter optimization
pip install neural-dsl[hpo]

# Visualization tools
pip install neural-dsl[visualization]

# NeuralDbg dashboard
pip install neural-dsl[dashboard]

# Cloud integration (Kaggle, Colab, AWS)
pip install neural-dsl[cloud]
```

## Verify Installation

Test your installation:

```bash
neural --version
```

You should see the version number displayed.

## Development Setup

For contributors:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

## Next Steps

- [Quick Start Guide](quick-start) - Build your first model
- [First Model Tutorial](first-model) - Step-by-step walkthrough
- [CLI Reference](/docs/api/cli) - Command-line options

## Troubleshooting

### Import Errors

If you see import errors, ensure you have the required backends installed:

```bash
pip install neural-dsl[backends]
```

### Permission Issues

Use `--user` flag if you don't have admin privileges:

```bash
pip install --user neural-dsl
```

### Virtual Environment

We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install neural-dsl[full]
```

## Getting Help

- Check the [FAQ](/docs/faq)
- Join our [Discord](https://discord.gg/KFku4KvS)
- Open an issue on [GitHub](https://github.com/Lemniscate-world/Neural/issues)
