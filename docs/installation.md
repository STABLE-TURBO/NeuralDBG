# Installation Guide

Get Neural DSL up and running on your system.

## Quick Install

### Basic Installation

```bash
# Core functionality only (~20 MB)
pip install neural-dsl

# Full installation with all features (~2.5 GB)
pip install neural-dsl[full]
```

### Feature-Specific Installation

Install only what you need:

```bash
# ML Framework Backends (TensorFlow, PyTorch, ONNX)
pip install neural-dsl[backends]

# Hyperparameter Optimization (Optuna, scikit-learn)
pip install neural-dsl[hpo]

# AutoML and Neural Architecture Search
pip install neural-dsl[automl]

# Distributed Computing (Ray, Dask)
pip install neural-dsl[distributed]

# Visualization (matplotlib, graphviz, plotly, networkx)
pip install neural-dsl[visualization]

# Dashboard (Dash, Flask) - NeuralDbg interface
pip install neural-dsl[dashboard]

# Cloud Integrations (pygithub, selenium)
pip install neural-dsl[cloud]

# ML Platform Integrations (requests, boto3, google-cloud, azure)
pip install neural-dsl[integrations]
```

### Combined Features

Combine multiple feature groups:

```bash
pip install neural-dsl[backends,visualization,hpo]
pip install neural-dsl[dashboard,visualization]
pip install neural-dsl[automl,distributed]
```

## From Source

### Development Installation

```bash
# Clone the repository
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate
# Linux/macOS:
source .venv/bin/activate

# Install core dependencies only
pip install -e .

# Or install with all optional dependencies
pip install -e ".[full]"

# Install development dependencies (recommended for contributors)
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

## Dependency Groups Reference

### Core Dependencies (Always Installed)
- **click**: CLI framework
- **lark**: DSL parser
- **numpy**: Numerical operations
- **pyyaml**: Configuration files

### Optional Feature Groups

#### Backends
- **torch**: PyTorch backend support
- **tensorflow**: TensorFlow backend support
- **onnx**: ONNX export and runtime

#### HPO (Hyperparameter Optimization)
- **optuna**: HPO framework
- **scikit-learn**: ML utilities and algorithms

#### AutoML
- **optuna**: AutoML and NAS
- **scikit-learn**: Model selection
- **scipy**: Scientific computing

#### Distributed
- **ray**: Distributed computing
- **dask**: Parallel computing

#### Visualization
- **matplotlib**: Charts and plots
- **graphviz**: Architecture diagrams
- **plotly**: Interactive visualizations
- **networkx**: Graph visualizations

#### Dashboard
- **dash**: Web dashboard framework
- **flask**: Web server

#### Cloud
- **pygithub**: GitHub integration
- **selenium**: Browser automation

#### Integrations
- **requests**: HTTP client
- **boto3**: AWS integration
- **google-cloud**: Google Cloud integration
- **azure**: Azure integration

#### Dev (Development Tools)
- **pytest**: Testing framework
- **ruff**: Linting
- **pylint**: Additional linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

## Verification

Verify your installation:

```bash
# Check version
neural --version

# Display help
neural --help

# Test with an example (if you have backends installed)
neural compile examples/mnist.neural --backend tensorflow
```

## Common Use Cases

| Use Case | Installation Command |
|----------|---------------------|
| Learning DSL syntax | `pip install neural-dsl` |
| PyTorch development | `pip install neural-dsl[backends]` |
| TensorFlow development | `pip install neural-dsl[backends]` |
| Using NeuralDbg dashboard | `pip install neural-dsl[dashboard,visualization]` |
| HPO experiments | `pip install neural-dsl[backends,hpo]` |
| AutoML with distributed computing | `pip install neural-dsl[automl,distributed]` |
| ML platform integrations | `pip install neural-dsl[integrations]` |
| Contributing to Neural | `pip install -r requirements-dev.txt` |
| Everything | `pip install neural-dsl[full]` |

## Troubleshooting

### Module Not Found Errors

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | `pip install neural-dsl[backends]` |
| `ModuleNotFoundError: tensorflow` | `pip install neural-dsl[backends]` |
| `ModuleNotFoundError: dash` | `pip install neural-dsl[dashboard]` |
| `ModuleNotFoundError: optuna` | `pip install neural-dsl[hpo]` |
| `ModuleNotFoundError: ray` | `pip install neural-dsl[distributed]` |

### Large Installation Size

If disk space is a concern, install only core:

```bash
pip install neural-dsl
```

Then add features as needed:

```bash
pip install torch  # PyTorch only
pip install tensorflow  # TensorFlow only
```

### Virtual Environment Issues

**Windows PowerShell execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Permission errors:**
```bash
# Use --user flag
pip install --user neural-dsl
```

### Version Conflicts

If you encounter dependency conflicts:

```bash
# Create fresh virtual environment
python -m venv .venv_fresh
# Activate and install
.\.venv_fresh\Scripts\Activate  # Windows
pip install neural-dsl[full]
```

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt
- May require Visual C++ Build Tools for some dependencies
- Download from: https://visualstudio.microsoft.com/downloads/

### macOS

- Xcode Command Line Tools may be required:
  ```bash
  xcode-select --install
  ```

### Linux

- May require development headers:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-dev
  
  # Fedora
  sudo dnf install python3-devel
  ```

## Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade neural-dsl
```

### Upgrade with Features

```bash
pip install --upgrade neural-dsl[full]
```

## Uninstallation

```bash
pip uninstall neural-dsl
```

## Next Steps

After installation:

1. **Quick Start**: See [Quick Start Guide](quickstart.md) or [website/docs/getting-started/quick-start.md](../website/docs/getting-started/quick-start.md)
2. **First Model**: Create your first neural network
3. **Dashboard**: Launch NeuralDbg: `neural debug model.neural`
4. **No-Code Interface**: Try the visual builder: `neural --no_code`
5. **Documentation**: Explore the [docs/](.) directory

## Support

- **Documentation**: [Full docs](.)
- **Issues**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- **Discord**: [Join our community](https://discord.gg/KFku4KvS)
- **Twitter**: [@NLang4438](https://x.com/NLang4438)
