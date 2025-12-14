# Neural DSL - Setup Status

## ✓ Completed Steps

### 1. Virtual Environment
- **Location**: `.venv/` (follows convention from .gitignore and AGENTS.md)
- **Python Version**: 3.14.0
- **Status**: ✓ Created and functional

### 2. Core Package Installation
- **Method**: Installed in editable mode with `pip install -e .`
- **Status**: ✓ Successfully installed
- **Installed Packages**:
  - neural-dsl 0.3.0 (editable)
  - click 8.3.1
  - lark 1.3.1
  - numpy 2.3.5
  - pyyaml 6.0.3
  - colorama 0.4.6

## Core Dependencies Status
The following core dependencies are already available:
- ✅ click >= 8.1.3 (version 8.3.1 installed)
- ✅ lark >= 1.1.5 (version 1.3.1 installed)
- ✅ numpy >= 1.23.0 (version 2.3.5 installed)
- ✅ pyyaml >= 6.0.1 (version 6.0.3 installed)

## Additional Dependencies Available
Many optional dependencies are also available:
- ✅ torch >= 1.10.0 (version 2.9.1 installed)
- ✅ onnx >= 1.10 (version 1.19.1 installed)
- ✅ optuna >= 3.0 (version 4.6.0 installed)
- ✅ scikit-learn >= 1.0 (version 1.7.2 installed)
- ✅ scipy >= 1.7 (version 1.16.3 installed)
- ✅ matplotlib (version 3.10.7 installed)
- ✅ graphviz (version 0.21 installed)
- ✅ networkx >= 2.8.8 (version 3.6 installed)
- ✅ plotly >= 5.18 (version 6.5.0 installed)
- ✅ dash >= 2.18.2 (version 3.3.0 installed)
- ✅ flask >= 3.0 (version 3.1.2 installed)
- ✅ flask-cors >= 3.1 (version 6.0.1 installed)
- ✅ pytest >= 7.0.0 (version 9.0.1 installed)

## ⚠️ Pending: Development Dependencies

Development dependencies from `requirements-dev.txt` need to be installed manually:

```powershell
.\.venv\Scripts\Activate.ps1
pip install pytest>=7.0.0 pytest-cov>=4.0.0 ruff>=0.1.0 pylint>=2.15.0 mypy>=1.0.0 flake8>=6.0.0 pre-commit>=3.0.0 pip-audit>=2.0.0
```

Or use the install script:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Or install the dev extras:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Missing Development Tools
- ❌ ruff >= 0.1.0 (not installed - required for linting)
- ❌ pylint >= 2.15.0 (not installed - alternative linter)
- ❌ mypy >= 1.0.0 (not installed - type checking)

## Alternative: Use Existing .venv312

Note: There's an existing `.venv312/` directory that already contains all development tools but doesn't have neural-dsl installed. To use it:

```powershell
.\.venv312\Scripts\Activate.ps1
pip install -e .
```

## Next Steps

To complete the setup, you need to:

### 1. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install the Neural DSL package in editable mode
```powershell
pip install -e .
```

### 3. Install development dependencies
```powershell
pip install -r requirements-dev.txt
```

Or install everything at once:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[full]"
pip install -r requirements-dev.txt
```

## Testing the Installation

After installation, you can verify it works:

```powershell
# Test the CLI
neural --help

# Run tests
python -m pytest tests/ -v

# Run linter (after installing ruff)
python -m ruff check .

# Or with pylint
python -m pylint neural/
```

## Verification Commands

Once dev dependencies are installed, verify with:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify installation
pip list

# Test imports
python -c "import neural; import pytest; import ruff; print('All imports successful!')"

# Run tests (from AGENTS.md)
python -m pytest tests/ -v

# Run linter (from AGENTS.md)
python -m ruff check .

# Or use pylint
python -m pylint neural/
```

## Build, Lint, and Test Commands

As documented in AGENTS.md:
- **Build**: N/A (pure Python, no build step)
- **Lint**: `python -m ruff check .` or `python -m pylint neural/`
- **Test**: `python -m pytest tests/ -v` or `pytest --cov=neural --cov-report=term`
- **Dev Server**: 
  - NeuralDbg: `python neural/dashboard/dashboard.py` (port 8050)
  - No-code GUI: `python neural/no_code/no_code.py` (port 8051)

## Additional Feature Groups

(Optional) Install additional feature groups as needed:
- `pip install -e ".[hpo]"` - Hyperparameter optimization
- `pip install -e ".[automl]"` - AutoML and NAS
- `pip install -e ".[backends]"` - TensorFlow, PyTorch, ONNX
- `pip install -e ".[ai]"` - AI features (language detection)
- `pip install -e ".[full]"` - All features
