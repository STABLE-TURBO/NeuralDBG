# Neural DSL Repository - Setup Status

## Completed Steps

### ✅ Completed
1. **Virtual Environment Created**: `.venv` directory has been created using `python -m venv .venv`
   - Location: `./.venv/`
   - Python version: 3.14.0
   - Python executable: `./.venv/Scripts/python.exe` (Windows)
   - Status: Ready for use
   - Convention follows `.gitignore` specification (`.venv/` is ignored)

### ⏳ Pending (Requires Manual Execution)

Due to security restrictions in the current environment, the following steps could not be completed automatically and must be run manually:

1. **Activate Virtual Environment**
2. **Install Core Package** 
3. **Install Development Dependencies**

## How to Complete Setup

### Option 1: Manual Commands (Recommended)

**Windows PowerShell:**
```powershell
# Step 1: Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Step 2: Upgrade pip
python -m pip install --upgrade pip

# Step 3: Install the package in editable mode
python -m pip install -e .

# Step 4: Install development dependencies  
python -m pip install -r requirements-dev.txt
```

**Windows Command Prompt:**
```cmd
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-dev.txt
```

### Option 2: Use the Automated Script

A Python script has been created to automate the installation:

```powershell
python complete_setup.py
```

This script will:
- Verify the .venv exists
- Upgrade pip in the venv
- Install neural-dsl in editable mode
- Install all development dependencies

### Option 3: Use Batch Files (Windows)

```cmd
# Activate venv first
.venv\Scripts\activate.bat

# Then use the provided batch files
install.bat         # Installs the package
install_dev.bat     # Installs dev dependencies
```

### Alternative: Install with All Features

If you want all optional features (backends, HPO, AutoML, etc.):
```powershell
pip install -e ".[full]"
pip install -r requirements-dev.txt
```

## Verification Commands

After completing the manual steps, verify the installation:

```powershell
# Activate venv (if not already activated)
.\.venv\Scripts\Activate.ps1

# Check Neural CLI
neural --version

# Test the installation
pytest tests/ -v

# Run linting
python -m ruff check .

# Test the CLI
neural --help
```

## Repository Structure

```
Neural/
├── .venv/                    # ✅ Virtual environment (created)
├── neural/                   # Source code
├── tests/                    # Test files
├── examples/                 # Example DSL files
├── docs/                     # Documentation
├── setup.py                  # ✅ Package configuration (verified)
├── requirements.txt          # ✅ Core dependencies (verified)
├── requirements-dev.txt      # ✅ Dev dependencies (verified)
├── complete_setup.py         # ✅ Automated setup script (created)
├── setup_repo.ps1            # ✅ PowerShell setup script (created)
└── AGENTS.md                 # ✅ Developer guide (verified)
```

## Dependencies to be Installed

### Core (from requirements.txt)
- click>=8.1.3
- lark>=1.1.5
- numpy>=1.23.0
- pyyaml>=6.0.1

### Development (from requirements-dev.txt)
- pytest>=7.0.0
- pytest-cov>=4.0.0
- ruff>=0.1.0
- pylint>=2.15.0
- mypy>=1.0.0
- flake8>=6.0.0
- pre-commit>=3.0.0
- pip-audit>=2.0.0

### Optional Features
Available via `pip install -e ".[feature]"`:
- `[backends]` - PyTorch, TensorFlow, ONNX
- `[hpo]` - Hyperparameter optimization (Optuna)
- `[automl]` - AutoML and NAS
- `[visualization]` - Matplotlib, Graphviz, Plotly
- `[dashboard]` - Dash, Flask dashboards
- `[full]` - All features

## Why Manual Setup is Required

The current environment has security restrictions that prevent:
- Activation of virtual environments via scripts
- Execution of pip install commands programmatically
- Modification of environment variables (PATH, VIRTUAL_ENV)
- Running PowerShell scripts or batch files

This is a protective measure and is normal in restricted environments.

## Support

For more information:
- See `AGENTS.md` for development workflows
- See `INSTALL.md` for detailed installation guide
- See `README.md` for project overview

## Next Steps

1. **Complete Setup**: Run one of the manual setup options above
2. **Install Pre-commit Hooks**: `pre-commit install` (after setup)
3. **Run Tests**: `pytest tests/ -v`
4. **Review Documentation**: Read `AGENTS.md` for development commands

---

**Summary**: The virtual environment is ready. You just need to activate it and run the pip install commands to complete the setup.
