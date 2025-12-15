# Neural DSL - Agent Guide

## Setup
```bash
python -m venv .venv                    # Create venv (convention: .venv or venv)
.\.venv\Scripts\Activate                # Windows activation
pip install -e .                        # Install core dependencies
pip install -r requirements-dev.txt     # Install development dependencies (recommended)
```

## Dependency Groups
- **Core**: click, lark, numpy, pyyaml (minimal DSL functionality)
- **Backends**: torch, tensorflow, onnx (ML framework support)
- **HPO**: optuna, scikit-learn (hyperparameter optimization)
- **Visualization**: matplotlib, graphviz, plotly, networkx (charts and diagrams)
- **Dashboard**: dash, flask (NeuralDbg interface)
- **AI**: langdetect (natural language processing and language detection)
- **Dev**: pytest, ruff, pylint, mypy, pre-commit (development tools)

Install specific feature groups: `pip install -e ".[hpo]"`, `pip install -e ".[automl]"`, `pip install -e ".[distributed]"`, or `pip install -e ".[ai]"`

**Note**: The API server module has been removed as per v0.4.0. For REST API functionality, wrap Neural in FastAPI/Flask.

## Commands
- **Build**: N/A (pure Python, no build step)
- **Lint**: `python -m ruff check .`
- **Type Check**: `python -m mypy neural/ --ignore-missing-imports`
- **Test**: `python -m pytest tests/ -v`
- **Test with Coverage**: `pytest tests/ -v --cov=neural --cov-report=term --cov-report=html`
- **Generate Coverage Report**: `python generate_test_coverage_summary.py` (creates TEST_COVERAGE_SUMMARY.md)
- **Security**: `python -m bandit -r neural/ -ll` or `python -m pip_audit -l`
- **Dev Server**: `python neural/dashboard/dashboard.py` (NeuralDbg on :8050)

## Tech Stack
- **Language**: Python 3.8+ with type hints
- **Core**: Lark (DSL parser), Click (CLI), Flask/Dash (dashboards)
- **ML Backends**: TensorFlow, PyTorch, ONNX (all optional)
- **Tools**: pytest, ruff/pylint, pre-commit, mypy

## Architecture
Core modules focused on essential DSL functionality:

- `neural/parser/` - DSL parser and AST transformer
- `neural/code_generation/` - Multi-backend code generators (TensorFlow/PyTorch/ONNX)
- `neural/shape_propagation/` - Shape validation and propagation
- `neural/cli/` - CLI commands (compile, run, visualize, debug)
- `neural/dashboard/` - NeuralDbg real-time debugger
- `neural/hpo/` - Hyperparameter optimization (Optuna integration)
- `neural/visualization/` - Model visualization and graph generation

## Code Style
- Follow PEP 8, 100-char line length (Ruff configured)
- Use type hints (`from __future__ import annotations` for forward refs)
- Docstrings with numpy-style parameters
- No comments unless complex logic requires context
- Functional over classes where reasonable

## CI/CD Workflows
1. **ci.yml** - Continuous integration (runs on push/PR)
   - Lint with Ruff
   - Type check with Mypy
   - Tests on Python 3.8, 3.11, 3.12 (Ubuntu & Windows)
   - Security scanning (Bandit, Safety, pip-audit)
   - Code coverage reporting

2. **release.yml** - Release automation (runs on version tags)
   - Build distributions
   - Publish to PyPI
   - Create GitHub releases

3. **codeql.yml** - Security analysis (weekly + PR)
   - CodeQL scanning for Python and JavaScript

4. **validate-examples.yml** - Example validation (daily + changes to examples/)
   - Validate DSL syntax
   - Test compilation

## Repository Structure
Focus areas for development:
- **Core DSL**: `neural/parser/`, `neural/code_generation/`
- **Shape Validation**: `neural/shape_propagation/`
- **Multi-Backend**: TensorFlow, PyTorch, ONNX code generators
- **Visualization**: Model graphs and architecture diagrams
- **HPO**: Hyperparameter optimization workflows
- **Testing**: `tests/` - comprehensive test coverage expected

Peripheral features (lower priority):
- AutoML, HPO, Integrations
- Aquarium IDE (consider separate repository)
- Marketing automation, blog generation

Note: Teams module (multi-tenancy, RBAC, billing, analytics) has been removed as it was not actively used by core features.

## Cleanup
The repository has been cleaned to remove 200+ redundant files:
- Archive directory (`docs/archive/`) removed (22 redundant files)
- Workflows consolidated from 20+ to 4 essential ones
- Development scripts removed or consolidated
- `.gitignore` comprehensively updated

### Cache and Artifacts Cleanup
Remove cache directories, virtual environments, and test artifacts:
- **Windows PowerShell**: `.\cleanup_cache_and_artifacts.ps1`
- **Windows Command Prompt**: `cleanup_cache_and_artifacts.bat`
- **Unix/Linux/macOS**: `./cleanup_cache_and_artifacts.sh`
- **Python (cross-platform)**: `python cleanup_cache_and_artifacts.py`

See [CLEANUP_README.md](CLEANUP_README.md) for detailed instructions and manual cleanup commands.

All cache patterns are already in `.gitignore`:
- `__pycache__/`, `.pytest_cache/`, `.hypothesis/`, `.mypy_cache/`, `.ruff_cache/`
- `.venv*/`, `venv*/`
- Test artifacts: `test_*.html`, `test_*.png`
- Temporary scripts: `sample_*.py`, `test_*.py` (root only)
