# Neural DSL - Agent Guide

## Setup
```bash
python -m venv .venv                    # Create venv (convention: .venv or venv)
.\.venv\Scripts\Activate                # Windows activation
pip install -e .                        # Install core dependencies only
pip install -e ".[full]"                # Install with all optional dependencies
pip install -r requirements-dev.txt     # Install development dependencies (recommended)
```

## Dependency Groups
- **Core**: click, lark, numpy, pyyaml (minimal DSL functionality)
- **Backends**: torch, tensorflow, onnx (ML framework support)
- **HPO**: optuna, scikit-learn (hyperparameter optimization)
- **AutoML**: optuna, scikit-learn, scipy (automated ML and NAS)
- **Distributed**: ray, dask (distributed computing for AutoML)
- **Visualization**: matplotlib, graphviz, plotly, networkx (charts and diagrams)
- **Dashboard**: dash, flask (NeuralDbg interface)
- **Cloud**: pygithub, selenium (cloud integrations)
- **Integrations**: None (base classes only, no external dependencies)
- **Teams**: click, pyyaml (multi-tenancy and team management)
- **AI**: langdetect (natural language processing and language detection)
- **Dev**: pytest, ruff, pylint, mypy, pre-commit (development tools)

Install specific feature groups: `pip install -e ".[hpo]"`, `pip install -e ".[automl]"`, `pip install -e ".[integrations]"`, `pip install -e ".[distributed]"`, `pip install -e ".[teams]"`, or `pip install -e ".[ai]"`

## Commands
- **Build**: N/A (pure Python, no build step)
- **Lint**: `python -m ruff check .`
- **Type Check**: `python -m mypy neural/ --ignore-missing-imports`
- **Test**: `python -m pytest tests/ -v`
- **Test with Coverage**: `pytest tests/ -v --cov=neural --cov-report=term --cov-report=html`
- **Generate Coverage Report**: `python generate_test_coverage_summary.py` (creates TEST_COVERAGE_SUMMARY.md)
- **Security**: `python -m bandit -r neural/ -ll` or `python -m pip_audit -l`
- **Dev Server**: `python neural/dashboard/dashboard.py` (NeuralDbg on :8050) or `python neural/no_code/no_code.py` (No-code GUI on :8051)

## Tech Stack
- **Language**: Python 3.8+ with type hints
- **Core**: Lark (DSL parser), Click (CLI), Flask/Dash (dashboards)
- **ML Backends**: TensorFlow, PyTorch, ONNX (all optional)
- **Tools**: pytest, ruff/pylint, pre-commit, mypy

## Architecture
- `neural/cli/` - CLI commands (compile, run, visualize, debug)
- `neural/parser/` - DSL parser and AST transformer
- `neural/code_generation/` - Multi-backend code generators (TF/PyTorch/ONNX)
- `neural/shape_propagation/` - Shape validation and propagation
- `neural/dashboard/` - NeuralDbg real-time debugger
- `neural/no_code/` - No-code web interface
- `neural/hpo/` - Hyperparameter optimization (Optuna integration)
- `neural/automl/` - AutoML and Neural Architecture Search (NAS)
- `neural/integrations/` - Base classes for custom ML platform connectors
- `neural/teams/` - Multi-tenancy, team management, RBAC, quotas, analytics, and billing

## Code Style
- Follow PEP 8, 100-char line length (Ruff configured)
- Use type hints (`from __future__ import annotations` for forward refs)
- Docstrings with numpy-style parameters
- No comments unless complex logic requires context
- Functional over classes where reasonable
ine (runs on push/PR)
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
- **Testing**: `tests/` - comprehensive test coverage expected

Peripheral features (lower priority):
- AutoML, HPO, Integrations, Teams
- Aquarium IDE (consider separate repository)
- Marketing automation, blog generation

## Cleanup
The repository has been cleaned to remove 200+ redundant files:
- Implementation summaries archived to `docs/archive/`
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
