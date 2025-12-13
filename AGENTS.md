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
- **Integrations**: requests, boto3, google-cloud, azure (ML platform connectors)
- **Teams**: click, pyyaml (multi-tenancy and team management)
- **Dev**: pytest, ruff, pylint, mypy, pre-commit (development tools)

Install specific feature groups: `pip install -e ".[hpo]"`, `pip install -e ".[automl]"`, `pip install -e ".[integrations]"`, or `pip install -e ".[distributed]"`

## Commands
- **Build**: N/A (pure Python, no build step)
- **Lint**: `python -m ruff check .` or `python -m pylint neural/`
- **Test**: `python -m pytest tests/ -v` or `pytest --cov=neural --cov-report=term`
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
- `neural/integrations/` - ML platform connectors (Databricks, SageMaker, Vertex AI, Azure ML, Paperspace, Run:AI)
- `neural/teams/` - Multi-tenancy, team management, RBAC, quotas, analytics, and billing

## Code Style
- Follow PEP 8, 100-char line length (Ruff configured)
- Use type hints (`from __future__ import annotations` for forward refs)
- Docstrings with numpy-style parameters
- No comments unless complex logic requires context
- Functional over classes where reasonable
