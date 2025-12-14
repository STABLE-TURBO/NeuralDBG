# Neural DSL - Agent Guide

## Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate                # Windows
# source .venv/bin/activate             # Linux/macOS

# Install in editable mode
pip install -e .                        # Core dependencies only
pip install -e ".[full]"                # With all optional dependencies

# For development (includes editable install + dev tools)
pip install -r requirements-dev.txt     # Recommended for contributors
```

## Dependency Groups

### Core Features (Maintained)
- **Core**: click, lark, numpy, pyyaml (minimal DSL functionality)
- **Backends**: torch, tensorflow, onnx (ML framework support)
- **Visualization**: matplotlib, graphviz, plotly, networkx (charts and diagrams)
- **Dashboard**: dash, flask (NeuralDbg interface)
- **HPO**: optuna, scikit-learn (hyperparameter optimization)
- **AutoML**: optuna, scikit-learn, scipy (architecture search - simplified)
- **Integrations**: boto3, google-cloud, azure (AWS/GCP/Azure only)
- **Dev**: pytest, ruff, pylint, mypy, pre-commit (development tools)

### Deprecated Features (Will Be Removed/Extracted)
- ~~**Collaboration**~~: Use Git instead (DEPRECATED v0.3.0)
- ~~**Federated**~~: Will be extracted to separate repo (DEPRECATED v0.3.0)
- ~~**Aquarium IDE**~~: Will be extracted to separate repo (DEPRECATED v0.3.0)
- ~~**Marketplace**~~: Use HuggingFace Hub (DEPRECATED v0.3.0)

### Experimental Features (Limited Support)
- **Monitoring**: Basic Prometheus integration (experimental)
- **API**: REST API interface (experimental)
- **Cloud**: Being simplified (use with caution)

Install examples:
```bash
pip install -e ".[core]"           # Recommended starting point
pip install -e ".[hpo]"            # Add HPO support
pip install -e ".[automl]"         # Add AutoML support
pip install -e ".[integrations]"   # Add cloud platform support
```

## Commands
- **Build**: N/A (pure Python, no build step)
- **Lint**: `python -m ruff check .` or `python -m pylint neural/`
- **Test**: `python -m pytest tests/ -v` or `pytest --cov=neural --cov-report=term`
- **Dev Server**: `neural server start` (Unified interface on :8050 with Debug/Build/Monitor tabs)
  - Legacy (deprecated): `python neural/dashboard/dashboard.py` or `python neural/no_code/no_code.py`

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
- `neural/federated/` - Federated learning (client-server architecture, differential privacy, secure aggregation, compression)
- `neural/education/` - Interactive tutorials, assignments, grading, progress tracking, LMS integration, teacher dashboard

## Code Style
- Follow PEP 8, 100-char line length (Ruff configured)
- Use type hints (`from __future__ import annotations` for forward refs)
- Docstrings with numpy-style parameters
- No comments unless complex logic requires context
- Functional over classes where reasonable
