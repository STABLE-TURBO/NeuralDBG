"""
Neural DSL - A domain-specific language for neural networks.

This package provides a declarative syntax for defining neural networks
with multi-backend compilation (TensorFlow, PyTorch, ONNX) and automatic
shape validation.

Core Features:
- Declarative DSL syntax for neural network definition
- Multi-backend code generation (TensorFlow, PyTorch, ONNX)
- Automatic shape inference and validation
- Network visualization
- Hyperparameter optimization (optional)
- AutoML and Neural Architecture Search (optional)

Modules are imported optionally - if a dependency is missing, the module
is set to None and a warning is emitted.
"""

import warnings
from typing import Dict


# Package metadata
__version__ = "0.4.0"
__author__ = "Lemniscate-SHA-256/SENOUVO Jacques-Charles Gad"
__email__ = "Lemniscate_zero@proton.me"

# Import exceptions
from .exceptions import (
    CodeGenException,
    ConfigurationError,
    DependencyError,
    DSLSyntaxError,
    DSLValidationError,
    ExecutionError,
    FileOperationError,
    HPOException,
    HPOSearchError,
    InvalidHPOConfigError,
    InvalidParameterError,
    InvalidShapeError,
    NeuralException,
    ParserException,
    ShapeException,
    ShapeMismatchError,
    UnsupportedBackendError,
    UnsupportedLayerError,
    VisualizationException,
)


# Core modules - CLI
try:
    from . import cli
except Exception as e:
    cli = None
    warnings.warn(f"CLI module unavailable: {e}. Install 'click' to enable.")

# Core modules - Parser
try:
    from . import parser
except Exception as e:
    parser = None
    warnings.warn(f"Parser module unavailable: {e}. Install 'lark' to enable.")

# Core modules - Shape propagation
try:
    from . import shape_propagation
except Exception as e:
    shape_propagation = None
    warnings.warn(f"Shape propagation module unavailable: {e}. Install 'numpy' to enable.")

# Core modules - Code generation
try:
    from . import code_generation
except Exception as e:
    code_generation = None
    warnings.warn(f"Code generation module unavailable: {e}")

# Core modules - Visualization
try:
    from . import visualization
except Exception as e:
    visualization = None
    warnings.warn(f"Visualization module unavailable: {e}. Install visualization dependencies.")

# Core modules - Utils
try:
    from . import utils
except Exception as e:
    utils = None
    warnings.warn(f"Utils module unavailable: {e}")

# Optional modules - Training
try:
    from . import training
except Exception as e:
    training = None
    warnings.warn(f"Training module unavailable: {e}")

# Optional modules - Metrics
try:
    from . import metrics
except Exception as e:
    metrics = None
    warnings.warn(f"Metrics module unavailable: {e}")

# Optional modules - Dashboard (simplified debugging interface)
try:
    from . import dashboard
except Exception as e:
    dashboard = None
    warnings.warn(f"Dashboard module unavailable: {e}. Install dashboard dependencies.")

# Optional modules - HPO
try:
    from . import hpo
except Exception as e:
    hpo = None
    warnings.warn(f"HPO module unavailable: {e}. Install 'optuna' and 'scikit-learn'.")

# Optional modules - AutoML
try:
    from . import automl
except Exception as e:
    automl = None
    warnings.warn(f"AutoML module unavailable: {e}. Install automl dependencies.")


def check_dependencies() -> Dict[str, bool]:
    """
    Check module availability and return status dictionary.

    Returns
    -------
    Dict[str, bool]
        Mapping of module name to availability status

    Examples
    --------
    >>> import neural
    >>> deps = neural.check_dependencies()
    >>> if deps['parser']:
    ...     from neural.parser import create_parser
    """
    return {
        "cli": cli is not None,
        "parser": parser is not None,
        "shape_propagation": shape_propagation is not None,
        "code_generation": code_generation is not None,
        "visualization": visualization is not None,
        "utils": utils is not None,
        "training": training is not None,
        "metrics": metrics is not None,
        "dashboard": dashboard is not None,
        "hpo": hpo is not None,
        "automl": automl is not None,
    }


__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    # Exceptions
    "NeuralException",
    "ParserException",
    "DSLSyntaxError",
    "DSLValidationError",
    "CodeGenException",
    "UnsupportedLayerError",
    "UnsupportedBackendError",
    "ShapeException",
    "ShapeMismatchError",
    "InvalidShapeError",
    "InvalidParameterError",
    "HPOException",
    "InvalidHPOConfigError",
    "HPOSearchError",
    "VisualizationException",
    "FileOperationError",
    "DependencyError",
    "ConfigurationError",
    "ExecutionError",
    # Core modules
    "cli",
    "parser",
    "shape_propagation",
    "code_generation",
    "visualization",
    "utils",
    # Optional modules
    "training",
    "metrics",
    "dashboard",
    "hpo",
    "automl",
    # Helper
    "check_dependencies",
]
