"""
Neural DSL - A domain-specific language for neural networks.

This package provides a declarative syntax for defining, training, debugging,
and deploying neural networks with cross-framework support.

Modules are imported optionally - if a dependency is missing, the module
is set to None and a warning is emitted. Check module availability before use.
"""

# Standard library imports
import warnings  # For emitting optional dependency warnings
from typing import Dict  # For type hints

# Package metadata - always available
__version__ = "0.3.0"  # Current stable version
__author__ = "Lemniscate-SHA-256/SENOUVO Jacques-Charles Gad"  # Package author
__email__ = "Lemniscate_zero@proton.me"  # Contact email

# Import exceptions - always available
from .exceptions import (
    NeuralException,
    ParserException,
    DSLSyntaxError,
    DSLValidationError,
    CodeGenException,
    UnsupportedLayerError,
    UnsupportedBackendError,
    ShapeException,
    ShapeMismatchError,
    InvalidShapeError,
    InvalidParameterError,
    HPOException,
    InvalidHPOConfigError,
    HPOSearchError,
    TrackingException,
    ExperimentNotFoundError,
    MetricLoggingError,
    CloudException,
    CloudConnectionError,
    CloudExecutionError,
    VisualizationException,
    FileOperationError,
    DependencyError,
    ConfigurationError,
    ExecutionError,
    MLOpsException,
    ModelRegistryError,
    ApprovalWorkflowError,
    DeploymentError,
    ABTestError,
    AuditLogError,
    CollaborationException,
    WorkspaceError,
    ConflictError,
    SyncError,
    AccessControlError,
)

# Core modules - imported optionally to handle missing dependencies gracefully
# Each module is wrapped in try/except so the package can be imported
# even when some dependencies are missing.
# Modules are set to None if import fails.

# CLI module - requires 'click' package
try:
    from . import cli  # Command-line interface using click framework
except Exception as e:
    cli = None  # Mark as unavailable
    warnings.warn(f"CLI module unavailable: {e}. Install 'click' to enable.")

# Parser module - requires 'lark' package
try:
    from . import parser  # DSL parser using lark-parser grammar
except Exception as e:
    parser = None  # Mark as unavailable
    warnings.warn(f"Parser module unavailable: {e}. Install 'lark' to enable.")

# Shape propagation module - requires 'numpy' package
try:
    from . import shape_propagation  # Tensor shape inference and validation
except Exception as e:
    shape_propagation = None  # Mark as unavailable
    warnings.warn(f"Shape propagation module unavailable: {e}. Install 'numpy' to enable.")

# Code generation module - may require various backends
try:
    from . import code_generation  # Generate PyTorch/TensorFlow code from DSL
except Exception as e:
    code_generation = None  # Mark as unavailable
    warnings.warn(f"Code generation module unavailable: {e}")

# Visualization module - requires 'matplotlib' package
try:
    from . import visualization  # Network architecture visualization
except Exception as e:
    visualization = None  # Mark as unavailable
    warnings.warn(f"Visualization module unavailable: {e}. Install 'matplotlib' to enable.")

# Dashboard module - requires 'flask' and 'flask-socketio' packages
try:
    from . import dashboard  # Real-time training dashboard
except Exception as e:
    dashboard = None  # Mark as unavailable
    warnings.warn(f"Dashboard module unavailable: {e}. Install 'flask' and 'flask-socketio' to enable.")

# HPO module - hyperparameter optimization
try:
    from . import hpo  # Hyperparameter optimization utilities
except Exception as e:
    hpo = None  # Mark as unavailable
    warnings.warn(f"HPO module unavailable: {e}")

# Cloud module - cloud execution features
try:
    from . import cloud  # Cloud training and deployment
except Exception as e:
    cloud = None  # Mark as unavailable
    warnings.warn(f"Cloud module unavailable: {e}")

# Utility modules - minimal dependencies
try:
    from . import utils  # Common utility functions
except Exception as e:
    utils = None  # Mark as unavailable
    warnings.warn(f"Utils module unavailable: {e}")

# MLOps module - enterprise ML operations
try:
    from . import mlops  # MLOps capabilities (registry, deployment, A/B testing, audit)
except Exception as e:
    mlops = None  # Mark as unavailable
    warnings.warn(f"MLOps module unavailable: {e}")

# Collaboration module - real-time collaborative editing
try:
    from . import collaboration  # Collaborative editing with WebSockets and Git integration
except Exception as e:
    collaboration = None  # Mark as unavailable
    warnings.warn(f"Collaboration module unavailable: {e}")


def check_dependencies() -> Dict[str, bool]:
    """
    Check module availability and return status dictionary.

    Returns
    -------
    Dict[str, bool]
        Mapping of module name to availability status (True/False)

    Examples
    --------
    >>> import neural
    >>> deps = neural.check_dependencies()
    >>> if deps['parser']:
    ...     from neural.parser import create_parser
    ...     parser = create_parser('network')
    >>> print(deps)  # doctest: +SKIP
    {'cli': True, 'parser': True, 'shape_propagation': True, ...}
    
    Notes
    -----
    Use this function to check if optional dependencies are installed
    before importing specific modules. If a module shows as False,
    install the required dependencies:
    
    - parser: requires 'lark'
    - dashboard: requires 'flask', 'dash'
    - hpo: requires 'optuna'
    - collaboration: requires 'websockets'
    - Full installation: pip install neural-dsl[full]
    """
    return {
        "cli": cli is not None,  # CLI available?
        "parser": parser is not None,  # Parser available?
        "shape_propagation": shape_propagation is not None,  # Shape prop available?
        "code_generation": code_generation is not None,  # Code gen available?
        "visualization": visualization is not None,  # Visualization available?
        "dashboard": dashboard is not None,  # Dashboard available?
        "hpo": hpo is not None,  # HPO available?
        "cloud": cloud is not None,  # Cloud available?
        "utils": utils is not None,  # Utils available?
        "mlops": mlops is not None,  # MLOps available?
        "collaboration": collaboration is not None,  # Collaboration available?
    }


# Export list - all public names
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
    "TrackingException",
    "ExperimentNotFoundError",
    "MetricLoggingError",
    "CloudException",
    "CloudConnectionError",
    "CloudExecutionError",
    "VisualizationException",
    "FileOperationError",
    "DependencyError",
    "ConfigurationError",
    "ExecutionError",
    "MLOpsException",
    "ModelRegistryError",
    "ApprovalWorkflowError",
    "DeploymentError",
    "ABTestError",
    "AuditLogError",
    "CollaborationException",
    "WorkspaceError",
    "ConflictError",
    "SyncError",
    "AccessControlError",
    # Modules (may be None if dependencies missing)
    "cli",
    "parser",
    "shape_propagation",
    "code_generation",
    "visualization",
    "dashboard",
    "hpo",
    "cloud",
    "utils",
    "mlops",
    "collaboration",
    # Helper function
    "check_dependencies",
]
