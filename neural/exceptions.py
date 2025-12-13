"""
Custom exception hierarchy for the Neural DSL framework.

This module defines a comprehensive exception hierarchy that provides:
- Specific exception types for different error categories
- Rich error context with helpful messages
- Line/column information for parsing errors
- Suggestions for common mistakes

Example Usage:
    
    # Parser exceptions with location information
    from neural.exceptions import DSLSyntaxError
    raise DSLSyntaxError(
        message="Missing colon after layer definition",
        line=10,
        column=25,
        suggestion="Add ':' after the layer name"
    )
    
    # Shape validation errors
    from neural.exceptions import ShapeMismatchError
    raise ShapeMismatchError(
        message="Conv2D kernel size exceeds input dimensions",
        input_shape=(None, 3, 28, 28),
        layer_type='Conv2D'
    )
    
    # Code generation errors
    from neural.exceptions import UnsupportedBackendError
    raise UnsupportedBackendError(
        backend='jax',
        available_backends=['tensorflow', 'pytorch', 'onnx']
    )
    
    # Parameter validation
    from neural.exceptions import InvalidParameterError
    raise InvalidParameterError(
        parameter='units',
        value=-10,
        layer_type='Dense',
        expected='positive integer'
    )
    
    # Dependency errors
    from neural.exceptions import DependencyError
    raise DependencyError(
        dependency='torch',
        feature='PyTorch code generation',
        install_hint='pip install torch'
    )
"""

from typing import Optional, Any, Dict
from enum import Enum
from dataclasses import dataclass


class Severity(Enum):
    """Error severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


class NeuralException(Exception):
    """
    Base exception for all Neural DSL errors.
    
    Provides common functionality for all Neural exceptions including
    error context, severity levels, and formatted error messages.
    
    Attributes:
        message: Error message
        severity: Error severity level
        context: Additional context about the error
    """
    
    def __init__(
        self,
        message: str,
        severity: Severity = Severity.ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with context."""
        msg = f"[{self.severity.name}] {self.message}"
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" (Context: {ctx_str})"
        return msg
    
    def __str__(self) -> str:
        return self._format_message()


@dataclass
class ErrorLocation:
    """Holds location information for parsing errors."""
    line: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None


class ParserException(NeuralException):
    """
    Base exception for all parsing-related errors.
    
    Raised when the DSL parser encounters syntax errors, invalid tokens,
    or malformed input.
    
    Attributes:
        location: Error location details
    """
    
    def __init__(
        self,
        message: str,
        location: Optional[ErrorLocation] = None,
        severity: Severity = Severity.ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        self.location = location or ErrorLocation()
        
        context = context or {}
        if self.location.line is not None:
            context['line'] = self.location.line
        if self.location.column is not None:
            context['column'] = self.location.column
        
        super().__init__(message, severity, context)
    
    def _format_message(self) -> str:
        """Format parser error with location information."""
        parts = [f"[{self.severity.name}]"]
        
        if self.location.line is not None and self.location.column is not None:
            parts.append(f"Line {self.location.line}, Column {self.location.column}:")
        elif self.location.line is not None:
            parts.append(f"Line {self.location.line}:")
        
        parts.append(self.message)
        
        msg = " ".join(parts)
        
        if self.location.code_snippet:
            msg += f"\n\n{self.location.code_snippet}"
        
        if self.location.suggestion:
            msg += f"\n\n💡 Suggestion: {self.location.suggestion}"
        
        return msg


class DSLSyntaxError(ParserException):
    """
    Raised for syntax errors in the Neural DSL.
    
    Examples:
        - Missing colons, parentheses, or brackets
        - Invalid token sequences
        - Malformed layer definitions
    """


class DSLValidationError(ParserException):
    """
    Raised for semantic validation errors in the DSL.
    
    Examples:
        - Invalid parameter values
        - Incompatible layer configurations
        - Missing required parameters
    """


class CodeGenException(NeuralException):
    """
    Base exception for code generation errors.
    
    Raised when generating backend-specific code (TensorFlow, PyTorch, ONNX)
    encounters errors.
    
    Attributes:
        backend: The backend being targeted (tensorflow, pytorch, onnx)
        layer_type: Optional layer type that caused the error
    """
    
    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        layer_type: Optional[str] = None,
        severity: Severity = Severity.ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        self.backend = backend
        self.layer_type = layer_type
        
        context = context or {}
        if backend:
            context['backend'] = backend
        if layer_type:
            context['layer_type'] = layer_type
        
        super().__init__(message, severity, context)


class UnsupportedLayerError(CodeGenException):
    """
    Raised when a layer type is not supported by the target backend.
    
    Examples:
        - Using TensorFlow-specific layers with PyTorch backend
        - Custom layers without proper backend implementation
    """
    
    def __init__(
        self,
        layer_type: str,
        backend: str,
        suggestion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Layer type '{layer_type}' is not supported by backend '{backend}'"
        if suggestion:
            message += f". {suggestion}"
        super().__init__(message, backend=backend, layer_type=layer_type, context=context)


class UnsupportedBackendError(CodeGenException):
    """
    Raised when an unsupported backend is specified.
    
    Examples:
        - Invalid backend name
        - Backend not installed or available
    """
    
    def __init__(
        self,
        backend: str,
        available_backends: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Unsupported backend: '{backend}'"
        if available_backends:
            message += f". Available backends: {', '.join(available_backends)}"
        super().__init__(message, backend=backend, context=context)


class ShapeException(NeuralException):
    """
    Base exception for shape propagation and validation errors.
    
    Raised when tensor shapes are incompatible or invalid during
    shape propagation through the network.
    
    Attributes:
        input_shape: The input shape that caused the error
        output_shape: The expected/computed output shape
        layer_type: The layer type where the error occurred
    """
    
    def __init__(
        self,
        message: str,
        input_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None,
        layer_type: Optional[str] = None,
        severity: Severity = Severity.ERROR,
        context: Optional[Dict[str, Any]] = None
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer_type = layer_type
        
        context = context or {}
        if input_shape is not None:
            context['input_shape'] = input_shape
        if output_shape is not None:
            context['output_shape'] = output_shape
        if layer_type:
            context['layer_type'] = layer_type
        
        super().__init__(message, severity, context)


class ShapeMismatchError(ShapeException):
    """
    Raised when tensor shapes are incompatible between layers.
    
    Examples:
        - Dense layer receiving 4D input without flattening
        - Kernel size exceeding input dimensions
        - Incompatible shapes in concatenation/addition operations
    """


class InvalidShapeError(ShapeException):
    """
    Raised when a shape is invalid or malformed.
    
    Examples:
        - Negative dimensions
        - Empty shapes
        - Invalid shape format
    """


class InvalidParameterError(NeuralException):
    """
    Raised when a layer parameter has an invalid value.
    
    Examples:
        - Negative filter counts
        - Zero or negative units
        - Invalid activation function names
        - Missing required parameters
    """
    
    def __init__(
        self,
        parameter: str,
        value: Any,
        layer_type: Optional[str] = None,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.parameter = parameter
        self.value = value
        self.layer_type = layer_type
        
        message = f"Invalid parameter '{parameter}': {value}"
        if layer_type:
            message = f"Invalid parameter '{parameter}' for layer '{layer_type}': {value}"
        if expected:
            message += f". Expected: {expected}"
        
        context = context or {}
        context['parameter'] = parameter
        context['value'] = value
        if layer_type:
            context['layer_type'] = layer_type
        
        super().__init__(message, context=context)


class HPOException(NeuralException):
    """
    Base exception for hyperparameter optimization errors.
    
    Raised during HPO configuration, search, or optimization.
    """


class InvalidHPOConfigError(HPOException):
    """
    Raised when HPO configuration is invalid.
    
    Examples:
        - Invalid search space definition
        - Incompatible HPO strategy
        - Missing required HPO parameters
    """


class HPOSearchError(HPOException):
    """
    Raised when HPO search fails.
    
    Examples:
        - Search timeout
        - No valid trials found
        - Optimization failure
    """


class TrackingException(NeuralException):
    """
    Base exception for experiment tracking errors.
    
    Raised when logging experiments, metrics, or artifacts fails.
    """


class ExperimentNotFoundError(TrackingException):
    """
    Raised when an experiment cannot be found.
    """
    
    def __init__(
        self,
        experiment_id: str,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Experiment not found: {experiment_id}"
        context = context or {}
        context['experiment_id'] = experiment_id
        super().__init__(message, context=context)


class MetricLoggingError(TrackingException):
    """
    Raised when logging metrics fails.
    
    Examples:
        - Invalid metric values
        - Backend unavailable
        - Serialization errors
    """


class CloudException(NeuralException):
    """
    Base exception for cloud execution errors.
    
    Raised when executing models on cloud platforms (SageMaker, etc.).
    """


class CloudConnectionError(CloudException):
    """
    Raised when connection to cloud service fails.
    
    Examples:
        - Authentication failure
        - Network timeout
        - Invalid credentials
    """


class CloudExecutionError(CloudException):
    """
    Raised when cloud execution fails.
    
    Examples:
        - Resource unavailable
        - Execution timeout
        - Runtime errors on cloud platform
    """


class VisualizationException(NeuralException):
    """
    Base exception for visualization errors.
    
    Raised when generating visualizations, graphs, or dashboards fails.
    """


class FileOperationError(NeuralException):
    """
    Raised when file operations fail.
    
    Examples:
        - File not found
        - Permission denied
        - Invalid file format
        - Write errors
    """
    
    def __init__(
        self,
        operation: str,
        filepath: str,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"File {operation} failed: {filepath}"
        if reason:
            message += f". Reason: {reason}"
        
        context = context or {}
        context['operation'] = operation
        context['filepath'] = filepath
        
        super().__init__(message, context=context)


class DependencyError(NeuralException):
    """
    Raised when a required dependency is missing or incompatible.
    
    Examples:
        - Missing optional dependencies (torch, tensorflow, etc.)
        - Incompatible versions
        - Import errors
    """
    
    def __init__(
        self,
        dependency: str,
        feature: Optional[str] = None,
        install_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        message = f"Missing dependency: {dependency}"
        if feature:
            message += f" (required for {feature})"
        if install_hint:
            message += f". Install with: {install_hint}"
        
        context = context or {}
        context['dependency'] = dependency
        
        super().__init__(message, context=context)


class ConfigurationError(NeuralException):
    """
    Raised when configuration is invalid or missing.
    
    Examples:
        - Missing required configuration
        - Invalid configuration values
        - Configuration conflicts
    """


class ExecutionError(NeuralException):
    """
    Raised when model execution fails.
    
    Examples:
        - Runtime errors during training
        - Inference failures
        - Device errors (GPU/CPU)
    """


class MLOpsException(NeuralException):
    """
    Base exception for MLOps-related errors.
    
    Raised during model registry, deployment, A/B testing, or audit operations.
    """


class ModelRegistryError(MLOpsException):
    """
    Raised when model registry operations fail.
    
    Examples:
        - Model not found
        - Version conflict
        - Registration failure
    """


class ApprovalWorkflowError(MLOpsException):
    """
    Raised when approval workflow operations fail.
    
    Examples:
        - Approval not found
        - Invalid approval status
        - Unauthorized approval attempt
    """


class DeploymentError(MLOpsException):
    """
    Raised when deployment operations fail.
    
    Examples:
        - Deployment creation failure
        - Health check failure
        - Rollback failure
    """


class ABTestError(MLOpsException):
    """
    Raised when A/B testing operations fail.
    
    Examples:
        - Invalid test configuration
        - Test not found
        - Statistical analysis failure
    """


class AuditLogError(MLOpsException):
    """
    Raised when audit logging operations fail.
    
    Examples:
        - Event logging failure
        - Query failure
        - Report generation failure
    """


class CollaborationException(NeuralException):
    """
    Base exception for collaboration-related errors.
    
    Raised during collaborative editing, workspace management, or synchronization.
    """


class WorkspaceError(CollaborationException):
    """
    Raised when workspace operations fail.
    
    Examples:
        - Workspace not found
        - Access denied
        - Invalid workspace configuration
    """


class ConflictError(CollaborationException):
    """
    Raised when edit conflicts occur during collaboration.
    
    Examples:
        - Concurrent edits to the same lines
        - Merge conflicts
        - Incompatible changes
    """


class SyncError(CollaborationException):
    """
    Raised when synchronization fails.
    
    Examples:
        - Network errors
        - Version mismatch
        - Sync timeout
    """


class AccessControlError(CollaborationException):
    """
    Raised when access control validation fails.
    
    Examples:
        - Insufficient permissions
        - Invalid credentials
        - Token expired
    """
    """


# Convenience functions for common error scenarios

def raise_parser_error(
    message: str,
    line: Optional[int] = None,
    column: Optional[int] = None,
    suggestion: Optional[str] = None
) -> None:
    """Raise a parser exception with location information."""
    location = ErrorLocation(line=line, column=column, suggestion=suggestion)
    raise ParserException(message, location=location)


def raise_shape_error(
    message: str,
    input_shape: Optional[tuple] = None,
    layer_type: Optional[str] = None
) -> None:
    """Raise a shape exception with shape information."""
    raise ShapeException(message, input_shape=input_shape, layer_type=layer_type)


def raise_codegen_error(
    message: str,
    backend: Optional[str] = None,
    layer_type: Optional[str] = None
) -> None:
    """Raise a code generation exception with backend information."""
    raise CodeGenException(message, backend=backend, layer_type=layer_type)


# Exception hierarchy overview (for documentation):
"""
Exception Hierarchy:
--------------------

NeuralException (base)
├── ParserException
│   ├── DSLSyntaxError
│   └── DSLValidationError
├── CodeGenException
│   ├── UnsupportedLayerError
│   └── UnsupportedBackendError
├── ShapeException
│   ├── ShapeMismatchError
│   └── InvalidShapeError
├── InvalidParameterError
├── HPOException
│   ├── InvalidHPOConfigError
│   └── HPOSearchError
├── TrackingException
│   ├── ExperimentNotFoundError
│   └── MetricLoggingError
├── CloudException
│   ├── CloudConnectionError
│   └── CloudExecutionError
├── VisualizationException
├── FileOperationError
├── DependencyError
├── ConfigurationError
├── ExecutionError
├── MLOpsException
│   ├── ModelRegistryError
│   ├── ApprovalWorkflowError
│   ├── DeploymentError
│   ├── ABTestError
│   └── AuditLogError
└── CollaborationException
    ├── WorkspaceError
    ├── ConflictError
    ├── SyncError
    └── AccessControlError

Usage Examples:
---------------
See module docstring for detailed examples of each exception type.
"""
