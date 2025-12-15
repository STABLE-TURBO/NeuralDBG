"""
Enhanced error messages with actionable suggestions for common mistakes.

This module provides helpful error messages, suggestions, and fixes for common
user errors in the Neural DSL framework.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Any
import re


class ErrorSuggestion:
    """
    Provides context-aware suggestions for common errors.
    """
    
    # Common parameter typos and their corrections
    PARAMETER_TYPOS: Dict[str, str] = {
        "unit": "units",
        "filter": "filters",
        "kernal_size": "kernel_size",
        "kernelsize": "kernel_size",
        "pool_size": "pool_size",
        "poolsize": "pool_size",
        "activaton": "activation",
        "actvation": "activation",
        "return_sequence": "return_sequences",
        "learning_rt": "learning_rate",
        "learningrate": "learning_rate",
        "dropout_rate": "rate",
        "padding_type": "padding",
        "num_filter": "filters",
        "num_units": "units",
        "num_heads": "num_heads",
        "head_count": "num_heads",
    }
    
    # Common layer name typos
    LAYER_TYPOS: Dict[str, str] = {
        "Dense2D": "Dense",
        "Conv1D": "Conv1D",
        "Conv3D": "Conv3D",
        "Convolution2D": "Conv2D",
        "MaxPool2D": "MaxPooling2D",
        "MaxPool": "MaxPooling2D",
        "AvgPool2D": "AveragePooling2D",
        "AvgPool": "AveragePooling2D",
        "BatchNorm": "BatchNormalization",
        "BN": "BatchNormalization",
        "RELU": "ReLU",
        "Relu": "ReLU",
        "Lstm": "LSTM",
        "Gru": "GRU",
        "Embedding": "Embedding",
        "Attention": "Attention",
        "MultiHeadAttention": "MultiHeadAttention",
        "GlobalAvgPool2D": "GlobalAveragePooling2D",
        "GlobalMaxPool2D": "GlobalMaxPooling2D",
    }
    
    # Common activation function typos
    ACTIVATION_TYPOS: Dict[str, str] = {
        "Relu": "relu",
        "RELU": "relu",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
        "Softmax": "softmax",
        "elu": "elu",
        "selu": "selu",
        "leakyrelu": "leaky_relu",
        "leaky-relu": "leaky_relu",
    }
    
    # Common optimizer typos
    OPTIMIZER_TYPOS: Dict[str, str] = {
        "adam": "Adam",
        "sgd": "SGD",
        "rmsprop": "RMSprop",
        "rmsProp": "RMSprop",
        "adamw": "AdamW",
        "adamW": "AdamW",
    }
    
    # Common loss function typos
    LOSS_TYPOS: Dict[str, str] = {
        "crossentropy": "categorical_crossentropy",
        "cross_entropy": "categorical_crossentropy",
        "binarycrossentropy": "binary_crossentropy",
        "binary_cross_entropy": "binary_crossentropy",
        "mean_square_error": "mse",
        "mean_squared_error": "mse",
        "mean_absolute_error": "mae",
    }
    
    @staticmethod
    def suggest_parameter_fix(param_name: str, layer_type: str) -> Optional[str]:
        """Suggest correction for misspelled parameter names."""
        if param_name in ErrorSuggestion.PARAMETER_TYPOS:
            correct = ErrorSuggestion.PARAMETER_TYPOS[param_name]
            return f"Did you mean '{correct}' instead of '{param_name}'?"
        
        # Check for common patterns
        if "size" in param_name.lower() and layer_type in ["Conv2D", "MaxPooling2D"]:
            if "kernel" in param_name.lower():
                return "Use 'kernel_size' for convolution layers"
            elif "pool" in param_name.lower():
                return "Use 'pool_size' for pooling layers"
        
        # Check for singular "unit" (but not "units" which is correct)
        if param_name.lower() == "unit" and layer_type in ["Dense", "LSTM", "GRU"]:
            return "Use 'units' (plural) to specify the number of neurons"
        
        return None
    
    @staticmethod
    def suggest_layer_fix(layer_name: str) -> Optional[str]:
        """Suggest correction for misspelled layer names."""
        if layer_name in ErrorSuggestion.LAYER_TYPOS:
            correct = ErrorSuggestion.LAYER_TYPOS[layer_name]
            return f"Did you mean '{correct}' instead of '{layer_name}'?"
        
        # Check for case sensitivity issues
        common_layers = [
            "Dense", "Conv2D", "LSTM", "GRU", "Dropout", "BatchNormalization",
            "MaxPooling2D", "AveragePooling2D", "Flatten", "Embedding",
            "Attention", "MultiHeadAttention", "LayerNormalization"
        ]
        
        for correct_layer in common_layers:
            if layer_name.lower() == correct_layer.lower():
                return f"Layer names are case-sensitive. Use '{correct_layer}' instead of '{layer_name}'"
        
        return None
    
    @staticmethod
    def suggest_activation_fix(activation: str) -> Optional[str]:
        """Suggest correction for misspelled activation functions."""
        if activation in ErrorSuggestion.ACTIVATION_TYPOS:
            correct = ErrorSuggestion.ACTIVATION_TYPOS[activation]
            return f"Did you mean '{correct}' instead of '{activation}'?"
        
        # Common activations should be lowercase
        valid_activations = ["relu", "sigmoid", "tanh", "softmax", "elu", "selu", "leaky_relu", "swish", "gelu"]
        for valid in valid_activations:
            if activation.lower() == valid:
                return f"Activation functions should be lowercase: use '{valid}' instead of '{activation}'"
        
        return None
    
    @staticmethod
    def suggest_optimizer_fix(optimizer: str) -> Optional[str]:
        """Suggest correction for misspelled optimizer names."""
        if optimizer in ErrorSuggestion.OPTIMIZER_TYPOS:
            correct = ErrorSuggestion.OPTIMIZER_TYPOS[optimizer]
            return f"Did you mean '{correct}' instead of '{optimizer}'?"
        
        # Optimizers should be PascalCase
        valid_optimizers = ["Adam", "SGD", "RMSprop", "AdamW", "Adagrad", "Adadelta", "Adamax"]
        for valid in valid_optimizers:
            if optimizer.lower() == valid.lower():
                return f"Optimizer names should use PascalCase: use '{valid}' instead of '{optimizer}'"
        
        return None
    
    @staticmethod
    def suggest_loss_fix(loss: str) -> Optional[str]:
        """Suggest correction for misspelled loss functions."""
        if loss in ErrorSuggestion.LOSS_TYPOS:
            correct = ErrorSuggestion.LOSS_TYPOS[loss]
            return f"Did you mean '{correct}' instead of '{loss}'?"
        
        return None
    
    @staticmethod
    def suggest_shape_fix(input_shape: tuple, layer_type: str) -> Optional[str]:
        """Suggest fixes for common shape-related errors."""
        if not input_shape or len(input_shape) == 0:
            return "Input shape cannot be empty. Specify at least one dimension."
        
        # Check for negative dimensions first (more critical)
        if any(dim is not None and dim < 0 for dim in input_shape):
            return "Shape dimensions cannot be negative. Use None for variable-length dimensions."
        
        # Check for common mistakes
        if layer_type == "Dense" and len(input_shape) > 2:
            return "Dense layers expect 2D input (batch, features). Consider adding Flatten() before Dense layer."
        
        if layer_type == "Conv2D" and len(input_shape) < 3:
            return "Conv2D expects at least 3D input (height, width, channels) or 4D with batch dimension."
        
        if layer_type in ["LSTM", "GRU"] and len(input_shape) < 2:
            return f"{layer_type} expects at least 2D input (sequence_length, features) or 3D with batch dimension."
        
        return None
    
    @staticmethod
    def suggest_parameter_value_fix(param: str, value: Any, layer_type: str) -> Optional[str]:
        """Suggest fixes for invalid parameter values."""
        suggestions = []
        
        # Units/filters must be positive
        if param in ["units", "filters"] and (not isinstance(value, int) or value <= 0):
            suggestions.append(f"{param.capitalize()} must be a positive integer, got {value}")
        
        # Dropout rate validation
        if param == "rate" and layer_type == "Dropout":
            if not isinstance(value, (int, float)) or value < 0 or value >= 1:
                suggestions.append(f"Dropout rate must be between 0 and 1 (exclusive), got {value}")
        
        # Kernel size validation
        if param == "kernel_size":
            if isinstance(value, int) and value <= 0:
                suggestions.append(f"Kernel size must be positive, got {value}")
            elif isinstance(value, (list, tuple)) and any(k <= 0 for k in value):
                suggestions.append(f"All kernel dimensions must be positive, got {value}")
        
        # Pool size validation
        if param == "pool_size":
            if isinstance(value, int) and value <= 0:
                suggestions.append(f"Pool size must be positive, got {value}")
            elif isinstance(value, (list, tuple)) and any(p <= 0 for p in value):
                suggestions.append(f"All pool dimensions must be positive, got {value}")
        
        # Learning rate validation
        if param == "learning_rate":
            if not isinstance(value, (int, float)) or value <= 0:
                suggestions.append(f"Learning rate must be positive, got {value}. Typical range: 1e-5 to 1e-1")
        
        return suggestions[0] if suggestions else None
    
    @staticmethod
    def suggest_dependency_install(dependency: str) -> str:
        """Provide installation instructions for missing dependencies."""
        install_commands = {
            "torch": "pip install torch torchvision",
            "tensorflow": "pip install tensorflow",
            "onnx": "pip install onnx onnxruntime",
            "optuna": "pip install 'neural-dsl[hpo]'",
            "ray": "pip install 'neural-dsl[distributed]'",
            "dash": "pip install 'neural-dsl[dashboard]'",
            "matplotlib": "pip install 'neural-dsl[visualization]'",
            "boto3": "pip install 'neural-dsl[integrations]'",
            "sklearn": "pip install 'neural-dsl[automl]'",
        }
        
        base_msg = f"Missing dependency: {dependency}"
        if dependency in install_commands:
            return f"{base_msg}. Install with: {install_commands[dependency]}"
        
        return f"{base_msg}. Install with: pip install {dependency}"
    
    @staticmethod
    def suggest_backend_fix(backend: str) -> str:
        """Suggest available backends when an invalid one is specified."""
        available = ["tensorflow", "pytorch", "onnx"]
        
        # Check for common typos
        typos = {
            "tf": "tensorflow",
            "torch": "pytorch",
            "pt": "pytorch",
        }
        
        if backend.lower() in typos:
            correct = typos[backend.lower()]
            return f"Use '{correct}' instead of '{backend}'. Available backends: {', '.join(available)}"
        
        return f"Unsupported backend '{backend}'. Available backends: {', '.join(available)}"
    
    @staticmethod
    def format_validation_error(
        error_type: str,
        details: Dict[str, Any],
        suggestion: Optional[str] = None
    ) -> str:
        """Format a validation error with helpful context."""
        lines = [f"Validation Error: {error_type}"]
        
        for key, value in details.items():
            lines.append(f"  {key}: {value}")
        
        if suggestion:
            lines.append(f"\n💡 Suggestion: {suggestion}")
        
        return "\n".join(lines)
    
    @staticmethod
    def suggest_syntax_fix(error_msg: str, line: Optional[int] = None) -> Optional[str]:
        """Suggest fixes for common syntax errors."""
        suggestions = []
        
        if "expected" in error_msg.lower() and ":" in error_msg:
            suggestions.append("Check for missing colons (:) after network/layer definitions")
        
        if "unexpected" in error_msg.lower() and "{" in error_msg:
            suggestions.append("Check that all opening braces { have matching closing braces }")
        
        if "parenthes" in error_msg.lower():
            suggestions.append("Check that all parentheses are properly matched")
        
        if "comma" in error_msg.lower():
            suggestions.append("Check for missing or extra commas between parameters")
        
        if "quote" in error_msg.lower() or "string" in error_msg.lower():
            suggestions.append("Check that all string values are properly quoted")
        
        return suggestions[0] if suggestions else None


class ErrorFormatter:
    """Format errors with rich context and suggestions."""
    
    @staticmethod
    def format_parser_error(
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        code_snippet: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> str:
        """Format a parser error with location and suggestion."""
        parts = ["Parse Error"]
        
        if line is not None:
            if column is not None:
                parts.append(f"at line {line}, column {column}")
            else:
                parts.append(f"at line {line}")
        
        error_msg = f"{parts[0]} {parts[1] if len(parts) > 1 else ''}: {message}"
        
        if code_snippet:
            error_msg += f"\n\n{code_snippet}"
            if column is not None:
                pointer = " " * (column - 1) + "^"
                error_msg += f"\n{pointer}"
        
        if suggestion:
            error_msg += f"\n\n💡 Suggestion: {suggestion}"
        elif not suggestion:
            # Try to auto-generate a suggestion
            auto_suggestion = ErrorSuggestion.suggest_syntax_fix(message, line)
            if auto_suggestion:
                error_msg += f"\n\n💡 Suggestion: {auto_suggestion}"
        
        return error_msg
    
    @staticmethod
    def format_shape_error(
        message: str,
        input_shape: Optional[tuple] = None,
        expected_shape: Optional[tuple] = None,
        layer_type: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> str:
        """Format a shape propagation error."""
        error_msg = f"Shape Error: {message}"
        
        if input_shape:
            error_msg += f"\n  Input shape: {input_shape}"
        if expected_shape:
            error_msg += f"\n  Expected: {expected_shape}"
        if layer_type:
            error_msg += f"\n  Layer type: {layer_type}"
        
        if suggestion:
            error_msg += f"\n\n💡 Suggestion: {suggestion}"
        elif layer_type and input_shape:
            auto_suggestion = ErrorSuggestion.suggest_shape_fix(input_shape, layer_type)
            if auto_suggestion:
                error_msg += f"\n\n💡 Suggestion: {auto_suggestion}"
        
        return error_msg
    
    @staticmethod
    def format_parameter_error(
        parameter: str,
        value: Any,
        layer_type: str,
        reason: Optional[str] = None,
        expected: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> str:
        """Format a parameter validation error."""
        error_msg = f"Invalid Parameter: '{parameter}' in {layer_type} layer"
        error_msg += f"\n  Received: {value}"
        
        if reason:
            error_msg += f"\n  Reason: {reason}"
        if expected:
            error_msg += f"\n  Expected: {expected}"
        
        if suggestion:
            error_msg += f"\n\n💡 Suggestion: {suggestion}"
        else:
            auto_suggestion = ErrorSuggestion.suggest_parameter_value_fix(parameter, value, layer_type)
            if auto_suggestion:
                error_msg += f"\n\n💡 Suggestion: {auto_suggestion}"
        
        return error_msg
    
    @staticmethod
    def format_dependency_error(
        dependency: str,
        feature: Optional[str] = None,
        import_error: Optional[str] = None
    ) -> str:
        """Format a dependency error with installation instructions."""
        error_msg = f"Missing Dependency: {dependency}"
        
        if feature:
            error_msg += f"\n  Required for: {feature}"
        
        if import_error:
            error_msg += f"\n  Error: {import_error}"
        
        install_hint = ErrorSuggestion.suggest_dependency_install(dependency)
        error_msg += f"\n\n💡 {install_hint}"
        
        return error_msg
