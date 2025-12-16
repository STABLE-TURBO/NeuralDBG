"""
Utility functions for Neural's shape propagation system.

This module provides helper functions for parameter extraction, shape validation,
error detection, and actionable error messages used by the shape propagator.

Enhanced Features:
1. format_error_message: Creates detailed error messages with fix suggestions
2. suggest_layer_fix: Provides layer-specific fix recommendations
3. diagnose_shape_flow: Analyzes entire network shape flow for issues
4. detect_shape_issues: Identifies potential problems (bottlenecks, large tensors)
5. suggest_optimizations: Recommends architectural improvements

The error messages focus on:
- Actionable suggestions with specific parameter values
- Common use cases and examples
- Step-by-step fix instructions
- Context about why the error occurred
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


def extract_param(params: Dict[str, Any],
                 key: str,
                 default: Any = None,
                 transform: Optional[Callable] = None) -> Any:
    """Extract parameter with HPO handling and optional transformation.

    Args:
        params: Dictionary of parameters
        key: Parameter key to extract
        default: Default value if parameter is not found
        transform: Optional function to transform the parameter value

    Returns:
        The extracted parameter value
    """
    value = params.get(key, default)

    # Handle HPO dictionary format
    if isinstance(value, dict):
        if 'value' in value:
            value = value['value']
        else:
            value = default

    # Apply transformation if provided
    if transform and value is not None:
        value = transform(value)

    return value

def calculate_output_dims(input_dims: Tuple[int, ...],
                         kernel_size: Tuple[int, ...],
                         stride: Tuple[int, ...],
                         padding: Union[str, int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Calculate output dimensions for convolution or pooling operations.

    Args:
        input_dims: Input spatial dimensions (can contain None for dynamic dims)
        kernel_size: Kernel size for each dimension
        stride: Stride for each dimension
        padding: Padding mode ('valid', 'same') or explicit padding

    Returns:
        Output spatial dimensions (with None preserved for dynamic dims)
    """
    output_dims = []

    for i, dim in enumerate(input_dims):
        # Handle None dimensions (dynamic/unknown sizes)
        if dim is None:
            output_dims.append(None)
            continue

        k = kernel_size[i] if i < len(kernel_size) else kernel_size[0]
        s = stride[i] if i < len(stride) else stride[0]

        if padding == 'valid':
            p = 0
        elif padding == 'same':
            # For 'same' padding with stride > 1, calculate proper padding
            # Output size should be ceil(input_size / stride)
            output_size = (dim + s - 1) // s
            total_padding = max(0, (output_size - 1) * s + k - dim)
            p = total_padding // 2
        elif isinstance(padding, (int, float)):
            p = int(padding)
        elif isinstance(padding, (tuple, list)):
            p = padding[i] if i < len(padding) else padding[0]
        else:
            p = 0

        output_dim = (dim + 2 * p - k) // s + 1
        output_dim = max(1, output_dim)  # Ensure at least 1
        output_dims.append(output_dim)

    return tuple(output_dims)

def detect_shape_issues(shape_history: List[Tuple[str, Tuple[int, ...]]]) -> List[Dict[str, Any]]:
    """Detect potential issues in shape propagation.

    Args:
        shape_history: List of (layer_name, output_shape) tuples

    Returns:
        List of detected issues with type, message, and layer index
    """
    issues = []

    # Check for extreme tensor size changes
    for i in range(1, len(shape_history)):
        prev_layer, prev_shape = shape_history[i-1]
        curr_layer, curr_shape = shape_history[i]

        prev_size = np.prod([dim for dim in prev_shape if dim is not None])
        curr_size = np.prod([dim for dim in curr_shape if dim is not None])

        if curr_size > prev_size * 10:
            issues.append({
                'type': 'warning',
                'message': f'Large tensor size increase at {curr_layer}: {prev_size} → {curr_size}',
                'layer_index': i
            })
        elif curr_size * 100 < prev_size and prev_size > 1000:
            issues.append({
                'type': 'info',
                'message': f'Significant tensor size reduction at {curr_layer}: {prev_size} → {curr_size}',
                'layer_index': i
            })

    # Check for very large tensors
    for i, (layer_name, shape) in enumerate(shape_history):
        size = np.prod([dim for dim in shape if dim is not None])
        memory_mb = size * 4 / (1024 * 1024)  # Assuming float32

        if memory_mb > 1000:
            issues.append({
                'type': 'warning',
                'message': f'Very large tensor at {layer_name}: {shape} ({memory_mb:.2f} MB)',
                'layer_index': i
            })

    # Check for potential bottlenecks
    for i in range(1, len(shape_history) - 1):
        prev_size = np.prod([dim for dim in shape_history[i-1][1] if dim is not None])
        curr_size = np.prod([dim for dim in shape_history[i][1] if dim is not None])
        next_size = np.prod([dim for dim in shape_history[i+1][1] if dim is not None])

        if curr_size < prev_size * 0.1 and curr_size < next_size * 0.1:
            issues.append({
                'type': 'bottleneck',
                'message': f'Potential information bottleneck at {shape_history[i][0]}: {curr_size} elements',
                'layer_index': i
            })

    return issues

def suggest_optimizations(shape_history: List[Tuple[str, Tuple[int, ...]]]) -> List[Dict[str, Any]]:
    """Suggest optimizations based on shape analysis.

    Args:
        shape_history: List of (layer_name, output_shape) tuples

    Returns:
        List of optimization suggestions with type, message, and layer index
    """
    suggestions = []

    # Look for opportunities to reduce dimensions
    for i, (layer_name, shape) in enumerate(shape_history):
        if len(shape) == 4:  # Conv layers
            if shape[1] > 100 and shape[2] > 100 and shape[3] > 64:
                suggestions.append({
                    'type': 'optimization',
                    'message': f'Consider adding pooling after {layer_name} to reduce spatial dimensions {shape[1]}x{shape[2]}',
                    'layer_index': i
                })

    # Check for potential over-parameterization
    for i, (layer_name, shape) in enumerate(shape_history):
        if 'Dense' in layer_name and len(shape) == 2:
            if shape[1] > 4096:
                suggestions.append({
                    'type': 'optimization',
                    'message': f'Consider reducing units in {layer_name} ({shape[1]} units may be excessive)',
                    'layer_index': i
                })

    return suggestions

def format_error_message(error_type: str, details: Dict[str, Any]) -> str:
    """Format user-friendly error messages with actionable suggestions.

    Args:
        error_type: Type of error
        details: Dictionary with error details

    Returns:
        Formatted error message with fix suggestions
    """
    error_templates = {
        'invalid_input_shape': {
            'message': f"Invalid input shape: {details.get('shape')}. Expected a tuple with positive dimensions.",
            'suggestions': [
                "Use format: (batch_size, height, width, channels) for TensorFlow",
                "Use format: (batch_size, channels, height, width) for PyTorch",
                "Example: input: (None, 28, 28, 1) for MNIST grayscale images"
            ]
        },
        'kernel_too_large': {
            'message': f"Kernel size {details.get('kernel_size')} is too large for input dimensions {details.get('input_dims')}.",
            'suggestions': [
                f"Reduce kernel_size to fit within {details.get('input_dims')}",
                "Try kernel_size=(3, 3) or smaller",
                "Or increase input dimensions before this layer",
                "Add padding to accommodate larger kernels"
            ]
        },
        'missing_parameter': {
            'message': f"Missing required parameter '{details.get('param')}' for {details.get('layer_type')} layer.",
            'suggestions': [
                f"Add {details.get('param')} parameter to {details.get('layer_type')} layer",
                f"Example: {details.get('layer_type')}({details.get('param')}=...)",
                f"Check {details.get('layer_type')} layer documentation for required parameters"
            ]
        },
        'incompatible_shapes': {
            'message': f"Incompatible shapes: cannot connect {details.get('from_shape')} to {details.get('to_shape')}.",
            'suggestions': [
                "Add a Flatten() layer to convert multi-dimensional to 1D",
                "Use Reshape() to change tensor dimensions",
                "Check that output of previous layer matches expected input",
                "Use 'neural visualize' to see shape flow through network"
            ]
        },
        'negative_stride': {
            'message': f"Stride must be positive, got {details.get('stride')} for {details.get('layer_type')} layer.",
            'suggestions': [
                "Change stride to a positive integer (typically 1 or 2)",
                "stride=1 means no downsampling",
                "stride=2 means 2x downsampling"
            ]
        },
        'negative_filters': {
            'message': f"Filters must be positive, got {details.get('filters')} for Conv2D layer.",
            'suggestions': [
                "Set filters to a positive integer (e.g., 32, 64, 128)",
                "More filters = more feature detection but higher computation",
                "Common progression: 32 -> 64 -> 128 -> 256"
            ]
        },
        'output_dimension_too_small': {
            'message': f"Output dimension {details.get('output_dim')} is too small after layer {details.get('layer_name')}.",
            'suggestions': [
                "Reduce stride or pool_size to preserve spatial dimensions",
                "Add padding to maintain size",
                "Check if you have too many pooling layers",
                f"Input was {details.get('input_shape')}, became {details.get('output_shape')}"
            ]
        },
        'zero_output_dimension': {
            'message': f"Layer {details.get('layer_type')} produced zero output dimension.",
            'suggestions': [
                "Kernel size or pool_size is too large for input",
                "Reduce kernel_size or pool_size",
                "Increase input dimensions",
                "Add padding to prevent dimension collapse"
            ]
        }
    }

    error_info = error_templates.get(error_type, {
        'message': f"Error: {details}",
        'suggestions': ["Check layer parameters and input shapes"]
    })

    message_parts = [
        "\n" + "="*70,
        "SHAPE PROPAGATION ERROR",
        "="*70,
        f"\n❌ {error_info['message']}",
        "\n🔧 Fix Suggestions:"
    ]

    for i, suggestion in enumerate(error_info['suggestions'], 1):
        message_parts.append(f"   {i}. {suggestion}")

    message_parts.append("\n" + "="*70)

    return "\n".join(message_parts)

def calculate_memory_usage(shape: Tuple[int, ...], dtype: str = 'float32') -> int:
    """Calculate memory usage for a tensor with given shape and dtype.

    Args:
        shape: Tensor shape
        dtype: Data type of tensor elements

    Returns:
        Memory usage in bytes
    """
    bytes_per_element = {
        'float32': 4,
        'float16': 2,
        'int32': 4,
        'int64': 8,
        'uint8': 1,
        'bool': 1
    }.get(dtype, 4)

    num_elements = np.prod([dim for dim in shape if dim is not None])
    return int(num_elements * bytes_per_element)

def format_memory_size(bytes: int) -> str:
    """Format memory size in human-readable form.

    Args:
        bytes: Memory size in bytes

    Returns:
        Formatted memory size string
    """
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.2f} GB"


def suggest_layer_fix(layer_type: str, error_context: Dict[str, Any]) -> List[str]:
    """Suggest specific fixes based on layer type and error context.
    
    Args:
        layer_type: Type of layer that encountered an error
        error_context: Dictionary with error context (input_shape, params, etc.)
        
    Returns:
        List of actionable fix suggestions
    """
    suggestions = []
    input_shape = error_context.get('input_shape')
    params = error_context.get('params', {})

    if layer_type == 'Conv2D':
        if input_shape and len(input_shape) != 4:
            suggestions.append(f"Conv2D expects 4D input, got {len(input_shape)}D: {input_shape}")
            suggestions.append("Ensure input format: (batch, height, width, channels) or (batch, channels, height, width)")

        kernel_size = params.get('kernel_size')
        if kernel_size and input_shape:
            suggestions.append(f"Current kernel_size: {kernel_size}, input spatial dims: {input_shape[1:3]}")
            suggestions.append("Try reducing kernel_size if it exceeds input dimensions")

    elif layer_type == 'Dense':
        if input_shape and len(input_shape) > 2:
            suggestions.append(f"Dense expects 2D input (batch, features), got {len(input_shape)}D")
            suggestions.append("Add Flatten() or GlobalAveragePooling2D before Dense layer")
            suggestions.append("Example: ...Conv2D(...) -> Flatten() -> Dense(...)")

    elif layer_type == 'MaxPooling2D':
        pool_size = params.get('pool_size')
        if pool_size and input_shape:
            suggestions.append(f"pool_size: {pool_size}, input: {input_shape}")
            suggestions.append("Reduce pool_size if it exceeds spatial dimensions")
            suggestions.append("Common pool_size values: (2,2), (3,3)")

    elif layer_type == 'Flatten':
        if input_shape:
            flattened_size = np.prod([d for d in input_shape[1:] if d is not None])
            suggestions.append(f"Flatten will convert {input_shape} to (batch, {flattened_size})")

    # Generic suggestions if none specific found
    if not suggestions:
        suggestions.append(f"Check {layer_type} layer parameters and input shape compatibility")
        suggestions.append("Use 'neural visualize <your_file>.neural' to see shape flow")

    return suggestions


def diagnose_shape_flow(shape_history: List[Tuple[str, Tuple[int, ...]]]) -> Dict[str, Any]:
    """Diagnose potential issues in shape flow through network.
    
    Args:
        shape_history: List of (layer_name, output_shape) tuples
        
    Returns:
        Dictionary with diagnostic information and suggestions
    """
    diagnostics: Dict[str, Any] = {
        'warnings': [],
        'errors': [],
        'suggestions': [],
        'shape_flow': []
    }

    for i, (layer_name, shape) in enumerate(shape_history):
        # Check for dimension collapse
        if any(dim is not None and dim <= 0 for dim in shape):
            diagnostics['errors'].append({
                'layer': layer_name,
                'issue': 'zero_or_negative_dimension',
                'message': f"Layer {layer_name} produced invalid dimensions: {shape}",
                'suggestions': [
                    "Check previous layer parameters (stride, kernel_size, pool_size)",
                    "Dimensions may have collapsed due to aggressive downsampling",
                    "Consider reducing stride or adding padding"
                ]
            })

        # Check for extreme size reductions
        if i > 0:
            prev_shape = shape_history[i-1][1]
            prev_size = np.prod([d for d in prev_shape if d is not None and d > 0])
            curr_size = np.prod([d for d in shape if d is not None and d > 0])

            if curr_size < prev_size * 0.01:  # >99% reduction
                diagnostics['warnings'].append({
                    'layer': layer_name,
                    'issue': 'extreme_dimension_reduction',
                    'message': f"Layer {layer_name} reduced tensor size by >99%: {prev_size} -> {curr_size}",
                    'suggestions': [
                        "This may cause information loss",
                        "Consider more gradual downsampling",
                        "Review pooling and stride parameters"
                    ]
                })

        diagnostics['shape_flow'].append({
            'layer': layer_name,
            'shape': shape,
            'size': np.prod([d for d in shape if d is not None and d > 0]),
            'memory_mb': calculate_memory_usage(shape) / (1024 * 1024)
        })

    # Overall suggestions
    if diagnostics['errors']:
        diagnostics['suggestions'].append("Fix dimension errors before proceeding")
    if diagnostics['warnings']:
        diagnostics['suggestions'].append("Review warnings for potential architecture improvements")

    return diagnostics
