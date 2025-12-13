"""
Layer handlers for Neural's shape propagation system.

This module provides handler functions for different layer types to calculate
output shapes based on input shapes and layer parameters.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .utils import calculate_output_dims, extract_param

def handle_conv1d(input_shape: Tuple[int, ...],
                 params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Conv1D layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    data_format = params.get('data_format', 'channels_last')
    filters = extract_param(params, 'filters', 32)
    kernel_size = extract_param(params, 'kernel_size', 3,
                              transform=lambda x: (x,) if isinstance(x, int) else x)
    stride = extract_param(params, 'stride', 1,
                         transform=lambda x: (x,) if isinstance(x, int) else x)
    padding = extract_param(params, 'padding', 'same')

    if data_format == 'channels_last':  # (batch, steps, channels)
        if len(input_shape) < 3:
            return input_shape  # Invalid input shape, return unchanged

        steps = input_shape[1]
        new_steps = calculate_output_dims((steps,), kernel_size, stride, padding)[0]
        return (input_shape[0], new_steps, filters)
    else:  # channels_first: (batch, channels, steps)
        if len(input_shape) < 3:
            return input_shape  # Invalid input shape, return unchanged

        steps = input_shape[2]
        new_steps = calculate_output_dims((steps,), kernel_size, stride, padding)[0]
        return (input_shape[0], filters, new_steps)

def handle_conv3d(input_shape: Tuple[int, ...],
                 params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Conv3D layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    data_format = params.get('data_format', 'channels_last')
    filters = extract_param(params, 'filters', 32)
    kernel_size = extract_param(params, 'kernel_size', 3,
                              transform=lambda x: (x, x, x) if isinstance(x, int) else x)
    stride = extract_param(params, 'stride', 1,
                         transform=lambda x: (x, x, x) if isinstance(x, int) else x)
    padding = extract_param(params, 'padding', 'same')

    if data_format == 'channels_last':  # (batch, depth, height, width, channels)
        if len(input_shape) < 5:
            return input_shape  # Invalid input shape, return unchanged

        spatial_dims = input_shape[1:4]
        new_spatial_dims = calculate_output_dims(spatial_dims, kernel_size, stride, padding)
        return (input_shape[0], *new_spatial_dims, filters)
    else:  # channels_first: (batch, channels, depth, height, width)
        if len(input_shape) < 5:
            return input_shape  # Invalid input shape, return unchanged

        spatial_dims = input_shape[2:5]
        new_spatial_dims = calculate_output_dims(spatial_dims, kernel_size, stride, padding)
        return (input_shape[0], filters, *new_spatial_dims)

def handle_lstm(input_shape: Tuple[int, ...],
               params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle LSTM layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    units = extract_param(params, 'units', 128)
    return_sequences = extract_param(params, 'return_sequences', False)

    if len(input_shape) < 3:
        return input_shape  # Invalid input shape, return unchanged

    batch_size = input_shape[0]
    time_steps = input_shape[1]

    if return_sequences:
        return (batch_size, time_steps, units)
    else:
        return (batch_size, units)

def handle_dropout(input_shape: Tuple[int, ...],
                  params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Dropout layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape (same as input)
    """
    # Dropout doesn't change the shape
    return input_shape

def handle_batch_normalization(input_shape: Tuple[int, ...],
                              params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle BatchNormalization layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape (same as input)
    """
    # BatchNormalization doesn't change the shape
    return input_shape

def handle_concatenate(input_shapes: List[Tuple[int, ...]],
                      params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Concatenate layer shape propagation.

    Args:
        input_shapes: List of input tensor shapes
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    if not input_shapes:
        return tuple()  # No inputs

    axis = extract_param(params, 'axis', -1)

    # Convert negative axis to positive
    if axis < 0:
        axis = len(input_shapes[0]) + axis

    # Check if all shapes are compatible for concatenation
    for shape in input_shapes[1:]:
        if len(shape) != len(input_shapes[0]):
            return input_shapes[0]  # Incompatible shapes, return first input shape

        for i in range(len(shape)):
            if i != axis and shape[i] != input_shapes[0][i]:
                return input_shapes[0]  # Incompatible shapes, return first input shape

    # Calculate concatenated dimension
    concat_dim = sum(shape[axis] for shape in input_shapes)

    # Create output shape
    output_shape = list(input_shapes[0])
    output_shape[axis] = concat_dim

    return tuple(output_shape)

def handle_add(input_shapes: List[Tuple[int, ...]],
              params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Add layer shape propagation.

    Args:
        input_shapes: List of input tensor shapes
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    if not input_shapes:
        return tuple()  # No inputs

    # For addition, all shapes must be identical
    # Return the first shape (they should all be the same)
    return input_shapes[0]

def handle_positional_encoding(input_shape: Tuple[int, ...],
                               params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle PositionalEncoding layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape (same as input)
    """
    # PositionalEncoding adds positional information but doesn't change shape
    # Expected input: (batch, seq_len, d_model)
    # Output: (batch, seq_len, d_model)
    return input_shape

def handle_global_average_pooling1d(input_shape: Tuple[int, ...],
                                   params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle GlobalAveragePooling1D layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    data_format = params.get('data_format', 'channels_last')

    if len(input_shape) < 3:
        return input_shape  # Invalid input shape, return unchanged

    if data_format == 'channels_last':  # (batch, steps, channels)
        return (input_shape[0], input_shape[2])
    else:  # channels_first: (batch, channels, steps)
        return (input_shape[0], input_shape[1])

def handle_reshape(input_shape: Tuple[int, ...],
                  params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Reshape layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    target_shape = extract_param(params, 'target_shape', None)

    if target_shape is None:
        return input_shape  # No target shape specified, return unchanged

    # Calculate total elements in input
    input_elements = np.prod([dim for dim in input_shape if dim is not None])

    # Handle -1 in target shape
    if -1 in target_shape:
        # Calculate the size of the -1 dimension
        neg_one_index = target_shape.index(-1)
        other_elements = np.prod([dim for i, dim in enumerate(target_shape) if i != neg_one_index and dim is not None])
        target_shape_list = list(target_shape)
        target_shape_list[neg_one_index] = input_elements // other_elements
        target_shape = tuple(target_shape_list)

    # Return shape with batch dimension preserved
    return (input_shape[0], *target_shape)

def handle_permute(input_shape: Tuple[int, ...],
                  params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Permute layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    pattern = extract_param(params, 'pattern', None)

    if pattern is None or len(pattern) != len(input_shape) - 1:
        return input_shape  # Invalid pattern, return unchanged

    # Create output shape by permuting dimensions according to pattern
    # Keep batch dimension (0) fixed
    output_shape = [input_shape[0]]
    for i in pattern:
        output_shape.append(input_shape[i + 1])  # +1 because pattern is 0-indexed but we skip batch dim

    return tuple(output_shape)

def handle_zero_padding2d(input_shape: Tuple[int, ...],
                         params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle ZeroPadding2D layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    padding = extract_param(params, 'padding', ((1, 1), (1, 1)))
    data_format = params.get('data_format', 'channels_last')

    if len(input_shape) < 4:
        return input_shape  # Invalid input shape, return unchanged

    if data_format == 'channels_last':  # (batch, height, width, channels)
        height = input_shape[1] + padding[0][0] + padding[0][1]
        width = input_shape[2] + padding[1][0] + padding[1][1]
        return (input_shape[0], height, width, input_shape[3])
    else:  # channels_first: (batch, channels, height, width)
        height = input_shape[2] + padding[0][0] + padding[0][1]
        width = input_shape[3] + padding[1][0] + padding[1][1]
        return (input_shape[0], input_shape[1], height, width)

def handle_cropping2d(input_shape: Tuple[int, ...],
                     params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Cropping2D layer shape propagation.

    Args:
        input_shape: Input tensor shape
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    cropping = extract_param(params, 'cropping', ((1, 1), (1, 1)))
    data_format = params.get('data_format', 'channels_last')

    if len(input_shape) < 4:
        return input_shape  # Invalid input shape, return unchanged

    if data_format == 'channels_last':  # (batch, height, width, channels)
        height = input_shape[1] - cropping[0][0] - cropping[0][1]
        width = input_shape[2] - cropping[1][0] - cropping[1][1]
        return (input_shape[0], height, width, input_shape[3])
    else:  # channels_first: (batch, channels, height, width)
        height = input_shape[2] - cropping[0][0] - cropping[0][1]
        width = input_shape[3] - cropping[1][0] - cropping[1][1]
        return (input_shape[0], input_shape[1], height, width)
