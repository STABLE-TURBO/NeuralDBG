"""
Layer handlers for Neural's shape propagation system.

This module provides handler functions for different layer types to calculate
output shapes based on input shapes and layer parameters.
"""

from typing import Any, Dict, List, Tuple


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
        input_shape: Input tensor shape (batch, seq_len, input_size)
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
        # Return full sequence: (batch, seq_len, units)
        return (batch_size, time_steps, units)
    else:
        # Return only last output: (batch, units)
        return (batch_size, units)

def handle_gru(input_shape: Tuple[int, ...],
              params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle GRU layer shape propagation.

    Args:
        input_shape: Input tensor shape (batch, seq_len, input_size)
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
        # Return full sequence: (batch, seq_len, units)
        return (batch_size, time_steps, units)
    else:
        # Return only last output: (batch, units)
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
    """Handle Concatenate layer shape propagation with improved None handling.

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
    # All dimensions except concat axis must match (or be None)
    for shape in input_shapes[1:]:
        if len(shape) != len(input_shapes[0]):
            return input_shapes[0]  # Incompatible shapes, return first input shape

        for i in range(len(shape)):
            if i != axis:
                # Check dimension compatibility, handling None values
                dim1, dim2 = input_shapes[0][i], shape[i]
                if dim1 is not None and dim2 is not None and dim1 != dim2:
                    return input_shapes[0]  # Incompatible shapes, return first input shape

    # Calculate concatenated dimension, handling None values
    concat_dim = 0
    has_none = False
    for shape in input_shapes:
        if shape[axis] is None:
            has_none = True
        else:
            concat_dim += shape[axis]

    # If any input has None at concat axis, output is None
    if has_none:
        concat_dim = None

    # Create output shape, preserving None in non-concat dimensions
    output_shape = list(input_shapes[0])
    output_shape[axis] = concat_dim

    return tuple(output_shape)

def handle_add(input_shapes: List[Tuple[int, ...]],
              params: Dict[str, Any]) -> Tuple[int, ...]:
    """Handle Add layer shape propagation with improved broadcasting support.

    Args:
        input_shapes: List of input tensor shapes
        params: Layer parameters

    Returns:
        Output tensor shape
    """
    if not input_shapes:
        return tuple()  # No inputs

    # For addition with broadcasting, we need to find the output shape
    # Start with first shape
    output_shape = list(input_shapes[0])

    # Check all other shapes for compatibility
    for shape in input_shapes[1:]:
        # Shapes must be broadcastable
        if len(shape) != len(output_shape):
            # Different ranks - would need complex broadcasting logic
            # For now, require same rank
            return input_shapes[0]

        # Check each dimension
        for i in range(len(shape)):
            dim1, dim2 = output_shape[i], shape[i]

            # Handle None (batch) dimensions
            if dim1 is None or dim2 is None:
                output_shape[i] = None
            # Check broadcasting compatibility
            elif dim1 == 1:
                output_shape[i] = dim2
            elif dim2 == 1:
                output_shape[i] = dim1
            elif dim1 != dim2:
                # Incompatible dimensions
                return input_shapes[0]

    return tuple(output_shape)

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

    # Calculate total elements in input (excluding batch dimension)
    input_elements = 1
    for i, dim in enumerate(input_shape):
        if i == 0:
            continue  # Skip batch dimension
        if dim is not None:
            input_elements *= dim

    # Handle -1 in target shape (infer dimension)
    if -1 in target_shape:
        # Calculate the size of the -1 dimension
        neg_one_index = target_shape.index(-1)
        other_elements = 1
        for i, dim in enumerate(target_shape):
            if i != neg_one_index and dim is not None and dim != -1:
                other_elements *= dim

        target_shape_list = list(target_shape)
        if other_elements > 0:
            target_shape_list[neg_one_index] = input_elements // other_elements
        else:
            target_shape_list[neg_one_index] = 1
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
        h, w = input_shape[1], input_shape[2]
        new_height = h + padding[0][0] + padding[0][1] if h is not None else None
        new_width = w + padding[1][0] + padding[1][1] if w is not None else None
        return (input_shape[0], new_height, new_width, input_shape[3])
    else:  # channels_first: (batch, channels, height, width)
        h, w = input_shape[2], input_shape[3]
        new_height = h + padding[0][0] + padding[0][1] if h is not None else None
        new_width = w + padding[1][0] + padding[1][1] if w is not None else None
        return (input_shape[0], input_shape[1], new_height, new_width)

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
        h, w = input_shape[1], input_shape[2]
        new_height = h - cropping[0][0] - cropping[0][1] if h is not None else None
        new_width = w - cropping[1][0] - cropping[1][1] if w is not None else None
        return (input_shape[0], new_height, new_width, input_shape[3])
    else:  # channels_first: (batch, channels, height, width)
        h, w = input_shape[2], input_shape[3]
        new_height = h - cropping[0][0] - cropping[0][1] if h is not None else None
        new_width = w - cropping[1][0] - cropping[1][1] if w is not None else None
        return (input_shape[0], input_shape[1], new_height, new_width)
