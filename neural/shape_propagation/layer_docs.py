"""
Layer documentation for Neural's shape propagation system.

This module provides comprehensive documentation for each layer type supported
by the shape propagator, including parameter descriptions and shape transformation
explanations.
"""

LAYER_DOCUMENTATION = {
    'Conv2D': {
        'description': 'Two-dimensional convolutional layer',
        'parameters': {
            'filters': 'Number of output filters',
            'kernel_size': 'Size of the convolution kernel (int or tuple of 2 ints)',
            'stride': 'Stride of the convolution (int or tuple of 2 ints)',
            'padding': 'Padding mode ("valid", "same", or integer)',
            'data_format': 'Data format ("channels_last" or "channels_first")'
        },
        'shape_transformation': 'For channels_last format with input shape (batch, height, width, channels), ' +
                               'output shape is (batch, new_height, new_width, filters) where new dimensions ' +
                               'depend on padding and stride.'
    },
    'Dense': {
        'description': 'Fully connected layer',
        'parameters': {
            'units': 'Number of output neurons',
            'activation': 'Activation function to use'
        },
        'shape_transformation': 'Transforms input shape (batch, features) to output shape (batch, units).'
    },
    'MaxPooling2D': {
        'description': 'Max pooling operation for 2D spatial data',
        'parameters': {
            'pool_size': 'Size of the pooling window (int or tuple of 2 ints)',
            'stride': 'Stride of the pooling (int or tuple of 2 ints)',
            'padding': 'Padding mode ("valid", "same", or integer)',
            'data_format': 'Data format ("channels_last" or "channels_first")'
        },
        'shape_transformation': 'Reduces spatial dimensions according to pool_size and stride.'
    },
    'Flatten': {
        'description': 'Flattens the input',
        'parameters': {},
        'shape_transformation': 'Transforms input shape (batch, dim1, dim2, ...) to (batch, dim1*dim2*...).'
    },
    'Output': {
        'description': 'Output layer (similar to Dense)',
        'parameters': {
            'units': 'Number of output neurons',
            'activation': 'Activation function to use'
        },
        'shape_transformation': 'Transforms input shape (batch, features) to output shape (batch, units).'
    },
    'GlobalAveragePooling2D': {
        'description': 'Global average pooling operation for 2D spatial data',
        'parameters': {
            'data_format': 'Data format ("channels_last" or "channels_first")'
        },
        'shape_transformation': 'Reduces spatial dimensions to 1x1, resulting in shape (batch, channels).'
    },
    'UpSampling2D': {
        'description': 'Upsampling operation for 2D spatial data',
        'parameters': {
            'size': 'Upsampling factors (int or tuple of 2 ints)',
            'data_format': 'Data format ("channels_last" or "channels_first")'
        },
        'shape_transformation': 'Increases spatial dimensions according to size factors.'
    },
    'TransformerEncoder': {
        'description': 'Transformer encoder layer',
        'parameters': {
            'num_heads': 'Number of attention heads',
            'd_model': 'Dimension of the model',
            'dff': 'Dimension of the feed-forward network'
        },
        'shape_transformation': 'Preserves input shape in TensorFlow, returns (batch, seq_len) in PyTorch.'
    },
    'TransformerDecoder': {
        'description': 'Transformer decoder layer with cross-attention for encoder-decoder architectures',
        'parameters': {
            'num_heads': 'Number of attention heads',
            'd_model': 'Dimension of the model (default: same as ff_dim)',
            'ff_dim': 'Dimension of the feed-forward network',
            'dropout': 'Dropout rate for regularization (default: 0.1)',
            'use_causal_mask': 'Whether to apply causal masking for autoregressive decoding (default: True)'
        },
        'shape_transformation': 'Preserves input shape through self-attention and cross-attention. ' +
                               'Input shape (batch, seq_len, d_model) remains unchanged. ' +
                               'Requires encoder_output for cross-attention in encoder-decoder architectures.'
    },
    'Conv1D': {
        'description': 'One-dimensional convolutional layer',
        'parameters': {
            'filters': 'Number of output filters',
            'kernel_size': 'Size of the convolution kernel (int)',
            'stride': 'Stride of the convolution (int)',
            'padding': 'Padding mode ("valid", "same", or integer)',
            'data_format': 'Data format ("channels_last" or "channels_first")'
        },
        'shape_transformation': 'For channels_last format with input shape (batch, steps, channels), ' +
                               'output shape is (batch, new_steps, filters).'
    },
    'Conv3D': {
        'description': 'Three-dimensional convolutional layer',
        'parameters': {
            'filters': 'Number of output filters',
            'kernel_size': 'Size of the convolution kernel (int or tuple of 3 ints)',
            'stride': 'Stride of the convolution (int or tuple of 3 ints)',
            'padding': 'Padding mode ("valid", "same", or integer)',
            'data_format': 'Data format ("channels_last" or "channels_first")'
        },
        'shape_transformation': 'For channels_last format with input shape (batch, depth, height, width, channels), ' +
                               'output shape is (batch, new_depth, new_height, new_width, filters).'
    },
    'LSTM': {
        'description': 'Long Short-Term Memory layer',
        'parameters': {
            'units': 'Number of output units',
            'return_sequences': 'Whether to return the full sequence or just the last output'
        },
        'shape_transformation': 'With input shape (batch, timesteps, features), returns (batch, units) if ' +
                               'return_sequences=False, otherwise (batch, timesteps, units).'
    },
    'Dropout': {
        'description': 'Dropout layer for regularization',
        'parameters': {
            'rate': 'Fraction of input units to drop (between 0 and 1)'
        },
        'shape_transformation': 'Preserves input shape.'
    },
    'Embedding': {
        'description': 'Embedding layer for converting integer indices to dense vectors',
        'parameters': {
            'input_dim': 'Size of the vocabulary (number of unique tokens)',
            'output_dim': 'Dimension of the dense embedding vectors',
            'input_length': 'Optional. Length of input sequences (for fixed-length sequences)'
        },
        'shape_transformation': 'Transforms input shape (batch, sequence_length) to output shape ' +
                               '(batch, sequence_length, output_dim). Commonly used for text and sequence data.'
    },
    'BatchNormalization': {
        'description': 'Batch normalization layer',
        'parameters': {
            'axis': 'The axis to normalize (typically the features axis)',
            'momentum': 'Momentum for the moving average'
        },
        'shape_transformation': 'Preserves input shape.'
    },
    'Concatenate': {
        'description': 'Layer that concatenates a list of inputs',
        'parameters': {
            'axis': 'The axis along which to concatenate'
        },
        'shape_transformation': 'Concatenates input shapes along the specified axis.'
    },
    'Add': {
        'description': 'Layer that adds a list of inputs',
        'parameters': {},
        'shape_transformation': 'All inputs must have the same shape, which is preserved in the output.'
    }
}

def get_layer_documentation(layer_type):
    """Return documentation for a specific layer type.

    Args:
        layer_type (str): The type of layer to get documentation for.

    Returns:
        dict: Documentation for the specified layer type, or a default message if not found.
    """
    return LAYER_DOCUMENTATION.get(layer_type, {
        'description': 'No documentation available for this layer type',
        'parameters': {},
        'shape_transformation': 'Shape transformation behavior is not documented.'
    })

def format_layer_documentation(layer_type):
    """Format layer documentation as a readable string.

    Args:
        layer_type (str): The type of layer to format documentation for.

    Returns:
        str: Formatted documentation string.
    """
    doc = get_layer_documentation(layer_type)

    result = f"# {layer_type}\n\n"
    result += f"{doc['description']}\n\n"

    if doc['parameters']:
        result += "## Parameters\n\n"
        for param, desc in doc['parameters'].items():
            result += f"- **{param}**: {desc}\n"
        result += "\n"

    result += "## Shape Transformation\n\n"
    result += f"{doc['shape_transformation']}\n"

    return result
