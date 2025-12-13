# MultiHeadAttention Implementation Summary

## Overview
Implemented MultiHeadAttention as a standalone layer in the Neural DSL, supporting both self-attention and cross-attention modes with configurable key/query/value projections.

## Files Modified

### 1. Grammar Definition (`neural/parser/grammar.py`)
- Added `MULTIHEADATTENTION` token to the grammar
- Added `"multiheadattention"i` to the `LAYER_TYPE` token list
- This allows the parser to recognize `MultiHeadAttention(...)` in DSL files

### 2. Parser (`neural/parser/parser.py`)
- Added `'MULTIHEADATTENTION': 'multiheadattention'` to `layer_type_map`
- Added `multiheadattention()` method to process MultiHeadAttention layers
- Parameter validation for `num_heads` and `key_dim` (must be positive integers)
- Supports all attention parameters: num_heads, key_dim, value_dim, dropout, use_bias, mode
- Updated `MACRO_NAME` pattern to exclude MultiHeadAttention from being treated as a macro

### 3. TensorFlow Generator (`neural/code_generation/tensorflow_generator.py`)
- Added MultiHeadAttention generation in `generate_layer()` method
- Parameters mapped to `tf.keras.layers.MultiHeadAttention`:
  - `num_heads`: Number of attention heads
  - `key_dim`: Dimension of key vectors
  - `value_dim`: Optional dimension for value vectors
  - `dropout`: Dropout rate for attention weights
  - `use_bias`: Whether to use bias in projections
  - `mode`: "self" (default) or "cross" for cross-attention
- Self-attention: `layers.MultiHeadAttention(...)(x, x)`
- Cross-attention: `layers.MultiHeadAttention(...)(x, context)`

### 4. PyTorch Generator (`neural/code_generation/pytorch_generator.py`)
- Added MultiHeadAttention in `generate_pytorch_layer()` function
- Parameters mapped to `torch.nn.MultiheadAttention`:
  - `embed_dim`: Inferred from input shape if not provided
  - `num_heads`: Number of attention heads
  - `dropout`: Dropout rate
  - `batch_first`: Always set to True for consistency
- Added special handling in main generation loop:
  - Creates layer instance in `__init__`
  - Forward pass unpacks tuple: `x, _ = self.layer(x, x, x)` for self-attention
  - Forward pass for cross-attention: `x, _ = self.layer(x, context, context)`

### 5. ONNX Generator (`neural/code_generation/onnx_generator.py`)
- Added MultiHeadAttention export using ONNX `Attention` operator
- Maps `num_heads` parameter to ONNX attribute

### 6. Shape Propagator (`neural/shape_propagation/shape_propagator.py`)
- Added `_handle_multiheadattention()` method
- MultiHeadAttention preserves input shape (sequence-to-sequence transformation)
- Returns input shape unchanged

### 7. Layer Processors (`neural/parser/layer_processors.py`)
- Added `map_positional_to_multiheadattention_params()` function
- Maps positional parameters: num_heads, key_dim, value_dim

### 8. Example Files
- **`examples/multihead_attention.neural`**: Comprehensive examples demonstrating:
  - Self-attention model
  - Cross-attention model
  - Stacked multi-layer attention
  - Minimal attention model
- **`examples/transformer.neural`**: Updated with AttentionModel example
- **`examples/README_MULTIHEADATTENTION.md`**: Complete documentation

## Features Implemented

### Self-Attention Mode
```neural
MultiHeadAttention(num_heads:8, key_dim:64, dropout:0.1)
```
Attends to the input sequence itself.

### Cross-Attention Mode
```neural
MultiHeadAttention(num_heads:4, key_dim:32, mode:"cross", dropout:0.1)
```
Attends to a different context sequence (encoder-decoder attention).

### Key Parameters
1. **num_heads**: Number of parallel attention heads (4, 8, 12, 16 typical)
2. **key_dim**: Dimension of key/query vectors
3. **value_dim**: Optional separate dimension for value vectors
4. **dropout**: Regularization rate for attention weights
5. **use_bias**: Enable/disable bias in linear projections
6. **mode**: Switch between "self" and "cross" attention

## Backend Support

### TensorFlow/Keras
- Uses `tf.keras.layers.MultiHeadAttention`
- Full parameter support
- Clean functional API integration

### PyTorch
- Uses `torch.nn.MultiheadAttention`
- Handles tuple output (output, attention_weights)
- Automatic embed_dim inference from input shape
- batch_first=True for consistent shape handling

### ONNX
- Exports using Attention operator
- Suitable for deployment and optimization

## Shape Behavior
- **Input**: `(batch_size, sequence_length, embedding_dim)`
- **Output**: `(batch_size, sequence_length, embedding_dim)`
- Shape preservation enables stacking multiple attention layers

## Integration with Existing Layers
MultiHeadAttention works seamlessly with:
- **LayerNormalization**: Pre/post normalization patterns
- **Dense**: Feed-forward networks after attention
- **Dropout**: Additional regularization
- **Residual connections**: Via manual implementation or wrapper layers

## Validation
- Parameter validation in parser (positive integers required)
- HPO parameter support (can be tuned)
- Device specification support (@"cuda:0")
- Shape propagation for dimension checking

## Documentation
Complete user-facing documentation provided in `examples/README_MULTIHEADATTENTION.md` covering:
- Parameter descriptions
- Usage examples
- Self-attention vs cross-attention
- Backend-specific details
- Best practices
- Comparison with TransformerEncoder

## Example Usage

```neural
network SelfAttentionModel {
  input: (128, 512)
  layers:
    MultiHeadAttention(num_heads:8, key_dim:64, dropout:0.1)
    LayerNormalization()
    Dense(units:256, activation:"relu")
    Dropout(rate:0.1)
    Output(units:10, activation:"softmax")

  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate:0.001)
}
```

## Testing Considerations
To properly test this implementation:
1. Parse DSL with MultiHeadAttention layers
2. Generate code for TensorFlow, PyTorch, and ONNX backends
3. Verify shape propagation
4. Test both self-attention and cross-attention modes
5. Validate parameter handling and error messages
