# MultiHeadAttention Layer

The `MultiHeadAttention` layer is a standalone attention mechanism that supports both self-attention and cross-attention modes with configurable key/query/value projections.

## Overview

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. It's a core component of Transformer architectures and can be used independently.

## Syntax

```neural
MultiHeadAttention(num_heads=<int>, key_dim=<int>, [optional_params])
```

## Parameters

### Required Parameters
- **num_heads** (int): Number of attention heads. Must be a positive integer that divides embed_dim evenly.
- **key_dim** (int): Dimension of key vectors. For TensorFlow, this is required and typically set to embed_dim / num_heads.

### Optional Parameters
- **value_dim** (int): Dimension of value vectors. If not specified, defaults to key_dim.
- **dropout** (float): Dropout rate applied to attention weights. Range: [0.0, 1.0]. Default: 0.0
- **use_bias** (bool): Whether to use bias in linear projections. Default: true
- **mode** (string): Attention mode - "self" for self-attention or "cross" for cross-attention. Default: "self"
- **embed_dim** (int): Total embedding dimension (PyTorch only). If not specified, inferred from input shape.
- **batch_first** (bool): Whether batch dimension comes first (PyTorch only). Default: true

## Modes

### Self-Attention Mode
In self-attention, the layer attends to itself. Query, key, and value all come from the same input sequence.

```neural
MultiHeadAttention(num_heads:8, key_dim:64, dropout:0.1)
```

**Generated TensorFlow:**
```python
layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x, x)
```

**Generated PyTorch:**
```python
nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)
# Forward pass: x, _ = layer(x, x, x)
```

### Cross-Attention Mode
In cross-attention, the layer attends to a different context sequence. Query comes from one sequence, while key and value come from another.

```neural
MultiHeadAttention(num_heads:4, key_dim:32, mode:"cross", dropout:0.1)
```

**Generated TensorFlow:**
```python
layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, context)
```

**Generated PyTorch:**
```python
nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1, batch_first=True)
# Forward pass: x, _ = layer(x, context, context)
```

## Examples

### Basic Self-Attention
```neural
network SelfAttentionModel {
  input: (128, 512)
  layers:
    MultiHeadAttention(num_heads:8, key_dim:64, dropout:0.1)
    LayerNormalization()
    Dense(units:256, activation:"relu")
    Output(units:10, activation:"softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate:0.001)
}
```

### Stacked Attention Layers
```neural
network StackedAttentionModel {
  input: (100, 384)
  layers:
    MultiHeadAttention(num_heads:6, key_dim:64, value_dim:64, dropout:0.1)
    LayerNormalization()
    Dense(units:384, activation:"gelu")
    Dropout(rate:0.1)
    MultiHeadAttention(num_heads:6, key_dim:64, dropout:0.1)
    LayerNormalization()
    Dense(units:192, activation:"relu")
    Output(units:20, activation:"softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate:0.0001)
}
```

### Cross-Attention for Encoder-Decoder
```neural
network CrossAttentionModel {
  input: (64, 256)
  layers:
    MultiHeadAttention(num_heads:4, key_dim:32, mode:"cross", dropout:0.1)
    LayerNormalization()
    Dense(units:128, activation:"relu")
    Output(units:5, activation:"softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate:0.0005)
}
```

## Shape Propagation

The MultiHeadAttention layer preserves the input shape:
- **Input:** `(batch_size, sequence_length, embedding_dim)`
- **Output:** `(batch_size, sequence_length, embedding_dim)`

The attention mechanism computes:
1. **Query, Key, Value projections:** Linear transformations split across num_heads
2. **Scaled dot-product attention:** Computes attention weights and applies to values
3. **Concatenation and output projection:** Combines heads and projects back to embedding_dim

## Backend-Specific Details

### TensorFlow/Keras
- Uses `tf.keras.layers.MultiHeadAttention`
- Requires `key_dim` parameter
- Supports `value_dim` to decouple value dimension from key dimension
- Returns attention output directly (attention weights can be accessed separately)

### PyTorch
- Uses `torch.nn.MultiheadAttention`
- Requires `embed_dim` (inferred from input shape if not provided)
- Must specify `batch_first=True` for consistent shape handling
- Returns tuple `(output, attention_weights)` - we extract the output with `x, _ = layer(...)`
- Requires three inputs: query, key, value (for self-attention, all are the same input)

### ONNX
- Uses `Attention` operator
- Exports with `num_heads` attribute
- Suitable for deployment and optimization

## Best Practices

1. **Head Dimension:** Set `key_dim = embed_dim / num_heads` for optimal performance
2. **Dropout:** Use 0.1-0.2 during training to prevent overfitting
3. **Layer Normalization:** Apply LayerNorm before or after attention for stable training
4. **Residual Connections:** Combine with residual connections for deep networks
5. **Multiple Heads:** Common choices: 4, 8, 12, 16 heads depending on model size

## Comparison with TransformerEncoder

| Feature | MultiHeadAttention | TransformerEncoder |
|---------|-------------------|-------------------|
| Attention | ✓ Self-attention only | ✓ Self-attention |
| Feed-forward | ✗ | ✓ |
| Layer Normalization | ✗ | ✓ (2 layers) |
| Residual Connections | ✗ | ✓ (2 residual paths) |
| Dropout | ✓ (attention only) | ✓ (attention + FFN) |
| Use Case | Flexible attention building block | Complete encoder block |

Use `MultiHeadAttention` when you need fine-grained control over the architecture, or `TransformerEncoder` for a complete transformer block.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- TensorFlow MultiHeadAttention: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
- PyTorch MultiheadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
