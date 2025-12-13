# TransformerEncoder Enhancements

This document describes the enhanced TransformerEncoder implementation with support for attention masks, multiple encoder layers, and configurable activation functions.

## Overview

The TransformerEncoder layer has been enhanced with three major features:

1. **Attention Mask Support** - Enable masking for padded sequences in variable-length inputs
2. **Multiple Encoder Layers Stacking** - Stack multiple transformer encoder layers for deeper models
3. **Configurable Activation Functions** - Customize activation functions in feed-forward networks

## Features

### 1. Attention Mask Support

Attention masks prevent the model from attending to padding tokens in variable-length sequences, improving both training efficiency and model quality.

**Parameter:** `use_attention_mask` (boolean, default: `false`)

**TensorFlow Implementation:**
- Uses the `attention_mask` parameter in `MultiHeadAttention` layer
- The mask should be provided as a tensor during model execution

**PyTorch Implementation:**
- Uses the `src_key_padding_mask` parameter in `TransformerEncoder`
- The mask should be a boolean tensor where `True` indicates positions to ignore

**Example:**
```neural
TransformerEncoder(num_heads=8, ff_dim=2048, use_attention_mask=true)
```

### 2. Multiple Encoder Layers

Stack multiple transformer encoder layers to create deeper models. Each layer contains:
- Multi-head self-attention mechanism
- Add & Normalize
- Position-wise feed-forward network
- Add & Normalize

**Parameter:** `num_layers` (integer, default: `1`)

**TensorFlow Implementation:**
- Generates a loop that creates `num_layers` encoder blocks
- Each iteration includes full attention and FFN sub-layers

**PyTorch Implementation:**
- Uses `nn.TransformerEncoder` with the `num_layers` parameter
- More efficient than manual stacking

**Example:**
```neural
TransformerEncoder(num_heads=8, ff_dim=2048, num_layers=6)
```

**Recommended Values:**
- Small models: 1-2 layers
- Medium models: 4-6 layers
- Large models: 12-24 layers
  - BERT-base: 12 layers
  - BERT-large: 24 layers
  - GPT-2: 12 layers
  - GPT-3: 96 layers

### 3. Configurable Activation Functions

Customize the activation function used in the feed-forward networks. Modern transformers often use GELU instead of ReLU.

**Parameter:** `activation` (string, default: `"relu"`)

**Supported Activations:**
- `"relu"` - Rectified Linear Unit (traditional choice)
- `"gelu"` - Gaussian Error Linear Unit (modern choice, used in BERT, GPT)
- `"tanh"` - Hyperbolic Tangent
- `"sigmoid"` - Sigmoid activation
- `"swish"` - Swish/SiLU activation (good for very deep networks)

**Example:**
```neural
TransformerEncoder(num_heads=8, ff_dim=2048, activation="gelu")
```

**Activation Function Comparison:**
- **ReLU**: Faster computation, but may suffer from "dead neurons"
- **GELU**: Smoother gradients, better for training deep models
- **Swish/SiLU**: Self-gated, effective for very deep architectures

## Complete Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | int | 8 | Number of attention heads |
| `ff_dim` | int | 512 | Dimension of feed-forward network |
| `dropout` | float | 0.1 | Dropout rate for regularization |
| `num_layers` | int | 1 | Number of stacked encoder layers |
| `activation` | string | "relu" | Activation function for FFN |
| `use_attention_mask` | boolean | false | Enable attention mask support |

## Usage Examples

### Basic Usage
```neural
network BasicTransformer {
  input: (100, 512)
  layers:
    TransformerEncoder(num_heads=8, ff_dim=2048)
    GlobalAveragePooling1D()
    Output(units=10, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
}
```

### Deep Transformer with GELU
```neural
network DeepTransformer {
  input: (128, 512)
  layers:
    TransformerEncoder(num_heads=8, ff_dim=2048, num_layers=6, activation="gelu")
    GlobalAveragePooling1D()
    Output(units=20, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

### Transformer with Attention Mask
```neural
network MaskedTransformer {
  input: (64, 256)
  layers:
    TransformerEncoder(num_heads=4, ff_dim=1024, num_layers=3, use_attention_mask=true)
    GlobalAveragePooling1D()
    Output(units=5, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
}
```

### Full Configuration
```neural
network FullTransformer {
  input: (128, 512)
  layers:
    TransformerEncoder(
      num_heads=8,
      ff_dim=2048,
      dropout=0.2,
      num_layers=6,
      activation="gelu",
      use_attention_mask=true
    )
    GlobalAveragePooling1D()
    Output(units=10, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

## Generated Code Examples

### TensorFlow (with num_layers=2, activation="gelu", use_attention_mask=true)

```python
# TransformerEncoder block
# Attention mask should be provided as input
attention_mask = None  # Set this to your mask tensor

# Encoder Layer 1
x = layers.LayerNormalization(epsilon=1e-6)(x)
attn_output = layers.MultiHeadAttention(num_heads=8, key_dim=512)(x, x, attention_mask=attention_mask)
attn_output = layers.Dropout(0.1)(attn_output)
x = layers.Add()([x, attn_output])
x = layers.LayerNormalization(epsilon=1e-6)(x)
ffn_output = layers.Dense(512, activation='gelu')(x)
ffn_output = layers.Dense(512)(ffn_output)
ffn_output = layers.Dropout(0.1)(ffn_output)
x = layers.Add()([x, ffn_output])

# Encoder Layer 2
x = layers.LayerNormalization(epsilon=1e-6)(x)
attn_output = layers.MultiHeadAttention(num_heads=8, key_dim=512)(x, x, attention_mask=attention_mask)
attn_output = layers.Dropout(0.1)(attn_output)
x = layers.Add()([x, attn_output])
x = layers.LayerNormalization(epsilon=1e-6)(x)
ffn_output = layers.Dense(512, activation='gelu')(x)
ffn_output = layers.Dense(512)(ffn_output)
ffn_output = layers.Dropout(0.1)(ffn_output)
x = layers.Add()([x, ffn_output])
```

### PyTorch (with num_layers=6, activation="gelu")

```python
# In __init__
self.transformer = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu'
    ),
    num_layers=6
)

# In forward pass
x = self.transformer(x)

# With attention mask
x = self.transformer(x, src_key_padding_mask=mask)
```

## Best Practices

### Number of Layers
- **Small models (< 10M params)**: 1-2 layers
- **Medium models (10M-100M params)**: 4-6 layers
- **Large models (> 100M params)**: 12-24 layers
- Consider computational resources and training time

### Activation Functions
- Use **ReLU** for faster training and simpler models
- Use **GELU** for state-of-the-art performance (recommended for most cases)
- Use **Swish/SiLU** for very deep networks or when experimentation shows better results

### Attention Masks
- **Always use** for variable-length sequences with padding
- Improves model quality by preventing attention to padding tokens
- Essential for real-world NLP tasks (sentiment analysis, translation, etc.)

### Hyperparameter Guidelines
- **num_heads**: Must divide `d_model` (embedding dimension) evenly
  - Common values: 4, 8, 12, 16
  - BERT-base uses 12, GPT-2 uses 12, GPT-3 uses 96
- **ff_dim**: Typically 4× the `d_model`
  - If `d_model=512`, use `ff_dim=2048`
  - If `d_model=768`, use `ff_dim=3072`
- **dropout**: Start with 0.1, increase to 0.2-0.3 for regularization
  - Higher dropout for smaller datasets
  - Lower dropout for larger datasets

## Architecture Details

Each encoder layer follows this structure:

```
Input
  ↓
LayerNorm
  ↓
Multi-Head Self-Attention (with optional mask)
  ↓
Dropout
  ↓
Add & Residual Connection
  ↓
LayerNorm
  ↓
Feed-Forward Network (Dense → Activation → Dense)
  ↓
Dropout
  ↓
Add & Residual Connection
  ↓
Output
```

This structure is repeated `num_layers` times.

## Testing

The implementation includes comprehensive tests in `tests/code_generator/test_code_generator.py`:

- `test_transformer_generation()` - Basic functionality
- `test_transformer_with_multiple_layers()` - Multiple layers stacking
- `test_transformer_with_custom_activation()` - Custom activations
- `test_transformer_with_attention_mask()` - Attention mask support
- `test_transformer_all_features()` - All features combined

Run tests with:
```bash
pytest tests/code_generator/test_code_generator.py -v -k transformer
```

## Files Modified

1. **neural/code_generation/tensorflow_generator.py**
   - Enhanced `generate_layer()` method for TransformerEncoder
   - Added support for multiple layers loop
   - Added attention mask handling
   - Added configurable activation functions

2. **neural/code_generation/pytorch_generator.py**
   - Enhanced `generate_pytorch_layer()` function for TransformerEncoder
   - Added `nn.TransformerEncoder` support for multiple layers
   - Added attention mask parameter handling
   - Added configurable activation functions

3. **neural/code_generation/pytorch_generator.py** (generate method)
   - Added TransformerEncoder handling in layer generation loop
   - Added forward pass code generation with mask support

4. **tests/code_generator/test_code_generator.py**
   - Added comprehensive test cases for new features

5. **examples/transformer.neural**
   - Updated with comprehensive examples and documentation
   - Added multiple example networks demonstrating features

## Backward Compatibility

All enhancements are backward compatible. Existing TransformerEncoder definitions will continue to work with default values:
- `num_layers=1` (single layer)
- `activation="relu"` (traditional ReLU)
- `use_attention_mask=false` (no masking)

## References

- [Attention Is All You Need (Original Transformer Paper)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
