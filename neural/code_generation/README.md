# Neural Code Generation

<p align="center">
  <img src="../../docs/images/code_generation_flow.png" alt="Code Generation Flow" width="600"/>
</p>

## Overview

The Code Generation module is responsible for transforming Neural DSL model representations into executable code for various backend frameworks. It uses a strategy pattern to support multiple backends and modular architecture for easy extension.

## Architecture

The module is structured using the Strategy Pattern with the following components:

### Base Generator (`base_generator.py`)

Abstract base class that defines the interface for all backend generators:
- `generate()`: Main method to generate complete code
- `generate_layer()`: Generate code for individual layers
- `expand_layers()`: Handle layer multiplication feature

### Backend-Specific Generators

Each backend has its own generator class implementing the strategy:

1. **TensorFlow Generator** (`tensorflow_generator.py`): Generates TensorFlow/Keras code
2. **PyTorch Generator** (`pytorch_generator.py`): Generates PyTorch code  
3. **ONNX Generator** (`onnx_generator.py`): Generates ONNX model definitions

### Shape Policy Helpers (`shape_policy_helpers.py`)

Extracted policy functions for ensuring correct tensor shapes:
- `ensure_2d_before_dense_tf()`: TensorFlow Dense layer policy
- `ensure_2d_before_dense_pt()`: PyTorch Linear layer policy
- `get_rank_non_batch()`: Calculate tensor rank excluding batch dimension

### Main Interface (`code_generator.py`)

The main entry point that:
- Validates model data format
- Selects the appropriate backend generator using strategy pattern
- Provides utility functions (`save_file`, `load_file`, `to_number`)
- Handles DSL optimization with HPO results

## Supported Backends

The Code Generation module currently supports the following backends:

1. **TensorFlow/Keras**: Generates TensorFlow 2.x code using the Keras API
2. **PyTorch**: Generates PyTorch code with nn.Module classes
3. **ONNX**: Generates ONNX model definitions for cross-framework compatibility

## Supported Layers

### Transformer Layers

#### TransformerEncoder
Self-attention based encoder layer with feed-forward network.

**Parameters:**
- `num_heads`: Number of attention heads (default: 8)
- `d_model`: Model dimension (default: 512)
- `ff_dim`: Feed-forward dimension (default: 2048)
- `dropout`: Dropout rate (default: 0.1)

#### TransformerDecoder
Decoder layer with self-attention, cross-attention, and feed-forward network for encoder-decoder architectures.

**Parameters:**
- `num_heads`: Number of attention heads (default: 8)
- `d_model`: Model dimension (default: same as ff_dim)
- `ff_dim`: Feed-forward dimension (default: 512)
- `dropout`: Dropout rate (default: 0.1)
- `use_causal_mask`: Apply causal masking for autoregressive decoding (default: true)

**Features:**
- Self-attention with optional causal masking for autoregressive decoding
- Cross-attention with encoder output for encoder-decoder architectures
- Residual connections around attention and feed-forward blocks
- Layer normalization for stable training

**Shape Propagation:**
- Input: `(batch, seq_len, d_model)` for TensorFlow
- Input: `(batch, seq_len)` for PyTorch
- Output: Same as input shape (preserves sequence length and model dimension)

**Example Usage:**
```python
# DSL
TransformerDecoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1, use_causal_mask=true)

# Full encoder-decoder architecture
network Seq2Seq {
  input: (None, 100, 512)
  layers:
    TransformerEncoder(num_heads=8, d_model=512, ff_dim=2048)
    TransformerDecoder(num_heads=8, d_model=512, ff_dim=2048, use_causal_mask=true)
    Dense(units=10000, activation=softmax)
}
```

**Implementation Notes:**
- **TensorFlow**: Implements multi-head self-attention with causal masking and cross-attention using `layers.MultiHeadAttention`. Requires `encoder_output` variable in scope for cross-attention.
- **PyTorch**: Uses `nn.TransformerDecoderLayer` with standard decoder architecture.
- **ONNX**: Uses stacked `MultiHeadAttention` operators for both self-attention and cross-attention.
- **Causal Masking**: Prevents attention to future tokens in autoregressive decoding (e.g., language generation).
- **Cross-Attention**: Enables conditioning on encoder representations for sequence-to-sequence tasks.

## Usage

### Basic Code Generation

```python
from neural.code_generation import generate_code

# Model data from the parser
model_data = {
    "name": "MNIST",
    "input": {"shape": [28, 28, 1]},
    "layers": [
        {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "activation": "relu"}},
        {"type": "MaxPooling2D", "params": {"pool_size": 2}},
        {"type": "Flatten"},
        {"type": "Dense", "params": {"units": 128, "activation": "relu"}},
        {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
    ],
    "loss": "sparse_categorical_crossentropy",
    "optimizer": {"type": "Adam", "params": {"learning_rate": 0.001}},
    "metrics": ["accuracy"]
}

# Generate TensorFlow code
tensorflow_code = generate_code(model_data, backend="tensorflow")

# Generate PyTorch code
pytorch_code = generate_code(model_data, backend="pytorch")

# Generate code with hyperparameter optimization results
optimized_code = generate_code(
    model_data, 
    backend="tensorflow", 
    best_params={"learning_rate": 0.0005, "batch_size": 128}
)

# Auto-flatten higher-rank inputs before Dense layers
code = generate_code(model_data, backend="tensorflow", auto_flatten_output=True)
```

### Using Backend Generators Directly

```python
from neural.code_generation import TensorFlowGenerator, PyTorchGenerator

# Create generator instance
tf_generator = TensorFlowGenerator(model_data, best_params={"learning_rate": 0.001})

# Generate code
code = tf_generator.generate()

# Generate individual layer
layer_code = tf_generator.generate_layer("Conv2D", {"filters": 32, "kernel_size": 3})
```

### Hyperparameter Optimization Integration

```python
from neural.code_generation import generate_optimized_dsl

# Original Neural DSL code with HPO annotations
neural_code = """
network MNIST {
  input: (28, 28, 1)
  layers:
    Conv2D(32, kernel_size=3, activation="relu")
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(128, activation="relu")
    Output(10, activation="softmax")

  optimizer: Adam(learning_rate=HPO(log_range(0.0001, 0.01)))
  training_config:
    batch_size: HPO(choice(32, 64, 128))
}
"""

# Optimized hyperparameters from HPO
best_params = {
    "learning_rate": 0.0005,
    "batch_size": 128
}

# Generate optimized DSL code with best hyperparameters
optimized_dsl = generate_optimized_dsl(neural_code, best_params)
```

### Using Shape Policy Helpers

```python
from neural.code_generation.shape_policy_helpers import (
    ensure_2d_before_dense_tf,
    ensure_2d_before_dense_pt,
    get_rank_non_batch
)
from neural.shape_propagation import ShapePropagator

propagator = ShapePropagator(debug=False)
current_shape = (None, 8, 8, 3)

# Get rank excluding batch dimension
rank = get_rank_non_batch(current_shape)  # Returns 3

# TensorFlow policy for Dense layers
insert_code, new_shape = ensure_2d_before_dense_tf(
    rank_non_batch=rank,
    auto_flatten_output=True,
    propagator=propagator,
    current_input_shape=current_shape
)

# PyTorch policy for Linear layers
forward_code = []
new_shape = ensure_2d_before_dense_pt(
    rank_non_batch=rank,
    auto_flatten_output=True,
    forward_code_body=forward_code,
    propagator=propagator,
    current_input_shape=current_shape
)
```

## Extension Guide

### Adding a New Backend

1. Create a new generator class inheriting from `BaseCodeGenerator`:

```python
from neural.code_generation.base_generator import BaseCodeGenerator

class MyBackendGenerator(BaseCodeGenerator):
    def generate(self) -> str:
        # Implement full code generation
        pass
    
    def generate_layer(self, layer_type: str, params: Dict[str, Any]) -> str:
        # Implement layer-specific code generation
        pass
```

2. Update `code_generator.py` to include the new backend:

```python
from neural.code_generation.mybackend_generator import MyBackendGenerator

def generate_code(model_data, backend, ...):
    if backend == "mybackend":
        generator = MyBackendGenerator(model_data, best_params, auto_flatten_output)
        return generator.generate()
    # ... existing backends
```

3. Export from `__init__.py`:

```python
from neural.code_generation.mybackend_generator import MyBackendGenerator

__all__ = [..., 'MyBackendGenerator']
```

### Adding Custom Layers

Extend the `generate_layer` method in the appropriate backend generator:

```python
def generate_layer(self, layer_type: str, params: Dict[str, Any]) -> str:
    if layer_type == "MyCustomLayer":
        custom_param = params.get("custom_param", default_value)
        return f"layers.MyCustomLayer({custom_param})"
    # ... existing layers
```

### Example: TransformerDecoder Implementation

The TransformerDecoder layer demonstrates how to implement complex layers with multiple attention mechanisms:

**TensorFlow Implementation:**
```python
elif layer_type == "TransformerDecoder":
    num_heads = params.get("num_heads", 8)
    ff_dim = params.get("ff_dim", 512)
    dropout = params.get("dropout", 0.1)
    d_model = params.get("d_model", ff_dim)
    use_causal_mask = params.get("use_causal_mask", True)
    code = [
        "# TransformerDecoder block with cross-attention",
        f"decoder_norm1 = layers.LayerNormalization(epsilon=1e-6)(x)",
    ]
    if use_causal_mask:
        code.append(f"self_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model}, use_causal_mask=True)(decoder_norm1, decoder_norm1)")
    else:
        code.append(f"self_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model})(decoder_norm1, decoder_norm1)")
    code.extend([
        f"x = layers.Add()([x, layers.Dropout({dropout})(self_attn_output)])",
        f"decoder_norm2 = layers.LayerNormalization(epsilon=1e-6)(x)",
        f"cross_attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={d_model})(decoder_norm2, encoder_output, encoder_output)",
        f"x = layers.Add()([x, layers.Dropout({dropout})(cross_attn_output)])",
        f"decoder_norm3 = layers.LayerNormalization(epsilon=1e-6)(x)",
        f"ff_output = layers.Dense({ff_dim}, activation='relu')(decoder_norm3)",
        f"ff_output = layers.Dense({d_model})(ff_output)",
        f"x = layers.Add()([x, layers.Dropout({dropout})(ff_output)])"
    ])
    return "\n".join(code)
```

**PyTorch Implementation:**
```python
elif layer_type == "TransformerDecoder":
    d_model = params.get("d_model", 512)
    nhead = params.get("num_heads", 8)
    dim_feedforward = params.get("ff_dim", 2048)
    dropout = params.get("dropout", 0.1)
    return f"nn.TransformerDecoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout})"
```

**Shape Propagation:**
```python
if layer_type == 'TransformerDecoder':
    if framework == 'tensorflow':
        return input_shape  # Preserves shape through attention
    elif framework == 'pytorch':
        return (input_shape[0], input_shape[1])  # (seq_len, d_model)
```

## Design Patterns

### Strategy Pattern

The module uses the Strategy Pattern to:
- Separate backend-specific logic into dedicated classes
- Allow easy addition of new backends without modifying existing code
- Enable runtime selection of code generation strategy

### Template Method Pattern

The `BaseCodeGenerator` defines the skeleton of the code generation algorithm:
- `expand_layers()`: Common preprocessing step
- `generate()`: Backend-specific implementation
- `generate_layer()`: Backend-specific layer handling

## Related Components

- **Parser**: Provides the model representation used for code generation
- **Shape Propagation**: Provides shape information used in the generated code
- **HPO**: Provides optimized hyperparameters for code generation
- **Experiment Tracking**: Integration for logging metrics during training

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ONNX Documentation](https://onnx.ai/onnx/index.html)
- [Neural DSL Reference](../../docs/DSL.md)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
