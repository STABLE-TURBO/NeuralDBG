# Attention Visualization for Neural DSL

Comprehensive documentation for attention visualization capabilities in Neural DSL.

## Overview

The attention visualization module provides tools to extract, analyze, and visualize attention weights from transformer models compiled from Neural DSL. It seamlessly integrates with the Neural CLI and supports both TensorFlow and PyTorch backends.

## Features

- **Seamless DSL Integration**: Compile and visualize Neural DSL models directly
- **Multi-Backend Support**: Works with TensorFlow and PyTorch
- **Multiple Visualization Formats**: Static heatmaps, grid views, and interactive HTML
- **Attention Analysis**: Entropy, max attention, diagonal strength metrics
- **Layer & Head Selection**: Focus on specific layers or attention heads
- **Token Labels**: Add semantic meaning to attention patterns
- **Programmatic API**: Python API for custom workflows

## Installation

### Minimal Installation

```bash
pip install matplotlib seaborn
```

### Full Installation (with all backends)

```bash
pip install -e ".[full]"
```

### Optional Interactive Visualizations

```bash
pip install plotly
```

## Quick Start

### CLI Usage

```bash
# Basic attention visualization
neural visualize examples/transformer.neural --attention

# With custom input data
neural visualize model.neural --attention --data input.npy

# With token labels
neural visualize model.neural --attention --tokens "The,cat,sat,on,the,mat"

# Specific layer and head
neural visualize model.neural --attention --layer transformer_encoder_0 --head 2

# PyTorch backend
neural visualize model.neural --attention --backend pytorch
```

### Python API

```python
from neural.explainability.attention_visualizer import AttentionVisualizer
import numpy as np

# Initialize visualizer
visualizer = AttentionVisualizer(model=None, backend='tensorflow')

# Visualize from DSL file
input_data = np.random.randn(1, 50, 256).astype(np.float32)
results = visualizer.visualize_from_dsl(
    dsl_file='model.neural',
    input_data=input_data,
    backend='tensorflow',
    output_dir='attention_outputs'
)

# Access attention weights
for layer_name, weights in results['attention_weights'].items():
    print(f"{layer_name}: {weights.shape}")
    
    # Analyze patterns
    analysis = visualizer.analyze_attention_patterns(weights)
    print(f"  Entropy: {analysis['avg_entropy']:.3f}")
    print(f"  Diagonal strength: {analysis['avg_diagonal_strength']:.3f}")
```

## AttentionVisualizer Class

### Constructor

```python
AttentionVisualizer(model, backend='tensorflow')
```

**Parameters:**
- `model`: The model with attention layers (can be None if using `visualize_from_dsl`)
- `backend`: ML framework ('tensorflow' or 'pytorch')

### Methods

#### extract_attention_weights

```python
extract_attention_weights(input_data, layer_names=None)
```

Extract attention weights from specified layers.

**Parameters:**
- `input_data`: Input data (numpy array)
- `layer_names`: List of layer names to extract (None = all attention layers)

**Returns:** Dictionary mapping layer names to attention weights

#### visualize

```python
visualize(input_data, layer_name=None, head_index=None, 
          tokens=None, output_path=None)
```

Visualize attention patterns.

**Parameters:**
- `input_data`: Input data
- `layer_name`: Specific layer to visualize (None = all)
- `head_index`: Specific attention head (None = average all heads)
- `tokens`: Token labels for axes
- `output_path`: Path to save visualization

**Returns:** Dictionary with attention weights and visualizations

#### visualize_from_dsl

```python
visualize_from_dsl(dsl_file, input_data, backend='tensorflow',
                   layer_name=None, head_index=None, 
                   tokens=None, output_dir='attention_outputs')
```

Compile and visualize a Neural DSL model.

**Parameters:**
- `dsl_file`: Path to .neural file
- `input_data`: Input data
- `backend`: Backend to compile with
- `layer_name`: Specific layer to visualize
- `head_index`: Specific attention head
- `tokens`: Token labels
- `output_dir`: Output directory for visualizations

**Returns:** Dictionary with attention weights and visualizations

#### plot_attention_heads

```python
plot_attention_heads(attention_weights, tokens=None, output_path=None)
```

Plot all attention heads in a grid.

**Parameters:**
- `attention_weights`: Attention weights (num_heads, seq_len, seq_len)
- `tokens`: Token labels
- `output_path`: Path to save plot

**Returns:** Matplotlib figure

#### create_interactive_visualization

```python
create_interactive_visualization(attention_weights, tokens=None,
                                 output_path='attention_interactive.html')
```

Create interactive Plotly visualization.

**Parameters:**
- `attention_weights`: Dictionary of attention weights by layer
- `tokens`: Token labels
- `output_path`: Path to save HTML file

**Returns:** Path to HTML file

#### analyze_attention_patterns

```python
analyze_attention_patterns(attention_weights)
```

Analyze attention patterns for insights.

**Parameters:**
- `attention_weights`: Attention weights to analyze

**Returns:** Dictionary with analysis metrics:
- `num_heads`: Number of attention heads
- `attention_entropy`: Entropy per head (lower = more focused)
- `max_attention_per_head`: Max attention weight per head
- `attention_distribution`: Mean, std, min, max per head
- `avg_entropy`: Average entropy across heads
- `avg_max_attention`: Average max attention
- `diagonal_attention_strength`: Diagonal attention relative to average
- `avg_diagonal_strength`: Average diagonal strength

## CLI Integration

The `neural visualize` command has been enhanced with attention visualization options:

### Options

- `--attention`: Enable attention visualization
- `--backend {tensorflow,pytorch}`: Backend for compilation (default: tensorflow)
- `--data PATH`: Input data file (.npy format)
- `--tokens TEXT`: Comma-separated token labels
- `--layer TEXT`: Specific attention layer to visualize
- `--head INT`: Specific attention head to visualize

### Examples

```bash
# Basic usage
neural visualize model.neural --attention

# With all options
neural visualize model.neural \
  --attention \
  --backend pytorch \
  --data input_sequence.npy \
  --tokens "Hello,world,this,is,a,test" \
  --layer transformer_encoder_1 \
  --head 0
```

## Understanding Attention Metrics

### Entropy

**Range:** 0 to log(sequence_length)

**Interpretation:**
- Low entropy (< 2): Attention is focused on few positions
- Medium entropy (2-4): Balanced attention distribution
- High entropy (> 4): Uniform/diffuse attention

### Max Attention

**Range:** 0 to 1

**Interpretation:**
- High values (> 0.5): Strong focus on specific positions
- Low values (< 0.2): Distributed attention

### Diagonal Strength

**Range:** Typically 0.5 to 3.0

**Interpretation:**
- High values (> 1.5): Strong self-attention (attends to own position)
- Values ~1.0: Balanced self vs cross attention
- Low values (< 0.5): Minimal self-attention

## Attention Pattern Types

### Local Attention
- **Characteristics**: High diagonal strength, low entropy
- **Use case**: Syntactic relationships, nearby token dependencies
- **Visualization**: Diagonal pattern in heatmap

### Global Attention
- **Characteristics**: High entropy, uniform distribution
- **Use case**: Long-range dependencies, document-level context
- **Visualization**: Uniform heatmap

### Prefix/Suffix Attention
- **Characteristics**: Focused on start/end of sequence
- **Use case**: Special tokens (CLS, SEP), positional markers
- **Visualization**: Vertical/horizontal bands

### Sparse Attention
- **Characteristics**: Low entropy, low max attention
- **Use case**: Efficient transformers, learned patterns
- **Visualization**: Isolated bright spots

## Neural DSL Model Requirements

For attention visualization to work, your Neural DSL model must include attention layers:

### Supported Layer Types

- `TransformerEncoder(num_heads, ff_dim, dropout)`
- `MultiHeadAttention(num_heads, key_dim)`
- Custom attention layers with 'attention' in the name

### Example Model

```
network MyTransformer {
  input: (100, 512)
  layers:
    Embedding(input_dim=10000, output_dim=512)
    TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)
    GlobalAveragePooling1D()
    Dense(units=128, activation="relu")
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

## Input Data Format

### For Models with Embedding Layer

If your model starts with an `Embedding` layer:

```python
# Shape: (batch_size, sequence_length, embedding_dim)
input_data = np.random.randn(1, 100, 512).astype(np.float32)
```

### For Token ID Input

Some models expect token indices:

```python
# Shape: (batch_size, sequence_length)
token_ids = np.random.randint(0, 10000, size=(1, 100), dtype=np.int32)
```

### Generating Sample Data

Use the provided script:

```bash
cd examples/attention_examples
python generate_sample_data.py
```

## Output Files

When running attention visualization, several files are created:

### Static Visualizations

- `attention_heatmap.png`: Average attention across all heads
- `attention_heads_{layer_name}.png`: Grid of all attention heads
- `compiled_model_{backend}.py`: Generated model code

### Interactive Visualizations

- `attention_interactive.html`: Interactive Plotly dashboard

### Analysis Files

Console output includes:
- Number of attention layers
- Shape of attention weights
- Entropy per head
- Max attention per head
- Diagonal strength

## Advanced Usage

### Comparing Multiple Layers

```python
visualizer = AttentionVisualizer(model=None, backend='tensorflow')

for layer_idx in range(3):
    results = visualizer.visualize_from_dsl(
        dsl_file='multilayer_transformer.neural',
        input_data=input_data,
        layer_name=f'transformer_encoder_{layer_idx}',
        output_dir=f'layer_{layer_idx}_outputs'
    )
```

### Custom Analysis Pipeline

```python
# Extract attention
visualizer = AttentionVisualizer(model, backend='tensorflow')
attention_weights = visualizer.extract_attention_weights(input_data)

# Custom analysis
for layer_name, weights in attention_weights.items():
    analysis = visualizer.analyze_attention_patterns(weights)
    
    # Custom metric: attention sparsity
    sparsity = (weights < 0.01).sum() / weights.size
    
    print(f"{layer_name}:")
    print(f"  Sparsity: {sparsity:.2%}")
    print(f"  Entropy: {analysis['avg_entropy']:.3f}")
```

### Batch Processing

```python
import glob

for model_file in glob.glob('models/*.neural'):
    try:
        results = visualizer.visualize_from_dsl(
            dsl_file=model_file,
            input_data=input_data,
            output_dir=f'outputs/{Path(model_file).stem}'
        )
        print(f"✓ {model_file}")
    except Exception as e:
        print(f"✗ {model_file}: {e}")
```

## Troubleshooting

### No Attention Layers Found

**Problem:** "No attention layers found in the model"

**Solutions:**
1. Verify model contains `TransformerEncoder` or `MultiHeadAttention` layers
2. Check layer names contain 'attention', 'transformer', or 'multihead'
3. Review generated code in `compiled_model_{backend}.py`

### Shape Mismatch

**Problem:** Input shape doesn't match model expectations

**Solutions:**
1. Check model's `input:` specification in .neural file
2. Verify input data shape matches: (batch_size, seq_len, embed_dim)
3. Use `generate_sample_data.py` to create correctly-shaped data

### Import Errors

**Problem:** Missing matplotlib, seaborn, or plotly

**Solution:**
```bash
pip install matplotlib seaborn plotly
```

### Backend Issues

**Problem:** TensorFlow or PyTorch not available

**Solution:**
```bash
# For TensorFlow
pip install tensorflow

# For PyTorch
pip install torch

# Or install full suite
pip install -e ".[full]"
```

## Best Practices

1. **Use Token Labels**: Makes visualizations much more interpretable
2. **Start Simple**: Begin with single-layer models before analyzing complex architectures
3. **Compare Heads**: Different heads often learn complementary patterns
4. **Save Outputs**: Keep visualizations for comparison across training runs
5. **Analyze Metrics**: Use entropy and diagonal strength to understand attention behavior

## Examples

See the `examples/attention_examples/` directory for:
- `basic_transformer.neural`: Simple single-layer model
- `multi_layer_transformer.neural`: Deep transformer with 3 layers
- `text_classification.neural`: Practical NLP example
- `visualize_attention_demo.py`: Comprehensive Python API demo
- `generate_sample_data.py`: Data generation utilities

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Analyzing Multi-Head Self-Attention](https://arxiv.org/abs/1905.09418) - Attention head analysis
- [BERTology Meets Biology](https://arxiv.org/abs/2006.15222) - Practical attention analysis

## Contributing

To enhance attention visualization:

1. Fork the repository
2. Add new visualization methods to `attention_visualizer.py`
3. Update this documentation
4. Add examples to `examples/attention_examples/`
5. Submit a pull request

## License

Same as Neural DSL main project (see LICENSE.md).
