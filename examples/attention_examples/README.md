# Attention Visualization Examples

This directory contains Neural DSL examples demonstrating attention visualization for transformer models.

## Examples

### 1. Basic Transformer (`basic_transformer.neural`)
A simple transformer model with:
- Single TransformerEncoder layer with 4 attention heads
- Input sequence length of 50 tokens
- Embedding dimension of 256
- Classification task with 5 classes

**Usage:**
```bash
# Visualize architecture
neural visualize examples/attention_examples/basic_transformer.neural

# Visualize attention weights
neural visualize examples/attention_examples/basic_transformer.neural --attention --backend tensorflow
```

### 2. Multi-Layer Transformer (`multi_layer_transformer.neural`)
A deeper transformer model with:
- 3 stacked TransformerEncoder layers
- 8 attention heads per layer
- Input sequence length of 128 tokens
- Embedding dimension of 512
- Classification task with 20 classes

**Usage:**
```bash
# Visualize specific layer's attention
neural visualize examples/attention_examples/multi_layer_transformer.neural --attention --layer transformer_encoder_1

# Visualize specific attention head
neural visualize examples/attention_examples/multi_layer_transformer.neural --attention --head 0
```

### 3. Text Classification (`text_classification.neural`)
A text classification transformer with:
- 2 TransformerEncoder layers with 4 heads each
- Designed for text sequences up to 200 tokens
- Includes embedding and dropout layers
- 3-class classification (e.g., sentiment analysis)

**Usage:**
```bash
# With custom input data
neural visualize examples/attention_examples/text_classification.neural --attention --data input_tokens.npy

# With token labels for better visualization
neural visualize examples/attention_examples/text_classification.neural --attention --tokens "This,is,an,example,sentence"
```

## Advanced Usage

### Visualize with Custom Input Data

1. Create input data file (NumPy format):
```python
import numpy as np

# For basic_transformer.neural
# Shape: (batch_size, sequence_length, embedding_dim)
input_data = np.random.randn(1, 50, 256).astype(np.float32)
np.save('transformer_input.npy', input_data)
```

2. Run visualization:
```bash
neural visualize examples/attention_examples/basic_transformer.neural --attention --data transformer_input.npy
```

### Visualize Specific Layers and Heads

```bash
# Visualize only first transformer layer
neural visualize examples/attention_examples/multi_layer_transformer.neural --attention --layer transformer_encoder_0

# Visualize specific head in first layer
neural visualize examples/attention_examples/multi_layer_transformer.neural --attention --layer transformer_encoder_0 --head 3
```

### PyTorch Backend

```bash
# Use PyTorch instead of TensorFlow
neural visualize examples/attention_examples/basic_transformer.neural --attention --backend pytorch
```

## Output Files

When running attention visualization, the following files are generated in `attention_outputs/`:

- `attention_heatmap.png` - Average attention weights visualization
- `attention_heads_{layer_name}.png` - Grid visualization of all attention heads
- `attention_interactive.html` - Interactive plotly visualization (if plotly is installed)
- `compiled_model_{backend}.py` - Generated Python code for the model

## Understanding Attention Patterns

The visualization includes several metrics to help understand attention behavior:

- **Entropy**: Lower entropy means the model focuses on fewer positions (more peaked attention)
- **Max Attention**: Average maximum attention weight per position
- **Diagonal Strength**: How much the model attends to its own position (self-attention)

Example output:
```
Attention Analysis:
  Layers with attention: 1
  transformer_encoder: (1, 4, 50, 50)
    - Num heads: 4
    - Avg entropy: 3.456
    - Avg max attention: 0.234
    - Diagonal strength: 1.543
```

## Tips

1. **Token Labels**: Providing token labels makes visualizations much more interpretable:
   ```bash
   --tokens "The,cat,sat,on,the,mat"
   ```

2. **Multiple Layers**: For multi-layer transformers, visualize each layer separately to see how attention patterns evolve

3. **Head Comparison**: Different attention heads often learn different patterns (e.g., syntactic vs semantic relationships)

4. **Interactive Mode**: Use the generated HTML file for interactive exploration of attention weights

## Requirements

Install visualization dependencies:
```bash
pip install matplotlib seaborn plotly
```

For full functionality with TensorFlow/PyTorch:
```bash
pip install -e ".[full]"
```
