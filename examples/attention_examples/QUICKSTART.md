# Quick Start: Attention Visualization

Get started with attention visualization in Neural DSL in just a few steps!

## 1. Install Dependencies

```bash
pip install matplotlib seaborn plotly
```

For full Neural DSL functionality:
```bash
pip install -e ".[full]"
```

## 2. Run Your First Visualization

### Basic Command

```bash
neural visualize examples/attention_examples/basic_transformer.neural --attention
```

This will:
- Compile the transformer model
- Generate synthetic input data
- Extract attention weights
- Create visualization heatmaps
- Save outputs to `attention_outputs/`

### Expected Output

```
Visualizing examples/attention_examples/basic_transformer.neural in html format
✓ Parsing Neural DSL file
✓ Propagating shapes through the network...
✓ Generating visualizations
✓ Extracting and visualizing attention weights

Attention Analysis:
  Layers with attention: 1
  transformer_encoder: (1, 4, 50, 50)
    - Num heads: 4
    - Avg entropy: 3.912
    - Avg max attention: 0.156
    - Diagonal strength: 1.234

Output files:
  - attention_outputs/attention_heatmap.png
  - attention_outputs/attention_heads_transformer_encoder.png
  - attention_outputs/attention_interactive.html (interactive)
```

## 3. View Results

Open the generated files:

1. **Static Heatmap**: `attention_outputs/attention_heatmap.png`
   - Shows average attention across all heads
   
2. **All Heads Grid**: `attention_outputs/attention_heads_transformer_encoder.png`
   - Shows all attention heads side-by-side

3. **Interactive HTML**: `attention_outputs/attention_interactive.html`
   - Open in browser for interactive exploration

## 4. Customize Visualization

### Use Custom Input Data

```bash
# First, generate sample data
python examples/attention_examples/generate_sample_data.py

# Then visualize with custom data
neural visualize examples/attention_examples/basic_transformer.neural \
  --attention \
  --data data/basic_transformer_input.npy
```

### Add Token Labels

```bash
neural visualize examples/attention_examples/basic_transformer.neural \
  --attention \
  --tokens "The,quick,brown,fox,jumps,over,the,lazy,dog"
```

This makes the heatmap more interpretable by showing which tokens attend to which.

### Select Specific Layer

For multi-layer models:

```bash
neural visualize examples/attention_examples/multi_layer_transformer.neural \
  --attention \
  --layer transformer_encoder_1
```

### Select Specific Head

```bash
neural visualize examples/attention_examples/basic_transformer.neural \
  --attention \
  --head 0
```

### Use PyTorch Backend

```bash
neural visualize examples/attention_examples/basic_transformer.neural \
  --attention \
  --backend pytorch
```

## 5. Programmatic Usage

For more control, use the Python API:

```python
from neural.explainability.attention_visualizer import AttentionVisualizer
import numpy as np

# Create visualizer
visualizer = AttentionVisualizer(model=None, backend='tensorflow')

# Generate input
input_data = np.random.randn(1, 50, 256).astype(np.float32)

# Visualize from DSL file
results = visualizer.visualize_from_dsl(
    dsl_file='examples/attention_examples/basic_transformer.neural',
    input_data=input_data,
    backend='tensorflow',
    output_dir='my_attention_outputs'
)

# Analyze patterns
for layer_name, weights in results['attention_weights'].items():
    analysis = visualizer.analyze_attention_patterns(weights)
    print(f"{layer_name}: entropy={analysis['avg_entropy']:.3f}")
```

## 6. Run Demo Script

See all features in action:

```bash
python examples/attention_examples/visualize_attention_demo.py
```

This demonstrates:
- Basic visualization
- Token labels
- Multi-layer analysis
- Attention head comparison
- Interactive visualizations

## Common Issues

### No Attention Layers Found

**Problem**: "No attention layers found in the model"

**Solution**: Make sure your Neural DSL model includes:
- `TransformerEncoder` layers
- `MultiHeadAttention` layers
- Or other attention-based layers

Example:
```
network MyModel {
  input: (100, 512)
  layers:
    TransformerEncoder(num_heads=8, ff_dim=2048)
    ...
}
```

### Module Not Found

**Problem**: `ImportError: No module named 'matplotlib'`

**Solution**: Install visualization dependencies:
```bash
pip install matplotlib seaborn plotly
```

### Shape Mismatch

**Problem**: Input data shape doesn't match model

**Solution**: Ensure input data matches the model's expected input shape:
- Check `input:` in the .neural file
- Generate matching data with `generate_sample_data.py`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore different transformer architectures
- Try visualizing your own models
- Experiment with different attention patterns
- Compare attention across training epochs

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Ensure all dependencies are installed
3. Verify your .neural file syntax
4. Review the examples in this directory
5. Open an issue on GitHub

Happy visualizing! 🎨
