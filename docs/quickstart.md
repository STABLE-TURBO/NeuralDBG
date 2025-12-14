# Neural DSL Quickstart Guide

Get up and running with Neural DSL in under 5 minutes.

## Installation

```bash
# Minimal installation
pip install neural-dsl

# Full installation with all features
pip install neural-dsl[full]

# Selective installation
pip install neural-dsl[backends]      # TensorFlow, PyTorch, ONNX
pip install neural-dsl[hpo]           # Hyperparameter optimization
pip install neural-dsl[dashboard]     # NeuralDbg interface
pip install neural-dsl[visualization] # Graphviz, Plotly, Matplotlib
```

## Your First Model

### 1. Create a Model File

Create `mnist.neural`:

```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  
  layers:
    Conv2D(32, (3,3), "relu")
    MaxPooling2D((2,2))
    Conv2D(64, (3,3), "relu")
    MaxPooling2D((2,2))
    Flatten()
    Dense(128, "relu")
    Dropout(0.5)
    Output(10, "softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 10
    batch_size: 64
  }
}
```

### 2. Compile to Python

```bash
# TensorFlow
neural compile mnist.neural --backend tensorflow --output mnist_tf.py

# PyTorch
neural compile mnist.neural --backend pytorch --output mnist_pt.py

# ONNX
neural compile mnist.neural --backend onnx --output model.onnx
```

### 3. Visualize Architecture

```bash
# Generate interactive HTML visualization
neural visualize mnist.neural --format html

# Generate PNG diagram
neural visualize mnist.neural --format png
```

This creates:
- `architecture.svg` - Network architecture diagram
- `shape_propagation.html` - Interactive shape analysis
- `tensor_flow.html` - Data flow animation

### 4. Debug Your Model

```bash
# Launch NeuralDbg dashboard
neural debug mnist.neural --dashboard --port 8050
```

Open `http://localhost:8050` to see:
- Real-time execution traces
- Gradient flow visualization
- Dead neuron detection
- Memory profiling

### 5. Experiment Tracking

```bash
# Initialize experiment tracking
neural track init mnist_experiment

# Train and log metrics
neural run mnist.neural --backend tensorflow

# View experiment results
neural track list
neural track show <experiment_id>
neural track plot <experiment_id>
```

## Common Commands

```bash
# Show help
neural --help
neural compile --help

# Check version
neural --version

# Compile with options
neural compile model.neural \
  --backend tensorflow \
  --output model.py \
  --dry-run

# Run with HPO
neural compile model.neural --hpo --backend pytorch

# Clean generated files
neural clean --yes

# Export for deployment
neural export model.neural --format onnx --optimize
```

## Feature Groups Reference

| Group | Installation | Use Case |
|-------|--------------|----------|
| Core | `pip install neural-dsl` | DSL parsing only |
| backends | `[backends]` | TensorFlow, PyTorch, ONNX |
| hpo | `[hpo]` | Hyperparameter optimization |
| automl | `[automl]` | Neural Architecture Search |
| dashboard | `[dashboard]` | NeuralDbg interface |
| visualization | `[visualization]` | Charts and diagrams |
| cloud | `[cloud]` | Cloud platform support |
| integrations | `[integrations]` | MLflow, W&B, etc. |
| full | `[full]` | All features |

## Next Steps

- [DSL Language Reference](dsl.md) - Complete syntax
- [Examples](../examples/README.md) - Working models
- [CLI Reference](cli.md) - All commands
- [Getting Started Guide](../GETTING_STARTED.md) - Detailed walkthrough
- [Deployment Guide](deployment.md) - Production export

## Quick Examples

### Image Classification

```yaml
network ImageClassifier {
  input: (224, 224, 3)
  layers:
    Conv2D(64, (3,3), "relu")
    BatchNormalization()
    MaxPooling2D((2,2))
    Conv2D(128, (3,3), "relu")
    BatchNormalization()
    MaxPooling2D((2,2))
    Flatten()
    Dense(512, "relu")
    Dropout(0.5)
    Output(1000, "softmax")
  optimizer: Adam(learning_rate=0.001)
  loss: "categorical_crossentropy"
}
```

### Text Classification

```yaml
network SentimentAnalyzer {
  input: (100,)
  layers:
    Embedding(vocab_size=10000, embedding_dim=128)
    LSTM(128)
    Dense(64, "relu")
    Dropout(0.5)
    Output(3, "softmax")
  optimizer: Adam(learning_rate=0.001)
  loss: "categorical_crossentropy"
}
```

### Transformer

```yaml
network TransformerModel {
  input: (512,)
  layers:
    Embedding(vocab_size=10000, embedding_dim=256)
    TransformerEncoder(
      num_heads=8,
      ff_dim=512,
      num_blocks=3
    )
    GlobalAveragePooling1D()
    Dense(128, "relu")
    Output(10, "softmax")
  optimizer: Adam(learning_rate=0.001)
  loss: "sparse_categorical_crossentropy"
}
```

## Troubleshooting

**Command not found?**
```bash
# Use module invocation
python -m neural.cli compile model.neural
```

**Import errors?**
```bash
# Install specific feature group
pip install neural-dsl[backends]
```

**Visualization fails?**
```bash
# Install graphviz system package
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# Download from graphviz.org
```

## Getting Help

- **Documentation**: [docs/](.)
- **GitHub Issues**: [Report bugs](https://github.com/Lemniscate-world/Neural/issues)
- **Discussions**: [Ask questions](https://github.com/Lemniscate-world/Neural/discussions)
- **Discord**: [Join community](https://discord.gg/KFku4KvS)

Ready to build? Check out the [examples](../examples/README.md) directory for more models!
