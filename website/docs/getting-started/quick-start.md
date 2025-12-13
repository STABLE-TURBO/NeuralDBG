---
sidebar_position: 2
---

# Quick Start

Get started with Neural DSL in 5 minutes.

## 1. Create Your First Model

Create a file named `mnist.neural`:

```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  
  layers:
    Conv2D(filters=32, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  
  train {
    epochs: 15
    batch_size: 64
    validation_split: 0.2
  }
}
```

## 2. Compile to Your Framework

### TensorFlow

```bash
neural compile mnist.neural --backend tensorflow --output mnist_tf.py
```

### PyTorch

```bash
neural compile mnist.neural --backend pytorch --output mnist_torch.py
```

### ONNX

```bash
neural export mnist.neural --format onnx --output mnist.onnx
```

## 3. Run Your Model

```bash
neural run mnist.neural --backend tensorflow
```

Neural DSL will:
1. ✅ Validate your model architecture
2. ✅ Check shape propagation
3. ✅ Generate framework-specific code
4. ✅ Execute training
5. ✅ Track experiments automatically

## 4. Visualize Architecture

```bash
neural visualize mnist.neural --format png
```

This generates:
- `architecture.png` - Model architecture diagram
- `shape_propagation.html` - Interactive shape flow
- `tensor_flow.html` - Detailed tensor transformations

## 5. Debug Your Model

Start the NeuralDbg debugger:

```bash
neural debug mnist.neural
```

Open http://localhost:8050 in your browser to see:
- Real-time execution traces
- Gradient flow analysis
- Memory and FLOP profiling
- Anomaly detection
- Dead neuron detection

## Common Commands

### Compile Only

```bash
neural compile mnist.neural --backend tensorflow
```

### Export for Deployment

```bash
# ONNX with optimization
neural export mnist.neural --format onnx --optimize

# TensorFlow Lite with quantization
neural export mnist.neural --format tflite --quantize

# TorchScript
neural export mnist.neural --backend pytorch --format torchscript
```

### Generate Documentation

```bash
neural docs mnist.neural --output model.md
```

### Clean Generated Files

```bash
neural clean --yes --all
```

## Try the No-Code Interface

Launch the visual model builder:

```bash
neural --no_code
```

Open http://localhost:8051 to build models with drag-and-drop.

## What's Next?

- [First Model Tutorial](first-model) - Detailed walkthrough
- [DSL Syntax Guide](/docs/concepts/dsl-syntax) - Learn the language
- [Tutorial Series](/docs/tutorial/basics) - In-depth guides
- [Examples](/docs/guides/mnist) - Real-world examples

## Need Help?

- [Documentation](/docs)
- [Discord Community](https://discord.gg/KFku4KvS)
- [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- [FAQ](/docs/faq)
