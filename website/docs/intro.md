---
sidebar_position: 1
slug: /
---

# Introduction to Neural DSL

Welcome to **Neural DSL** - the modern neural network programming language that simplifies deep learning development.

## What is Neural DSL?

Neural DSL is a domain-specific language designed for defining, training, debugging, and deploying neural networks. It provides:

- **Clean, declarative syntax** - No framework boilerplate
- **Cross-framework support** - TensorFlow, PyTorch, ONNX
- **Built-in debugging** - NeuralDbg with real-time tracing
- **Shape validation** - Catch errors before runtime
- **Production ready** - Export to multiple formats

## Quick Example

```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  
  layers:
    Conv2D(32, (3,3), "relu")
    MaxPooling2D((2,2))
    Flatten()
    Dense(128, "relu")
    Dropout(0.5)
    Output(10, "softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 15
    batch_size: 64
  }
}
```

## Why Neural DSL?

### 🎯 Simplicity
Write models in minutes with intuitive syntax. No need to remember framework-specific APIs.

### 🔄 Flexibility
Switch between TensorFlow, PyTorch, and ONNX with a single flag. No code rewrite needed.

### ✅ Reliability
Catch shape mismatches and configuration errors before runtime with automatic validation.

### 🐛 Debuggability
Built-in NeuralDbg provides real-time execution tracing, gradient analysis, and anomaly detection.

### 🚀 Productivity
Focus on model architecture, not boilerplate code. Export for deployment with one command.

## Getting Started

Ready to start? Head over to the [Installation Guide](getting-started/installation) to set up Neural DSL, or try it in the [Playground](/playground) right now!

## Community

Join our growing community:

- [Discord](https://discord.gg/KFku4KvS) - Chat with developers
- [GitHub](https://github.com/Lemniscate-world/Neural) - Source code and issues
- [Twitter](https://x.com/NLang4438) - Latest updates

## Support

- **Community Support**: Discord and GitHub Discussions
- **Documentation**: Comprehensive guides and API reference
- **Enterprise Support**: Available with Team and Enterprise plans

Let's build something amazing together! 🚀
