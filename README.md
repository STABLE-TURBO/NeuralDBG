<div align="center">
  <img src="https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b" alt="Neural Logo" width="200"/>
  <h1>Neural: A Neural Network Programming Language</h1>
  <p><strong>Simplify deep learning development with a powerful DSL, cross-framework support, and built-in debugging</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
  [![Discord](https://img.shields.io/badge/Chat-Discord-7289DA)](https://discord.gg/KFku4KvS)
  [![Pylint](https://github.com/Lemniscate-world/Neural/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/Lemniscate-world/Neural/actions/workflows/pylint.yml)
  [![Python package](https://github.com/Lemniscate-world/Neural/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Lemniscate-world/Neural/actions/workflows/python-package.yml)
  [![CodeQL Advanced](https://github.com/Lemniscate-world/Neural/actions/workflows/codeql.yml/badge.svg)](https://github.com/Lemniscate-world/Neural/actions/workflows/codeql.yml)
  [![CI](https://github.com/Lemniscate-world/Neural/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lemniscate-world/Neural/actions/workflows/ci.yml)
  [![Coverage](https://img.shields.io/codecov/c/github/Lemniscate-world/Neural)](https://codecov.io/gh/Lemniscate-world/Neural)

  <a href="https://www.producthunt.com/posts/neural-2?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-neural&#0045;2" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=945073&theme=dark&t=1742808173867" alt="Neural - DSL for defining, training, debugging neural networks | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</div>

[![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)](#contributing)

### Quick commands (end-to-end)

```
# Compile DSL to TensorFlow/PyTorch code
neural compile examples/mnist.neural --backend tensorflow --output mnist_tensorflow.py

# Run the generated script
python mnist_tensorflow.py

# Export for production deployment
neural export examples/mnist.neural --format onnx --optimize

# Generate docs (Markdown, optional PDF with --pdf)
neural docs examples/mnist.neural --output model.md

# Visualize architecture & shapes
neural visualize examples/mnist.neural --format png

# Clean generated artifacts (dry-run by default; add --yes to apply)
neural clean --yes --all
```


> ⚠️ **BETA STATUS**: Neural-dsl v0.2.9 is under active development—bugs may exist, feedback welcome! Not yet recommended for production use.

![Neural Demo](https://github.com/user-attachments/assets/ecbcce19-73df-4696-ace2-69e32d02709f)

## 📋 Table of Contents
- [Overview](#overview)
- [Pain Points Solved](#pain-points-solved)
- [Key Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Debugging with NeuralDbg](#-debugging-with-neuraldbg)
- [Cloud Integration](#cloud-integration)
- [Why Neural?](#why-neural)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [Community](#community)
- [Support](#support)

## Overview
Neural is a domain-specific language (DSL) designed for defining, training, debugging, and deploying neural networks. With declarative syntax, cross-framework support, and built-in execution tracing (NeuralDbg), it simplifies deep learning development whether via code, CLI, or a no-code interface.

##  Pain Points Solved

Neural addresses deep learning challenges across **Criticality** (how essential) and **Impact Scope** (how transformative):

| Criticality / Impact | Low Impact                  | Medium Impact                       | High Impact                         |
|----------------------|-----------------------------|-------------------------------------|-------------------------------------|
| **High**             |                             |                                     | - **Shape Mismatches**: Pre-runtime validation stops runtime errors.<br>- **Debugging Complexity**: Real-time tracing & anomaly detection. |
| **Medium**           |                             | - **Steep Learning Curve**: No-code GUI eases onboarding. | - **Framework Switching**: One-flag backend swaps.<br>- **HPO Inconsistency**: Unified tuning across frameworks. |
| **Low**              | - **Boilerplate**: Clean DSL syntax saves time. | - **Model Insight**: FLOPs & diagrams.<br>- **Config Fragmentation**: Centralized setup. |                                     |

### Why It Matters
- **Core Value**: Fix critical blockers like shape errors and debugging woes with game-changing tools.
- **Strategic Edge**: Streamline framework switches and HPO for big wins.
- **User-Friendly**: Lower barriers and enhance workflows with practical features.

## Feedback

Help us improve Neural DSL! Share your feedback: [Typeform link](https://form.typeform.com/to/xcibBdKD#name=xxxxx&email=xxxxx&phone_number=xxxxx&user_id=xxxxx&product_id=xxxxx&auth_code=xxxxx).



## Features

- **🤖 AI-Powered Development**: Build models using natural language! Describe what you want in any language, and Neural generates the DSL code automatically. [Learn more](docs/ai_integration_guide.md)
- **YAML-like Syntax**: Define models intuitively without framework boilerplate.
- **Shape Propagation**: Catch dimension mismatches *before* runtime.
  - ✅ Interactive shape flow diagrams included.
- **Multi-Framework HPO**: Optimize hyperparameters for both PyTorch and TensorFlow with a single DSL config (#434).
![Peek06-04-202517-00-ezgif com-speed](https://github.com/user-attachments/assets/5c4f51b5-e40f-47b3-872d-445f71c6582f)
- **Enhanced HPO Support**: Added HPO tracking for Conv2D kernel_size and improved ExponentialDecay parameter handling (v0.2.7).
- **Automated Issue Management**: Improved GitHub workflows for automatically creating and closing issues based on test results (v0.2.8).
- **Aquarium IDE**: Specialized IDE for neural network development with visual design and real-time shape propagation (v0.2.9).
- **Enhanced Dashboard UI**: Improved NeuralDbg dashboard with a more aesthetic dark theme design (#452).
- **Blog Support**: Infrastructure for blog content with markdown support and Dev.to integration (#445).
- **NeuralPaper.ai**: Interactive model visualization platform with annotation capabilities (in development).
- **Multi-Backend Export**: Generate code for **TensorFlow**, **PyTorch**, or **ONNX**.
- **🚀 Production-Ready Deployment**: Export to ONNX, TensorFlow Lite, TorchScript, and SavedModel with built-in optimization and quantization. Includes deployment configs for TensorFlow Serving and TorchServe. [Learn more](docs/deployment.md)
- **Training Orchestration**: Configure optimizers, schedulers, and metrics in one place.
- **Visual Debugging**: Render interactive 3D architecture diagrams.
- **Extensible**: Add custom layers/losses via Python plugins.
- **NeuralDbg**: Built-in Neural Network Debugger and Visualizer.
- **🔄 Fully Automated**: Automated releases, blog posts, tests, and maintenance. [Learn more](AUTOMATION_GUIDE.md)
- **No-Code Interface**: Quick Prototyping for researchers and an educational, accessible tool for beginners.

---

### **NeuralDbg: Built-in Neural Network Debugger**
NeuralDbg provides **real-time execution tracing, profiling, and debugging**, allowing you to visualize and analyze deep learning models in action. Now with an enhanced dark theme UI for better visualization (#452).

✅ **Real-Time Execution Monitoring** – Track activations, gradients, memory usage, and FLOPs.
![test_trace_graph](https://github.com/user-attachments/assets/15b1edd2-2643-4587-9843-aa4697ed2e4b)
![test_flops_memory_chart](https://github.com/user-attachments/assets/de1f6504-787b-4948-b543-fe3d2f8bfd74)
![test_trace_graph_stacked](https://github.com/user-attachments/assets/529fc487-fb31-48ad-bb11-b0c64ab330ed)
![test_trace_graph_heatmap](https://github.com/user-attachments/assets/debef7d5-9989-45da-ae91-7cef19aac2b0)
![test_anomaly_chart](https://github.com/user-attachments/assets/b57d3142-6da8-4d57-94f0-486d1797e92c)
![test_dead_neurons](https://github.com/user-attachments/assets/f4629b4f-2988-410e-8b49-3dde225f926f)
![test_gradient_chart](https://github.com/user-attachments/assets/ca6b9f20-7dd8-4c72-9ee8-a3f35af6208b)


✅ **Shape Propagation Debugging** – Visualize tensor transformations at each layer.
✅ **Gradient Flow Analysis** – Detect **vanishing & exploding gradients**.
✅ **Dead Neuron Detection** – Identify inactive neurons in deep networks.
✅ **Anomaly Detection** – Spot **NaNs, extreme activations, and weight explosions**.
✅ **Step Debugging Mode** – Pause execution and inspect tensors manually.


## Installation

**Prerequisites**: Python 3.8+, pip

### Option 1: Install from PyPI (Recommended)

```bash
# Install the latest stable version
pip install neural-dsl

# Or specify a version
pip install neural-dsl==0.2.9  # Latest version with Aquarium IDE integration
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Define a Model

Create a file named `mnist.neural` with your model definition:

```yaml
network MNISTClassifier {
  input: (28, 28, 1)  # Channels-last format

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

### 2. Run or Compile the Model

```bash
# Generate and run TensorFlow code
neural run mnist.neural --backend tensorflow --output mnist_tf.py

# Or generate and run PyTorch code
neural run mnist.neural --backend pytorch --output mnist_torch.py
```

### 3. Visualize Architecture

```bash
neural visualize mnist.neural --format png
```

This will create visualization files for inspecting the network structure and shape propagation:
- `architecture.png`: Visual representation of your model
- `shape_propagation.html`: Interactive tensor shape flow diagram
- `tensor_flow.html`: Detailed tensor transformations

### 4. Debug with NeuralDbg

```bash
neural debug mnist.neural
```

Open your browser to http://localhost:8050 to monitor execution traces, gradients, and anomalies interactively.

### 5. Use the No-Code Interface

```bash
neural --no_code
```




Open your browser to http://localhost:8051 to build and compile models via a graphical interface.

### 6. Export for Deployment

```bash
# Export to ONNX with optimization
neural export mnist.neural --format onnx --optimize

# Export to TensorFlow Lite with quantization
neural export mnist.neural --backend tensorflow --format tflite --quantize --quantization-type int8

# Export to TorchScript for PyTorch production
neural export mnist.neural --backend pytorch --format torchscript

# Generate TensorFlow Serving deployment
neural export mnist.neural --backend tensorflow --format savedmodel --deployment tfserving --model-name mnist_model
```

For complete deployment guides, see [docs/deployment.md](docs/deployment.md) and [docs/DEPLOYMENT_QUICK_START.md](docs/DEPLOYMENT_QUICK_START.md).

---

## **🛠 Debugging with NeuralDbg**

### **🔹 1️⃣ Start Real-Time Execution Tracing**
```bash
neural debug mnist.neural
```
**Features:**


✅ Layer-wise execution trace
✅ Memory & FLOP profiling
✅ Live performance monitoring

### **🔹 2️⃣ Analyze Gradient Flow**
```bash
neural debug --gradients mnist.neural
```
 **Detect vanishing/exploding gradients** with interactive charts.

### **🔹 3️⃣ Identify Dead Neurons**
```bash
neural debug --dead-neurons mnist.neural
```
🛠 **Find layers with inactive neurons (common in ReLU networks).**

### **🔹 4️⃣ Detect Training Anomalies**
```bash
neural debug --anomalies mnist.neural
```
 **Flag NaNs, weight explosions, and extreme activations.**

### **🔹 5️⃣ Step Debugging (Interactive Tensor Inspection)**
```bash
neural debug --step mnist.neural
```
🔍 **Pause execution at any layer and inspect tensors manually.**

---

## **☁️ Cloud Integration**

Neural now supports running in cloud environments like Kaggle, Google Colab, and AWS SageMaker, with both direct execution in the cloud and remote control from your local terminal.

### **🔹 1️⃣ Run in Kaggle or Colab**

In your Kaggle notebook or Google Colab:

```python
# Install Neural DSL
!pip install neural-dsl==0.2.9

# Import the cloud module
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()
print(f"Detected environment: {executor.environment}")
print(f"GPU available: {executor.is_gpu_available}")

# Define a model
dsl_code = """
network MnistCNN {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
"""

# Compile and run the model
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST')

# Start the NeuralDbg dashboard with ngrok tunnel
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard URL: {dashboard_info['tunnel_url']}")
```

### **🔹 2️⃣ Run in AWS SageMaker**

```python
# In your SageMaker notebook
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()  # Automatically detects SageMaker environment

# Define and run your model as above
```

### **🔹 3️⃣ Connect from Local Terminal**

Control cloud environments directly from your local terminal:

```bash
# Connect to a cloud platform
neural cloud connect kaggle

# Start an interactive shell connected to Kaggle
neural cloud connect kaggle --interactive

# Execute a Neural DSL file on Kaggle
neural cloud execute kaggle my_model.neural

# Run Neural in cloud mode with remote access
neural cloud run --setup-tunnel
```

### **🔹 4️⃣ Example Notebooks**

Ready-to-use notebooks are available for:
- [Kaggle](neural/cloud/examples/neural_kaggle_example.ipynb)
- [Google Colab](neural/cloud/examples/neural_colab_example.ipynb)

---

## 🔬 Experiment Tracking

Neural DSL now automatically tracks all your training runs! Every model you train is logged with hyperparameters, metrics, and artifacts.

### Features
- **Automatic Logging**: No manual tracking code needed
- **Metrics History**: Loss, accuracy, and custom metrics tracked per epoch
- **Hyperparameter Logging**: Optimizer, learning rate, and model config automatically saved
- **Experiment Storage**: All experiments saved to `neural_experiments/` directory
- **CLI Commands**: View, compare, and plot experiments

### Usage
```bash
# Run a model (tracking is automatic)
neural run examples/mnist.neural --backend tensorflow

# List all experiments  
neural track list

# Show experiment details
neural track show <experiment_id>

# Plot metrics
neural track plot <experiment_id>

# Compare experiments
neural track compare <exp_id_1> <exp_id_2>
```

All generated code automatically includes tracking logic - no setup required!

---

## 🏗️ Neural Ecosystem Projects

Neural is part of a larger ecosystem within the **λ-S (Lambda-Section)** startup. The following projects have been extracted to separate repositories:

### Related Projects
- **Aquarium** - Tauri-based desktop IDE for Neural DSL
  - Specialized development environment with visual design tools
  - Real-time shape propagation visualization
  - Repository: [TBD]

- **NeuralPaper.ai** - Interactive model visualization platform
  - Web-based model annotation and sharing
  - Collaborative features for research
  - Repository: [TBD]

- **Neural-Research** - Historical neural network paper analysis
  - McCulloch-Pitts 1943 paper annotations
  - Jupyter notebooks and research content
  - Repository: [TBD]

- **Lambda-sec Models** - Production neural network models for λ-S startup
  - MathLang models (Newton differential equations)
  - Transformer architectures with training data
  - Repository: [TBD]

These projects remain integrated with Neural through shared APIs and tooling.

---

##  Why Neural?

| Feature               | Neural      | Raw TensorFlow/PyTorch |
|-----------------------|-------------|-------------------------|
| Shape Validation      | ✅ Auto     | ❌ Manual               |
| Framework Switching   | 1-line flag | Days of rewriting       |
| Architecture Diagrams | Built-in    | Third-party tools       |
| Training Config       | Unified     | Fragmented configs      |
| Experiment Tracking   | ✅ Automatic | ❌ Manual setup         |


### **🔄 Cross-Framework Code Generation**
| Neural DSL          | TensorFlow Output          | PyTorch Output            |
|---------------------|----------------------------|---------------------------|
| `Conv2D(filters=32)`| `tf.keras.layers.Conv2D(32)`| `nn.Conv2d(in_channels, 32)` |
| `Dense(units=128)`  | `tf.keras.layers.Dense(128)`| `nn.Linear(in_features, 128)`|

##  Benchmarks
| Task                 | Neural | Baseline (TF/PyTorch) |
|----------------------|--------|-----------------------|
| MNIST Training       | 1.2x ⚡| 1.0x                  |
| Debugging Setup      | 5min 🕒| 2hr+                  |

##  Documentation

### Core Documentation
- [DSL Documentation](docs/dsl.md)
- [AI Integration Guide](docs/ai_integration_guide.md) - 🤖 **NEW!** Natural language to DSL
- [Automation Guide](AUTOMATION_GUIDE.md) - 🔄 **NEW!** Automated releases and blog posts
- [Contributing Guide](CONTRIBUTING.md) - **NEW!** How to contribute
- [Blog](docs/blog/README.md)

### Explore advanced features:
- [Custom Layers Guide](docs/examples/custom_layers.md) (Coming soon)
- [ONNX Export Tutorial](docs/examples/onnx_export.md) (Coming soon)
- [Training Configuration](docs/examples/training_config.md) (Coming soon)
- [NeuralDbg Debugging Features](docs/examples/neuraldbg_features.md) (Coming soon)
- [HPO Configuration Guide](docs/examples/hpo_guide.md) (Coming soon)

##  Examples

Explore common use cases in `examples/` with step-by-step guides in `docs/examples/`:
- [MNIST Classifier Guide](docs/examples/mnist_guide.md)
- [Sentiment Analysis Guide](docs/examples/sentiment_guide.md)
- [Transformer for NLP Guide](docs/examples/transformer_guide.md)

## 🕸 Architecture Graphs

![classes](https://github.com/Lemniscate-world/Neural/blob/main/classes.png)
![packages](https://github.com/Lemniscate-world/Neural/blob/main/packages.png)

*Note: You may need to zoom in to see details in these architecture diagrams.*

## Repository Structure

The Neural repository is organized into the following main directories:

- **`docs/`**: Documentation files
- **`examples/`**: Example Neural DSL files
- **`neural/`**: Main source code
  - **`neural/cli/`**: Command-line interface
  - **`neural/parser/`**: Neural DSL parser
  - **`neural/shape_propagation/`**: Shape propagation and validation
  - **`neural/code_generation/`**: Code generation for different backends
  - **`neural/visualization/`**: Visualization tools
  - **`neural/dashboard/`**: NeuralDbg dashboard
  - **`neural/hpo/`**: Hyperparameter optimization
  - **`neural/cloud/`**: Cloud integration (Kaggle, Colab, SageMaker)
  - **`neural/tracking/`**: Experiment tracking system
- **`profiler/`**: Performance profiling tools
- **`tests/`**: Test suite

For a detailed explanation of the repository structure, see [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md).

Each directory contains its own README with detailed documentation:

- [neural/cli](neural/cli/README.md): Command-line interface
- [neural/parser](neural/parser/README.md): Neural DSL parser
- [neural/code_generation](neural/code_generation/README.md): Code generation
- [neural/shape_propagation](neural/shape_propagation/README.md): Shape propagation
- [neural/visualization](neural/visualization/README.md): Visualization tools
- [neural/dashboard](neural/dashboard/README.md): NeuralDbg dashboard
- [neural/hpo](neural/hpo/README.md): Hyperparameter optimization
- [neural/cloud](neural/cloud/README.md): Cloud integration
- [neural/tracking](neural/tracking/README.md): Experiment tracking
- [profiler](profiler/README.md): Performance profiling tools
- [docs](docs/README.md): Documentation
- [examples](examples/README.md): Example models
- [tests](tests/README.md): Test suite


## Reproducibility

To get deterministic results across Python, NumPy, PyTorch, and TensorFlow, use the built-in seeding utility:

```python
from neural.utils.seed import set_seed
set_seed(42)
```

This sets Python's random, NumPy, PyTorch (including CUDA if present), and TensorFlow seeds.


## Optional Dependencies

These packages enable optional features and backends. Install only what you need:

- torch: PyTorch backend for code generation and execution
- tensorflow: TensorFlow backend for code generation and execution
- onnx: Export or interop with ONNX
- jax: Experimental backend and numerical utilities
- optuna: Hyperparameter optimization
- dash, flask: Dashboard and API/visualization components
- scikit-learn: Metrics, datasets, and HPO utilities

Examples:

```bash
# PyTorch backend
pip install torch

# TensorFlow backend
pip install tensorflow

# ONNX export support
pip install onnx

# HPO and metrics utilities
pip install optuna scikit-learn

# Dashboard/visualization
pip install dash flask
```




---


##  Contributing

We welcome contributions! See our:
- [Contributing Guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Roadmap](ROADMAP.md)

To set up a development environment:
```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -r requirements-dev.txt  # Includes linter, formatter, etc.
pre-commit install  # Auto-format code on commit
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Lemniscate-world/Neural&type=Timeline)](https://www.star-history.com/#Lemniscate-world/Neural&Timeline)

## Support

If you find Neural useful, please consider supporting the project:

- ⭐ **Star the repository**: Help us reach more developers by starring the project on GitHub
- 🔄 **Share with others**: Spread the word on social media, blogs, or developer communities
- 🐛 **Report issues**: Help us improve by reporting bugs or suggesting features
- 🤝 **Contribute**: Submit pull requests to help us enhance Neural (see [Contributing](#contributing))

### Repository Status

This repository has been cleaned and optimized for better performance. Large files have been removed from the Git history to ensure a smoother experience when cloning or working with the codebase.

## Community

Join our growing community of developers and researchers:

- [Discord Server](https://discord.gg/KFku4KvS): Chat with developers, get help, and share your projects
- [Twitter @NLang4438](https://x.com/NLang4438): Follow for updates, announcements, and community highlights
- [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions): Participate in discussions about features, use cases, and best practices

<div align="center">
  <img src="https://github.com/user-attachments/assets/9edd42b3-dd23-4f4a-baad-422e690d687c" alt="Neural Logo" width="150"/>
  <p><em>Building the future of neural network development, one line of DSL at a time.</em></p>
</div>


## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for detailed information on how to get started.

**Quick Start:**
1. Fork and clone the repository
2. Install in development mode: `pip install -e .`
3. Run tests: `python -m pytest tests/ -v`
4. Make your changes and submit a pull request!

**Areas to Contribute:**
- 🤖 AI Integration enhancements
- 📝 Documentation improvements
- 🐛 Bug fixes
- ✨ New features (see [ROADMAP.md](ROADMAP.md) for priorities)
- 📚 More examples

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

### Development Workflow (Legacy)

This section outlines a minimal, fast local workflow to lint, type‑check, test, and audit changes before opening a PR.

### 1) Environment setup (Windows PowerShell)

- Create and activate a virtual environment

```
python -m venv .venv
.\.venv\Scripts\Activate
```

- Install the project (editable) and dev tools used by CI

```
pip install -e .
pip install ruff mypy pip-audit pytest
```

### 2) Common checks (fast)

- Lint (Ruff)

```
python -m ruff check .
```

- Type check (mypy)

Fast, scoped type check for currently‑hardened modules:
```
python -m mypy neural/code_generation neural/utils
```
Full project type check (may show many findings; tighten gradually):
```
python -m mypy .
```

- Tests (targeted and full)

Run fast, targeted tests:
```
python -m pytest -q tests/test_seed.py tests/code_generator/test_policy_and_parity.py tests/code_generator/test_policy_helpers.py -rA
```
Run full test suite (may require optional deps such as torch/tensorflow/onnx):
```
python -m pytest -q -rA
```

- Supply‑chain audit

```
python -m pip_audit -l --progress-spinner off
```

### 3) Commit & PR hygiene

- Keep PRs small and focused; include context in the description.
- Run lint, type check (scoped or full), tests, and pip‑audit locally before pushing.
- Do not commit secrets/keys. Use environment variables; keep .env or credentials out of Git.
- Follow the shape/policy rules in codegen; add or update tests for any policy changes.

### 4) Optional dependencies

Install only what you need for the tests you are running (examples):
```
# PyTorch backend
your-shell> pip install torch
# TensorFlow backend
your-shell> pip install tensorflow
# ONNX export
your-shell> pip install onnx
```

If you have questions or want guidance on tightening typing or adding new policy checks, open a discussion or draft PR.
