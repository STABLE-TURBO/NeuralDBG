<div align="center">
  <img src="https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b" alt="Neural Logo" width="200"/>
  <h1>Neural: A Neural Network Programming Language</h1>
  <p><strong>Simplify deep learning development with a powerful DSL, cross-framework support, and built-in debugging</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
  [![PyPI version](https://img.shields.io/pypi/v/neural-dsl.svg)](https://pypi.org/project/neural-dsl/)
  [![Downloads](https://img.shields.io/pypi/dm/neural-dsl.svg)](https://pypi.org/project/neural-dsl/)
  
  [![CI](https://github.com/Lemniscate-world/Neural/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lemniscate-world/Neural/actions/workflows/ci.yml)
  [![Python package](https://github.com/Lemniscate-world/Neural/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Lemniscate-world/Neural/actions/workflows/python-package.yml)
  [![Pylint](https://github.com/Lemniscate-world/Neural/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/Lemniscate-world/Neural/actions/workflows/pylint.yml)
  [![CodeQL Advanced](https://github.com/Lemniscate-world/Neural/actions/workflows/codeql.yml/badge.svg)](https://github.com/Lemniscate-world/Neural/actions/workflows/codeql.yml)
  [![Coverage](https://img.shields.io/codecov/c/github/Lemniscate-world/Neural)](https://codecov.io/gh/Lemniscate-world/Neural)
  
  [![Discord](https://img.shields.io/badge/Chat-Discord-7289DA)](https://discord.gg/KFku4KvS)
  [![Twitter Follow](https://img.shields.io/twitter/follow/NLang4438?style=social)](https://x.com/NLang4438)
  [![GitHub Discussions](https://img.shields.io/github/discussions/Lemniscate-world/Neural)](https://github.com/Lemniscate-world/Neural/discussions)

  <a href="https://www.producthunt.com/posts/neural-2?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-neural&#0045;2" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=945073&theme=dark&t=1742808173867" alt="Neural - DSL for defining, training, debugging neural networks | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</div>

---

## 🚀 **What's New in v0.3.0**

<div align="center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
  
### 🎉 **Major Release: Neural v0.3.0 is Here!**

**Transform your neural network development with cutting-edge automation and AI-powered tools**

</div>

### ✨ **Key Highlights**

🤖 **AI-Powered Development**
- Natural language to DSL code generation - describe your model in plain English (or any language)
- Intelligent model suggestions based on your use case
- Automatic architecture optimization recommendations
- [Learn more →](docs/ai_integration_guide.md)

🚀 **Production-Ready Deployment**
- One-command export to ONNX, TensorFlow Lite, TorchScript, SavedModel
- Built-in optimization and quantization (INT8, FP16)
- Deployment configs for TensorFlow Serving and TorchServe
- Cloud-ready containerization support
- [Learn more →](docs/deployment.md)

🔄 **Complete Automation Suite**
- Automated CI/CD pipelines with GitHub Actions
- Automatic versioning and release management
- Automated blog post generation and publishing
- Scheduled maintenance and dependency updates
- [Learn more →](AUTOMATION_GUIDE.md)

📊 **Marketing Automation**
- Automated social media integration (Twitter, Dev.to)
- Product Hunt launch automation
- Documentation auto-generation with examples
- Community engagement tracking

---

[![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)](#contributing)

## 📋 Table of Contents
- [Overview](#overview)
- [Feature Showcase](#-feature-showcase)
- [Aquarium IDE](#-aquarium-ide)
- [Neural DSL vs Raw Frameworks](#-neural-dsl-vs-raw-frameworks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pain Points Solved](#pain-points-solved)
- [Debugging with NeuralDbg](#-debugging-with-neuraldbg)
- [Cloud Integration](#cloud-integration)
- [Documentation](#documentation)
- [Examples](#examples)
- [Community & Support](#-community--support)
- [Contributing](#-contributing)
- [Star History](#star-history)

---

## Overview

Neural is a **domain-specific language (DSL)** designed for defining, training, debugging, and deploying neural networks. With declarative syntax, cross-framework support, and built-in execution tracing (NeuralDbg), it simplifies deep learning development whether via code, CLI, or a no-code interface.

**Why Neural?** Stop wrestling with boilerplate code, shape mismatches, and framework lock-in. Neural provides a unified, intuitive interface for modern deep learning—from prototype to production.

![Neural Demo](https://github.com/user-attachments/assets/ecbcce19-73df-4696-ace2-69e32d02709f)

---

## 🌟 **Feature Showcase**

### 🤖 **AI-Powered Model Generation**
Describe your model in natural language and let Neural generate the DSL code for you:

```python
# Just describe what you want
from neural.ai import generate_model

model_code = generate_model("""
    I need a CNN for image classification with 10 classes.
    Use 3 convolutional layers with increasing filters (32, 64, 128).
    Add dropout for regularization.
""")
```

**Result:** Complete, optimized Neural DSL code ready to compile!

[📖 Read the AI Integration Guide →](docs/ai_integration_guide.md)

---

### 🔍 **Real-Time Debugging with NeuralDbg**

Monitor your model's behavior in real-time with interactive visualizations:

<div align="center">

**Execution Trace Graph**
![test_trace_graph](https://github.com/user-attachments/assets/15b1edd2-2643-4587-9843-aa4697ed2e4b)

**FLOPs & Memory Usage**
![test_flops_memory_chart](https://github.com/user-attachments/assets/de1f6504-787b-4948-b543-fe3d2f8bfd74)

**Gradient Flow Analysis**
![test_gradient_chart](https://github.com/user-attachments/assets/ca6b9f20-7dd8-4c72-9ee8-a3f35af6208b)

**Dead Neuron Detection**
![test_dead_neurons](https://github.com/user-attachments/assets/f4629b4f-2988-410e-8b49-3dde225f926f)

**Anomaly Detection**
![test_anomaly_chart](https://github.com/user-attachments/assets/b57d3142-6da8-4d57-94f0-486d1797e92c)

</div>

**Features:**
- ✅ Real-time execution monitoring
- ✅ Gradient flow visualization
- ✅ Dead neuron detection
- ✅ NaN/Inf anomaly detection
- ✅ Memory & FLOPs profiling
- ✅ Interactive step debugging

```bash
neural debug mnist.neural
# Open http://localhost:8050 for live dashboard
```

---

### 🎯 **Shape Propagation & Validation**

Catch dimension mismatches **before runtime**:

![Peek06-04-202517-00-ezgif com-speed](https://github.com/user-attachments/assets/5c4f51b5-e40f-47b3-872d-445f71c6582f)

```bash
neural visualize mnist.neural --format png
# Generates: shape_propagation.html, tensor_flow.html
```

**Benefits:**
- 🚫 No more runtime shape errors
- 📊 Visual tensor flow diagrams
- ⚡ Instant validation feedback

---

### 🔄 **Cross-Framework Support**

Write once, deploy anywhere. Switch frameworks with a single flag:

```bash
# Same model, different frameworks
neural compile model.neural --backend tensorflow
neural compile model.neural --backend pytorch
neural compile model.neural --backend onnx
```

**Supported Backends:**
- TensorFlow / Keras
- PyTorch
- ONNX (cross-platform)

---

### 🚀 **Production Deployment**

Export optimized models for production with built-in optimization:

```bash
# Export to ONNX with optimization
neural export model.neural --format onnx --optimize

# TensorFlow Lite with INT8 quantization
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type int8

# TorchScript for PyTorch production
neural export model.neural --backend pytorch --format torchscript

# TensorFlow Serving deployment
neural export model.neural --backend tensorflow --format savedmodel --deployment tfserving
```

**Deployment Options:**
- ✅ ONNX Runtime
- ✅ TensorFlow Lite (mobile/edge)
- ✅ TorchScript (PyTorch production)
- ✅ TensorFlow Serving
- ✅ TorchServe

[📖 Full Deployment Guide →](docs/deployment.md)

---

### 📊 **Automatic Experiment Tracking**

Every training run is automatically logged—no manual setup required:

```bash
# Run model (tracking is automatic)
neural run model.neural --backend tensorflow

# View all experiments
neural track list

# Compare experiments
neural track compare exp_1 exp_2

# Plot metrics
neural track plot exp_1
```

**What's Tracked:**
- Hyperparameters
- Loss & accuracy per epoch
- Training time & resources
- Model architecture
- Dataset info

---

### 🎨 **No-Code Interface**

Build models visually with the no-code GUI:

```bash
neural --no_code
# Open http://localhost:8051
```

**Perfect for:**
- 🎓 Students & educators
- 🔬 Rapid prototyping
- 👥 Non-technical stakeholders
- 🧪 Quick experiments

---

## 🎨 **Aquarium IDE**

**A Modern Web-Based IDE for Neural DSL** - Write, compile, train, and debug models in your browser!

<div align="center">

![Aquarium IDE](https://github.com/user-attachments/assets/aquarium-ide-banner.png)

**[📥 Download](#installation) • [📚 Documentation](docs/aquarium/README.md) • [🎥 Video Tutorials](docs/aquarium/video-tutorials.md) • [🚀 Quick Start](docs/aquarium/user-manual.md#getting-started)**

</div>

### ✨ **Key Features**

🎨 **Intuitive DSL Editor**
- Syntax-highlighted code editor
- Real-time parse validation
- 8+ built-in example models
- Dark theme for comfort

🔧 **Multi-Backend Support**
- TensorFlow, PyTorch, ONNX
- One-click backend switching
- Consistent API across frameworks

🚀 **One-Click Training**
- Compile DSL to Python instantly
- Execute training in-browser
- Real-time console output
- Live metrics monitoring

📊 **Dataset Integration**
- Built-in: MNIST, CIFAR10, CIFAR100, ImageNet
- Custom dataset support
- Auto-download on first use

🐛 **Integrated Debugging**
- NeuralDbg integration
- Layer-by-layer inspection
- Gradient flow visualization
- Performance profiling

📦 **Export & Share**
- Save compiled scripts
- Open in external IDE
- Version control ready
- Share with team

### 🚀 **Quick Start**

**1. Install Aquarium**:
```bash
# Install with dashboard support
pip install neural-dsl[dashboard]

# Or full package with all features
pip install neural-dsl[full]
```

**2. Launch**:
```bash
# Start Aquarium IDE
python -m neural.aquarium.aquarium

# Custom port
python -m neural.aquarium.aquarium --port 8053

# Debug mode
python -m neural.aquarium.aquarium --debug
```

**3. Open Browser**:
```
http://localhost:8052
```

**4. Build Your First Model**:
- Click **"Load Example"** to insert a pre-built model
- Click **"Parse DSL"** to validate syntax
- Select **Backend**: TensorFlow, **Dataset**: MNIST, **Epochs**: 5
- Click **"Compile"** to generate Python code
- Click **"Run"** to start training
- Watch real-time logs and metrics!

### 📥 **Installation Options**

| Method | Command | Use Case |
|--------|---------|----------|
| **Quick Install** | `pip install neural-dsl[dashboard]` | Most users (includes Aquarium) |
| **Full Install** | `pip install neural-dsl[full]` | All features + backends |
| **From Source** | `git clone ...` + `pip install -e ".[dashboard]"` | Development |

### 📚 **Documentation**

| Guide | Description | Link |
|-------|-------------|------|
| **Installation Guide** | Complete setup instructions | [📖 Read](docs/aquarium/installation.md) |
| **User Manual** | Comprehensive usage guide | [📖 Read](docs/aquarium/user-manual.md) |
| **Keyboard Shortcuts** | Productivity shortcuts | [📖 Read](docs/aquarium/keyboard-shortcuts.md) |
| **Troubleshooting** | Common issues & solutions | [📖 Read](docs/aquarium/troubleshooting.md) |
| **Architecture** | System design & components | [📖 Read](docs/aquarium/architecture.md) |
| **Plugin Development** | Extend Aquarium | [📖 Read](docs/aquarium/plugin-development.md) |
| **Video Tutorials** | Visual learning resources | [📖 Read](docs/aquarium/video-tutorials.md) |

### 🎥 **Video Tutorials** (Coming Soon)

Learn Aquarium through hands-on video tutorials:

| Tutorial | Duration | Level | Link |
|----------|----------|-------|------|
| Installation & Setup | 5 min | Beginner | Coming Soon |
| Your First Model | 10 min | Beginner | Coming Soon |
| Compiling Models | 8 min | Beginner | Coming Soon |
| Running Training | 12 min | Beginner | Coming Soon |
| Debugging Techniques | 25 min | Intermediate | Coming Soon |
| Plugin Development | 45 min | Advanced | Coming Soon |

**Full tutorial library**: [Video Tutorials Guide](docs/aquarium/video-tutorials.md)

### 🎯 **Use Cases**

**For Students & Educators**:
- Learn deep learning without framework complexity
- Focus on concepts, not boilerplate
- Visual feedback and debugging

**For Researchers**:
- Rapid prototyping
- Easy architecture experimentation
- Multi-framework compatibility

**For Engineers**:
- Quick model development
- Production-ready code generation
- Team collaboration

### 💡 **Example Workflow**

```neural
# Write DSL in editor
network MNISTClassifier {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

**Then**:
1. Parse DSL ✓
2. Select TensorFlow backend ✓
3. Choose MNIST dataset ✓
4. Compile to Python ✓
5. Run training ✓
6. Export trained model ✓

### 🔗 **Quick Links**

- **Main Documentation**: [Aquarium README](docs/aquarium/README.md)
- **5-Minute Quick Start**: [Getting Started](docs/aquarium/user-manual.md#getting-started)
- **Troubleshooting**: [Common Issues](docs/aquarium/troubleshooting.md)
- **GitHub Issues**: [Report Bugs](https://github.com/Lemniscate-world/Neural/issues)
- **Discord**: [Join Community](https://discord.gg/KFku4KvS)

### 🌟 **Why Aquarium?**

| Traditional Approach | With Aquarium IDE |
|---------------------|-------------------|
| Install Python, TF/PyTorch, Jupyter | `pip install neural-dsl[dashboard]` |
| Write 100+ lines of boilerplate | 15 lines of DSL |
| Manual dataset loading | Built-in dataset integration |
| Command-line compilation | One-click compile & run |
| Scattered debugging tools | Integrated debugger |
| Hours of setup | Minutes to first model |

**Result**: Focus on models, not infrastructure!

---

## 📊 **Neural DSL vs Raw Frameworks**

### **Quick Comparison**

| Feature | Neural DSL | Raw TensorFlow/PyTorch |
|---------|------------|------------------------|
| **Shape Validation** | ✅ Automatic, pre-runtime | ❌ Manual, runtime errors |
| **Framework Switching** | ✅ One flag | ❌ Days of rewriting |
| **Boilerplate Code** | ✅ Minimal DSL syntax | ❌ Verbose imports & setup |
| **Built-in Debugging** | ✅ NeuralDbg dashboard | ❌ Custom tools required |
| **Architecture Visualization** | ✅ Auto-generated diagrams | ❌ Third-party tools |
| **Experiment Tracking** | ✅ Automatic logging | ❌ Manual setup (MLflow, etc.) |
| **Training Configuration** | ✅ Unified in DSL | ❌ Scattered across files |
| **Deployment Export** | ✅ One command | ❌ Complex conversion scripts |
| **Learning Curve** | ✅ Intuitive YAML-like | ❌ Framework-specific APIs |

### **Code Comparison**

<table>
<tr>
<th>Neural DSL</th>
<th>PyTorch Equivalent</th>
</tr>
<tr>
<td>

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
  
  loss: "categorical_crossentropy"
  optimizer: Adam(lr=0.001)
  
  train {
    epochs: 10
    batch_size: 64
  }
}
```

</td>
<td>

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5408, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc2(x))
        return x

model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=0.001
)

# Plus 50+ lines of training loop...
```

</td>
</tr>
</table>

**Result:** Neural DSL reduces code by **~80%** while adding powerful features like shape validation and automatic tracking.

### **Performance Benchmarks**

| Task | Neural DSL | Raw Framework | Speedup |
|------|-----------|---------------|---------|
| **Model Definition** | 20 lines | 100+ lines | 5x faster |
| **Debugging Setup** | 1 command | 2+ hours | 120x faster |
| **Framework Switch** | 1 flag change | 1-2 days | 10x faster |
| **Runtime Performance** | 1.0-1.2x | 1.0x | Near-native |

*Benchmarks based on MNIST classification task. Runtime overhead is minimal (~0-20%) due to code generation approach.*

---

## Installation

**Prerequisites:** Python 3.8+, pip

### 🎯 **Quick Install (Recommended)**

```bash
# Install with all features (recommended for most users)
pip install neural-dsl[full]
```

### 📦 **Feature-Specific Installation**

Neural DSL uses a modular architecture. Install only what you need:

#### **Core Only (Minimal)**
For basic DSL parsing and CLI functionality:
```bash
pip install neural-dsl
```
**Includes:** Click, Lark parser, NumPy, PyYAML (~20 MB)

#### **ML Backends**
For TensorFlow, PyTorch, or ONNX support:
```bash
pip install neural-dsl[backends]
```
**Adds:** TensorFlow, PyTorch, ONNX Runtime

#### **Hyperparameter Optimization**
For automated HPO with Optuna:
```bash
pip install neural-dsl[hpo]
```
**Adds:** Optuna, scikit-learn

#### **Visualization**
For charts, graphs, and architecture diagrams:
```bash
pip install neural-dsl[visualization]
```
**Adds:** Matplotlib, Plotly, Graphviz, NetworkX

#### **NeuralDbg Dashboard**
For the real-time debugging interface:
```bash
pip install neural-dsl[dashboard]
```
**Adds:** Dash, Flask

#### **Cloud Integration**
For Kaggle, Colab, and AWS support:
```bash
pip install neural-dsl[cloud]
```
**Adds:** PyGithub, Selenium, cloud SDKs

#### **Development Tools**
For contributors:
```bash
pip install neural-dsl[dev]
```
**Adds:** pytest, ruff, pylint, mypy, pre-commit

### 🔧 **Install from Source**

```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\Activate   # Windows

# Install
pip install -e ".[full]"   # All features
# or
pip install -e .           # Core only
```

### 📚 **Feature Groups Summary**

| Group | Command | Size | Use Case |
|-------|---------|------|----------|
| **Core** | `pip install neural-dsl` | ~20 MB | DSL parsing, CLI basics |
| **Backends** | `...[backends]` | ~2 GB | TF/PyTorch/ONNX support |
| **HPO** | `...[hpo]` | ~100 MB | Hyperparameter tuning |
| **Visualization** | `...[visualization]` | ~150 MB | Charts & diagrams |
| **Dashboard** | `...[dashboard]` | ~50 MB | NeuralDbg interface |
| **Cloud** | `...[cloud]` | ~100 MB | Kaggle/Colab/AWS |
| **Full** | `...[full]` | ~2.5 GB | Everything |
| **Dev** | `...[dev]` | ~50 MB | Development tools |

### 🔄 **Migrating from v0.2.x?**

Previous versions installed all dependencies by default. For the same behavior:
```bash
pip install neural-dsl[full]
```

See [Migration Guide](MIGRATION_GUIDE_DEPENDENCIES.md) for details.

---

## Quick Start

### **End-to-End Example: MNIST Classification**

Let's build, train, and deploy a neural network in **under 5 minutes**:

#### **Step 1: Define Your Model**

Create `mnist.neural`:

```yaml
network MNISTClassifier {
  input: (28, 28, 1)  # Grayscale MNIST images
  
  layers:
    # Feature extraction
    Conv2D(filters=32, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Conv2D(filters=64, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    
    # Classification head
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  
  train {
    epochs: 10
    batch_size: 64
    validation_split: 0.2
  }
}
```

#### **Step 2: Validate & Visualize**

```bash
# Check shape propagation
neural visualize mnist.neural --format png
```

**Output:**
- ✅ `architecture.png` - Model structure diagram
- ✅ `shape_propagation.html` - Interactive tensor flow
- ✅ `tensor_flow.html` - Detailed shape transformations

#### **Step 3: Train Your Model**

```bash
# Compile and run with TensorFlow
neural run mnist.neural --backend tensorflow --output mnist_tf.py

# Or use PyTorch
neural run mnist.neural --backend pytorch --output mnist_torch.py
```

**Automatic Features:**
- ✅ Data loading & preprocessing
- ✅ Training loop with progress bars
- ✅ Validation & metrics tracking
- ✅ Model checkpointing
- ✅ Experiment logging

#### **Step 4: Debug in Real-Time**

```bash
neural debug mnist.neural
```

Open `http://localhost:8050` to:
- Monitor training progress
- Visualize gradient flow
- Detect dead neurons
- Profile memory & FLOPs
- Inspect layer activations

#### **Step 5: Deploy to Production**

```bash
# Export optimized ONNX model
neural export mnist.neural --format onnx --optimize

# Or mobile deployment (TFLite with quantization)
neural export mnist.neural --backend tensorflow \
  --format tflite --quantize --quantization-type int8

# Or server deployment (TensorFlow Serving)
neural export mnist.neural --backend tensorflow \
  --format savedmodel --deployment tfserving --model-name mnist_model
```

**Result:** Production-ready model with deployment configs!

#### **Bonus: Use AI to Generate Models**

Don't want to write DSL? Just describe what you need:

```python
from neural.ai import generate_model

model_code = generate_model("""
    Build a CNN for MNIST digit classification.
    Use 2 conv layers with 32 and 64 filters.
    Add dropout and dense layers for classification.
""")

# Save and use
with open("mnist_generated.neural", "w") as f:
    f.write(model_code)
```

### **Quick Commands Reference**

```bash
# Compile DSL to Python code
neural compile model.neural --backend [tensorflow|pytorch|onnx] --output model.py

# Run model (compile + execute)
neural run model.neural --backend tensorflow

# Visualize architecture
neural visualize model.neural --format [png|svg|html]

# Debug with dashboard
neural debug model.neural

# Export for production
neural export model.neural --format [onnx|tflite|torchscript|savedmodel]

# Generate documentation
neural docs model.neural --output model_docs.md

# View experiments
neural track list
neural track compare exp_1 exp_2

# No-code interface
neural --no_code

# Clean artifacts
neural clean --yes --all
```

---

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

### Feedback

Help us improve Neural DSL! Share your feedback: [Typeform Survey](https://form.typeform.com/to/xcibBdKD#name=xxxxx&email=xxxxx&phone_number=xxxxx&user_id=xxxxx&product_id=xxxxx&auth_code=xxxxx)

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

Neural supports running in cloud environments like Kaggle, Google Colab, and AWS SageMaker, with both direct execution and remote control from your local terminal.

### **🔹 1️⃣ Run in Kaggle or Colab**

```python
# Install Neural DSL
!pip install neural-dsl==0.3.0

# Import and use
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()
print(f"Environment: {executor.environment}")
print(f"GPU available: {executor.is_gpu_available}")

# Define and run model
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

model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST')

# Start NeuralDbg with tunnel
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard URL: {dashboard_info['tunnel_url']}")
```

### **🔹 2️⃣ Control from Local Terminal**

```bash
# Connect to cloud platform
neural cloud connect kaggle

# Execute model remotely
neural cloud execute kaggle my_model.neural

# Start with remote access
neural cloud run --setup-tunnel
```

### **🔹 3️⃣ Example Notebooks**

- [Kaggle Example](neural/cloud/examples/neural_kaggle_example.ipynb)
- [Google Colab Example](neural/cloud/examples/neural_colab_example.ipynb)

---

## 🔬 Experiment Tracking

Neural DSL automatically tracks all training runs—no setup required!

### Features
- **Automatic Logging**: Hyperparameters, metrics, and artifacts tracked automatically
- **Metrics History**: Loss, accuracy per epoch
- **CLI Commands**: View, compare, and plot experiments

### Usage
```bash
# Run model (tracking automatic)
neural run examples/mnist.neural --backend tensorflow

# List experiments  
neural track list

# Show details
neural track show <experiment_id>

# Plot metrics
neural track plot <experiment_id>

# Compare experiments
neural track compare <exp_id_1> <exp_id_2>
```

---

##  Documentation

### Core Guides
- [DSL Documentation](docs/dsl.md) - Complete language reference
- [AI Integration Guide](docs/ai_integration_guide.md) - Natural language to DSL
- [Deployment Guide](docs/deployment.md) - Production deployment options
- [Automation Guide](AUTOMATION_GUIDE.md) - CI/CD and automation features
- [Contributing Guide](CONTRIBUTING.md) - How to contribute

### Advanced Topics
- [Custom Layers Guide](docs/examples/custom_layers.md) (Coming soon)
- [ONNX Export Tutorial](docs/examples/onnx_export.md) (Coming soon)
- [Training Configuration](docs/examples/training_config.md) (Coming soon)
- [NeuralDbg Features](docs/examples/neuraldbg_features.md) (Coming soon)
- [HPO Configuration](docs/examples/hpo_guide.md) (Coming soon)

### Resources
- [Blog](docs/blog/README.md) - Articles and tutorials
- [Roadmap](ROADMAP.md) - Future plans
- [Security Policy](SECURITY.md) - Security guidelines

---

##  Examples

Explore common use cases with step-by-step guides:

| Example | Description | Guide |
|---------|-------------|-------|
| **MNIST Classifier** | Handwritten digit recognition | [Guide](docs/examples/mnist_guide.md) |
| **Sentiment Analysis** | Text classification with LSTM | [Guide](docs/examples/sentiment_guide.md) |
| **Transformer NLP** | Attention-based language model | [Guide](docs/examples/transformer_guide.md) |

Browse more in [`examples/`](examples/) directory.

---

## 🕸 Architecture

<div align="center">

**Class Diagram**
![classes](https://github.com/Lemniscate-world/Neural/blob/main/classes.png)

**Package Structure**
![packages](https://github.com/Lemniscate-world/Neural/blob/main/packages.png)

</div>

*Note: Click images to zoom in for details.*

### Repository Structure

```
Neural/
├── docs/              # Documentation
├── examples/          # Example DSL files
├── neural/            # Main source code
│   ├── cli/           # Command-line interface
│   ├── parser/        # DSL parser
│   ├── code_generation/  # Multi-backend code generators
│   ├── shape_propagation/  # Shape validation
│   ├── dashboard/     # NeuralDbg debugger
│   ├── hpo/           # Hyperparameter optimization
│   ├── cloud/         # Cloud integrations
│   └── tracking/      # Experiment tracking
├── profiler/          # Performance profiling
└── tests/             # Test suite
```

See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for details.

---

## 🌍 **Community & Support**

Join our growing community of developers and researchers!

### 💬 **Get Help & Connect**

<div align="center">

| Platform | Purpose | Link |
|----------|---------|------|
| 💬 **Discord** | Real-time chat, Q&A, share projects | [![Discord](https://img.shields.io/badge/Chat-Discord-7289DA?style=for-the-badge&logo=discord)](https://discord.gg/KFku4KvS) |
| 🐦 **Twitter** | Updates, announcements, tips | [![Twitter](https://img.shields.io/badge/Follow-@NLang4438-1DA1F2?style=for-the-badge&logo=twitter)](https://x.com/NLang4438) |
| 💭 **GitHub Discussions** | Feature requests, best practices | [![Discussions](https://img.shields.io/badge/Join-Discussions-181717?style=for-the-badge&logo=github)](https://github.com/Lemniscate-world/Neural/discussions) |
| 📧 **Email** | Security issues, private inquiries | [Lemniscate_zero@proton.me](mailto:Lemniscate_zero@proton.me) |

</div>

### 🆘 **Get Support**

- **Questions?** Ask in [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions) or [Discord](https://discord.gg/KFku4KvS)
- **Bugs?** Open an [issue](https://github.com/Lemniscate-world/Neural/issues)
- **Feature Requests?** Start a [discussion](https://github.com/Lemniscate-world/Neural/discussions)
- **Security Issues?** Email [Lemniscate_zero@proton.me](mailto:Lemniscate_zero@proton.me) with `[SECURITY]` in subject

### 📣 **Stay Updated**

- ⭐ **Star the repo** to get notified of new releases
- 👁️ **Watch the repo** for issue discussions
- 🐦 **Follow [@NLang4438](https://x.com/NLang4438)** for daily tips and updates

---

## 🤝 **Contributing**

We welcome contributions of all kinds! Whether you're fixing bugs, adding features, improving docs, or sharing feedback—every contribution matters.

### 🚀 **Quick Start for Contributors**

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Neural.git
cd Neural

# 2. Set up development environment
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate on Windows
pip install -r requirements-dev.txt
pre-commit install

# 3. Run tests
python -m pytest tests/ -v

# 4. Make your changes and submit a PR!
```

### 🎯 **Areas to Contribute**

| Area | Examples | Difficulty |
|------|----------|-----------|
| 🐛 **Bug Fixes** | Fix shape validation edge cases | Beginner-Intermediate |
| 📝 **Documentation** | Improve guides, add examples | Beginner |
| ✨ **Features** | New layer types, optimizers | Intermediate-Advanced |
| 🤖 **AI Integration** | Enhance NL model generation | Advanced |
| 🧪 **Testing** | Add test coverage | Beginner-Intermediate |
| 🎨 **UI/UX** | Improve NeuralDbg dashboard | Intermediate |
| 📊 **Visualization** | New chart types | Intermediate |
| ☁️ **Cloud** | AWS/GCP integrations | Advanced |

### 📚 **Contribution Guidelines**

- Read our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)
- Check the [Roadmap](ROADMAP.md) for priority features
- Join [Discord](https://discord.gg/KFku4KvS) to discuss ideas before implementing

### 🎁 **Recognition**

All contributors are recognized in:
- Our [Contributors](https://github.com/Lemniscate-world/Neural/graphs/contributors) page
- Release notes for their contributions
- Our community showcase on social media

**First-time contributors welcome!** Look for issues tagged with [`good first issue`](https://github.com/Lemniscate-world/Neural/labels/good%20first%20issue).

---

## Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Lemniscate-world/Neural&type=Timeline)](https://www.star-history.com/#Lemniscate-world/Neural&Timeline)

</div>

---

## 🙏 **Support the Project**

If Neural has helped you, consider supporting us:

- ⭐ **Star the repository** - Help us reach more developers
- 🔄 **Share on social media** - Twitter, Reddit, LinkedIn
- 📝 **Write a blog post** - Share your experience
- 🐛 **Report issues** - Help us improve quality
- 💡 **Suggest features** - Share your ideas
- 🤝 **Contribute code** - See [Contributing](#-contributing)
- ☕ **Sponsor development** - Support continued development (coming soon)

Every star, share, and contribution helps Neural grow! 🚀

---

## 🏗️ Neural Ecosystem Projects

Neural is part of the **λ-S (Lambda-Section)** ecosystem:

### Related Projects
- **Aquarium** - Tauri-based desktop IDE with visual design tools
- **NeuralPaper.ai** - Interactive model visualization and annotation platform
- **Neural-Research** - Historical neural network paper analysis
- **Lambda-sec Models** - Production models for λ-S startup

See [EXTRACTED_PROJECTS.md](EXTRACTED_PROJECTS.md) for details.

---

## 🔒 Security

Neural DSL takes security seriously. We use:
- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability scanner
- **pip-audit**: Package vulnerability auditor
- **git-secrets**: Secret scanning

**Report vulnerabilities:** Email [Lemniscate_zero@proton.me](mailto:Lemniscate_zero@proton.me) with `[SECURITY]` in subject.

See [SECURITY.md](SECURITY.md) for full policy.

---

## 📜 License

Neural is released under the [MIT License](LICENSE.md).

---

### Development Workflow

This section outlines a minimal, fast local workflow to lint, type‑check, test, and audit changes before opening a PR.

### 1) Environment setup (Windows PowerShell)

- Create and activate a virtual environment

```
python -m venv .venv
.\.venv\Scripts\Activate
```

- Install the project with development dependencies

```
pip install -r requirements-dev.txt
```

This installs the core package in editable mode plus all development tools (ruff, mypy, pylint, pytest, pre-commit, pip-audit).

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

### 4) Optional dependencies for testing

Install only what you need for the tests you are running (examples):
```
# PyTorch backend tests
pip install neural-dsl[backends]

# Or install specific backends individually
pip install torch           # PyTorch only
pip install tensorflow      # TensorFlow only
pip install onnx           # ONNX only

# HPO tests
pip install neural-dsl[hpo]

# Dashboard tests
pip install neural-dsl[dashboard]

# Full feature set (for comprehensive testing)
pip install neural-dsl[full]
```

If you have questions or want guidance on tightening typing or adding new policy checks, open a discussion or draft PR.
