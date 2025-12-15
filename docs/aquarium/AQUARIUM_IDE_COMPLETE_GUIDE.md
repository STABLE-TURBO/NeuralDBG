# Aquarium IDE - Complete Guide

**Version**: 1.0.0 | **Last Updated**: December 2024 | **License**: MIT

**The Comprehensive, All-in-One Guide to Aquarium IDE**

---

## 📖 Table of Contents

### Part I: Getting Started
1. [Introduction](#1-introduction)
2. [Installation & Setup](#2-installation--setup)
3. [Quick Start Guide](#3-quick-start-guide)
4. [User Interface Overview](#4-user-interface-overview)

### Part II: Core Features
5. [DSL Editor](#5-dsl-editor)
6. [Model Compilation](#6-model-compilation)
7. [Training Execution](#7-training-execution)
8. [Real-Time Debugging](#8-real-time-debugging)
9. [Export & Integration](#9-export--integration)

### Part III: Advanced Features
10. [Welcome Screen & Tutorials](#10-welcome-screen--tutorials)
11. [Plugin System](#11-plugin-system)
12. [Hyperparameter Optimization](#12-hyperparameter-optimization)
13. [Keyboard Shortcuts](#13-keyboard-shortcuts)

### Part IV: Reference & Troubleshooting
14. [API Reference](#14-api-reference)
15. [Troubleshooting Guide](#15-troubleshooting-guide)
16. [Performance Optimization](#16-performance-optimization)
17. [FAQ](#17-faq)

### Part V: Developer Resources
18. [Architecture Overview](#18-architecture-overview)
19. [Plugin Development](#19-plugin-development)
20. [Contributing](#20-contributing)

---

## 1. Introduction

### 1.1 What is Aquarium IDE?

Aquarium IDE is a comprehensive web-based Integrated Development Environment for Neural DSL. It provides an intuitive interface for writing, compiling, executing, and debugging neural network models across multiple backends (TensorFlow, PyTorch, ONNX).

**Key Benefits:**
- 🎨 **Visual Development**: Write DSL code with syntax highlighting and real-time validation
- 🔧 **Multi-Backend**: Switch between TensorFlow, PyTorch, and ONNX without rewriting code
- 🚀 **One-Click Training**: Compile and train models with a single click
- 📊 **Live Monitoring**: Watch training metrics update in real-time
- 🐛 **Integrated Debugging**: Deep inspection with NeuralDbg integration
- 📦 **Easy Export**: Generate standalone Python scripts
- 🎓 **Built-in Learning**: Interactive tutorials, examples, and documentation

### 1.2 Why Aquarium?

**Problem**: Traditional neural network development requires juggling multiple tools:
- Text editor for code
- Terminal for execution
- Browser for TensorBoard
- Debugger for issues
- Documentation in separate tabs

**Solution**: Aquarium unifies everything in one interface. Write your model, click Run, and watch it train—all in one place.

### 1.3 Who Should Use Aquarium?

- **ML Researchers**: Fast prototyping and experimentation
- **Students**: Learning neural networks with guided tutorials
- **Engineers**: Production model development and testing
- **Data Scientists**: Quick model iterations and comparisons
- **Educators**: Teaching neural network concepts visually

### 1.4 System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 4GB RAM
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for CDN resources)

**Recommended:**
- Python 3.9+
- 8GB+ RAM
- GPU with CUDA support (for faster training)
- 16GB+ disk space (for datasets)

---

## 2. Installation & Setup

### 2.1 Quick Installation

```bash
# Install with dashboard support
pip install neural-dsl[dashboard]

# Or install full package with all features
pip install neural-dsl[full]
```

### 2.2 Installation from Source

```bash
# Clone repository
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e ".[dashboard]"
```

### 2.3 Verify Installation

```bash
# Check installation
python -c "import neural.aquarium; print('✓ Aquarium installed')"

# Launch Aquarium
python -m neural.aquarium.aquarium

# Should see: "Running on http://127.0.0.1:8052"
```

### 2.4 Backend Configuration

**For TensorFlow:**
```bash
pip install tensorflow>=2.13.0
```

**For PyTorch:**
```bash
pip install torch torchvision
```

**For ONNX:**
```bash
pip install onnx onnxruntime
```

### 2.5 First-Time Setup

Create `~/.neural/aquarium/config.yaml`:

```yaml
# Aquarium Configuration
server:
  host: localhost
  port: 8052
  debug: false

ui:
  theme: darkly
  show_welcome: true

backends:
  default: tensorflow
  
training:
  default_epochs: 10
  default_batch_size: 32
  default_validation_split: 0.2

paths:
  examples: ~/.neural/examples/
  exports: ~/.neural/exports/
  temp: ~/.neural/temp/
```

### 2.6 Platform-Specific Notes

**Windows:**
- Use PowerShell or Command Prompt
- Activate venv: `.venv\Scripts\activate`
- Default browser will open automatically

**macOS:**
- Use Terminal
- Activate venv: `source .venv/bin/activate`
- May need to allow firewall access

**Linux:**
- Use bash/zsh
- Activate venv: `source .venv/bin/activate`
- Ensure port 8052 is available

### 2.7 Troubleshooting Installation

**Issue: pip install fails**
```bash
# Update pip
python -m pip install --upgrade pip

# Try with verbose output
pip install -v neural-dsl[dashboard]
```

**Issue: Module not found**
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall in fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install neural-dsl[dashboard]
```

**Issue: Permission denied**
```bash
# Install in user directory
pip install --user neural-dsl[dashboard]
```

---

## 3. Quick Start Guide

### 3.1 Launch Aquarium

```bash
# Default launch (port 8052)
python -m neural.aquarium.aquarium

# Custom port
python -m neural.aquarium.aquarium --port 8053

# Debug mode (detailed logging)
python -m neural.aquarium.aquarium --debug
```

**Access in browser**: `http://localhost:8052`

### 3.2 Your First Model (5 Minutes)

**Step 1: Load an Example**
1. Click **"Load Example"** button in the left sidebar
2. A pre-built MNIST classifier will populate the editor

**Step 2: Parse the DSL**
1. Click **"Parse DSL"** button
2. Verify green success message
3. Check Model Information panel for details

**Step 3: Configure Training**
1. In the Runner tab:
   - Backend: **TensorFlow**
   - Dataset: **MNIST**
   - Epochs: **5** (for quick testing)
   - Batch Size: **32**

**Step 4: Compile & Run**
1. Click **"Compile"** button
2. Wait for "Compilation successful!" message
3. Click **"Run"** button
4. Watch training progress in console

**Step 5: Export Your Model**
1. After training completes
2. Click **"Export Script"**
3. Enter filename: `my_first_model.py`
4. Click **"Export"**

**🎉 Congratulations!** You've trained your first model in Aquarium.

### 3.3 Understanding the Output

**Console Output Example:**
```
[COMPILE] Starting compilation...
[COMPILE] Backend: TensorFlow
[COMPILE] Generating model code...
[COMPILE] ✓ Compilation successful!

[RUN] Starting execution...
[RUN] Loading MNIST dataset...
[RUN] Train samples: 48000, Val samples: 12000

Epoch 1/5
1500/1500 [==============================] - 12s 8ms/step
  loss: 0.3245 - accuracy: 0.8921
  val_loss: 0.2156 - val_accuracy: 0.9234

[SUCCESS] Training completed!
[SUCCESS] Final accuracy: 0.9834
```

---

## 4. User Interface Overview

### 4.1 Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     Header Bar                              │
│  🐠 Aquarium IDE    [New] [Open] [Save] [Help] [⚙️]         │
├─────────────┬───────────────────────────────────────────────┤
│             │  ┌─────────────────────────────────────────┐  │
│             │  │      Tab Navigation                     │  │
│   DSL       │  │  [Runner] Debugger  Viz  Docs          │  │
│  Editor     │  └─────────────────────────────────────────┘  │
│             │                                               │
│  [Parse]    │         Runner Panel                          │
│  [Viz]      │  ┌─────────────────────────────────────────┐ │
│  [Example]  │  │ Backend: [TensorFlow ▼]                 │ │
│             │  │ Dataset: [MNIST ▼]                      │ │
│   Model     │  │ Epochs: [10]  Batch: [32]              │ │
│   Info      │  │                                         │ │
│             │  │ [Compile] [Run] [Stop]                  │ │
│  Network:   │  │                                         │ │
│  MNISTNet   │  │ Console:                                │ │
│  Input:     │  │ ┌─────────────────────────────────────┐ │ │
│  (28,28,1)  │  │ │ Training output appears here...     │ │ │
│  Layers: 8  │  │ └─────────────────────────────────────┘ │ │
│             │  └─────────────────────────────────────────┘ │
└─────────────┴───────────────────────────────────────────────┘
```

### 4.2 Header Bar Components

- **New**: Create new model (clears editor)
- **Open**: Load model from file (.neural format)
- **Save**: Save current model to file
- **Help**: Access documentation and tutorials
- **Settings** (⚙️): Configure IDE preferences

### 4.3 Left Sidebar - Editor Panel

**DSL Editor:**
- Monospace text area for Neural DSL code
- Syntax highlighting
- Auto-indent support
- Line numbers
- Dark theme

**Action Buttons:**
- **Parse DSL**: Validate and parse model
- **Visualize**: Generate architecture diagram
- **Load Example**: Insert pre-built model

**Model Information Panel:**
- Network name
- Input shape
- Number of layers
- Loss function
- Optimizer details
- Layer-by-layer summary

### 4.4 Right Panel - Tabbed Interface

**Runner Tab**: Compilation and execution
- Backend selection
- Dataset configuration
- Training parameters
- Execution controls
- Console output
- Metrics visualization

**Debugger Tab**: NeuralDbg integration
- Launch debugger
- Layer inspection
- Gradient analysis
- Memory profiling

**Visualization Tab**: Architecture diagrams
- Network topology
- Shape propagation
- Connection graphs

**Documentation Tab**: Quick reference
- DSL syntax guide
- Layer documentation
- Best practices

---

## 5. DSL Editor

### 5.1 Writing Neural DSL Code

**Basic Model Structure:**

```neural
network ModelName {
    input: (None, height, width, channels)
    
    layers:
        # Convolutional layers
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        
        # Dense layers
        Flatten()
        Dense(units=128, activation=relu)
        Dropout(rate=0.5)
        
        # Output layer
        Output(units=num_classes, activation=softmax)
    
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

### 5.2 Editor Features

**Keyboard Shortcuts:**
- `Ctrl+A`: Select all
- `Ctrl+C`: Copy
- `Ctrl+V`: Paste
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `Tab`: Indent
- `Shift+Tab`: Unindent

### 5.3 Model Validation

**Parse Button:**
- Validates syntax
- Extracts model structure
- Updates Model Info panel
- Enables compilation

**Validation Feedback:**
- ✅ Success: Green alert with model details
- ❌ Error: Red alert with error message and line number
- ⚠️ Warning: Yellow alert for potential issues

### 5.4 Built-in Examples

1. **MNIST Classifier** (Beginner) - Simple CNN for digit recognition
2. **CIFAR10 CNN** (Intermediate) - Deep CNN for image classification
3. **LSTM Text Classifier** (Intermediate) - Recurrent network for sequences
4. **ResNet-Style** (Advanced) - Residual connections
5. **Transformer** (Advanced) - Attention mechanisms
6. **Autoencoder** (Advanced) - Encoder-decoder architecture

### 5.5 Common DSL Patterns

**Image Classification:**
```neural
network ImageClassifier {
    input: (None, 224, 224, 3)
    layers:
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Conv2D(filters=128, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=512, activation=relu)
        Output(units=1000, activation=softmax)
    loss: categorical_crossentropy
    optimizer: SGD(learning_rate=0.01, momentum=0.9)
}
```

**Text Classification:**
```neural
network TextClassifier {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64, return_sequences=false)
        Dense(units=64, activation=relu)
        Output(units=2, activation=sigmoid)
    loss: binary_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

---

## 6. Model Compilation

### 6.1 Backend Selection

**Supported Backends:**

| Backend | Pros | Cons | Best For |
|---------|------|------|----------|
| **TensorFlow** | Production-ready, Keras integration | Can be verbose | Deployment, serving |
| **PyTorch** | Research-friendly, dynamic graphs | Less production tooling | Research, experimentation |
| **ONNX** | Cross-platform, optimized inference | Limited training features | Model exchange, inference |

### 6.2 Dataset Configuration

**Built-in Datasets:**
- **MNIST** (28x28x1, 10 classes)
- **CIFAR10** (32x32x3, 10 classes)
- **CIFAR100** (32x32x3, 100 classes)
- **ImageNet** (224x224x3, 1000 classes)

**Custom Dataset:**
1. Select "Custom" from dropdown
2. Enter dataset path
3. Ensure proper directory structure

### 6.3 Training Configuration

**Key Parameters:**
- **Epochs**: Number of complete passes (1-1000)
- **Batch Size**: Samples processed together (1-2048)
- **Validation Split**: Portion for validation (0.0-1.0)

### 6.4 Compilation Process

**Steps:**
1. Validation of model and configuration
2. Code generation for selected backend
3. Script creation with training loop
4. Status update and preparation for execution

---

## 7. Training Execution

### 7.1 Execution Controls

- **Run Button**: Starts training process
- **Stop Button**: Terminates running process
- **Clear Button**: Clears console output

### 7.2 Console Output

**Color-Coded Messages:**
- `[COMPILE]` - Cyan - Compilation messages
- `[RUN]` - Green - Execution messages
- `[METRICS]` - Blue - Training metrics
- `[SUCCESS]` - Green - Success messages
- `[ERROR]` - Red - Error messages
- `[WARNING]` - Yellow - Warning messages

### 7.3 Training Progress

Watch live training metrics:
- Loss (training and validation)
- Accuracy (training and validation)
- Epoch progress
- Batch progress
- Time estimates

---

## 8. Real-Time Debugging

### 8.1 NeuralDbg Integration

**Launch Debugger:**
1. Switch to **Debugger** tab
2. Click **"Launch NeuralDbg"**
3. Opens in new browser tab: `http://localhost:8050`

### 8.2 Debugging Features

- **Layer-by-Layer Inspection**: View output shapes and tensor values
- **Gradient Flow Analysis**: Visualize gradient magnitudes
- **Dead Neuron Detection**: Find inactive neurons
- **Memory & FLOP Profiling**: Resource usage analysis
- **Anomaly Detection**: NaN/Inf value detection

### 8.3 Common Debugging Scenarios

**Model Not Converging:**
- Check gradient flow
- Verify learning rate
- Add batch normalization

**Shape Mismatch:**
- Use Visualize button
- Trace shape propagation
- Adjust layer configuration

**Out of Memory:**
- Reduce batch size
- Simplify architecture
- Enable gradient checkpointing

---

## 9. Export & Integration

### 9.1 Exporting Scripts

**Export Process:**
1. Ensure model is compiled
2. Click **"Export Script"**
3. Enter filename
4. Choose export location
5. Click **"Export"**

**Exported File Contains:**
- Complete Python code
- Dataset loading
- Model architecture
- Training loop
- Evaluation code
- Model saving

### 9.2 Integration Examples

**Use with CI/CD:**
```bash
python -m neural.aquarium.aquarium --headless \
  --compile model.neural \
  --backend tensorflow \
  --export output.py
```

**Use with Jupyter:**
```python
from neural.aquarium.src.components.runner import ExecutionManager

manager = ExecutionManager()
manager.compile_model('model.neural', backend='tensorflow')
manager.run_script()
```

---

## 10. Welcome Screen & Tutorials

### 10.1 Welcome Screen Overview

**First Launch Experience:**
- Quick-start templates
- Interactive tutorial
- Example gallery
- Documentation browser
- Video tutorials

### 10.2 Quick Start Templates

Available templates:
1. Image Classification (Beginner)
2. Text Classification (Beginner)
3. Time Series Forecasting (Intermediate)
4. Autoencoder (Intermediate)
5. Sequence-to-Sequence (Advanced)
6. GAN Generator (Advanced)

### 10.3 Interactive Tutorial

**9-Step Guided Tour:**
1. Welcome to Aquarium
2. AI Assistant
3. Quick Start Templates
4. Example Gallery
5. Visual Network Designer
6. DSL Code Editor
7. Real-time Debugger
8. Multi-backend Export
9. Completion

---

## 11. Plugin System

### 11.1 Plugin Overview

**Plugin Types:**
- Panel Plugins (custom UI)
- Theme Plugins (color schemes)
- Command Plugins (custom commands)
- Visualization Plugins (custom viz)
- Integration Plugins (external services)
- Language Support (DSL extensions)

### 11.2 Installing Plugins

**From Marketplace:**
1. Open Plugins menu
2. Browse/search plugins
3. Click "Install"

**From npm/PyPI:**
```bash
npm install @neural/plugin-name
# or
pip install neural-aquarium-plugin-name
```

### 11.3 Example Plugins

1. **GitHub Copilot Integration** - AI-powered code completion
2. **Custom Visualizations** - 3D architecture models
3. **Dark Ocean Theme** - Beautiful dark theme

---

## 12. Hyperparameter Optimization

### 12.1 HPO Overview

Enable automated hyperparameter tuning with Optuna:
- Bayesian optimization
- Pruning strategies
- Multi-objective optimization

### 12.2 Configuration

**Hyperparameters to Tune:**
- Learning rate
- Batch size
- Number of layers
- Layer units
- Dropout rate

**Example:**
```yaml
hpo:
  n_trials: 50
  direction: maximize
  metric: val_accuracy
  search_space:
    learning_rate: [0.0001, 0.01, log]
    batch_size: [16, 32, 64, 128]
```

---

## 13. Keyboard Shortcuts

### 13.1 Essential Shortcuts

| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Parse DSL | `Ctrl+P` | `⌘P` |
| Compile | `Ctrl+B` | `⌘B` |
| Run | `Ctrl+R` | `⌘R` |
| Stop | `Ctrl+C` | `⌘C` |
| Save | `Ctrl+S` | `⌘S` |
| Load Example | `Ctrl+E` | `⌘E` |

### 13.2 Editor Shortcuts

| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Select All | `Ctrl+A` | `⌘A` |
| Copy | `Ctrl+C` | `⌘C` |
| Paste | `Ctrl+V` | `⌘V` |
| Undo | `Ctrl+Z` | `⌘Z` |
| Find | `Ctrl+F` | `⌘F` |

---

## 14. API Reference

### 14.1 Python API

**ExecutionManager:**
```python
from neural.aquarium.src.components.runner import ExecutionManager

manager = ExecutionManager()

# Compile model
manager.compile_model(
    dsl_code='network MyModel {...}',
    backend='tensorflow',
    output_path='/tmp/model.py'
)

# Run training
manager.run_script(
    script_path='/tmp/model.py',
    dataset='mnist',
    epochs=10,
    batch_size=32
)
```

**PluginManager:**
```python
from neural.aquarium.src.plugins import PluginManager

manager = PluginManager()

# List plugins
plugins = manager.list_plugins()

# Enable plugin
manager.enable_plugin('github-copilot')
```

### 14.2 REST API

**Endpoints:**

```bash
# Compilation
POST /api/compile
{
  "dsl_code": "network MyModel {...}",
  "backend": "tensorflow"
}

# Execution
POST /api/run
{
  "script_path": "/tmp/model.py",
  "config": {"dataset": "mnist", "epochs": 10}
}

# Status
GET /api/status

# Examples
GET /api/examples/list
GET /api/examples/load?path=mnist_cnn.neural

# Plugins
GET /api/plugins/list
POST /api/plugins/enable
{"plugin_id": "github-copilot"}
```

---

## 15. Troubleshooting Guide

### 15.1 Installation Issues

**Problem: pip install fails**
```bash
# Update pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -v neural-dsl[dashboard]
```

**Problem: Module not found**
```bash
# Verify installation
python -c "import neural.aquarium; print('OK')"

# Reinstall
pip uninstall neural-dsl
pip install neural-dsl[dashboard]
```

### 15.2 Launch Problems

**Problem: Port already in use**
```bash
# Use different port
python -m neural.aquarium.aquarium --port 8053

# Kill process on port (Linux/macOS)
lsof -ti:8052 | xargs kill -9

# Windows
netstat -ano | findstr :8052
taskkill /F /PID <PID>
```

**Problem: Browser can't connect**
- Verify server is running
- Try `http://127.0.0.1:8052`
- Check firewall settings
- Try incognito/private mode

### 15.3 Compilation Errors

**Problem: Parse error**
- Check DSL syntax
- Verify colons, commas, parentheses
- Use Load Example as template

**Problem: Backend not supported**
```bash
# Install missing backend
pip install tensorflow  # or pytorch, onnx
```

### 15.4 Execution Problems

**Problem: Out of memory**
- Reduce batch size
- Simplify model architecture
- Close other applications

**Problem: Training very slow**
```bash
# Check GPU availability
nvidia-smi

# Verify GPU usage
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 16. Performance Optimization

### 16.1 Training Speed

**Optimize Training:**
- Use GPU when available
- Increase batch size (if memory allows)
- Enable data prefetching
- Use mixed precision training

**GPU Configuration (TensorFlow):**
```python
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### 16.2 Memory Management

**Reduce Memory Usage:**
- Start with small batch sizes
- Use gradient checkpointing
- Clear Keras session between runs
- Limit dataset size for testing

---

## 17. FAQ

### 17.1 General Questions

**Q: What is Aquarium IDE?**
A: A web-based IDE for Neural DSL providing model editing, compilation, execution, and debugging.

**Q: Do I need to know Python?**
A: No, you only need Neural DSL. Aquarium generates Python code automatically.

**Q: Is it free?**
A: Yes, completely free and open source (MIT License).

**Q: Which browsers are supported?**
A: Chrome, Firefox, Safari, Edge (modern versions recommended).

### 17.2 Technical Questions

**Q: Can I train on GPU?**
A: Yes, if TensorFlow/PyTorch with GPU support is installed and GPU is available.

**Q: How do I save trained models?**
A: Enable "Save model weights" option. Weights saved to `.h5` (TF) or `.pth` (PyTorch).

**Q: Can I use custom datasets?**
A: Yes, select "Custom" dataset and provide path to your data.

**Q: Which backend should I use?**
A: TensorFlow for production, PyTorch for research, ONNX for cross-platform.

---

## 18. Architecture Overview

### 18.1 System Design

```
┌────────────────────────────────────────┐
│         Web Browser (Frontend)         │
│  ┌──────────┬──────────┬─────────┐    │
│  │ Editor   │ Runner   │ Debugger│    │
│  └──────────┴──────────┴─────────┘    │
└────────────────┬───────────────────────┘
                 │ HTTP/WebSocket
┌────────────────▼───────────────────────┐
│         Dash Application               │
│  ┌──────────────────────────────────┐ │
│  │  Callbacks & State Management    │ │
│  └──────────────────────────────────┘ │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│      Business Logic Layer              │
│  ┌─────────────┬──────────────────┐   │
│  │ Execution   │ Script Generator │   │
│  │ Manager     │                  │   │
│  └─────────────┴──────────────────┘   │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│      Neural DSL Core                   │
│  ┌─────────┬──────────┬────────────┐  │
│  │ Parser  │ CodeGen  │ ShapeProp  │  │
│  └─────────┴──────────┴────────────┘  │
└────────────────────────────────────────┘
```

### 18.2 Technology Stack

**Frontend:**
- Dash (Python web framework)
- Plotly (charts and visualization)
- Bootstrap (UI components)
- Font Awesome (icons)

**Backend:**
- Flask (web server)
- Neural DSL (parsing and code generation)
- Subprocess (execution management)

---

## 19. Plugin Development

### 19.1 Creating Plugins

**Minimal Plugin Structure:**
```
my-plugin/
├── plugin.json          # Manifest
├── main.py             # Plugin code
└── README.md           # Documentation
```

**plugin.json:**
```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "Plugin description",
  "capabilities": ["panel"],
  "license": "MIT",
  "min_aquarium_version": "0.3.0"
}
```

**main.py:**
```python
from plugin_base import PanelPlugin

class MyPlugin(PanelPlugin):
    def initialize(self):
        print("Initializing...")
    
    def activate(self):
        self._enabled = True
    
    def deactivate(self):
        self._enabled = False
    
    def get_panel_component(self):
        return "MyPanel"

def create_plugin(metadata):
    return MyPlugin(metadata)
```

### 19.2 Publishing Plugins

**To npm:**
```bash
npm init
npm publish
```

**To PyPI:**
```bash
python setup.py sdist
twine upload dist/*
```

---

## 20. Contributing

### 20.1 Ways to Contribute

1. **Report Bugs**: GitHub Issues
2. **Suggest Features**: Discussions
3. **Submit Pull Requests**: Bug fixes, features
4. **Write Tutorials**: Share knowledge
5. **Help Others**: Answer questions

### 20.2 Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Neural.git
cd Neural

# Install dev dependencies
pip install -r requirements-dev.txt
pre-commit install

# Run tests
pytest tests/ -v

# Run linter
ruff check .
```

### 20.3 Code Style

- Follow PEP 8
- 100-char line length
- Type hints
- Numpy-style docstrings
- No comments unless necessary

---

## Appendix

### A. Glossary

- **DSL**: Domain-Specific Language for neural networks
- **Backend**: ML framework (TensorFlow, PyTorch, ONNX)
- **Epoch**: One complete pass through training data
- **Batch Size**: Number of samples processed together
- **Validation Split**: Portion of data reserved for validation
- **HPO**: Hyperparameter Optimization

### B. Additional Resources

**Documentation:**
- [Neural DSL Docs](../dsl.md)
- [Examples](../../examples/README.md)

**Community:**
- [GitHub Repository](https://github.com/Lemniscate-world/Neural)
- [Discord Server](https://discord.gg/KFku4KvS)
- [Twitter](https://x.com/NLang4438)

### C. License

MIT License - see [LICENSE.md](../../LICENSE.md) for details

### D. Citation

```bibtex
@software{neural_aquarium,
  title = {Aquarium IDE: Web-Based IDE for Neural DSL},
  author = {Neural DSL Development Team},
  year = {2024},
  url = {https://github.com/Lemniscate-world/Neural},
  version = {1.0.0}
}
```

---

**Made with ❤️ by the Neural DSL Team**

[⭐ Star on GitHub](https://github.com/Lemniscate-world/Neural) • 
[📚 Documentation](https://github.com/Lemniscate-world/Neural/tree/main/docs) • 
[💬 Discord](https://discord.gg/KFku4KvS) • 
[🐦 Twitter](https://x.com/NLang4438)

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready  
**License**: MIT
