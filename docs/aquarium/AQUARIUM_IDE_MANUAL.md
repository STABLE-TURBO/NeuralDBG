# Aquarium IDE - Complete User Manual

**Version**: 1.0 | **Last Updated**: December 2024 | **License**: MIT

---

## 📚 Table of Contents

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

### 2.4 First-Time Setup

**1. Configure Backend Support:**

```bash
# For TensorFlow
pip install tensorflow>=2.13.0

# For PyTorch
pip install torch torchvision

# For ONNX
pip install onnx onnxruntime
```

**2. Download Example Datasets:**

```bash
# Optional: Pre-download common datasets
python -c "from tensorflow.keras.datasets import mnist; mnist.load_data()"
```

**3. Configure Settings:**

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

### 2.5 Troubleshooting Installation

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

Epoch 2/5
1500/1500 [==============================] - 10s 7ms/step
  loss: 0.1823 - accuracy: 0.9456
  val_loss: 0.1567 - val_accuracy: 0.9512

...

[SUCCESS] Training completed!
[SUCCESS] Final accuracy: 0.9834
[SUCCESS] Model saved to: model_weights.h5
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

**Logo & Title**: Aquarium IDE branding

**File Menu Buttons:**
- **New**: Create new model (clears editor)
- **Open**: Load model from file (.neural format)
- **Save**: Save current model to file
- **Help**: Access documentation and tutorials
- **Settings** (⚙️): Configure IDE preferences

### 4.3 Left Sidebar - Editor Panel

**DSL Editor:**
- Monospace text area for Neural DSL code
- Syntax highlighting (coming soon)
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

### 5.2 Syntax Highlighting

**Currently Supported:**
- Keyword highlighting (network, layers, loss, optimizer)
- String literals
- Numbers and parameters
- Comments

**Coming Soon:**
- Layer name highlighting
- Parameter validation highlighting
- Error underlining

### 5.3 Editor Features

**Keyboard Shortcuts:**
- `Ctrl+A`: Select all
- `Ctrl+C`: Copy
- `Ctrl+V`: Paste
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `Tab`: Indent
- `Shift+Tab`: Unindent

**Auto-formatting:**
- Consistent indentation
- Proper spacing
- Comment alignment

### 5.4 Model Validation

**Parse Button:**
- Validates syntax
- Extracts model structure
- Updates Model Info panel
- Enables compilation

**Validation Feedback:**
- ✅ Success: Green alert with model details
- ❌ Error: Red alert with error message and line number
- ⚠️ Warning: Yellow alert for potential issues

**Model Information Display:**
```
Model Details
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Network Name: MNISTClassifier
Input Shape: (None, 28, 28, 1)
Number of Layers: 8
Output Units: 10

Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loss: categorical_crossentropy
Optimizer: Adam
Learning Rate: 0.001

Layer Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Conv2D (32 filters, 3×3)
2. MaxPooling2D (2×2)
3. Conv2D (64 filters, 3×3)
4. MaxPooling2D (2×2)
5. Flatten
6. Dense (128 units)
7. Dropout (0.5)
8. Output (10 units, softmax)
```

### 5.5 Built-in Examples

**Click "Load Example" to insert:**

1. **MNIST Classifier** (Beginner)
   - Simple CNN for digit recognition
   - Conv → Pool → Dense → Output
   
2. **CIFAR10 CNN** (Intermediate)
   - Deep CNN for image classification
   - Multiple conv blocks with regularization
   
3. **LSTM Text Classifier** (Intermediate)
   - Recurrent network for sequences
   - Embedding → LSTM → Dense
   
4. **ResNet-Style** (Advanced)
   - Residual connections
   - Skip connections and identity blocks
   
5. **Transformer** (Advanced)
   - Attention mechanisms
   - Multi-head attention and positional encoding
   
6. **Autoencoder** (Advanced)
   - Encoder-decoder architecture
   - Dimensionality reduction

### 5.6 Common DSL Patterns

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

**Time Series Prediction:**
```neural
network TimeSeriesPredictor {
    input: (None, 30, 1)
    layers:
        LSTM(units=128, return_sequences=true)
        LSTM(units=64, return_sequences=false)
        Dense(units=32, activation=relu)
        Output(units=1, activation=linear)
    loss: mean_squared_error
    optimizer: Adam(learning_rate=0.001)
}
```

---

## 6. Model Compilation

### 6.1 Backend Selection

**Supported Backends:**

| Backend | Pros | Cons | Best For |
|---------|------|------|----------|
| **TensorFlow** | Production-ready, Keras integration, TensorBoard | Can be verbose | Deployment, serving, production |
| **PyTorch** | Research-friendly, dynamic graphs, flexible | Less production tooling | Research, experimentation |
| **ONNX** | Cross-platform, optimized inference | Limited training features | Model exchange, inference |

**Switching Backends:**
1. Select from dropdown in Runner panel
2. Click "Compile" to regenerate code
3. Code generation adapts automatically

**Backend-Specific Features:**

*TensorFlow:*
- Keras high-level API
- TensorBoard integration
- SavedModel format
- TensorFlow Serving

*PyTorch:*
- Dynamic computation graphs
- TorchScript compilation
- CUDA optimization
- Custom autograd functions

*ONNX:*
- Framework-agnostic format
- Optimized inference
- Hardware acceleration
- Cross-platform deployment

### 6.2 Dataset Configuration

**Built-in Datasets:**

**MNIST** (Handwritten Digits)
- Shape: (28, 28, 1)
- Classes: 10 (digits 0-9)
- Train: 60,000 images
- Test: 10,000 images

**CIFAR10** (Color Images)
- Shape: (32, 32, 3)
- Classes: 10 (airplane, car, bird, etc.)
- Train: 50,000 images
- Test: 10,000 images

**CIFAR100**
- Shape: (32, 32, 3)
- Classes: 100 (fine-grained categories)
- Train: 50,000 images
- Test: 10,000 images

**ImageNet**
- Shape: (224, 224, 3)
- Classes: 1000
- Large-scale dataset

**Custom Dataset:**
1. Select "Custom" from dropdown
2. Enter dataset path
3. Ensure data format:
   ```
   dataset/
   ├── train/
   │   ├── class1/
   │   └── class2/
   └── test/
       ├── class1/
       └── class2/
   ```

### 6.3 Training Configuration

**Epochs:**
- Range: 1-1000
- Default: 10
- Tip: Start with 5-10 for testing

**Batch Size:**
- Range: 1-2048
- Default: 32
- Tip: Powers of 2 are efficient (16, 32, 64, 128)
- Larger = faster but more memory

**Validation Split:**
- Range: 0.0-1.0
- Default: 0.2 (20%)
- Splits training data for validation

### 6.4 Training Options

**Auto-flatten Output** ✓
- Automatically flattens output to match dataset
- Recommended for most models

**HPO (Hyperparameter Optimization)** ☐
- Enables Optuna-based tuning
- Experimental feature
- Searches for optimal hyperparameters

**Verbose Output** ✓
- Shows detailed training logs
- Progress bars and metrics
- Recommended for monitoring

**Save Model Weights** ✓
- Saves trained weights to file
- Format: .h5 (TF), .pth (PyTorch)
- Enables model reuse

### 6.5 Compilation Process

**Click "Compile":**

1. **Validation**
   - Checks parsed model exists
   - Verifies backend is available
   - Validates configuration

2. **Code Generation**
   - Parses DSL to AST
   - Generates backend-specific code
   - Adds dataset loading
   - Creates training loop
   - Adds evaluation code

3. **Script Creation**
   - Writes to temporary file
   - Adds necessary imports
   - Includes helper functions
   - Sets up logging

4. **Status Update**
   - Updates console
   - Shows file path
   - Enables Run button

**Compilation Output:**
```
[COMPILE] Starting compilation...
[COMPILE] Backend: TensorFlow
[COMPILE] Parsing DSL...
[COMPILE] Generating model architecture...
[COMPILE] Adding dataset loader (MNIST)...
[COMPILE] Creating training loop (epochs=10, batch=32)...
[COMPILE] Writing to: /tmp/aquarium_model_abc123.py
[COMPILE] ✓ Compilation successful!
[COMPILE] Script size: 245 lines
[COMPILE] Ready to execute
```

---

## 7. Training Execution

### 7.1 Execution Controls

**Run Button:**
- Starts training process
- Spawns separate Python process
- Streams output to console
- Updates metrics in real-time

**Stop Button:**
- Terminates running process
- Cleans up resources
- Saves partial results

**Clear Button:**
- Clears console output
- Resets metrics display

### 7.2 Execution Process

**Click "Run":**

1. **Process Start**
   - Launches Python subprocess
   - Sets unbuffered output
   - Configures environment

2. **Dataset Loading**
   - Downloads if needed (first time)
   - Loads and preprocesses data
   - Splits train/validation

3. **Model Building**
   - Constructs architecture
   - Initializes weights
   - Compiles with optimizer

4. **Training Loop**
   - Iterates through epochs
   - Updates weights
   - Calculates metrics
   - Validates on val set

5. **Completion**
   - Saves model weights
   - Evaluates on test set
   - Reports final metrics

### 7.3 Console Output

**Color-Coded Messages:**

```
[COMPILE]  - Cyan    - Compilation messages
[RUN]      - Green   - Execution messages
[METRICS]  - Blue    - Training metrics
[SUCCESS]  - Green   - Success messages
[ERROR]    - Red     - Error messages
[WARNING]  - Yellow  - Warning messages
```

**Example Training Output:**
```
[RUN] Starting execution...
[RUN] Python interpreter: /usr/bin/python3
[RUN] Working directory: /tmp/aquarium/
[RUN] Loading MNIST dataset...
[RUN] ✓ Dataset loaded
[RUN] Train samples: 48000
[RUN] Validation samples: 12000
[RUN] Test samples: 10000

[RUN] Building model...
[RUN] Model: MNISTClassifier
[RUN] Total params: 1,199,882
[RUN] Trainable params: 1,199,882
[RUN] Non-trainable params: 0

[RUN] Starting training...

Epoch 1/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1500/1500 [==============================] - 12s 8ms/step
[METRICS] loss: 0.3245 - accuracy: 0.8921
[METRICS] val_loss: 0.2156 - val_accuracy: 0.9234

Epoch 2/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1500/1500 [==============================] - 10s 7ms/step
[METRICS] loss: 0.1823 - accuracy: 0.9456
[METRICS] val_loss: 0.1567 - val_accuracy: 0.9512

...

Epoch 10/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1500/1500 [==============================] - 10s 7ms/step
[METRICS] loss: 0.0234 - accuracy: 0.9923
[METRICS] val_loss: 0.0445 - val_accuracy: 0.9834

[RUN] Evaluating on test set...
313/313 [==============================] - 2s 6ms/step
[SUCCESS] Test accuracy: 0.9834
[SUCCESS] Test loss: 0.0445

[SUCCESS] Training completed!
[SUCCESS] Total time: 2m 15s
[SUCCESS] Model saved to: model_weights.h5
```

### 7.4 Metrics Visualization

**Real-Time Charts** (Coming Soon):
- Loss curves (train/val)
- Accuracy curves
- Learning rate schedule
- Epoch timing

**Current**: Metrics logged to console with color coding

### 7.5 Stopping Execution

**Use Cases:**
- Training taking too long
- Spotted error in configuration
- Want to try different parameters
- System resources needed

**How to Stop:**
1. Click "Stop" button
2. Wait for graceful shutdown
3. Check console for confirmation

**After Stopping:**
- Partial results may be saved
- Console shows termination message
- Can restart with new configuration

---

## 8. Real-Time Debugging

### 8.1 NeuralDbg Integration

**Launch Debugger:**
1. Switch to **Debugger** tab
2. Click **"Launch NeuralDbg"**
3. Opens in new browser tab: `http://localhost:8050`

### 8.2 Debugging Features

**Layer-by-Layer Inspection:**
- View output shape at each layer
- Inspect tensor values
- Check activation patterns

**Gradient Flow Analysis:**
- Visualize gradient magnitudes
- Detect vanishing/exploding gradients
- Identify problematic layers

**Dead Neuron Detection:**
- Find neurons with zero activations
- Highlight ineffective units
- Suggest architecture changes

**Memory & FLOP Profiling:**
- Memory usage per layer
- Computational requirements
- Bottleneck identification

**Anomaly Detection:**
- NaN/Inf value detection
- Out-of-range values
- Distribution shifts

### 8.3 Common Debugging Scenarios

**Scenario 1: Model Not Converging**

**Symptoms:**
- Loss stays constant
- Accuracy doesn't improve
- Gradients very small

**Debugging Steps:**
1. Launch NeuralDbg
2. Check gradient flow
3. Look for vanishing gradients
4. Verify learning rate

**Solutions:**
- Increase learning rate
- Use gradient clipping
- Add batch normalization
- Check data normalization

**Scenario 2: Shape Mismatch**

**Symptoms:**
- Error during compilation
- Shape incompatibility message

**Debugging Steps:**
1. Check Model Info panel
2. Use Visualize button
3. Trace shape propagation

**Solutions:**
- Adjust input shape
- Add Flatten layer before Dense
- Verify dataset dimensions

**Scenario 3: Out of Memory**

**Symptoms:**
- Training crashes
- ResourceExhaustedError

**Debugging Steps:**
1. Check FLOP profiler
2. Identify memory-heavy layers
3. Review batch size

**Solutions:**
- Reduce batch size
- Simplify model architecture
- Enable gradient checkpointing

---

## 9. Export & Integration

### 9.1 Exporting Scripts

**Export Process:**
1. Ensure model is compiled
2. Click **"Export Script"** button
3. Enter filename (e.g., `my_model.py`)
4. Choose export location
5. Click **"Export"**

**Exported File Contents:**
```python
"""
Generated by Aquarium IDE
Model: MNISTClassifier
Backend: TensorFlow
Date: 2024-12-13
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Save
model.save('model_weights.h5')
print('Model saved to: model_weights.h5')
```

### 9.2 Opening in External IDE

**Supported IDEs:**
- VS Code
- PyCharm
- Sublime Text
- Atom
- Notepad++
- Any default .py handler

**Process:**
1. Export script first
2. Click **"Open in IDE"**
3. Script opens in default editor

**Platform Support:**
- **Windows**: Uses `os.startfile()`
- **macOS**: Uses `open` command
- **Linux**: Uses `xdg-open`

### 9.3 File Organization

**Default Structure:**
```
~/.neural/aquarium/
├── compiled/           # Temporary compiled scripts
│   ├── model_abc123.py
│   └── model_xyz789.py
├── exported/           # User-exported scripts
│   ├── mnist_v1.py
│   ├── mnist_v2.py
│   └── production_model.py
├── models/             # Saved DSL files
│   ├── mnist.neural
│   ├── cifar10.neural
│   └── custom.neural
├── weights/            # Trained model weights
│   ├── model_weights.h5
│   └── checkpoint.pth
└── logs/               # Training logs
    ├── aquarium.log
    └── training_20241213.log
```

### 9.4 Integration Examples

**Use with CI/CD:**
```bash
# In your CI pipeline
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

**Use with Docker:**
```dockerfile
FROM python:3.9
RUN pip install neural-dsl[full]
COPY model.neural /app/
WORKDIR /app
CMD python -m neural.aquarium.aquarium --headless --compile model.neural
```

---

## 10. Welcome Screen & Tutorials

### 10.1 Welcome Screen Overview

**First Launch Experience:**
- Appears automatically on first startup
- Provides quick-start templates
- Interactive tutorial option
- Example gallery
- Documentation browser

**Tabs:**
1. **Quick Start**: Pre-built templates
2. **Examples**: Repository models
3. **Documentation**: Integrated docs
4. **Video Tutorials**: Step-by-step guides

### 10.2 Quick Start Templates

**Available Templates:**

1. **Image Classification** (Beginner)
   - CNN for MNIST/CIFAR-10
   - Conv2D → MaxPooling → Dense
   
2. **Text Classification** (Beginner)
   - LSTM for sentiment analysis
   - Embedding → LSTM → Dense
   
3. **Time Series Forecasting** (Intermediate)
   - Multi-layer LSTM
   - Sequence prediction
   
4. **Autoencoder** (Intermediate)
   - Encoder-decoder architecture
   - Dimensionality reduction
   
5. **Sequence-to-Sequence** (Advanced)
   - Machine translation
   - Attention mechanisms
   
6. **GAN Generator** (Advanced)
   - Generative adversarial network
   - Synthetic data generation

### 10.3 Interactive Tutorial

**9-Step Guided Tour:**

1. Welcome to Aquarium
2. AI Assistant introduction
3. Quick Start Templates
4. Example Gallery
5. Visual Network Designer
6. DSL Code Editor
7. Real-time Debugger
8. Multi-backend Export
9. Completion & next steps

**Features:**
- Element highlighting
- Progress tracking
- Skip option
- Previous/Next navigation

### 10.4 Example Gallery

**Browse Examples:**
- Filter by category (Vision, NLP, Generative)
- Search by name/description
- Load directly into editor

**Built-in Examples:**
- MNIST CNN
- LSTM Text Classifier
- ResNet Image Classifier
- Transformer Model
- Variational Autoencoder (VAE)

### 10.5 Documentation Browser

**Integrated Docs:**
- DSL Syntax Reference
- Layer Types Guide
- Best Practices
- Troubleshooting

**Features:**
- Markdown rendering
- Syntax highlighting
- Search functionality
- Quick navigation

---

## 11. Plugin System

### 11.1 Plugin Architecture

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
2. Click "Marketplace"
3. Browse/search plugins
4. Click "Install"

**From npm:**
```bash
npm install @neural/plugin-name
```

**From PyPI:**
```bash
pip install neural-aquarium-plugin-name
```

**Manual Installation:**
```bash
cp -r my-plugin ~/.neural/aquarium/plugins/
```

### 11.3 Example Plugins

**1. GitHub Copilot Integration**
- AI-powered code completion
- Context-aware suggestions
- Layer recommendations

**2. Custom Visualizations**
- 3D architecture models
- Interactive flow diagrams
- Advanced metrics charts

**3. Dark Ocean Theme**
- Beautiful dark theme
- Ocean color palette
- Reduced eye strain

### 11.4 Plugin Marketplace

**Features:**
- Search and filter
- Sort by rating/downloads
- One-click install
- User reviews
- Plugin details

---

## 12. Hyperparameter Optimization

### 12.1 HPO Overview

**Enable HPO:**
1. Check "HPO" option in Runner
2. Configure search space
3. Run optimization

**Backend: Optuna**
- Bayesian optimization
- Pruning strategies
- Multi-objective optimization

### 12.2 Search Space Configuration

**Hyperparameters to Tune:**
- Learning rate
- Batch size
- Number of layers
- Layer units
- Dropout rate
- Optimizer choice

**Example Configuration:**
```yaml
hpo:
  n_trials: 50
  direction: maximize
  metric: val_accuracy
  
  search_space:
    learning_rate: [0.0001, 0.01, log]
    batch_size: [16, 32, 64, 128]
    units: [64, 128, 256, 512]
    dropout: [0.1, 0.5]
```

### 12.3 Running HPO

**Process:**
1. Enable HPO option
2. Set number of trials
3. Click "Run"
4. Monitor study progress
5. View best hyperparameters

**Output:**
```
[HPO] Starting hyperparameter optimization...
[HPO] Study: mnist_optimization
[HPO] Trials: 50
[HPO] Metric: val_accuracy (maximize)

Trial 1/50: learning_rate=0.001, batch_size=32
  → val_accuracy=0.9234

Trial 2/50: learning_rate=0.0005, batch_size=64
  → val_accuracy=0.9312

...

[HPO] ✓ Optimization complete!
[HPO] Best trial: 23
[HPO] Best hyperparameters:
  learning_rate: 0.00075
  batch_size: 48
  units: 192
  dropout: 0.3
[HPO] Best val_accuracy: 0.9456
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
| Comment | `Ctrl+/` | `⌘/` |

### 13.3 View Shortcuts

| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Next Tab | `Ctrl+Tab` | `⌘⌥→` |
| Previous Tab | `Ctrl+Shift+Tab` | `⌘⌥←` |
| Zoom In | `Ctrl++` | `⌘+` |
| Zoom Out | `Ctrl+-` | `⌘-` |
| Fullscreen | `F11` | `⌘Ctrl+F` |

**Complete Reference**: See [Keyboard Shortcuts](#13-keyboard-shortcuts) section

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

# Stop execution
manager.stop_execution()

# Get status
status = manager.get_status()
print(status)  # 'idle', 'compiling', 'running', 'stopped'
```

**PluginManager:**
```python
from neural.aquarium.src.plugins import PluginManager

manager = PluginManager()

# List plugins
plugins = manager.list_plugins()

# Enable plugin
manager.enable_plugin('github-copilot')

# Get panels
panels = manager.get_panels()

# Execute command
result = manager.execute_command('my-command', {'arg': 'value'})
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
  "config": {
    "dataset": "mnist",
    "epochs": 10,
    "batch_size": 32
  }
}

# Stop
POST /api/stop

# Status
GET /api/status

# Examples
GET /api/examples/list
GET /api/examples/load?path=mnist_cnn.neural

# Plugins
GET /api/plugins/list
POST /api/plugins/enable
{
  "plugin_id": "github-copilot"
}
```

### 14.3 Component API

**DSL Parser:**
```python
from neural.parser import parse

ast = parse(dsl_code)
print(ast.network_name)
print(ast.input_shape)
print(ast.layers)
```

**Code Generator:**
```python
from neural.code_generation import TensorFlowGenerator

generator = TensorFlowGenerator()
code = generator.generate(ast)
print(code)
```

**Shape Propagation:**
```python
from neural.shape_propagation import propagate_shapes

shapes = propagate_shapes(ast)
for layer, shape in shapes.items():
    print(f"{layer}: {shape}")
```

---

## 15. Troubleshooting Guide

### 15.1 Installation Issues

**Problem: pip install fails**
```bash
# Update pip
python -m pip install --upgrade pip

# Install with verbose
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

# Kill process on port 8052
# Windows
netstat -ano | findstr :8052
taskkill /F /PID <PID>

# Linux/macOS
lsof -ti:8052 | xargs kill -9
```

**Problem: Browser can't connect**
- Verify server is running
- Try `http://127.0.0.1:8052`
- Check firewall settings
- Disable browser extensions
- Try incognito/private mode

### 15.3 Compilation Errors

**Problem: Parse error**
- Check DSL syntax carefully
- Verify colons, commas, parentheses
- Use Load Example as template
- Check indentation (use 4 spaces)

**Problem: Backend not supported**
```bash
# Install missing backend
pip install tensorflow  # or pytorch, onnx
```

### 15.4 Execution Problems

**Problem: Script won't run**
- Compile model first
- Check console for errors
- Verify dataset selection
- Ensure backend is installed

**Problem: Out of memory**
- Reduce batch size (try 16, 8, or 4)
- Simplify model architecture
- Close other applications
- Enable GPU memory growth (TF)

**Problem: Training very slow**
```bash
# Check GPU availability
nvidia-smi

# Verify GPU usage
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 15.5 UI Issues

**Problem: Layout broken**
- Hard refresh (Ctrl+Shift+R)
- Clear browser cache
- Try different browser
- Reset zoom (Ctrl+0)

**Problem: Icons not showing**
- Check internet connection (CDN)
- Disable ad blocker
- Clear cache

### 15.6 Export Issues

**Problem: Export fails**
```bash
# Check permissions
chmod +w ~/path/to/export/

# Verify path exists
mkdir -p ~/path/to/export/

# Check disk space
df -h
```

**Complete Troubleshooting**: See [Troubleshooting Guide](#15-troubleshooting-guide) section

---

## 16. Performance Optimization

### 16.1 Training Speed

**Optimize Training:**
- Use GPU when available
- Increase batch size (if memory allows)
- Enable data prefetching
- Use mixed precision training
- Optimize data loading

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

**Monitor Resources:**
```bash
# Windows
taskmgr

# macOS
Activity Monitor

# Linux
htop
```

### 16.3 UI Performance

**Improve UI Responsiveness:**
- Clear console regularly
- Close unused browser tabs
- Reduce console buffer size
- Disable verbose output for long runs

### 16.4 Benchmarking

**Measure Performance:**
```python
import time

start = time.time()
manager.run_script(...)
duration = time.time() - start

print(f"Training took {duration:.2f} seconds")
```

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

### 17.3 Troubleshooting Questions

**Q: Compilation fails, what do I do?**
A: Check parse status first. Verify DSL syntax. Check console for specific errors.

**Q: Training is slow, how to speed up?**
A: Use GPU if available. Increase batch size. Reduce model complexity.

**Q: Console not updating, why?**
A: Wait a moment (buffered output). Refresh browser if needed. Check process is running.

**Q: Export button grayed out, why?**
A: Compile model first. Ensure compilation succeeded. Check status indicator.

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

### 18.2 Component Structure

```
neural/aquarium/
├── aquarium.py              # Main application entry
├── config.py                # Configuration management
├── examples.py              # Built-in examples
├── src/
│   └── components/
│       ├── runner/          # Compilation & execution
│       │   ├── runner_panel.py
│       │   ├── execution_manager.py
│       │   └── script_generator.py
│       ├── welcome/         # Welcome screen & tutorials
│       │   ├── WelcomeScreen.tsx
│       │   ├── QuickStartTemplates.tsx
│       │   └── ExampleGallery.tsx
│       ├── debugger/        # Debugging interface
│       ├── editor/          # DSL editor
│       ├── settings/        # Configuration UI
│       ├── project/         # Project management
│       └── plugins/         # Plugin system
│           ├── plugin_manager.py
│           ├── plugin_loader.py
│           └── plugin_registry.py
└── backend/                 # Backend API (future)
    └── server.py
```

### 18.3 Data Flow

**Compilation Flow:**
```
DSL Code → Parser → AST → Code Generator → Python Script
                                               ↓
                                          Temp File
```

**Execution Flow:**
```
Python Script → Subprocess → stdout/stderr → Parser → Console
                                                         ↓
                                                    Metrics
```

**Plugin Flow:**
```
Plugin Manifest → Loader → Registry → Manager → Components
                                                     ↓
                                                 UI/API
```

### 18.4 Technology Stack

**Frontend:**
- Dash (Python web framework)
- Plotly (charts and visualization)
- Bootstrap (UI components)
- Font Awesome (icons)

**Backend:**
- Flask (web server)
- Neural DSL (parsing and code generation)
- Subprocess (execution management)

**Plugins:**
- Python plugin system
- npm/PyPI distribution
- Hook-based events

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
from plugin_base import PanelPlugin, PluginMetadata

class MyPlugin(PanelPlugin):
    def initialize(self):
        print("Initializing...")
    
    def activate(self):
        self._enabled = True
    
    def deactivate(self):
        self._enabled = False
    
    def get_panel_component(self):
        return "MyPanel"
    
    def get_panel_config(self):
        return {
            'title': 'My Panel',
            'position': 'right',
            'width': 400
        }

def create_plugin(metadata):
    return MyPlugin(metadata)
```

### 19.2 Plugin Types

**Panel Plugin**: Custom UI panels
**Theme Plugin**: Color schemes
**Command Plugin**: Custom commands
**Visualization Plugin**: Custom visualizations
**Integration Plugin**: External service connectors

### 19.3 Publishing Plugins

**To npm:**
```bash
npm init
# Add neuralAquariumPlugin field to package.json
npm publish
```

**To PyPI:**
```bash
python setup.py sdist
twine upload dist/*
```

**Complete Guide**: See [Plugin Development](#19-plugin-development) section

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

### 20.4 Submitting Changes

1. Create feature branch
2. Make changes
3. Add tests
4. Run linter
5. Submit pull request

---

## Appendix

### A. Glossary

**DSL**: Domain-Specific Language for neural networks
**Backend**: ML framework (TensorFlow, PyTorch, ONNX)
**Epoch**: One complete pass through training data
**Batch Size**: Number of samples processed together
**Validation Split**: Portion of data reserved for validation
**HPO**: Hyperparameter Optimization

### B. Additional Resources

**Documentation:**
- [Neural DSL Docs](../../docs/dsl.md)
- [API Reference](../../docs/api/README.md)
- [Examples](../../examples/README.md)

**Community:**
- [GitHub Repository](https://github.com/Lemniscate-world/Neural)
- [Discord Server](https://discord.gg/KFku4KvS)
- [Twitter](https://x.com/NLang4438)

**Support:**
- GitHub Issues
- GitHub Discussions
- Email: Lemniscate_zero@proton.me

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
**License**: MIT  
**Status**: Production Ready
