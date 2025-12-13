# Aquarium IDE User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [User Interface Overview](#user-interface-overview)
4. [DSL Editor](#dsl-editor)
5. [Model Compilation & Execution](#model-compilation--execution)
6. [Debugging](#debugging)
7. [Export & Integration](#export--integration)
8. [Advanced Features](#advanced-features)
9. [Best Practices](#best-practices)
10. [FAQ](#faq)

## Introduction

Aquarium IDE is a comprehensive web-based development environment for Neural DSL. It provides an intuitive interface for writing, compiling, executing, and debugging neural network models across multiple backends (TensorFlow, PyTorch, ONNX).

### Key Features

- 🎨 **Syntax-Highlighted DSL Editor** - Write models with ease
- 🔧 **Multi-Backend Support** - Switch between TensorFlow, PyTorch, and ONNX
- 🚀 **One-Click Compilation** - Generate executable Python code
- 📊 **Real-Time Training** - Execute and monitor training directly
- 🐛 **Integrated Debugging** - Debug with NeuralDbg integration
- 📦 **Export & Share** - Save and share your models
- 📚 **Built-in Examples** - Learn from pre-built templates

## Getting Started

### Launching Aquarium

```bash
# Default launch
python -m neural.aquarium.aquarium

# Custom port
python -m neural.aquarium.aquarium --port 8053

# Debug mode
python -m neural.aquarium.aquarium --debug
```

Access the IDE at: `http://localhost:8052`

### Your First Model

**Step 1**: Load an example
- Click the **"Load Example"** button
- A pre-built model will populate the editor

**Step 2**: Parse the DSL
- Click **"Parse DSL"** to validate the model
- Check the **Model Information** panel for details

**Step 3**: Configure execution
- Select **Backend**: TensorFlow
- Select **Dataset**: MNIST
- Set **Epochs**: 5 (for quick testing)

**Step 4**: Compile and run
- Click **"Compile"** to generate Python code
- Click **"Run"** to start training
- Watch progress in the console

**Step 5**: Export your model
- Click **"Export Script"**
- Enter a filename: `my_first_model.py`
- Choose export location
- Click **"Export"**

![First Model Workflow](../images/aquarium/first-model-workflow.png)

## User Interface Overview

### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     Header Bar                              │
│  Neural Aquarium IDE    [New] [Open] [Save] [Help] [⚙️]     │
├─────────────┬───────────────────────────────────────────────┤
│             │  ┌─────────────────────────────────────────┐  │
│             │  │         Tab Navigation                  │  │
│             │  │  Runner | Debugger | Viz | Docs        │  │
│             │  └─────────────────────────────────────────┘  │
│   DSL       │                                               │
│  Editor     │           Main Content Area                   │
│             │      (Context-dependent based on tab)         │
│             │                                               │
│ [Parse DSL] │                                               │
│ [Visualize] │                                               │
│ [Load Ex]   │                                               │
│             │                                               │
│   Model     │                                               │
│   Info      │                                               │
│   Panel     │                                               │
└─────────────┴───────────────────────────────────────────────┘
```

### Header Bar

- **New**: Create a new model (clears editor)
- **Open**: Load a model from file
- **Save**: Save current model to file
- **Help**: Open help documentation
- **Settings**: Configure IDE preferences

### Left Sidebar

#### DSL Editor Panel
- Text area for writing Neural DSL code
- Monospace font for readability
- Dark theme for reduced eye strain
- Vertical resize capability

#### Action Buttons
- **Parse DSL**: Validate and parse model
- **Visualize**: Generate architecture diagrams
- **Load Example**: Insert example model

#### Model Information Panel
- Displays parsed model details
- Shows input shape, layer count
- Lists loss function and optimizer
- Layer-by-layer summary

![UI Overview](../images/aquarium/ui-overview.png)

### Right Main Area - Tabs

#### 1. Runner Tab
Model compilation and execution interface

#### 2. Debugger Tab
NeuralDbg integration for debugging

#### 3. Visualization Tab
Model architecture visualization

#### 4. Documentation Tab
Quick reference for DSL syntax

## DSL Editor

### Writing Models

The DSL editor supports full Neural DSL syntax:

```neural
network MyClassifier {
    input: (None, 28, 28, 1)
    
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Conv2D(filters=64, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
    
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

### Editor Features

**Keyboard Shortcuts**:
- `Ctrl+A` / `Cmd+A`: Select all
- `Ctrl+C` / `Cmd+C`: Copy
- `Ctrl+V` / `Cmd+V`: Paste
- `Ctrl+Z` / `Cmd+Z`: Undo
- `Tab`: Insert indentation

**Tips**:
- Use consistent indentation (4 spaces recommended)
- Follow Neural DSL syntax exactly
- Check parse errors in Model Info panel
- Save frequently (use Save button)

![DSL Editor](../images/aquarium/dsl-editor.png)

### Parsing Models

**Parse DSL Button**:
- Validates syntax
- Extracts model structure
- Updates Model Information panel
- Enables compilation

**Parse Status**:
- ✅ **Success**: Green alert, model info displayed
- ❌ **Error**: Red alert with error message

**Model Information Display**:
```
Model Details
Input Shape: (None, 28, 28, 1)
Number of Layers: 8
Loss Function: categorical_crossentropy
Optimizer: Adam

Layer Summary
1. Conv2D
2. MaxPooling2D
3. Conv2D
4. MaxPooling2D
5. Flatten
6. Dense
7. Dropout
8. Output
```

### Loading Examples

Click **"Load Example"** to insert a random pre-built model:

**Available Examples**:
1. **MNIST Classifier** - Simple CNN for digit recognition
2. **CIFAR10 CNN** - Deep CNN for image classification
3. **Dense Network** - Fully connected architecture
4. **LSTM Text** - Recurrent network for sequences
5. **VGG-Style** - VGG-inspired architecture
6. **ResNet-Style** - Residual blocks
7. **Autoencoder** - Encoder-decoder architecture
8. **Transformer** - Attention-based model

![Load Examples](../images/aquarium/load-examples.png)

### Visualization

Click **"Visualize"** to generate:
- Architecture diagram (PNG/SVG)
- Shape propagation graph
- Layer connection graph

(Feature currently under development)

## Model Compilation & Execution

### Runner Panel Overview

The Runner tab provides complete model training capabilities:

```
┌─────────────────────────────────────────────┐
│  Backend: [TensorFlow ▼]                    │
│  Dataset: [MNIST ▼]                         │
│                                             │
│  Training Configuration                     │
│  Epochs: [10]                               │
│  Batch Size: [32]                          │
│  Validation Split: [0.2]                   │
│                                             │
│  Options                                    │
│  ☑ Auto-flatten output                     │
│  ☐ HPO (Hyperparameter Optimization)       │
│  ☑ Verbose output                          │
│  ☑ Save model weights                      │
│                                             │
│  [Compile] [Run] [Stop]                    │
│  [Export] [Open in IDE] [Clear]            │
│                                             │
│  Status: [Idle]                             │
│                                             │
│  Console Output                             │
│  ┌───────────────────────────────────────┐ │
│  │ Ready to compile and run models...    │ │
│  │                                       │ │
│  └───────────────────────────────────────┘ │
│                                             │
│  Training Metrics                           │
│  (Graph placeholder)                        │
└─────────────────────────────────────────────┘
```

![Runner Panel](../images/aquarium/runner-panel.png)

### Backend Selection

Choose your preferred ML framework:

| Backend | Pros | Cons | Use Case |
|---------|------|------|----------|
| **TensorFlow** | Production-ready, Keras integration | Can be verbose | Deployment, serving |
| **PyTorch** | Research-friendly, dynamic graphs | Less production tools | Research, experimentation |
| **ONNX** | Cross-platform, optimized | Limited training features | Model exchange, inference |

**Switching Backends**:
1. Select from dropdown
2. Recompile model
3. Code generation adapts automatically

### Dataset Selection

**Built-in Datasets**:
- **MNIST**: Handwritten digits (28×28×1, 10 classes)
- **CIFAR10**: Color images (32×32×3, 10 classes)
- **CIFAR100**: Color images (32×32×3, 100 classes)
- **ImageNet**: Large-scale (224×224×3, 1000 classes)

**Custom Dataset**:
1. Select "Custom" from dropdown
2. Enter dataset path
3. Ensure data format compatibility

**Dataset Requirements**:
- Match model input shape
- Proper train/test split
- Compatible data format (NumPy, TF Dataset, etc.)

![Dataset Selection](../images/aquarium/dataset-selection.png)

### Training Configuration

#### Epochs
- **Range**: 1-1000
- **Default**: 10
- **Tip**: Start small (5-10) for testing, increase for final training

#### Batch Size
- **Range**: 1-2048
- **Default**: 32
- **Tip**: Larger = faster but more memory. Powers of 2 are efficient (16, 32, 64, 128)

#### Validation Split
- **Range**: 0.0-1.0
- **Default**: 0.2 (20%)
- **Tip**: 0.2 = 80% train, 20% validation

### Training Options

**Auto-flatten Output** ✓
- Automatically flattens output to match dataset
- Recommended for most models

**HPO (Hyperparameter Optimization)** ☐
- Enables Optuna-based hyperparameter tuning
- Experimental feature

**Verbose Output** ✓
- Shows detailed training logs
- Recommended for debugging

**Save Model Weights** ✓
- Saves trained weights to file
- Enables model reuse

![Training Config](../images/aquarium/training-config.png)

### Compilation Process

**Click "Compile" to**:
1. Validate parsed model
2. Generate backend-specific code
3. Create training script
4. Save to temporary file

**Compile Output**:
```
[COMPILE] Starting compilation...
[COMPILE] Backend: TensorFlow
[COMPILE] Generating model code...
[COMPILE] Adding dataset loader...
[COMPILE] Creating training loop...
[COMPILE] Writing to: /tmp/aquarium_model_xyz.py
[COMPILE] ✓ Compilation successful!
[COMPILE] Script ready to execute
```

**Status Changes**: Idle → Compiled

### Execution Process

**Click "Run" to**:
1. Execute compiled script
2. Start training process
3. Stream output to console
4. Update metrics visualization

**Execution Output**:
```
[RUN] Starting execution...
[RUN] Loading MNIST dataset...
[RUN] Train samples: 48000, Val samples: 12000
[RUN] Starting training...

Epoch 1/10
[METRICS] Loss: 0.3245, Accuracy: 0.8921, Val Loss: 0.2156, Val Accuracy: 0.9234
Epoch 2/10
[METRICS] Loss: 0.1823, Accuracy: 0.9456, Val Loss: 0.1567, Val Accuracy: 0.9512
...
Epoch 10/10
[METRICS] Loss: 0.0234, Accuracy: 0.9923, Val Loss: 0.0445, Val Accuracy: 0.9834

[SUCCESS] Training completed!
[SUCCESS] Final accuracy: 0.9834
[SUCCESS] Model saved to: model_weights.h5
```

**Status Changes**: Compiled → Running → Idle

![Execution Process](../images/aquarium/execution-process.png)

### Stopping Execution

**Click "Stop" to**:
- Terminate running training process
- Clean up resources
- Reset status to Stopped

**Use Cases**:
- Training taking too long
- Noticed error in configuration
- Want to try different parameters

### Console Output

**Color Coding**:
- `[COMPILE]` - Cyan: Compilation messages
- `[RUN]` - Green: Execution messages
- `[METRICS]` - Blue: Training metrics
- `[SUCCESS]` - Green: Success messages
- `[ERROR]` - Red: Error messages

**Features**:
- Auto-scroll to latest output
- Monospace font for alignment
- Persistent across runs
- Clearable with "Clear" button

### Training Metrics

**Real-time Visualization** (Coming Soon):
- Loss curves (training & validation)
- Accuracy curves
- Epoch progression
- Learning rate schedule

**Current**: Metrics logged to console

![Training Metrics](../images/aquarium/training-metrics.png)

## Debugging

### NeuralDbg Integration

**Launch Debugger**:
1. Switch to **Debugger** tab
2. Click **"Launch NeuralDbg"**
3. Opens separate debugging dashboard

**Debugging Features**:
- Layer-by-layer execution trace
- Tensor shape inspection
- Gradient flow visualization
- Dead neuron detection
- Memory & FLOP profiling
- Anomaly detection (NaN/Inf)

**Debugger Dashboard**:
```
http://localhost:8050
```

**Use Cases**:
- Model not converging
- Shape mismatches
- Gradient issues
- Performance bottlenecks

![Debugger](../images/aquarium/debugger.png)

### Common Issues

**1. Shape Mismatch**
- **Symptom**: Error during compilation/execution
- **Solution**: Check input shape matches dataset
- **Tool**: Use Visualize to inspect shapes

**2. Loss Not Decreasing**
- **Symptom**: Loss stays constant or increases
- **Solution**: Check learning rate, data normalization
- **Tool**: Use NeuralDbg gradient analysis

**3. Out of Memory**
- **Symptom**: Process crashes during training
- **Solution**: Reduce batch size or model size
- **Tool**: Check FLOP profiler

## Export & Integration

### Exporting Scripts

**Export Process**:
1. Compile model successfully
2. Click **"Export Script"**
3. Enter filename (e.g., `my_model.py`)
4. Choose export location
5. Click **"Export"**

**Export Dialog**:
```
┌─────────────────────────────────────────┐
│  Export Training Script                 │
│                                         │
│  Filename: [my_model.py              ] │
│                                         │
│  Location: [/home/user/projects/      ] │
│            [Browse...]                  │
│                                         │
│  ☑ Include comments                    │
│  ☑ Include dataset loader              │
│  ☐ Export metadata file                │
│                                         │
│  [Cancel]             [Export]          │
└─────────────────────────────────────────┘
```

**Exported File Contents**:
- Complete training script
- Dataset loading code
- Model architecture
- Training loop
- Evaluation code
- Model saving logic

![Export Script](../images/aquarium/export-script.png)

### Opening in External Editor

**Click "Open in IDE" to**:
- Launch exported script in default editor
- Supports: VS Code, PyCharm, Sublime, Atom, Notepad++

**Platform Support**:
- **Windows**: Uses `os.startfile()`
- **macOS**: Uses `open` command
- **Linux**: Uses `xdg-open`

**Troubleshooting**:
- Set default program for `.py` files
- Or open manually from export location

### File Organization

**Default Structure**:
```
~/.neural/aquarium/
├── compiled/           # Compiled scripts
│   ├── model_abc.py
│   └── model_xyz.py
├── exported/           # Exported scripts
│   ├── my_model.py
│   └── production_model.py
└── temp/               # Temporary files
    └── temp_model.py
```

**Managing Files**:
- Exported files persist across sessions
- Compiled files are temporary
- Clean up old files periodically

## Advanced Features

### Keyboard Shortcuts

| Action | Shortcut | Description |
|--------|----------|-------------|
| Parse DSL | `Ctrl+P` | Validate current model |
| Compile | `Ctrl+B` | Build model code |
| Run | `Ctrl+R` | Execute training |
| Stop | `Ctrl+C` | Terminate execution |
| Save | `Ctrl+S` | Save current model |
| New | `Ctrl+N` | Create new model |
| Open | `Ctrl+O` | Open model file |
| Example | `Ctrl+E` | Load random example |

*(Note: Shortcuts under development)*

### Model Templates

**Using Templates**:
1. Click "Load Example"
2. Modify template for your needs
3. Parse and test
4. Save as new model

**Template Categories**:
- **Classification**: CNN, ResNet, VGG
- **Sequence**: LSTM, GRU, Transformer
- **Generative**: Autoencoder, VAE
- **Custom**: Build from scratch

### Batch Processing

**Running Multiple Models**:
1. Export first model
2. Clear editor
3. Load/write second model
4. Compile and run
5. Repeat

**Tip**: Use scripting for automation:
```python
from neural.aquarium.src.components.runner import ExecutionManager

manager = ExecutionManager()
for model_file in models:
    manager.compile_model(model_file, backend='tensorflow')
    manager.run_script()
```

### Configuration Profiles

**Create Profile** (via config file):
```yaml
# ~/.neural/aquarium/profiles/research.yaml
backends:
  default: pytorch
training:
  default_epochs: 50
  default_batch_size: 64
ui:
  theme: darkly
```

**Load Profile**:
```bash
python -m neural.aquarium.aquarium --profile research
```

## Best Practices

### Model Development Workflow

1. **Start Small**
   - Begin with simple architecture
   - Test with few epochs (5-10)
   - Verify training works

2. **Iterate Gradually**
   - Add layers incrementally
   - Test after each change
   - Monitor metrics

3. **Validate Frequently**
   - Parse after each edit
   - Check shape propagation
   - Run short training tests

4. **Export Regularly**
   - Save working versions
   - Keep model history
   - Document changes

### Performance Optimization

**Training Speed**:
- Use GPU when available
- Increase batch size (if memory allows)
- Reduce model complexity
- Use efficient data loaders

**Memory Management**:
- Start with small batch sizes
- Monitor memory usage
- Clear console regularly
- Close unused browser tabs

### Code Organization

**File Naming**:
```
model_mnist_v1.neural      # First version
model_mnist_v2.neural      # Improved version
model_mnist_final.neural   # Production version
```

**Comments in DSL**:
```neural
# MNIST Classifier v1
# Author: Your Name
# Date: 2024-12-13
# Description: Simple CNN for digit recognition

network MNISTv1 {
    # Input layer
    input: (None, 28, 28, 1)
    
    # Feature extraction
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        ...
```

### Debugging Tips

1. **Use Verbose Output**
   - Enable verbose logging
   - Watch for warnings
   - Check shapes at each layer

2. **Start with Examples**
   - Load working example
   - Modify incrementally
   - Keep working version

3. **Check Console First**
   - Read error messages carefully
   - Look for line numbers
   - Check syntax errors

4. **Use NeuralDbg**
   - Launch debugger for complex issues
   - Inspect layer outputs
   - Analyze gradients

## FAQ

### General Questions

**Q: What is Aquarium IDE?**
A: A web-based IDE for Neural DSL that provides model editing, compilation, execution, and debugging capabilities.

**Q: Do I need to know Python?**
A: No, you only need to know Neural DSL syntax. Aquarium generates Python code for you.

**Q: Can I use my own datasets?**
A: Yes, select "Custom" dataset and provide the path to your data.

**Q: Which backend should I use?**
A: TensorFlow for production, PyTorch for research, ONNX for cross-platform deployment.

### Technical Questions

**Q: Can I train on GPU?**
A: Yes, if TensorFlow/PyTorch with GPU support is installed and GPU is available.

**Q: How do I save my trained model?**
A: Enable "Save model weights" option. Weights saved to `model_weights.h5` (TF) or `model_weights.pth` (PyTorch).

**Q: Can I export to other formats?**
A: Yes, use `neural export` CLI command for ONNX, TFLite, TorchScript, etc.

**Q: Can I use Aquarium remotely?**
A: Yes, start with `--host 0.0.0.0` (requires firewall configuration).

### Troubleshooting Questions

**Q: Compilation fails, what do I do?**
A: Check parse status first. Ensure model syntax is valid. Check console for specific errors.

**Q: Training is very slow, how to speed up?**
A: Use GPU if available. Increase batch size. Reduce model complexity. Check dataset loading efficiency.

**Q: Console not updating, why?**
A: Wait a moment (buffered output). Refresh browser if needed. Check if process is running.

**Q: Export button grayed out, why?**
A: Compile model first. Ensure compilation succeeded. Check status indicator.

### Integration Questions

**Q: Can I use with Neural CLI?**
A: Yes, Aquarium is part of Neural DSL ecosystem. Models work across all tools.

**Q: Can I import existing models?**
A: Yes, use "Open" button or paste DSL code directly into editor.

**Q: Can I collaborate with others?**
A: Export models as `.neural` files and share. Collaborative editing coming soon.

**Q: Can I deploy trained models?**
A: Yes, use Neural CLI export commands or export script from Aquarium.

## Additional Resources

### Documentation
- [Installation Guide](installation.md)
- [Keyboard Shortcuts](keyboard-shortcuts.md)
- [Plugin Development](plugin-development.md)
- [Architecture Overview](architecture.md)
- [Troubleshooting Guide](troubleshooting.md)

### External Links
- [Neural DSL Documentation](../../docs/dsl.md)
- [GitHub Repository](https://github.com/Lemniscate-world/Neural)
- [Discord Community](https://discord.gg/KFku4KvS)
- [Video Tutorials](video-tutorials.md)

### Support
- **Documentation**: [README.md](../../neural/aquarium/README.md)
- **Issues**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
- **Email**: Lemniscate_zero@proton.me

---

**Version**: 1.0  
**Last Updated**: December 2024  
**License**: MIT
