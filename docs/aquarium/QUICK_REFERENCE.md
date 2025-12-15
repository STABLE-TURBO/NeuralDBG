# Aquarium IDE - Quick Reference Guide

**One-Page Cheat Sheet** | **Version**: 1.0.0

---

## 🚀 Quick Start (30 Seconds)

```bash
# 1. Install
pip install neural-dsl[dashboard]

# 2. Launch
python -m neural.aquarium.aquarium

# 3. Open browser
http://localhost:8052
```

**Your First Model:**
1. Click "Load Example" → 2. Click "Parse DSL" → 3. Click "Compile" → 4. Click "Run"

---

## 📋 Essential Commands

### File Operations
| Action | Shortcut | Description |
|--------|----------|-------------|
| New | `Ctrl+N` | Create new model |
| Open | `Ctrl+O` | Load model file |
| Save | `Ctrl+S` | Save current model |
| Example | `Ctrl+E` | Load random example |

### Build & Run
| Action | Shortcut | Description |
|--------|----------|-------------|
| Parse | `Ctrl+P` | Validate DSL |
| Compile | `Ctrl+B` | Generate code |
| Run | `Ctrl+R` | Start training |
| Stop | `Ctrl+C` | Terminate process |

### Editor
| Action | Shortcut | Description |
|--------|----------|-------------|
| Select All | `Ctrl+A` | Select all text |
| Copy | `Ctrl+C` | Copy selection |
| Paste | `Ctrl+V` | Paste from clipboard |
| Undo | `Ctrl+Z` | Undo last action |
| Find | `Ctrl+F` | Find in editor |
| Comment | `Ctrl+/` | Toggle comment |

---

## 📝 DSL Syntax Cheat Sheet

### Basic Structure
```neural
network ModelName {
    input: (None, height, width, channels)
    
    layers:
        # Your layers here
        
    loss: loss_function
    optimizer: Optimizer(params)
}
```

### Common Layers
```neural
# Convolutional
Conv2D(filters=32, kernel_size=(3,3), activation=relu)
MaxPooling2D(pool_size=(2,2))
AveragePooling2D(pool_size=(2,2))

# Dense
Dense(units=128, activation=relu)
Dropout(rate=0.5)
BatchNormalization()

# Recurrent
LSTM(units=64, return_sequences=true)
GRU(units=64, return_sequences=false)

# Special
Flatten()
Reshape(target_shape=(28, 28, 1))
Embedding(input_dim=1000, output_dim=128)

# Output
Output(units=10, activation=softmax)
```

### Activations
```neural
relu, sigmoid, tanh, softmax, linear, elu, selu, swish
```

### Loss Functions
```neural
categorical_crossentropy
binary_crossentropy
mean_squared_error
mean_absolute_error
sparse_categorical_crossentropy
```

### Optimizers
```neural
Adam(learning_rate=0.001)
SGD(learning_rate=0.01, momentum=0.9)
RMSprop(learning_rate=0.001)
Adagrad(learning_rate=0.01)
```

---

## 🎯 Common Patterns

### Image Classification
```neural
network ImageClassifier {
    input: (None, 224, 224, 3)
    layers:
        Conv2D(filters=64, kernel_size=(3,3), activation=relu)
        MaxPooling2D(pool_size=(2,2))
        Conv2D(filters=128, kernel_size=(3,3), activation=relu)
        MaxPooling2D(pool_size=(2,2))
        Flatten()
        Dense(units=512, activation=relu)
        Dropout(rate=0.5)
        Output(units=1000, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

### Text Classification
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

### Autoencoder
```neural
network Autoencoder {
    input: (None, 784)
    layers:
        # Encoder
        Dense(units=256, activation=relu)
        Dense(units=128, activation=relu)
        Dense(units=64, activation=relu)
        # Decoder
        Dense(units=128, activation=relu)
        Dense(units=256, activation=relu)
        Output(units=784, activation=sigmoid)
    loss: mean_squared_error
    optimizer: Adam(learning_rate=0.001)
}
```

---

## ⚙️ Configuration Quick Reference

### Backend Selection
| Backend | Use Case | Pros |
|---------|----------|------|
| TensorFlow | Production | Keras API, TensorBoard |
| PyTorch | Research | Dynamic graphs, flexible |
| ONNX | Inference | Cross-platform, optimized |

### Dataset Options
| Dataset | Shape | Classes |
|---------|-------|---------|
| MNIST | (28, 28, 1) | 10 |
| CIFAR10 | (32, 32, 3) | 10 |
| CIFAR100 | (32, 32, 3) | 100 |
| ImageNet | (224, 224, 3) | 1000 |
| Custom | Your data | Varies |

### Training Parameters
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Epochs | 1-1000 | 10 | Start small (5-10) |
| Batch Size | 1-2048 | 32 | Powers of 2 work best |
| Validation Split | 0.0-1.0 | 0.2 | 20% for validation |

---

## 🔍 Troubleshooting Quick Fixes

### Issue: Port already in use
```bash
python -m neural.aquarium.aquarium --port 8053
```

### Issue: Module not found
```bash
pip install neural-dsl[dashboard]
```

### Issue: Backend not available
```bash
pip install tensorflow  # or pytorch, onnx
```

### Issue: Out of memory
- Reduce batch size (try 16, 8, or 4)
- Simplify model architecture
- Close other applications

### Issue: Parse error
- Check DSL syntax carefully
- Verify colons, commas, parentheses
- Use "Load Example" as template
- Check indentation (use 4 spaces)

### Issue: Training slow
```bash
# Check GPU
nvidia-smi

# Verify GPU usage
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 📊 Console Output Guide

### Message Types
```
[COMPILE]  - Cyan    - Compilation messages
[RUN]      - Green   - Execution messages
[METRICS]  - Blue    - Training metrics
[SUCCESS]  - Green   - Success messages
[ERROR]    - Red     - Error messages
[WARNING]  - Yellow  - Warning messages
```

### Training Output Example
```
[RUN] Starting execution...
[RUN] Loading MNIST dataset...
[RUN] Train samples: 48000, Val samples: 12000

Epoch 1/10
1500/1500 [==============================] - 12s
[METRICS] loss: 0.3245 - accuracy: 0.8921
[METRICS] val_loss: 0.2156 - val_accuracy: 0.9234

[SUCCESS] Training completed!
[SUCCESS] Final accuracy: 0.9834
```

---

## 🐛 Debugging Workflow

1. **Parse First**: Click "Parse DSL" to validate
2. **Check Model Info**: Verify layers and shapes
3. **Start Simple**: Use 5 epochs for testing
4. **Watch Console**: Look for error messages
5. **Use NeuralDbg**: Launch debugger for deep inspection
6. **Iterate**: Fix issues one at a time

### Common Debug Checks
- ✅ Input shape matches dataset
- ✅ Output units match number of classes
- ✅ Activation functions appropriate
- ✅ Loss function matches task
- ✅ Learning rate reasonable (0.001-0.01)

---

## 📦 Export & Integration

### Export Script
1. Click "Export Script"
2. Enter filename: `my_model.py`
3. Choose location
4. Click "Export"

### File Locations
```
~/.neural/aquarium/
├── compiled/    # Temp compiled scripts
├── exported/    # Your exported scripts
├── models/      # Saved DSL files (.neural)
└── weights/     # Trained weights (.h5, .pth)
```

### Use Exported Script
```bash
# Run directly
python my_model.py

# Modify and reuse
# Script is fully standalone with all dependencies
```

---

## 🔌 Plugin Quick Start

### Install Plugin
```bash
# From npm
npm install @neural/plugin-name

# From PyPI
pip install neural-aquarium-plugin-name

# Manual
cp -r my-plugin ~/.neural/aquarium/plugins/
```

### Enable Plugin
1. Menu → Plugins → Marketplace
2. Find plugin
3. Click "Install" or "Enable"

### Popular Plugins
- **GitHub Copilot**: AI code completion
- **Custom Viz**: Advanced visualizations
- **Dark Ocean Theme**: Beautiful dark theme

---

## 🎓 Learning Path

### Week 1: Basics
- [ ] Install Aquarium
- [ ] Load example models
- [ ] Parse and compile
- [ ] Run first training
- [ ] Export script

### Week 2: DSL Mastery
- [ ] Write custom models
- [ ] Understand all layer types
- [ ] Master common patterns
- [ ] Debug parse errors
- [ ] Optimize hyperparameters

### Week 3: Advanced Features
- [ ] Use NeuralDbg
- [ ] Try all backends
- [ ] Custom datasets
- [ ] HPO experiments
- [ ] Plugin installation

### Week 4: Production
- [ ] Export for deployment
- [ ] Integrate with CI/CD
- [ ] Optimize performance
- [ ] Contribute plugins
- [ ] Help community

---

## 📞 Getting Help

### Quick Links
- 📖 [Complete Manual](AQUARIUM_IDE_MANUAL.md)
- 🔧 [API Reference](API_REFERENCE.md)
- 🐛 [Troubleshooting](troubleshooting.md)
- ⌨️ [Keyboard Shortcuts](keyboard-shortcuts.md)

### Community
- 💬 [Discord](https://discord.gg/KFku4KvS)
- 🐙 [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- 📧 Email: Lemniscate_zero@proton.me

---

## 💡 Pro Tips

1. **Start Simple**: Begin with examples, modify gradually
2. **Parse Often**: Validate after each change
3. **Small Epochs**: Test with 5 epochs before full training
4. **Save Frequently**: Export working versions
5. **Use Console**: Error messages are helpful
6. **GPU Matters**: Training is 10-100x faster on GPU
7. **Batch Size**: Powers of 2 (16, 32, 64, 128)
8. **Learn by Doing**: Try different architectures
9. **Read Docs**: Complete manual has everything
10. **Ask for Help**: Community is friendly!

---

## 🎯 Common Tasks

### Task: Change Learning Rate
```neural
optimizer: Adam(learning_rate=0.0001)  # Reduce if training unstable
```

### Task: Add Dropout for Regularization
```neural
Dense(units=128, activation=relu)
Dropout(rate=0.5)  # Add after Dense layers
```

### Task: Increase Model Capacity
```neural
# More filters
Conv2D(filters=64, ...)  # Instead of 32

# More units
Dense(units=256, ...)  # Instead of 128
```

### Task: Reduce Overfitting
```neural
# Add dropout
Dropout(rate=0.5)

# Add batch normalization
BatchNormalization()

# Use data augmentation (in dataset config)
```

### Task: Speed Up Training
```python
# Increase batch size
batch_size=64  # Instead of 32

# Reduce model size
# Fewer layers or smaller units

# Use GPU
# Ensure TensorFlow/PyTorch GPU version installed
```

---

## 🎨 UI Layout Reference

```
┌────────────────────────────────────────────────┐
│  Header: [New] [Open] [Save] [Help] [⚙️]      │
├──────────┬─────────────────────────────────────┤
│          │  Tabs: [Runner] Debugger Viz Docs   │
│  Editor  ├─────────────────────────────────────┤
│          │                                      │
│ [Parse]  │  Backend: [TensorFlow ▼]            │
│ [Viz]    │  Dataset: [MNIST ▼]                 │
│ [Example]│  Epochs: [10]  Batch: [32]         │
│          │                                      │
│  Model   │  [Compile] [Run] [Stop]             │
│  Info    │                                      │
│          │  Console:                            │
│          │  ┌───────────────────────────────┐  │
│          │  │ Output appears here...        │  │
│          │  └───────────────────────────────┘  │
└──────────┴─────────────────────────────────────┘
```

---

## 📈 Typical Workflow

```
Write/Load DSL
     ↓
Parse DSL (Ctrl+P)
     ↓
Review Model Info
     ↓
Configure Training
     ↓
Compile (Ctrl+B)
     ↓
Run (Ctrl+R)
     ↓
Monitor Console
     ↓
Export Script
     ↓
Deploy/Integrate
```

---

## 🎁 Bonus: Python API One-Liner

```python
from neural.aquarium.src.components.runner import ExecutionManager
ExecutionManager().compile_model(dsl_code, 'tensorflow').run_script('mnist')
```

---

**Print this page for quick reference!**

[⭐ Star on GitHub](https://github.com/Lemniscate-world/Neural) • 
[📚 Full Manual](AQUARIUM_IDE_MANUAL.md) • 
[💬 Get Help](https://discord.gg/KFku4KvS)

---

**Version**: 1.0.0 | **License**: MIT | **Made with ❤️ by Neural DSL Team**
