# Neural Aquarium - Quick Reference Card

## Launch Commands

```bash
# Standard launch
python -m neural.aquarium.aquarium

# Custom port
python -m neural.aquarium.aquarium --port 8052

# Debug mode
python -m neural.aquarium.aquarium --debug
```

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Neural Aquarium IDE Header                                 │
├─────────────┬───────────────────────────────────────────────┤
│             │  Runner Tab                                   │
│ DSL Editor  │  ┌──────────────────────────────────────────┐ │
│             │  │ Backend: [TensorFlow ▼]                  │ │
│ [Parse DSL] │  │ Dataset: [MNIST ▼]                       │ │
│ [Visualize] │  │ Epochs: [10] Batch: [32] Val: [0.2]     │ │
│ [Example]   │  │                                          │ │
│             │  │ [Compile] [Run] [Stop]                   │ │
│             │  │ [Export] [Open IDE] [Clear]              │ │
│ Model Info  │  ├──────────────────────────────────────────┤ │
│             │  │ Console Output                           │ │
│             │  │ Ready to compile and run models...       │ │
│             │  │                                          │ │
└─────────────┴──┴──────────────────────────────────────────┴─┘
```

## Workflow

```
1. Write DSL → 2. Parse → 3. Configure → 4. Compile → 5. Run
                                                         ↓
                                                    6. Monitor
                                                         ↓
                                              7. Export or Iterate
```

## Backend Selection

| Backend    | Language    | Use Case                |
|------------|-------------|-------------------------|
| TensorFlow | Python      | Production, deployment  |
| PyTorch    | Python      | Research, flexibility   |
| ONNX       | Cross-platform | Model exchange      |

## Dataset Options

| Dataset   | Shape           | Classes | Size    |
|-----------|-----------------|---------|---------|
| MNIST     | (28, 28, 1)     | 10      | 60K     |
| CIFAR10   | (32, 32, 3)     | 10      | 50K     |
| CIFAR100  | (32, 32, 3)     | 100     | 50K     |
| ImageNet  | (224, 224, 3)   | 1000    | 1M+     |
| Custom    | User-defined    | Any     | Any     |

## Training Configuration

| Parameter        | Range      | Default | Description              |
|------------------|------------|---------|--------------------------|
| Epochs           | 1-1000     | 10      | Training iterations      |
| Batch Size       | 1-2048     | 32      | Samples per update       |
| Validation Split | 0.0-1.0    | 0.2     | Fraction for validation  |

## Action Buttons

| Button      | Shortcut | Function                          |
|-------------|----------|-----------------------------------|
| Compile     | -        | Generate backend code             |
| Run         | -        | Execute training script           |
| Stop        | -        | Terminate running process         |
| Export      | -        | Save script to file               |
| Open in IDE | -        | Launch in default editor          |
| Clear       | -        | Reset console output              |

## Status Indicators

| Badge     | Color  | Meaning                        |
|-----------|--------|--------------------------------|
| Idle      | Gray   | Ready for compilation          |
| Compiled  | Green  | Code generated, ready to run   |
| Running   | Blue   | Training in progress           |
| Error     | Red    | Compilation or execution error |
| Stopped   | Yellow | Process stopped by user        |

## Console Log Prefixes

```
[COMPILE]  - Compilation stage messages
[RUN]      - Execution stage messages
[ERROR]    - Error messages
[SUCCESS]  - Success confirmations
[METRICS]  - Training metrics
```

## Common DSL Patterns

### Simple CNN
```neural
network SimpleCNN {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3,3), activation=relu)
        MaxPooling2D(pool_size=(2,2))
        Flatten()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

### Dense Network
```neural
network DenseNet {
    input: (None, 784)
    layers:
        Dense(units=256, activation=relu)
        Dropout(rate=0.3)
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

### LSTM Text
```neural
network TextLSTM {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64)
        Dense(units=64, activation=relu)
        Output(units=1, activation=sigmoid)
    loss: binary_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

## Keyboard Tips

- **Tab**: Navigate between fields
- **Enter**: Submit in text fields
- **Ctrl+A**: Select all in editor
- **Scroll**: Auto-scrolls console to bottom

## File Locations

```
~/.neural/aquarium/
├── compiled/      # Compiled scripts
├── exported/      # Exported scripts
└── temp/          # Temporary files
```

## Troubleshooting

| Issue                  | Solution                              |
|------------------------|---------------------------------------|
| Port in use            | Change port with `--port` flag        |
| Script won't run       | Check dataset matches input shape     |
| Logs not streaming     | Wait a moment, refresh if needed      |
| Can't export           | Check directory permissions           |
| Process won't stop     | Click Stop again, check Task Manager  |

## Tips & Tricks

✅ **DO**:
- Parse DSL before compiling
- Start with small epochs for testing
- Export successful models
- Use examples as templates
- Monitor console for errors

❌ **DON'T**:
- Run without parsing first
- Use very large batch sizes on small datasets
- Forget to select correct dataset
- Run multiple models simultaneously (stop first)

## Example Workflow

```bash
# 1. Launch Aquarium
python -m neural.aquarium.aquarium

# 2. In Browser
- Load example model
- Click "Parse DSL"
- Select "TensorFlow" backend
- Select "MNIST" dataset
- Set epochs to 5
- Click "Compile"
- Wait for success message
- Click "Run"
- Monitor training in console
- Click "Export" when done
- Enter filename: my_model.py
- Click "Export"
- Click "Open in IDE" to edit
```

## Getting Help

- **Documentation**: `neural/aquarium/README.md`
- **Quick Start**: `neural/aquarium/QUICKSTART.md`
- **Technical**: `neural/aquarium/IMPLEMENTATION.md`
- **Features**: `neural/aquarium/FEATURES.md`

## Version Info

```
Application: Neural Aquarium IDE
Version: 0.1.0
Python: 3.8+
Framework: Dash + Bootstrap
License: Same as Neural DSL
```

---

**Quick Links**:
- Main Docs: [README.md](README.md)
- Tutorial: [QUICKSTART.md](QUICKSTART.md)
- Features: [FEATURES.md](FEATURES.md)
