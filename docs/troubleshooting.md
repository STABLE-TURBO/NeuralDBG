# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when working with Neural DSL.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Parser Errors](#parser-errors)
- [Shape Propagation Errors](#shape-propagation-errors)
- [Code Generation Issues](#code-generation-issues)
- [Runtime Errors](#runtime-errors)
- [Dashboard Issues](#dashboard-issues)
- [HPO Issues](#hpo-issues)
- [Cloud Integration Issues](#cloud-integration-issues)
- [Performance Issues](#performance-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### Problem: `pip install neural-dsl` fails with dependency conflicts

**Symptoms:**
```
ERROR: Cannot install neural-dsl because these package versions have conflicting dependencies.
```

**Solutions:**

1. **Use a clean virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   pip install neural-dsl
   ```

2. **Install without optional dependencies first:**
   ```bash
   pip install neural-dsl --no-deps
   pip install lark click  # Install minimal required deps
   ```

3. **Specify backend-specific installation:**
   ```bash
   # For TensorFlow
   pip install neural-dsl tensorflow
   
   # For PyTorch
   pip install neural-dsl torch
   ```

### Problem: Import errors after installation

**Symptoms:**
```python
ModuleNotFoundError: No module named 'neural'
```

**Solutions:**

1. **Verify installation:**
   ```bash
   pip show neural-dsl
   python -c "import neural; print(neural.__version__)"
   ```

2. **Check Python path conflicts:**
   ```bash
   # Ensure you're using the correct Python
   which python  # Linux/macOS
   where python  # Windows
   ```

3. **Reinstall in editable mode (for development):**
   ```bash
   pip install -e .
   ```

### Problem: Version mismatch warnings

**Symptoms:**
```
WARNING: Version mismatch between CLI and package
```

**Solutions:**

1. **Update to latest version:**
   ```bash
   pip install --upgrade neural-dsl
   ```

2. **Clear pip cache:**
   ```bash
   pip cache purge
   pip install --force-reinstall neural-dsl
   ```

---

## Parser Errors

### Problem: Syntax errors in DSL files

**Symptoms:**
```
ERROR at line 5: Unexpected token 'Dense'
```

**Common Causes & Solutions:**

1. **Missing commas or colons:**
   ```yaml
   # ❌ Wrong
   network MyModel {
     input (28, 28, 1)  # Missing colon
   
   # ✅ Correct
   network MyModel {
     input: (28, 28, 1)
   ```

2. **Incorrect indentation:**
   ```yaml
   # ❌ Wrong - inconsistent indentation
   network MyModel {
     input: (28, 28, 1)
     layers:
        Dense(64)  # 4 spaces
      Dense(32)    # 2 spaces
   
   # ✅ Correct - consistent indentation
   network MyModel {
     input: (28, 28, 1)
     layers:
       Dense(64)
       Dense(32)
   ```

3. **Invalid parameter syntax:**
   ```yaml
   # ❌ Wrong
   Conv2D(filters=32 kernel_size=(3,3))  # Missing comma
   
   # ✅ Correct
   Conv2D(filters=32, kernel_size=(3,3))
   ```

### Problem: Layer not recognized

**Symptoms:**
```
ERROR: Unknown layer type 'Conv2d'
```

**Solutions:**

1. **Check layer name spelling and capitalization:**
   - Use `Conv2D` not `Conv2d` or `conv2d`
   - Use `Dense` not `dense` or `FC`

2. **Verify layer is supported:**
   ```bash
   # List all supported layers
   neural layers --list
   ```

3. **For custom layers, ensure they're defined:**
   ```yaml
   # Define custom layer first
   define CustomBlock {
     Conv2D(32, (3,3), "relu")
     BatchNormalization()
   }
   
   # Then use it
   network MyModel {
     layers:
       CustomBlock()
   }
   ```

### Problem: Parameter validation errors

**Symptoms:**
```
ERROR: Dense units must be positive integer, got -5
ERROR: Dropout rate must be in range [0, 1], got 1.5
```

**Solutions:**

1. **Check parameter constraints:**
   - `Dense.units` > 0
   - `Dropout.rate` ∈ [0, 1]
   - `Conv2D.filters` > 0
   - `Conv2D.kernel_size` > 0

2. **Use correct data types:**
   ```yaml
   # ❌ Wrong - units should be int
   Dense(units=64.5)
   
   # ✅ Correct
   Dense(units=64)
   ```

3. **Validate HPO ranges:**
   ```yaml
   # ❌ Wrong - min > max
   Dense(units=HPO(range(100, 10, step=10)))
   
   # ✅ Correct
   Dense(units=HPO(range(10, 100, step=10)))
   ```

---

## Shape Propagation Errors

### Problem: Shape mismatch between layers

**Symptoms:**
```
ERROR: Shape mismatch at layer 3 (Dense)
Expected input: (None, 128)
Got: (None, 32, 32, 64)
```

**Solutions:**

1. **Add Flatten layer before Dense:**
   ```yaml
   # ❌ Wrong
   Conv2D(64, (3,3))
   Dense(128)  # Can't connect 4D to Dense
   
   # ✅ Correct
   Conv2D(64, (3,3))
   Flatten()  # Convert 4D to 2D
   Dense(128)
   ```

2. **Use GlobalAveragePooling instead:**
   ```yaml
   Conv2D(64, (3,3))
   GlobalAveragePooling2D()  # Alternative to Flatten
   Dense(128)
   ```

3. **Check input shape format:**
   ```yaml
   # For TensorFlow (channels-last)
   input: (28, 28, 1)
   
   # For PyTorch (channels-first) - auto-converted
   input: (1, 28, 28)
   ```

### Problem: Invalid input shape

**Symptoms:**
```
ERROR: Input shape must be a tuple, got 28
```

**Solutions:**

1. **Always use tuple format:**
   ```yaml
   # ❌ Wrong
   input: 784
   
   # ✅ Correct
   input: (784,)
   # Or for images
   input: (28, 28, 1)
   ```

2. **Include None for batch dimension (optional):**
   ```yaml
   # Both are valid
   input: (None, 28, 28, 1)
   input: (28, 28, 1)  # None is implicit
   ```

### Problem: Shape propagation visualization errors

**Symptoms:**
```
ERROR: Cannot generate shape flow diagram
```

**Solutions:**

1. **Ensure model is valid before visualization:**
   ```bash
   # Validate first
   neural compile model.neural --dry-run
   
   # Then visualize
   neural visualize model.neural
   ```

2. **Check for circular dependencies in macros:**
   ```yaml
   # ❌ Wrong - circular reference
   define A {
     B()
   }
   define B {
     A()
   }
   ```

---

## Code Generation Issues

### Problem: Backend-specific generation fails

**Symptoms:**
```
ERROR: Cannot generate PyTorch code for layer 'TransformerEncoder'
```

**Solutions:**

1. **Check layer support per backend:**
   ```bash
   neural layers --backend pytorch
   neural layers --backend tensorflow
   ```

2. **Use alternative layer or custom implementation:**
   ```yaml
   # If TransformerEncoder not supported in PyTorch:
   # Implement using basic layers
   layers:
     MultiHeadAttention(num_heads=8)
     Dense(2048, activation="relu")
     Dense(512)
   ```

3. **Switch backend:**
   ```bash
   # Try TensorFlow instead
   neural compile model.neural --backend tensorflow
   ```

### Problem: Optimizer parameters not generated correctly

**Symptoms:**
Generated code has wrong optimizer configuration.

**Solutions:**

1. **Check HPO parameter format:**
   ```yaml
   # ✅ Correct
   optimizer: Adam(learning_rate=0.001)
   
   # ✅ With HPO
   optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
   ```

2. **Verify optimizer name:**
   ```yaml
   # Supported: Adam, SGD, RMSprop, Adagrad, Adadelta
   optimizer: Adam(learning_rate=0.001)
   ```

### Problem: Generated code has syntax errors

**Symptoms:**
```python
# Generated code:
model.add(Dense(units=))  # Missing value
```

**Solutions:**

1. **Update to latest version:**
   ```bash
   pip install --upgrade neural-dsl
   ```

2. **Report bug with minimal example:**
   ```bash
   # Create minimal reproducible DSL
   # Submit issue: https://github.com/Lemniscate-world/Neural/issues
   ```

---

## Runtime Errors

### Problem: CUDA out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   train {
     batch_size: 32  # Try 16 or 8
     epochs: 15
   }
   ```

2. **Use gradient accumulation:**
   ```python
   # In generated code, add:
   for i, (x, y) in enumerate(train_loader):
       loss = loss / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Enable mixed precision:**
   ```bash
   neural compile model.neural --mixed-precision
   ```

### Problem: NaN loss during training

**Symptoms:**
```
Epoch 1/10: loss=0.45
Epoch 2/10: loss=nan
```

**Solutions:**

1. **Lower learning rate:**
   ```yaml
   optimizer: Adam(learning_rate=0.0001)  # Was 0.001
   ```

2. **Check for gradient clipping:**
   ```yaml
   train {
     gradient_clip: 1.0  # Add gradient clipping
   }
   ```

3. **Use batch normalization:**
   ```yaml
   layers:
     Conv2D(32, (3,3))
     BatchNormalization()  # Add after Conv2D
     Activation("relu")
   ```

4. **Debug with NeuralDbg:**
   ```bash
   neural debug model.neural --anomalies
   ```

### Problem: Model not converging

**Symptoms:**
Training loss stays high or oscillates.

**Solutions:**

1. **Adjust learning rate schedule:**
   ```yaml
   optimizer: Adam(learning_rate=0.001)
   lr_schedule: ExponentialDecay(
     initial_lr=0.001,
     decay_steps=1000,
     decay_rate=0.96
   )
   ```

2. **Check data preprocessing:**
   ```python
   # Normalize inputs
   X = X / 255.0  # For images
   X = (X - mean) / std  # For general data
   ```

3. **Increase model capacity:**
   ```yaml
   # Add more layers or units
   Dense(256)  # Was 128
   ```

4. **Use different activation:**
   ```yaml
   # Try different activations
   Dense(128, activation="relu")  # Or "elu", "selu"
   ```

---

## Dashboard Issues

### Problem: Dashboard won't start

**Symptoms:**
```
ERROR: Failed to start dashboard on port 8050
```

**Solutions:**

1. **Check port availability:**
   ```bash
   # Windows
   netstat -ano | findstr :8050
   
   # Linux/macOS
   lsof -i :8050
   ```

2. **Use different port:**
   ```bash
   neural debug model.neural --port 8051
   ```

3. **Check firewall settings:**
   - Allow Python through firewall
   - On Windows: Windows Defender → Allow an app

### Problem: Dashboard shows no data

**Symptoms:**
Dashboard opens but displays empty charts.

**Solutions:**

1. **Ensure model is running:**
   ```bash
   # Start debug session with model execution
   neural debug model.neural --execute
   ```

2. **Check WebSocket connection:**
   - Open browser console (F12)
   - Look for WebSocket errors
   - Reload page if connection lost

3. **Verify tracing is enabled:**
   ```python
   # In generated code, ensure:
   from neural.dashboard import trace_execution
   trace_execution(enabled=True)
   ```

### Problem: Real-time updates not working

**Solutions:**

1. **Clear browser cache:**
   - Ctrl+Shift+Delete → Clear cache
   - Hard reload: Ctrl+F5

2. **Update Neural DSL:**
   ```bash
   pip install --upgrade neural-dsl
   ```

3. **Check browser compatibility:**
   - Use Chrome, Firefox, or Edge
   - Update to latest version

---

## HPO Issues

### Problem: HPO search takes too long

**Solutions:**

1. **Reduce search space:**
   ```yaml
   # ❌ Too many trials
   Dense(units=HPO(range(10, 1000, step=10)))  # 100 trials
   
   # ✅ Reasonable
   Dense(units=HPO(choice(32, 64, 128, 256)))  # 4 trials
   ```

2. **Use early stopping:**
   ```yaml
   hpo_config {
     trials: 50
     early_stopping: 5  # Stop if no improvement for 5 trials
   }
   ```

3. **Parallelize trials:**
   ```bash
   neural hpo model.neural --parallel 4
   ```

### Problem: HPO fails with parameter errors

**Symptoms:**
```
ERROR: Invalid HPO parameter for learning_rate
```

**Solutions:**

1. **Check HPO syntax:**
   ```yaml
   # ✅ Correct formats
   learning_rate=HPO(log_range(1e-4, 1e-2))
   units=HPO(choice(32, 64, 128))
   rate=HPO(range(0.1, 0.9, step=0.1))
   ```

2. **Verify parameter support:**
   ```bash
   # Check which parameters support HPO
   neural hpo --show-params
   ```

### Problem: Best parameters not applied

**Solutions:**

1. **Generate optimized DSL:**
   ```bash
   neural hpo model.neural --output optimized.neural
   ```

2. **Check HPO results:**
   ```bash
   neural hpo model.neural --show-best
   ```

---

## Cloud Integration Issues

### Problem: Cannot connect to cloud platform

**Symptoms:**
```
ERROR: Failed to connect to Kaggle
```

**Solutions:**

1. **Check credentials:**
   ```bash
   # For Kaggle
   cat ~/.kaggle/kaggle.json
   
   # For AWS
   aws configure list
   ```

2. **Install cloud dependencies:**
   ```bash
   pip install neural-dsl[cloud]
   ```

3. **Verify network connection:**
   ```bash
   ping kaggle.com
   ```

### Problem: Tunnel setup fails

**Symptoms:**
```
ERROR: ngrok tunnel failed to start
```

**Solutions:**

1. **Install ngrok:**
   ```bash
   # Windows
   choco install ngrok
   
   # Linux/macOS
   brew install ngrok
   ```

2. **Set ngrok auth token:**
   ```bash
   ngrok authtoken YOUR_TOKEN
   ```

3. **Use alternative tunneling:**
   ```bash
   neural cloud run --tunnel localtunnel
   ```

---

## Performance Issues

### Problem: Compilation is slow

**Solutions:**

1. **Use cached results:**
   ```bash
   neural compile model.neural --cache
   ```

2. **Disable validation:**
   ```bash
   neural compile model.neural --no-validate
   ```

3. **Profile compilation:**
   ```bash
   neural compile model.neural --profile
   ```

### Problem: Training is slow

**Solutions:**

1. **Enable GPU acceleration:**
   ```yaml
   device: "cuda"  # Or "auto"
   ```

2. **Use data augmentation efficiently:**
   ```python
   # Use built-in augmentation
   # Instead of custom preprocessing
   ```

3. **Optimize data loading:**
   ```python
   # Increase num_workers
   DataLoader(..., num_workers=4, pin_memory=True)
   ```

4. **Profile training:**
   ```bash
   neural profile model.neural
   ```

---

## Getting Help

### Before asking for help:

1. **Check version:**
   ```bash
   neural --version
   pip show neural-dsl
   ```

2. **Create minimal example:**
   - Reduce DSL to smallest failing case
   - Remove unnecessary layers/configs

3. **Collect error logs:**
   ```bash
   neural compile model.neural --verbose > debug.log 2>&1
   ```

### Where to get help:

1. **Documentation:**
   - [DSL Reference](dsl.md)
   - [CLI Reference](cli.md)
   - [Examples](../examples/README.md)

2. **Community:**
   - [Discord Server](https://discord.gg/KFku4KvS)
   - [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
   - [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)

3. **Report bugs:**
   - Create issue with:
     - Neural version
     - Python version
     - Operating system
     - Minimal DSL example
     - Full error message
     - Steps to reproduce

### Providing useful error reports:

```markdown
**Environment:**
- Neural DSL: 0.2.9
- Python: 3.9.7
- OS: Windows 10
- Backend: TensorFlow 2.12.0

**Issue:**
Shape mismatch error when using Flatten after Conv2D

**DSL Code:**
[Paste minimal DSL that reproduces issue]

**Error Message:**
[Paste full error traceback]

**Steps to Reproduce:**
1. Save DSL to file
2. Run: neural compile model.neural
3. See error

**Expected Behavior:**
Should compile successfully

**Actual Behavior:**
Throws shape mismatch error
```

---

## Common Error Messages Reference

| Error | Meaning | Solution |
|-------|---------|----------|
| `Unexpected token` | Syntax error in DSL | Check syntax, commas, colons |
| `Unknown layer type` | Layer not supported | Check spelling, verify support |
| `Shape mismatch` | Incompatible layer shapes | Add Flatten or check dimensions |
| `Parameter validation failed` | Invalid parameter value | Check constraints (e.g., rate ∈ [0,1]) |
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size or model size |
| `ModuleNotFoundError` | Missing dependency | Install required backend |
| `HPO parameter error` | Invalid HPO syntax | Check HPO format and ranges |
| `Backend not supported` | Feature unavailable | Switch backend or use alternative |

---

## Quick Diagnostics Checklist

- [ ] Using latest Neural DSL version
- [ ] Using correct Python version (3.8+)
- [ ] Virtual environment activated
- [ ] Required backend installed (TensorFlow/PyTorch)
- [ ] DSL syntax is valid
- [ ] Input shapes are correct format
- [ ] Layer parameters meet constraints
- [ ] No shape mismatches in network
- [ ] Port 8050 available (for dashboard)
- [ ] Sufficient GPU/CPU memory
- [ ] Network connectivity (for cloud features)

---

For additional help, visit our [Discord](https://discord.gg/KFku4KvS) or [create an issue](https://github.com/Lemniscate-world/Neural/issues/new).
