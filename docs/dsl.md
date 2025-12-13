# Neural DSL Documentation

## What's New in v0.2.9

### Key Improvements

We've added an early preview of Aquarium IDE - a visual tool for designing neural networks. It's very much in early development, but you can try it out if you're curious.

**What works:**
- Basic visual design for simple networks
- Tensor shape calculation for each layer
- Neural DSL code generation from your visual design
- Parameter counting

**Current limitations (and there are many):**
- Only supports a handful of layer types (Input, Conv2D, MaxPooling2D, Flatten, Dense, Output)
- Limited parameter configuration
- Shape calculation works for simple cases but hasn't been tested extensively
- Code generation is basic - no advanced features
- No support for complex architectures (multi-input/output, skip connections, etc.)
- Error checking is minimal

We also fixed trailing whitespace and missing newlines across the codebase. Not exciting, but it makes the code cleaner.

### Example: Using Aquarium IDE (Early Preview)

Aquarium provides a basic visual interface. Here's how to try it:

1. **Install and Launch:**
   ```bash
   # Clone if you haven't already
   git clone https://github.com/Lemniscate-world/Neural.git
   cd Neural

   # Get Aquarium submodule
   git submodule update --init --recursive

   # Install Rust (if you don't have it)
   # https://www.rust-lang.org/tools/install

   # Install Tauri CLI
   cargo install tauri-cli

   # Navigate to Aquarium
   cd Aquarium

   # Install dependencies
   npm install

   # Run dev server (first time takes a while)
   cargo tauri dev
   ```

2. **Try the Features:**
   - Add layers using the left panel buttons
   - Configure parameters in the properties panel
   - View shapes in the shape tab
   - See generated code in the code tab

3. **Export to Neural DSL:**
   - Copy code from the code tab
   - Save to a .neural file
   - Use Neural CLI to compile

### Example: Generated Neural DSL Code

```yaml
# Neural DSL Model

Input(shape=[28, 28, 1])
Conv2D(filters=32, kernel_size=[3, 3], padding="same", activation="relu")
MaxPooling2D(pool_size=[2, 2])
Flatten()
Dense(units=128, activation="relu")
Output(units=10, activation="softmax")
```

**Reality check:** The generated code is pretty basic. You'll probably want to edit it before using it in production.

## What's New in v0.2.8

### Key Improvements

Enhanced cloud integration for running Neural in Kaggle, Colab, and AWS SageMaker. This is useful if you don't have local GPU resources.

**New features:**
- Better environment detection
- Interactive shell for cloud platforms
- Improved HPO parameter handling (fixed some edge cases with log_range)
- Cleaner CLI output (fewer debug messages)

### Example: Cloud Integration

```python
# In a Colab notebook
!pip install neural-dsl==0.2.9

from neural.cloud.cloud_execution import CloudExecutor

# Initialize
executor = CloudExecutor()
print(f"Environment: {executor.environment}")
print(f"GPU available: {executor.is_gpu_available}")

# Define model
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

# Compile and run
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST')

# Start dashboard with ngrok tunnel (for remote access)
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard URL: {dashboard_info['tunnel_url']}")
```

### Example: Interactive Shell

```bash
# Connect to Kaggle with interactive shell
neural cloud connect kaggle --interactive

# In the shell:
neural-cloud> run my_model.neural --backend tensorflow
neural-cloud> visualize my_model.neural
neural-cloud> debug my_model.neural --setup-tunnel
neural-cloud> shell ls -la
neural-cloud> python print("Hello from Kaggle!")
```

**Note:** The interactive shell is convenient, but adds some overhead. For simple tasks, direct commands might be faster.

## What's New in v0.2.7

### Key Improvements

Better HPO support, especially for Conv2D layers and learning rate schedules.

**New HPO features:**
- Conv2D kernel_size parameter can now be optimized
- Padding parameter supports HPO
- ExponentialDecay schedules work better with HPO
- Consistent parameter naming (min/max instead of low/high)

### Example: Enhanced HPO for Conv2D

```yaml
network ConvHPOExample {
  input: (28, 28, 1)
  layers:
    # Both filters and kernel_size with HPO
    Conv2D(
      filters=HPO(choice(32, 64)),
      kernel_size=HPO(choice((3,3), (5,5))),
      padding=HPO(choice("same", "valid")),
      activation="relu"
    )
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(128, activation="relu")
    Output(10, activation="softmax")

  optimizer: Adam(learning_rate=0.001)
  loss: "sparse_categorical_crossentropy"
}
```

### Example: Improved ExponentialDecay

```yaml
network DecayExample {
  input: (28, 28, 1)
  layers:
    Conv2D(32, (3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(128, activation="relu")
    Output(10, activation="softmax")

  # ExponentialDecay with HPO parameters
  optimizer: Adam(
    learning_rate=ExponentialDecay(
      HPO(log_range(1e-3, 1e-1)),  # Initial learning rate
      HPO(choice(500, 1000, 2000)),  # Decay steps
      HPO(range(0.9, 0.99, step=0.01))  # Decay rate
    )
  )
  loss: "sparse_categorical_crossentropy"
}
```

## What's New in v0.2.6

### Key Improvements

- Enhanced dashboard with better dark theme
- Advanced HPO examples for complex configurations
- Blog infrastructure (more for us than you, but might be useful)
- Better error reporting (planning stage - not fully implemented yet)
- CLI version command now shows actual package version

### Example: Enhanced Dashboard UI

```bash
# Start with new dark theme
neuraldbg --theme dark

# Or use debug command
neural debug my_model.neural --gradients --theme dark
```

The new theme is easier on the eyes for long debugging sessions.

### Example: Advanced Nested HPO

```yaml
network AdvancedHPOExample {
  input: (28, 28, 1)
  layers:
    # Multiple HPO parameters
    Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))

    Conv2D(filters=HPO(choice(64, 128)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))

    Flatten()
    Dense(HPO(choice(128, 256, 512)), activation="relu")
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, "softmax")

  # Optimizer with complex HPO
  optimizer: SGD(
    learning_rate=ExponentialDecay(
      HPO(range(0.05, 0.2, step=0.05)),
      1000,
      HPO(range(0.9, 0.99, step=0.01))
    ),
    momentum=HPO(range(0.8, 0.99, step=0.01))
  )

  train {
    epochs: 20
    batch_size: HPO(choice(32, 64, 128))
    search_method: "bayesian"
  }
}
```

**Warning:** This many HPO parameters means a large search space. Budget at least 100 trials.

## What's New in v0.2.5

### Key Improvements

Multi-framework HPO support (works with both PyTorch and TensorFlow now), better optimizer handling, precision and recall metrics, improved error messages, and cleaner syntax for optimizer parameters.

### Example: Advanced HPO with Learning Rate Schedules

```yaml
optimizer: SGD(
  learning_rate=ExponentialDecay(
    HPO(range(0.05, 0.2, step=0.05)),
    1000,
    HPO(range(0.9, 0.99, step=0.01))
  ),
  momentum=HPO(range(0.8, 0.99, step=0.01))
)
```

This works now. Previous versions had issues with nested HPO in learning rate schedules.

## Table of Contents
- [What's New](#whats-new-in-v029)
- [Syntax Reference](#syntax-reference)
- [Core Components](#core-components)
- [Validation Rules](#validation-rules)
- [CLI Reference](#cli-reference)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Cloud Integration](#cloud-integration-v028)
- [Development Guide](#development-guide)

---

## Syntax Reference

### Network Structure

```yaml
network <ModelName> {
  input: <Shape>                # e.g., (224, 224, 3) or multiple inputs
  layers:
    <LayerType>(<Parameters>)   # Ordered or named parameters
    <LayerType>*<Count>         # Repeat a layer
  loss: <LossFunction>
  optimizer: <Optimizer>
  train {
    epochs: <Int>
    batch_size: <Int|HPO-range>
    validation_split: <0.0-1.0>
    search_method: "random"     # For HPO
  }
  execution {
    device: <String>            # "cpu", "cuda", "tpu"
  }
}
```

### Parameter Types

```yaml
# Ordered parameters (positional)
Conv2D(32, (3,3), "relu")

# Named parameters (explicit)
TransformerEncoder(num_heads=8, ff_dim=512)

# HPO parameters (for optimization)
Dense(HPO(choice(128, 256, 512)))

# Device placement
LSTM(units=128) @ "cuda:0"
```

---

## Core Components

### Layer Types

| Category         | Layers                                                                 |
|------------------|-----------------------------------------------------------------------|
| **Convolution**  | `Conv1D`, `Conv2D`, `DepthwiseConv2D`, `SeparableConv2D`              |
| **Recurrent**    | `LSTM`, `GRU`, `Bidirectional`, `ConvLSTM2D`                          |
| **Transformer**  | `TransformerEncoder`, `TransformerDecoder`                            |
| **Regularization**| `Dropout`, `SpatialDropout2D`, `GaussianNoise`, `BatchNormalization` |
| **Utility**      | `Flatten`, `Lambda`, `TimeDistributed`, `ResidualConnection`          |

**Note:** For detailed transformer documentation, see [Transformer Documentation](transformers_README.md).

---

## Validation Rules

### Parameter Constraints

| Parameter           | Rule                                  | Error Example              |
|---------------------|---------------------------------------|----------------------------|
| `num_heads`         | Must be > 0                          | `num_heads=0` → ERROR      |
| `filters`           | Positive integer                     | `filters=-32` → ERROR      |
| `kernel_size`       | Tuple of positive integers           | `(0,3)` → ERROR            |
| `rate`              | 0 ≤ value ≤ 1                        | `Dropout(1.2)` → ERROR     |
| `validation_split`  | 0 ≤ value ≤ 1                        | `1.1` → ERROR              |
| `device`            | Valid device identifier              | `device:"npu"` → CRITICAL  |

### Type Coercions

The parser tries to be helpful:
- Float → Int conversion with warning (e.g., `Dense(256.0)` becomes `Dense(256)`)
- Automatic tuple wrapping for single values (sometimes)

**Note:** Relying on automatic coercion makes your code less clear. Be explicit when you can.

---

## CLI Reference

### Command Overview

```bash
# Core commands
neural compile <file> [--backend tensorflow|pytorch] [--hpo]
neural run <file> [--device cpu|cuda]
neural debug <file> [--gradients] [--dead-neurons] [--step]

# Analysis and visualization
neural visualize <file> [--format png|svg|html]
neural profile <file> [--memory] [--latency]

# Cloud integration (v0.2.8+)
neural cloud connect <platform> [--interactive]
neural cloud execute <platform> <file>
neural cloud run [--setup-tunnel]

# Project management
neural clean
neural version
```

### Key Options

- `--dry-run` - Validate without generating code (fast check)
- `--hpo` - Enable hyperparameter optimization
- `--step` - Interactive debugging (step through layers)
- `--port` - Specify port for web interface
- `--theme` - Dashboard theme: light or dark (v0.2.6+)
- `--interactive` - Start interactive shell for cloud (v0.2.8+)
- `--setup-tunnel` - Set up ngrok tunnel for remote access (v0.2.8+)

---

## Error Handling

### Severity Levels

| Level     | Description                          | Example Case                      |
|-----------|--------------------------------------|-----------------------------------|
| CRITICAL  | Fatal error, can't continue         | Invalid device specification      |
| ERROR     | Model is broken                     | Missing required parameter        |
| WARNING   | Probably fine but suspicious        | Type coercion                     |
| INFO      | Just FYI                            | Shape propagation info            |

### Example Messages

```text
CRITICAL at line 5, column 12:
Invalid device 'npu' - must be one of [cpu, cuda, tpu]

ERROR at line 12, column 8:
Conv2D kernel_size requires positive integers. Got (0,3)

WARNING at line 8, column 15:
Converting 256.0 to integer 256 (consider using int directly)
```

### Planned Error Improvements (v0.2.6)

We're working on better error messages with more context. Currently in planning stage (#459).

**What we want to add:**
- Context about where in the model the error occurred
- Suggestions for fixing common mistakes
- Better handling of nested structure errors

**Current state:** Basic line/column info works. Contextual information is limited.

### Enhanced Error Messages (v0.2.5)

v0.2.5 improved error reporting:

```text
ERROR at line 15, column 10:
Dense units must be positive integers. Got -128

ERROR at line 22, column 14:
Dropout rate must be between 0 and 1. Got 1.5

ERROR at line 30, column 18:
TransformerEncoder num_heads must be positive. Got 0

WARNING at line 42, column 22:
HPO parameter 'learning_rate' should use log_range for better optimization
```

The messages now include:
- Exact location (line and column)
- Clear rule that was violated
- The actual problematic value
- Sometimes suggestions for fixing

---

## Examples

### Vision Transformer

```yaml
network ViT {
  input: (224, 224, 3)
  layers:
    Conv2D(64, (7,7), strides=2) @ "cuda:0"
    TransformerEncoder() * 12     # 12 blocks
    GlobalAveragePooling2D()
    Dense(1000, "softmax")
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=1e-4)
  train {
    epochs: 300
    batch_size: HPO(range(128, 512, step=64))
  }
}
```

**Reality check:** Training a ViT from scratch on ImageNet takes days even on multiple GPUs. Consider transfer learning instead.

### Hyperparameter Optimization

```yaml
network HPOExample {
  layers:
    Dense(HPO(choice(128, 256, 512)))
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
  optimizer: Adam(
    learning_rate=HPO(log_range(1e-4, 1e-2))
  )
  train {
    epochs: 100
    search_method: "bayesian"
  }
}
```

### HPO Parameter Types

```yaml
# Categorical choice - pick from discrete values
HPO(choice(128, 256, 512))

# Linear range with step
HPO(range(0.3, 0.7, step=0.1))

# Log-scale range (better for learning rates)
HPO(log_range(1e-4, 1e-2))
```

### Supported HPO Parameters

| Parameter | HPO Type | Example | Since |
|-----------|----------|---------|-------|
| Dense units | `choice` | `Dense(HPO(choice(64, 128, 256)))` | v0.2.5 |
| Dropout rate | `range` | `Dropout(HPO(range(0.3, 0.7, step=0.1)))` | v0.2.5 |
| Learning rate | `log_range` | `Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))` | v0.2.5 |
| Conv2D filters | `choice` | `Conv2D(filters=HPO(choice(32, 64)))` | v0.2.6 |
| Conv2D kernel_size | `choice` | `Conv2D(kernel_size=HPO(choice((3,3), (5,5))))` | v0.2.7 |
| Padding | `choice` | `Conv2D(padding=HPO(choice("same", "valid")))` | v0.2.7 |
| Decay steps | `choice` | `ExponentialDecay(0.1, HPO(choice(500, 1000)), 0.96)` | v0.2.7 |

### Validation Rules for HPO

- Dense units: positive integers only
- Dropout rate: between 0 and 1
- Learning rate: positive values
- Step size: required for range type (no default)

**Trade-offs:** More HPO parameters = larger search space = more trials needed. Start with the parameters that matter most.

### HPO Updates in v0.2.8

- Consistent parameter naming (min/max instead of low/high)
- Fixed Conv2D HPO issues
- Better handling of missing parameters
- Optimizer HPO works without quotes now

### HPO Updates in v0.2.5

- Multi-framework support (PyTorch and TensorFlow)
- All optimizer parameters support HPO
- Better scientific notation handling
- Cleaner syntax (no quotes needed)

---

## Cloud Integration (v0.2.8+)

Run Neural DSL in cloud environments when you don't have local GPU resources.

### Supported Platforms

| Platform | Description | Command |
|----------|-------------|---------|
| **Kaggle** | Data science competition platform | `neural cloud connect kaggle` |
| **Google Colab** | Free Jupyter environment | `neural cloud connect colab` |
| **AWS SageMaker** | Managed ML service | `neural cloud connect sagemaker` |

### Cloud Commands

```bash
# Connect to platform
neural cloud connect kaggle

# Interactive shell
neural cloud connect kaggle --interactive

# Execute file on platform
neural cloud execute kaggle my_model.neural

# Run with remote access
neural cloud run --setup-tunnel
```

### Python API

```python
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()
print(f"Environment: {executor.environment}")
print(f"GPU: {executor.is_gpu_available}")

# Compile and run
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST')

# Dashboard with tunnel
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard: {dashboard_info['tunnel_url']}")
```

### Interactive Shell

The shell gives you a Neural CLI-like experience on cloud platforms:

```bash
neural cloud connect kaggle --interactive

# In shell:
neural-cloud> run my_model.neural --backend tensorflow
neural-cloud> visualize my_model.neural
neural-cloud> debug my_model.neural --setup-tunnel
neural-cloud> shell ls -la
neural-cloud> python print("Hello!")
```

**Overhead note:** The shell adds a small latency to each command. For batch operations, direct execution is faster.

### Remote Dashboard Access

Use ngrok tunnels to access the dashboard from your browser:

```bash
# With tunnel
neural debug my_model.neural --setup-tunnel

# Or in Python
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"URL: {dashboard_info['tunnel_url']}")
```

**Security note:** The tunnel is public by default. Don't expose sensitive data.

---

## Training Configuration

### Basic Setup

```yaml
network MyModel {
    train {
        epochs: 100
        batch_size: 32
        validation_split: 0.2
        search_method: "bayesian"
    }
}
```

### Optimizer Configuration

```yaml
# Basic optimizer
optimizer: Adam(
    learning_rate=HPO(log_range(1e-4, 1e-2)),
    beta_1=0.9,
    beta_2=0.999
)

# With learning rate schedule
optimizer: SGD(
    learning_rate=ExponentialDecay(0.1, 1000, 0.96),
    momentum=0.9
)

# Schedule with HPO
optimizer: SGD(
    learning_rate=ExponentialDecay(
        HPO(range(0.05, 0.2, step=0.05)), 
        1000, 
        HPO(range(0.9, 0.99, step=0.01))
    )
)
```

### Learning Rate Schedules

Dynamic learning rate adjustment during training:

```yaml
# ExponentialDecay
learning_rate=ExponentialDecay(0.1, 1000, 0.96)

# With HPO
learning_rate=ExponentialDecay(
    HPO(range(0.05, 0.2, step=0.05)), 
    1000, 
    0.96
)
```

For backward compatibility, string-based schedules still work:

```yaml
learning_rate="ExponentialDecay(0.1, 1000, 0.96)"
```

**Supported schedules:**
- `ExponentialDecay` - exponential decay
- `PiecewiseConstantDecay` - piecewise constant
- `PolynomialDecay` - polynomial decay
- `InverseTimeDecay` - inverse time decay

**When to use schedules:** If your loss plateaus during training, a schedule can help. But they add complexity - start without one.

---

## Enhanced HPO Support (v0.2.7+)

v0.2.7 added better HPO for Conv2D and learning rate schedules.

### Conv2D with HPO

```yaml
# Basic (v0.2.6)
Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3))

# Enhanced (v0.2.7)
Conv2D(
  filters=HPO(choice(32, 64)),
  kernel_size=HPO(choice((3,3), (5,5))),
  padding=HPO(choice("same", "valid"))
)
```

### ExponentialDecay with HPO

```yaml
# Basic (v0.2.6)
ExponentialDecay(
  HPO(range(0.05, 0.2, step=0.05)),
  1000,
  HPO(range(0.9, 0.99, step=0.01))
)

# All parameters (v0.2.7)
ExponentialDecay(
  HPO(log_range(1e-3, 1e-1)),
  HPO(choice(500, 1000, 2000)),
  HPO(range(0.9, 0.99, step=0.01))
)
```

### Parameter Naming

```yaml
# Old (v0.2.6)
HPO(log_range(low=1e-4, high=1e-2))

# New (v0.2.7)
HPO(log_range(min=1e-4, max=1e-2))
```

More consistent with common conventions.

## Enhanced Dashboard UI (v0.2.6+)

v0.2.6 improved the NeuralDbg dashboard significantly.

### UI Improvements
- Dark theme (easier on eyes)
- Responsive layout
- Better tensor flow animations
- Fixed WebSocket issues
- Improved shape propagation charts

### Using the Dashboard

```bash
# Dark theme (default)
neural debug my_model.neural --theme dark

# Light theme
neural debug my_model.neural --theme light
```

### Dashboard Components
- **Execution Trace** - Layer activations in real-time
- **Gradient Flow** - Gradient magnitudes across layers
- **Dead Neuron Detection** - Find inactive neurons
- **Resource Monitoring** - CPU/GPU/Memory usage
- **Shape Propagation** - Interactive tensor shapes

**Performance note:** The dashboard adds overhead. For production training, consider disabling it.

---

## Aquarium IDE (v0.2.9+)

Early preview of a visual neural network designer. Very much alpha quality.

### Current Features

- Basic visual layer designer
- Shape calculation for simple networks
- Neural DSL code generation
- Parameter counting

### Technology

- Frontend: Tauri (cross-platform)
- Backend: Rust for shape calculation
- Integration: Neural's shape propagator

### Current Limitations

These are significant:
- Only supports basic layer types
- Limited parameter options
- Shape calculation not thoroughly tested
- Simple code generation only
- No complex architectures (skip connections, multi-input, etc.)
- Minimal error checking

### Trying Aquarium

See the installation steps in "What's New in v0.2.9" above. Note that first launch takes several minutes.

### Shape Calculation Example

```
Layer         | Input Shape      | Output Shape     | Parameters
--------------|------------------|------------------|------------
Input         | -                | [null,28,28,1]   | 0
Conv2D        | [null,28,28,1]   | [null,28,28,32]  | 320
MaxPooling2D  | [null,28,28,32]  | [null,14,14,32]  | 0
Flatten       | [null,14,14,32]  | [null,6272]      | 0
Dense         | [null,6272]      | [null,128]       | 802,944
Output        | [null,128]       | [null,10]        | 1,290
```

**Accuracy:** These calculations are correct for simple cases. Complex architectures may give wrong results.

### Roadmap

Things we're considering (no promises):
- More layer types
- Better shape propagation
- Improved error handling
- Visual connections between layers
- Save/load functionality
- Export to multiple formats

---

## Development Guide

### Setup & Testing

```bash
# Install dependencies
pip install -r requirements.txt
pre-commit install

# Run tests
pytest test_parser.py -v
pytest -k "transformer_validation" --log-level=DEBUG

# Coverage report
coverage run -m pytest && coverage html
```

### Linting Rules

- Type hints required
- Parameter validation checks
- HPO syntax verification
- Device configuration validation

### Contribution Requirements

1. Unit tests for new features (not optional)
2. Documentation updates (please)
3. Backward compatibility checks
4. Pre-commit hooks pass

---

## Migration Guide

### From v0.2.4 to v0.2.5

#### HPO Optimizer Changes

**Old (v0.2.4):**
```yaml
# Needed quotes
optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"

# Limited schedule support
```

**New (v0.2.5):**
```yaml
# No quotes needed
optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))

# Full schedule support
optimizer: SGD(
  learning_rate=ExponentialDecay(
    HPO(range(0.05, 0.2, step=0.05)),
    1000,
    HPO(range(0.9, 0.99, step=0.01))
  )
)
```

#### Multi-Framework Support

v0.2.5 HPO works with both backends:

```bash
# PyTorch
neural compile mnist_hpo.neural --backend pytorch --hpo

# TensorFlow
neural compile mnist_hpo.neural --backend tensorflow --hpo
```

### From v0.2.1 to v0.2.2

#### Old Style (Deprecated)

```yaml
network OldStyle {
    layers: Dense("64")  # String numbers
    optimizer: Adam(learning_rate="0.001")
}
```

#### New Style (Recommended)

```yaml
network NewStyle {
    layers: Dense(64)    # Actual numbers
    optimizer: Adam(learning_rate=0.001)
}
```

The old style still works but you'll get warnings. Update when you can.

---

## Limitations and Trade-offs

Let's be honest about what doesn't work well:

**Parser limitations:**
- Error messages could be better (we're working on it)
- Complex nested structures can be confusing
- Some edge cases in HPO parameter handling

**HPO limitations:**
- Large search spaces need many trials
- Multi-objective optimization is slow
- Parameter importance needs sufficient data

**Cloud integration:**
- Setup is somewhat involved
- Ngrok tunnels are public by default
- Some latency in interactive shell

**Aquarium IDE:**
- Very early stage (alpha quality)
- Limited layer support
- Basic functionality only
- Not well tested

**General:**
- Documentation could always be better
- Some features are experimental
- Not all combinations are tested

We're continuously improving, but these are the current realities.

---

[Report Issues](https://github.com/Lemniscate-world/Neural/issues)
