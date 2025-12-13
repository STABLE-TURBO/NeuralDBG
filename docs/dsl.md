# Neural DSL Documentation

## What's New in v0.2.9

### Key Improvements
- **Aquarium IDE (Early Preview)**: Added an early preview of a specialized IDE for neural network development with basic visual design tools.
- **Basic Shape Calculation**: View tensor dimensions for each layer in your network.
- **Simple Neural DSL Code Generation**: Generate basic Neural DSL code from your visual design.
- **Code Quality Improvements**: Fixed trailing whitespace and missing newlines at end of files across the codebase.

### Current Limitations
- Only supports a small set of layer types (Input, Conv2D, MaxPooling2D, Flatten, Dense, Output)
- Limited parameter configuration options
- Basic shape calculation that may not handle all edge cases
- Simple code generation without advanced features
- No support for complex network architectures (e.g., multi-input/output, skip connections)
- Limited error checking and validation

### Example: Using Aquarium IDE (Early Preview)

Aquarium IDE provides a basic visual interface for designing simple neural networks:

1. **Install and Launch Aquarium**:
   ```bash
   # Clone the Neural repository if you haven't already
   git clone https://github.com/Lemniscate-world/Neural.git
   cd Neural

   # Update submodules to get Aquarium
   git submodule update --init --recursive

   # Install Rust if you don't have it already
   # https://www.rust-lang.org/tools/install

   # Install Tauri CLI
   cargo install tauri-cli

   # Navigate to the Aquarium directory
   cd Aquarium

   # Install Node.js dependencies
   npm install

   # Run the development server (this may take a few minutes the first time)
   cargo tauri dev
   ```

2. **Try the Basic Features**:
   - Add layers using the buttons in the left panel
   - Configure basic parameters in the properties panel
   - View tensor shapes in the shape tab
   - See generated Neural DSL code in the code tab

3. **Export to Neural DSL**:
   - Copy the generated code from the code tab
   - Save it to a .neural file
   - Use the Neural CLI to compile and run the model

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

## What's New in v0.2.8

### Key Improvements
- **Cloud Integration**: Enhanced support for running Neural in cloud environments like Kaggle, Colab, and AWS SageMaker.
- **Interactive Shell for Cloud Platforms**: Added interactive shell capabilities when connecting to cloud platforms.
- **HPO Parameter Handling Fixes**: Improved handling of HPO log_range parameters with consistent min/max naming.
- **CLI Debug Messages**: Further reduced debug logs when starting the Neural CLI for a cleaner user experience.

### Example: Cloud Integration

```python
# Install Neural DSL in your Colab notebook
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

### Example: Interactive Shell for Cloud Platforms

```bash
# Connect to Kaggle with an interactive shell
neural cloud connect kaggle --interactive

# In the shell, you can run commands like:
neural-cloud> run my_model.neural --backend tensorflow
neural-cloud> visualize my_model.neural
neural-cloud> debug my_model.neural --setup-tunnel
neural-cloud> shell ls -la
neural-cloud> python print("Hello from Kaggle!")
```

## What's New in v0.2.7

### Key Improvements
- **Enhanced HPO Support for Conv2D Layers**: Added HPO tracking for kernel_size parameter in Conv2D layers.
- **Improved ExponentialDecay Parameter Structure**: Enhanced support for complex decay schedules with better parameter handling.
- **Extended Padding Options in Layers**: Added HPO expression support for padding parameters.
- **Parser Improvements**: Fixed metrics processing logic and improved HPO log_range parameter naming for consistency.

### Example: Enhanced HPO for Conv2D Layers
```yaml
network ConvHPOExample {
  input: (28, 28, 1)
  layers:
    # Conv2D with HPO for both filters and kernel_size
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

### Example: Improved ExponentialDecay Parameter Structure
```yaml
network DecayExample {
  input: (28, 28, 1)
  layers:
    Conv2D(32, (3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(128, activation="relu")
    Output(10, activation="softmax")

  # Enhanced ExponentialDecay with complex parameter structure
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
- **Enhanced Dashboard UI**: Improved NeuralDbg dashboard with a more aesthetic dark theme design (#452)
- **Advanced HPO Examples**: Added comprehensive examples for complex hyperparameter optimization configurations (#448)
- **Blog Support**: Infrastructure for blog content with markdown support and Dev.to integration (#445)
- **Improved Error Reporting**: Planning enhanced error context in validation messages with more precise line/column information (#459)
- **CLI Version Display**: Updated version command to dynamically fetch package version (#437)

### Example: Enhanced Dashboard UI
```bash
# Start the NeuralDbg dashboard with the new dark theme
neuraldbg --theme dark

# Or use the debug command with visualization
neural debug my_model.neural --gradients --theme dark
```

### Example: Advanced Nested HPO Configuration
```yaml
network AdvancedHPOExample {
  input: (28, 28, 1)
  layers:
    # Convolutional layers with HPO parameters
    Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))

    # Another conv block with HPO
    Conv2D(filters=HPO(choice(64, 128)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))

    # Flatten and dense layers
    Flatten()
    Dense(HPO(choice(128, 256, 512)), activation="relu")
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, "softmax")

  # Advanced optimizer configuration with HPO
  optimizer: SGD(
    learning_rate=ExponentialDecay(
      HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
      1000,                              # Decay steps
      HPO(range(0.9, 0.99, step=0.01))   # Decay rate
    ),
    momentum=HPO(range(0.8, 0.99, step=0.01))
  )

  # Training configuration with HPO
  train {
    epochs: 20
    batch_size: HPO(choice(32, 64, 128))
    validation_split: 0.2
    search_method: "bayesian"  # Use Bayesian optimization
  }
}
```

## What's New in v0.2.5

### Key Improvements
- **Multi-Framework HPO Support**: Seamless hyperparameter optimization across PyTorch and TensorFlow
- **Enhanced Optimizer Handling**: Improved parsing and validation of optimizer configurations
- **Precision & Recall Metrics**: Comprehensive metrics reporting in training loops
- **Error Message Improvements**: More detailed error messages with line/column information
- **No-Quote Syntax**: Cleaner syntax for optimizer parameters without quotes

### Example: Advanced HPO with Learning Rate Schedules
```yaml
optimizer: SGD(
  learning_rate=ExponentialDecay(
    HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
    1000,                              # Decay steps
    HPO(range(0.9, 0.99, step=0.01))   # Decay rate
  ),
  momentum=HPO(range(0.8, 0.99, step=0.01))
)
```

## Table of Contents
- [What's New in v0.2.9](#whats-new-in-v029)
- [What's New in v0.2.8](#whats-new-in-v028)
- [What's New in v0.2.7](#whats-new-in-v027)
- [What's New in v0.2.6](#whats-new-in-v026)
- [What's New in v0.2.5](#whats-new-in-v025)
- [Syntax Reference](#syntax-reference)
- [Core Components](#core-components)
- [Validation Rules](#validation-rules)
- [CLI Reference](#cli-reference)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Aquarium IDE (v0.2.9+)](#aquarium-ide-v029)
- [Cloud Integration (v0.2.8+)](#cloud-integration-v028)
- [Enhanced HPO Support (v0.2.7+)](#enhanced-hpo-support-v027)
- [Enhanced Dashboard UI (v0.2.6+)](#enhanced-dashboard-ui-v026)
- [Blog Support (v0.2.6+)](#blog-support-v026)
- [Development Guide](#development-guide)

---

## Syntax Reference

### Network Structure
```yaml
network <ModelName> {
  input: <Shape>                # e.g., (224, 224, 3) or multiple inputs
  layers:
    <LayerType>(<Parameters>)   # Supports ordered and named params
    <LayerType>*<Count>         # Layer repetition syntax
  loss: <LossFunction>
  optimizer: <Optimizer>
  train {
    epochs: <Int>
    batch_size: <Int|HPO-range>
    validation_split: <0.0-1.0>
    search_method: "random"     # Hyperparameter optimization
  }
  execution {
    device: <String>            # "cpu", "cuda", "tpu"
  }
}
```

### Parameter Types
```yaml
# Ordered parameters
Conv2D(32, (3,3), "relu")

# Named parameters
TransformerEncoder(num_heads=8, ff_dim=512)

# HPO parameters
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

**Note**: For comprehensive transformer documentation including architecture patterns, attention mechanisms, training best practices, and migration from TensorFlow/PyTorch, see the [Transformer Documentation](transformers_README.md).

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
- Float → Int conversion with warning (e.g., `Dense(256.0)`)
- Automatic tuple wrapping for single values

---

## CLI Reference

### Command Overview
```bash
# Core Commands
neural compile <file> [--backend tensorflow|pytorch] [--hpo]
neural run <file> [--device cpu|cuda]
neural debug <file> [--gradients] [--dead-neurons] [--step]

# Analysis & Visualization
neural visualize <file> [--format png|svg|html]
neural profile <file> [--memory] [--latency]

# Cloud Integration (v0.2.8+)
neural cloud connect <platform> [--interactive]  # Connect to cloud platform
neural cloud execute <platform> <file>  # Execute a file on cloud platform
neural cloud run [--setup-tunnel]  # Run Neural in cloud mode

# Project Management
neural clean  # Remove generated files
neural version  # Show version and dependencies info
```

### Key Options
- `--dry-run`: Validate without code generation
- `--hpo`: Enable hyperparameter optimization
- `--step`: Interactive debugging mode
- `--port`: Specify GUI port for no-code interface
- `--theme`: Set dashboard theme (light/dark, v0.2.6+)
- `--interactive`: Start an interactive shell (for cloud commands, v0.2.8+)
- `--setup-tunnel`: Set up an ngrok tunnel for remote access (v0.2.8+)

---

## Error Handling

### Severity Levels
| Level     | Description                          | Example Case                      |
|-----------|--------------------------------------|-----------------------------------|
| CRITICAL  | Fatal configuration error           | Invalid device specification      |
| ERROR     | Structural/model error              | Missing required parameter        |
| WARNING   | Non-fatal issue                     | Type coercion                     |
| INFO      | Diagnostic message                  | Shape propagation update          |

### Example Messages
```text
CRITICAL at line 5, column 12:
Invalid device 'npu' - must be one of [cpu, cuda, tpu]

ERROR at line 12, column 8:
Conv2D kernel_size requires positive integers. Got (0,3)

WARNING at line 8, column 15:
Implicit conversion of 256.0 to integer 256
```

### Planned Error Message Improvements (v0.2.6)

Version 0.2.6 plans to improve error messages with more context and better formatting (#459):

```text
# Current error format
ERROR at line 15, column 10:
Dense units must be positive integers. Got -128

# Planned improvements for future releases
# ERROR at line 15, column 10:
# Dense units must be positive integers. Got -128
# Context: In network 'MNISTClassifier', layer 2
# Suggestion: Use a positive integer value
```

Planned error message improvements include:
- Context information about where the error occurred
- Suggestions for fixing common issues
- Better handling of nested structures

### Enhanced Error Messages (v0.2.5)

Version 0.2.5 includes improved error messages with more context and better formatting:

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

Error messages now include:
- Precise line and column numbers
- Clear descriptions of the validation rule that was violated
- The actual value that caused the error
- Suggestions for fixing common issues

---

## Examples

### Vision Transformer
```yaml
network ViT {
  input: (224, 224, 3)
  layers:
    Conv2D(64, (7,7), strides=2) @ "cuda:0"
    TransformerEncoder() * 12     # 12 transformer blocks
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

### Parameter Types
```yaml
# Categorical Choice
HPO(choice(128, 256, 512))      # Select from discrete values

# Range with Step
HPO(range(0.3, 0.7, step=0.1))  # Linear range with step size

# Logarithmic Range
HPO(log_range(1e-4, 1e-2))      # Log-scale range for learning rates
```

### Supported Parameters
| Parameter | HPO Type | Example | Since Version |
|-----------|----------|---------|---------------|
| Dense units | `choice` | `Dense(HPO(choice(64, 128, 256)))` | v0.2.5 |
| Dropout rate | `range` | `Dropout(HPO(range(0.3, 0.7, step=0.1)))` | v0.2.5 |
| Learning rate | `log_range` | `Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))` | v0.2.5 |
| Conv2D filters | `choice` | `Conv2D(filters=HPO(choice(32, 64)))` | v0.2.6 |
| Conv2D kernel_size | `choice` | `Conv2D(kernel_size=HPO(choice((3,3), (5,5))))` | v0.2.7 |
| Padding | `choice` | `Conv2D(padding=HPO(choice("same", "valid")))` | v0.2.7 |
| Decay steps | `choice` | `ExponentialDecay(0.1, HPO(choice(500, 1000)), 0.96)` | v0.2.7 |

### Validation Rules
- Dense units must be positive integers
- Dropout rate must be between 0 and 1
- Learning rate must be positive
- Step size must be provided for range type

### HPO Parameters Updates (v0.2.6)
- **Advanced Examples**: Added comprehensive examples demonstrating complex HPO configurations (#448):
  - Multiple HPO parameters within the same layer
  - HPO parameters in both optimizer and learning rate schedules
  - Batch size optimization alongside model parameters
- **Planned Error Handling Improvements**: Planning better validation and error reporting for HPO parameters (#459):
  - More precise line/column information in error messages
  - Detailed context about parameter constraints
  - Validation for complex nested configurations

### HPO Parameters Updates (v0.2.8)
- **Consistent Parameter Naming**: Improved HPO log_range parameter naming from low/high to min/max for consistency.
- **Enhanced Conv2D Support**: Fixed issues with HPO parameters in Conv2D layers (filters, kernel_size, padding).
- **Optimizer Parameters**: Fixed issues with optimizer HPO parameters without quotes.
- **Missing Parameters Handling**: Added graceful handling of missing parameters in best_params during HPO optimization.

### HPO Parameters Updates (v0.2.5)
- **Multi-Framework Support**: HPO now works seamlessly across both PyTorch and TensorFlow backends.
- **Optimizer Parameters**: All optimizer parameters now support HPO, including:
  - `learning_rate` with `HPO(log_range(1e-4, 1e-2))` (#434)
  - `beta_1` and `beta_2` for Adam
  - `momentum` for SGD
- **Learning Rate Schedules**: HPO parameters can be nested within learning rate schedules.
- **String Representation**: Improved handling of scientific notation (e.g., `1e-4` vs `0.0001`).
- **No-Quote Syntax**: Parameters can be specified without quotes for cleaner syntax.

### Examples
#### Basic HPO Example
```yaml
network HPOExample {
  input: (28, 28, 1)
  layers:
    Dense(HPO(choice(128, 256)))
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, "softmax")
  optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
  train {
    epochs: 10
    search_method: "random"
  }
}
```

#### Advanced HPO with Learning Rate Schedules
```yaml
network AdvancedHPO {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3))
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(HPO(choice(128, 256, 512)))
    Output(10, "softmax")
  optimizer: SGD(
    learning_rate=ExponentialDecay(
      HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
      1000,                              # Decay steps
      HPO(range(0.9, 0.99, step=0.01))   # Decay rate
    ),
    momentum=HPO(range(0.8, 0.99, step=0.01))
  )
  train {
    epochs: 20
    batch_size: HPO(choice(32, 64, 128))
    search_method: "bayesian"
  }
}

---

## Cloud Integration (v0.2.8+)

Neural DSL v0.2.8 introduces enhanced support for running in cloud environments like Kaggle, Google Colab, and AWS SageMaker. This allows you to leverage cloud resources for training and debugging your models.

### Cloud Platforms

| Platform | Description | Command |
|----------|-------------|---------|
| **Kaggle** | Data science competition platform | `neural cloud connect kaggle` |
| **Google Colab** | Free Jupyter notebook environment | `neural cloud connect colab` |
| **AWS SageMaker** | Managed machine learning service | `neural cloud connect sagemaker` |

### Cloud Commands

```bash
# Connect to a cloud platform
neural cloud connect kaggle

# Connect with interactive shell
neural cloud connect kaggle --interactive

# Execute a Neural DSL file on a cloud platform
neural cloud execute kaggle my_model.neural

# Run Neural in cloud mode with remote access
neural cloud run --setup-tunnel
```

### Python API

```python
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()
print(f"Detected environment: {executor.environment}")
print(f"GPU available: {executor.is_gpu_available}")

# Compile and run a model
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST')

# Start the NeuralDbg dashboard with ngrok tunnel
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard URL: {dashboard_info['tunnel_url']}")
```

### Interactive Shell

The interactive shell provides a familiar Neural CLI experience but executes commands on the cloud platform:

```bash
# Connect to Kaggle with an interactive shell
neural cloud connect kaggle --interactive

# In the shell, you can run commands like:
neural-cloud> run my_model.neural --backend tensorflow
neural-cloud> visualize my_model.neural
neural-cloud> debug my_model.neural --setup-tunnel
neural-cloud> shell ls -la
neural-cloud> python print("Hello from Kaggle!")
```

### Remote Dashboard Access

You can access the NeuralDbg dashboard remotely through an ngrok tunnel:

```bash
# Start the dashboard with a tunnel
neural debug my_model.neural --setup-tunnel

# Or in Python
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard URL: {dashboard_info['tunnel_url']}")
```

---

## Training Configuration

### Basic Setup
```yaml
network MyModel {
    train {
        epochs: 100
        batch_size: 32
        validation_split: 0.2
        search_method: "bayesian"  # For HPO
    }
}
```

### Optimizer Configuration
```yaml
# Basic optimizer configuration
optimizer: Adam(
    learning_rate=HPO(log_range(1e-4, 1e-2)),
    beta_1=0.9,
    beta_2=0.999
)

# Learning rate schedules
optimizer: SGD(
    learning_rate=ExponentialDecay(0.1, 1000, 0.96),
    momentum=0.9
)

# Learning rate schedules with HPO
optimizer: SGD(
    learning_rate=ExponentialDecay(HPO(range(0.05, 0.2, step=0.05)), 1000, HPO(range(0.9, 0.99, step=0.01)))
)
```

### Learning Rate Schedules
Learning rate schedules allow you to dynamically adjust the learning rate during training. They can be specified directly in the `learning_rate` parameter of optimizers.

```yaml
# ExponentialDecay schedule
learning_rate=ExponentialDecay(0.1, 1000, 0.96)

# With HPO parameters
learning_rate=ExponentialDecay(HPO(range(0.05, 0.2, step=0.05)), 1000, 0.96)
```

For backward compatibility, string-based learning rate schedules are also supported:

```yaml
# String-based ExponentialDecay schedule
learning_rate="ExponentialDecay(0.1, 1000, 0.96)"
```

Supported schedules:
- `ExponentialDecay`: Applies exponential decay to the learning rate
- `PiecewiseConstantDecay`: Uses a piecewise constant decay schedule
- `PolynomialDecay`: Applies a polynomial decay to the learning rate
- `InverseTimeDecay`: Applies inverse time decay to the learning rate

---

## Enhanced HPO Support (v0.2.7+)

Version 0.2.7 introduces significant improvements to hyperparameter optimization support, particularly for Conv2D layers and learning rate schedules.

### Key HPO Improvements
- **Conv2D kernel_size HPO**: Added support for optimizing kernel sizes in convolutional layers
- **Padding Parameter HPO**: Extended HPO support to padding parameters
- **ExponentialDecay Parameter Structure**: Enhanced support for complex decay schedules
- **HPO Parameter Naming**: Improved parameter naming from low/high to min/max for consistency

### Conv2D with HPO Parameters
```yaml
# Basic Conv2D with HPO for filters only (supported in v0.2.6)
Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3))

# Enhanced Conv2D with HPO for both filters and kernel_size (new in v0.2.7)
Conv2D(
  filters=HPO(choice(32, 64)),
  kernel_size=HPO(choice((3,3), (5,5))),
  padding=HPO(choice("same", "valid"))
)
```

### ExponentialDecay with HPO Parameters
```yaml
# Basic ExponentialDecay with HPO (supported in v0.2.6)
ExponentialDecay(
  HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
  1000,                              # Fixed decay steps
  HPO(range(0.9, 0.99, step=0.01))   # Decay rate
)

# Enhanced ExponentialDecay with HPO for all parameters (new in v0.2.7)
ExponentialDecay(
  HPO(log_range(1e-3, 1e-1)),       # Initial learning rate
  HPO(choice(500, 1000, 2000)),      # Variable decay steps
  HPO(range(0.9, 0.99, step=0.01))   # Decay rate
)
```

### HPO Parameter Naming Improvements
```yaml
# Old style (v0.2.6)
HPO(log_range(low=1e-4, high=1e-2))

# New style (v0.2.7)
HPO(log_range(min=1e-4, max=1e-2))
```

### HPO Range Step Parameter
```yaml
# In v0.2.7, the step parameter is optional with a default value
HPO(range(0.1, 0.5))  # Uses default step

# Explicit step parameter
HPO(range(0.1, 0.5, step=0.1))
```

## Enhanced Dashboard UI (v0.2.6+)

Version 0.2.6 introduces a significantly improved NeuralDbg dashboard with a more aesthetic design and better visualization components (#452).

### Key UI Improvements
- **Dark Theme**: A modern, eye-friendly dark interface using Dash Bootstrap components
- **Responsive Design**: Better layout that adapts to different screen sizes
- **Improved Visualizations**: Enhanced tensor flow animations and shape propagation charts
- **Real-time Updates**: Fixed WebSocket connectivity for smoother data streaming

### Using the Enhanced Dashboard
```bash
# Start the dashboard with dark theme (default in v0.2.6+)
neural debug my_model.neural --theme dark

# Use light theme if preferred
neural debug my_model.neural --theme light
```

### Dashboard Components
- **Execution Trace**: Real-time visualization of layer activations
- **Gradient Flow**: Analysis of gradient magnitudes across layers
- **Dead Neuron Detection**: Identification of inactive neurons
- **Resource Monitoring**: CPU/GPU/Memory usage tracking
- **Shape Propagation**: Interactive visualization of tensor shapes

---

## Aquarium IDE (v0.2.9+)

Neural v0.2.9 introduces an early preview of Aquarium IDE, a new development environment for neural network design. In this initial version, it provides a basic visual interface for designing simple neural networks and viewing tensor shapes.

### Current Features

- **Basic Visual Designer**: Simple interface for adding and configuring common layer types
- **Shape Calculation**: View tensor dimensions for each layer in your network
- **Neural DSL Code Generation**: Generate basic Neural DSL code from your visual design
- **Parameter Estimation**: Basic calculation of parameter counts for each layer

### Technology Stack

- **Frontend**: Tauri with JavaScript/HTML/CSS for cross-platform compatibility
- **Backend**: Rust components for shape calculation
- **Neural Integration**: Integration with Neural's shape propagator for tensor dimension calculations

### Current Limitations

- Only supports a small set of layer types (Input, Conv2D, MaxPooling2D, Flatten, Dense, Output)
- Limited parameter configuration options
- Basic shape calculation that may not handle all edge cases
- Simple code generation without advanced features
- No support for complex network architectures (e.g., multi-input/output, skip connections)
- Limited error checking and validation

### Trying Aquarium IDE

1. **Install and Launch**:
   ```bash
   # Clone the Neural repository if you haven't already
   git clone https://github.com/Lemniscate-world/Neural.git
   cd Neural

   # Update submodules to get Aquarium
   git submodule update --init --recursive

   # Install Rust if you don't have it already
   # https://www.rust-lang.org/tools/install

   # Install Tauri CLI
   cargo install tauri-cli

   # Navigate to the Aquarium directory
   cd Aquarium

   # Install Node.js dependencies
   npm install

   # Run the development server (this may take a few minutes the first time)
   cargo tauri dev
   ```

2. **Try the Basic Features**:
   - Add layers using the buttons in the left panel
   - Configure basic parameters in the properties panel
   - View tensor shapes in the shape tab
   - See generated Neural DSL code in the code tab

### Shape Calculation

The current version calculates basic tensor dimensions for each layer in your network:

```
Layer         | Input Shape      | Output Shape     | Parameters
--------------|------------------|------------------|------------
Input Layer   | -                | [null,28,28,1]   | 0
Conv2D        | [null,28,28,1]   | [null,28,28,32]  | 320
MaxPooling2D  | [null,28,28,32]  | [null,14,14,32]  | 0
Flatten       | [null,14,14,32]  | [null,6272]      | 0
Dense         | [null,6272]      | [null,128]       | 802,944
Output        | [null,128]       | [null,10]        | 1,290
```

### Basic Code Generation

The current version generates simple Neural DSL code from your visual design:

```yaml
# Neural DSL Model

Input(shape=[28, 28, 1])
Conv2D(filters=32, kernel_size=[3, 3], padding="same", activation="relu")
MaxPooling2D(pool_size=[2, 2])
Flatten()
Dense(units=128, activation="relu")
Output(units=10, activation="softmax")
```

### Roadmap

Aquarium IDE is in very early development, and we have a long roadmap ahead. Some of the features we're planning to work on:

- **Support for More Layer Types**: Add support for additional layer types beyond the basic ones
- **Improved Shape Propagation**: More accurate and detailed shape calculations
- **Better Error Handling**: Provide more helpful error messages and validation
- **Visual Connections**: Allow creating connections between layers visually
- **Save/Load Functionality**: Save and load network designs
- **Export to Multiple Formats**: Export to different backends and formats

## Blog Support (v0.2.6+)

Neural DSL now includes infrastructure for blog content with markdown support and Dev.to integration (#445).

### Blog Directory Structure
```
docs/
  blog/
    README.md             # Blog overview and guidelines
    blog-list.json       # Metadata for all blog posts
    website_*.md         # Posts for the website
    devto_*.md           # Posts formatted for Dev.to
```

### Creating a Blog Post
1. Create a new markdown file in the `docs/blog/` directory
2. Use the `website_` prefix for website posts or `devto_` prefix for Dev.to posts
3. Add metadata to `blog-list.json`

### Dev.to Integration
Posts with the `devto_` prefix include special frontmatter for Dev.to:

```markdown
---
title: "Your Title Here"
published: true
description: "Brief description of your post"
tags: tag1, tag2, tag3
cover_image: https://url-to-your-cover-image.png
---

# Your Content Here
```

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

# Generate coverage report
coverage run -m pytest && coverage html
```

### Linting Rules
- Type hint enforcement
- Parameter validation checks
- HPO syntax verification
- Device configuration validation

### Contribution Requirements
1. Unit tests for new features
2. Documentation updates
3. Backward compatibility checks
4. Pre-commit hook validation

---

## Migration Guide

### From v0.2.4 to v0.2.5

#### HPO Optimizer Improvements

**Old Style (v0.2.4):**
```yaml
# Quoted optimizer parameters
optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"

# Limited support for nested HPO in learning rate schedules
```

**New Style (v0.2.5):**
```yaml
# No quotes needed for optimizer parameters
optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))

# Full support for nested HPO in learning rate schedules
optimizer: SGD(
  learning_rate=ExponentialDecay(
    HPO(range(0.05, 0.2, step=0.05)),
    1000,
    HPO(range(0.9, 0.99, step=0.01))
  )
)
```

#### Multi-Framework Support

In v0.2.5, HPO works seamlessly across both PyTorch and TensorFlow backends:

```bash
# Run HPO with PyTorch backend
neural compile mnist_hpo.neural --backend pytorch --hpo

# Run HPO with TensorFlow backend
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
    layers: Dense(64)    # Integer numbers
    optimizer: Adam(learning_rate=0.001)
}
```


[Report Issues](https://github.com/Lemniscate-world/Neural/issues)
