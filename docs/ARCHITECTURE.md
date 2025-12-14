# Neural DSL Architecture

## Overview

Neural DSL is a compiler and toolchain for neural network definition. This document explains the high-level architecture and data flow.

## System Architecture

```
┌─────────────────┐
│  .neural file   │  User writes declarative model definition
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Parser      │  Lark-based parser → AST
│  (neural/parser)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Shape Validator │  Propagate and validate tensor shapes
│ (neural/shape_  │
│   propagation)  │
└────────┬────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
  ┌───────────┐      ┌───────────┐      ┌───────────┐
  │ TensorFlow│      │  PyTorch  │      │   ONNX    │
  │ Generator │      │ Generator │      │ Generator │
  └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
        │                  │                  │
        ▼                  ▼                  ▼
  ┌───────────┐      ┌───────────┐      ┌───────────┐
  │ .py (TF)  │      │ .py (PT)  │      │  .onnx    │
  └───────────┘      └───────────┘      └───────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Execution  │  Train/evaluate model
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  NeuralDbg  │  Debug & visualize
                    │  Dashboard  │
                    └─────────────┘
```

## Core Components

### 1. Parser (`neural/parser/`)

**Purpose**: Convert `.neural` files to Abstract Syntax Tree (AST)

**Key Files**:
- `grammar.py` - Lark grammar definition
- `parser.py` - Main parser class
- `network_processors.py` - AST transformers
- `layer_handlers.py` - Layer-specific parsing

**Flow**:
```
.neural file → Lark Parser → Raw AST → Transformer → Structured AST
```

**Output**: Dictionary with:
- `network_name`
- `input_shape`
- `layers` (list of layer configs)
- `optimizer`, `loss`, `metrics`
- `training` config

### 2. Shape Propagator (`neural/shape_propagation/`)

**Purpose**: Validate tensor dimensions through the network

**Key Files**:
- `shape_propagator.py` - Main propagation logic
- `layer_handlers.py` - Per-layer shape calculations
- `utils.py` - Shape manipulation utilities

**Flow**:
```
AST + input_shape → Layer-by-layer propagation → Output shapes + validation
```

**What it catches**:
- Dimension mismatches
- Invalid kernel sizes
- Incompatible layer connections
- Flatten/reshape errors

**Example**:
```python
from neural.shape_propagation import ShapePropagator

propagator = ShapePropagator(input_shape=(28, 28, 1))
propagator.add_layer("Conv2D", {"filters": 32, "kernel_size": (3, 3)})
# Output: (26, 26, 32)
propagator.add_layer("MaxPooling2D", {"pool_size": (2, 2)})
# Output: (13, 13, 32)
propagator.add_layer("Flatten", {})
# Output: (5408,)
```

### 3. Code Generators (`neural/code_generation/`)

**Purpose**: Generate framework-specific Python code

**Key Files**:
- `base_generator.py` - Abstract base class
- `tensorflow_generator.py` - TensorFlow Keras code
- `pytorch_generator.py` - PyTorch code
- `onnx_generator.py` - ONNX export
- `shape_policy_helpers.py` - Shape handling policies

**Flow**:
```
AST → Generator.generate() → Framework-specific Python code
```

**Design**:
- Each generator inherits from `BaseGenerator`
- Implements `_generate_layer(layer_type, params)` for each layer
- Uses shape information from propagator
- Generates idiomatic, readable code

**Example Output (PyTorch)**:
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5408, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
```

### 4. CLI (`neural/cli/`)

**Purpose**: Command-line interface for all operations

**Key Files**:
- `cli.py` - Main Click commands
- `version.py` - Version management
- `lazy_imports.py` - Lazy loading for fast startup

**Commands**:
- `neural compile` - Generate code
- `neural run` - Compile + execute
- `neural visualize` - Generate diagrams
- `neural debug` - Start debugger
- `neural export` - Export for deployment

**Design Philosophy**:
- Fast startup (<500ms)
- Lazy imports for heavy dependencies
- Clear, actionable error messages
- Composable commands

### 5. Dashboard (`neural/dashboard/`)

**Purpose**: Real-time debugging and visualization (NeuralDbg)

**Key Files**:
- `dashboard.py` - Main Dash application
- `debugger_backend.py` - Backend API
- `tensor_flow.py` - Tensor flow visualization

**Features**:
- Execution traces
- Gradient flow monitoring
- Dead neuron detection
- Memory profiling
- Real-time updates via WebSockets

**Architecture**:
```
Flask/Dash Server (port 8050)
    │
    ├── WebSocket Manager (real-time updates)
    ├── Debugger Backend (data collection)
    └── Visualization Components (Plotly charts)
```

## Data Flow

### Compilation Flow

```
1. Read .neural file
2. Parse to AST
3. Validate syntax
4. Propagate shapes
5. Generate target code
6. Write output file
```

### Execution Flow

```
1. Compile to Python
2. Import generated module
3. Instantiate model
4. Load data
5. Train/evaluate
6. Log to debugger (optional)
```

### Debug Flow

```
1. Start dashboard server
2. Instrument model code
3. Execute training
4. Stream metrics to dashboard
5. Visualize in real-time
```

## Type System

### Shape Representation

Shapes are tuples of integers:
```python
(28, 28, 1)      # 2D grayscale image
(224, 224, 3)    # 2D RGB image
(None, 128)      # Sequence with variable length
(10,)            # 1D vector
```

Special values:
- `None` - Variable/unknown dimension
- `-1` - Inferred dimension (like NumPy)

### Layer Types

Layers are categorized:
- **Convolutional**: Conv2D, Conv1D, DepthwiseConv2D, etc.
- **Pooling**: MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
- **Dense**: Dense, Output
- **Normalization**: BatchNormalization, LayerNormalization
- **Regularization**: Dropout, SpatialDropout2D
- **Recurrent**: LSTM, GRU, SimpleRNN
- **Attention**: MultiHeadAttention, SelfAttention
- **Utility**: Flatten, Reshape, Concatenate

Each layer type has:
- Required parameters
- Optional parameters
- Shape transformation rules
- Backend-specific implementations

## Extension Points

### Adding New Layers

1. **Update grammar** (`neural/parser/grammar.py`)
2. **Add parser handler** (`neural/parser/layer_handlers.py`)
3. **Add shape rule** (`neural/shape_propagation/layer_handlers.py`)
4. **Implement generators**:
   - `neural/code_generation/tensorflow_generator.py`
   - `neural/code_generation/pytorch_generator.py`
   - `neural/code_generation/onnx_generator.py`

### Adding New Backends

1. **Create generator class** extending `BaseGenerator`
2. **Implement layer generation methods**
3. **Handle shape policies**
4. **Add to CLI options**
5. **Add tests**

### Adding Optimizations

1. **Create optimizer module** in `neural/optimization/`
2. **Hook into code generation pipeline**
3. **Add CLI flags for optimization passes**

## Performance Considerations

### Parser Performance
- Lark parser is fast: ~1-5ms for typical models
- AST transformation: ~5-10ms
- Total parse time: <20ms for most models

### Shape Propagation Performance
- O(n) where n = number of layers
- Typical time: <10ms for models with <100 layers
- Slowdown for very deep networks (>1000 layers)

### Code Generation Performance
- Template-based generation: ~5-20ms per model
- No compilation (just text generation)
- Instant for most models

### Startup Time
- Cold start: ~300-500ms (lazy imports)
- Warm start: ~50-100ms
- Most time spent importing TensorFlow/PyTorch (if needed)

## Error Handling

### Error Categories

1. **Syntax Errors**: Invalid DSL syntax
2. **Semantic Errors**: Valid syntax, wrong meaning
3. **Shape Errors**: Dimension mismatches
4. **Backend Errors**: Framework-specific issues
5. **Runtime Errors**: Execution failures

### Error Reporting Strategy

```
┌─────────────────────┐
│   Parse Error?      │  → Show line/column, expected syntax
└──────────┬──────────┘
           │ No
           ▼
┌─────────────────────┐
│   Shape Error?      │  → Show layer chain, expected vs actual shapes
└──────────┬──────────┘
           │ No
           ▼
┌─────────────────────┐
│  Generation Error?  │  → Show layer, suggest fix
└──────────┬──────────┘
           │ No
           ▼
┌─────────────────────┐
│   Runtime Error     │  → Framework-specific error + context
└─────────────────────┘
```

## Testing Strategy

### Unit Tests
- Parser: Test each layer type
- Shape propagation: Test each transformation
- Code generation: Test output correctness

### Integration Tests
- End-to-end: `.neural` → code → execution
- Cross-backend: Same model, multiple frameworks
- Error cases: Invalid inputs

### Performance Tests
- Parse time benchmarks
- Shape propagation benchmarks
- Generated code performance

## Security Considerations

1. **No `eval()` or `exec()`**: Generated code is written to file, not evaluated
2. **Path validation**: Prevent path traversal
3. **Input validation**: Sanitize user inputs
4. **Dependency scanning**: Regular security audits

## Future Architecture Changes

### Planned Improvements
1. **Incremental compilation**: Cache parsed models
2. **Optimization passes**: Dead code elimination, constant folding
3. **Plugin system**: User-defined layers and backends
4. **Language server**: IDE integration via LSP

### Deprecation Plans
See [DEPRECATIONS.md](DEPRECATIONS.md) for modules being removed/extracted.

## Related Documents

- [FOCUS.md](FOCUS.md) - Project scope
- [TYPE_SAFETY.md](TYPE_SAFETY.md) - Type checking guidelines
- [dsl.md](dsl.md) - DSL language reference

---

Questions? Open an issue or ask on [Discord](https://discord.gg/KFku4KvS).
