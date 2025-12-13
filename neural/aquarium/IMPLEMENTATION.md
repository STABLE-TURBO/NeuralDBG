# Neural Aquarium - Implementation Documentation

## Overview

Neural Aquarium is a comprehensive web-based IDE for the Neural DSL, providing model compilation, execution, and debugging capabilities. This document describes the complete implementation.

## Architecture

### Component Structure

```
neural/aquarium/
├── aquarium.py                    # Main application (Dash app)
├── __init__.py                    # Package exports
├── __main__.py                    # Module entry point
├── config.py                      # Configuration settings
├── examples.py                    # Example DSL models
├── README.md                      # User documentation
├── QUICKSTART.md                  # Quick start guide
├── CHANGELOG.md                   # Version history
├── IMPLEMENTATION.md              # This file
└── src/
    ├── __init__.py
    └── components/
        ├── __init__.py
        └── runner/
            ├── __init__.py
            ├── runner_panel.py       # Main UI component
            ├── execution_manager.py  # Process management
            ├── script_generator.py   # Training script generation
            └── utils.py              # Utility functions
```

## Core Components

### 1. Main Application (aquarium.py)

**Purpose**: Primary entry point and UI layout

**Key Features**:
- Dash application setup with dark theme
- DSL editor with textarea component
- Multi-tab interface (Runner, Debugger, Visualization, Documentation)
- Model parsing and validation
- Integration with runner panel

**Callbacks**:
- `parse_dsl`: Parses DSL content and extracts model information
- `load_example`: Loads random example models

**Technologies**:
- Dash framework
- Dash Bootstrap Components
- Plotly for visualizations

### 2. Runner Panel (runner_panel.py)

**Purpose**: Model compilation and execution interface

**Components**:

#### Backend Configuration
- Backend selection dropdown (TensorFlow, PyTorch, ONNX)
- Dataset selection (MNIST, CIFAR10, CIFAR100, ImageNet, Custom)
- Custom dataset path input
- Training parameters (epochs, batch size, validation split)
- Options checklist (auto-flatten, HPO, verbose, save weights)

#### Action Controls
- **Compile**: Generate backend-specific Python code
- **Run**: Execute compiled script
- **Stop**: Terminate running process
- **Export Script**: Save to custom location
- **Open in IDE**: Launch in default editor
- **Clear**: Reset console output

#### Output Display
- Console with real-time logs
- Status badge (Idle/Compiled/Running/Error/Stopped)
- Training metrics graph (placeholder)

**Callbacks**:
- `compile_model`: Generates code using Neural code generator
- `run_model`: Starts script execution in separate process
- `stop_execution`: Terminates running process
- `update_logs`: Streams output from execution queue
- `export_script`: Saves compiled script to file
- `open_in_ide`: Opens script in system editor

### 3. Execution Manager (execution_manager.py)

**Purpose**: Process lifecycle and output management

**Classes**:

#### ExecutionManager
- `compile_model()`: Wraps Neural code generation
- `run_script()`: Executes Python scripts in subprocess
- `stop_execution()`: Terminates processes
- `get_output_lines()`: Retrieves queued output
- `get_metrics()`: Extracts parsed metrics
- `export_script()`: Exports scripts with metadata
- `open_in_editor()`: System-specific file opening

**Features**:
- Threaded execution for non-blocking UI
- Queue-based output streaming
- Metrics parsing from training logs
- Cross-platform editor integration

#### DatasetManager
- Dataset information lookup
- Built-in dataset registry
- Custom dataset validation
- Dataset metadata (shape, classes, type)

### 4. Script Generator (script_generator.py)

**Purpose**: Generate complete training scripts

**Methods**:
- `generate_training_script()`: Main generation entry point
- `_generate_tensorflow_script()`: TensorFlow-specific template
- `_generate_pytorch_script()`: PyTorch-specific template
- `wrap_model_code_*()`: Code wrapping utilities

**Generated Script Components**:
- Dataset loading code
- Model building function
- Training loop
- Validation logic
- Metrics logging
- Model weight saving
- Command-line friendly output

### 5. Utilities (utils.py)

**Purpose**: Helper functions for runner panel

**Functions**:
- `parse_log_line()`: Extract information from logs
- `extract_metrics()`: Parse training metrics
- `format_console_line()`: Format output with prefixes
- `validate_training_config()`: Validate parameters
- `format_file_size()`: Human-readable sizes
- `estimate_training_time()`: Time estimation
- `truncate_output()`: Console output management
- Display name formatting functions

### 6. Configuration (config.py)

**Purpose**: Centralized configuration management

**Settings**:
- Application constants (name, version, ports)
- Path configuration (cache, exports, temp)
- Backend and dataset configuration
- Training parameter defaults and limits
- UI theme settings
- Feature flags
- Environment detection
- Example models registry

### 7. Examples (examples.py)

**Purpose**: Pre-built example models

**Available Examples**:
- MNIST Classifier
- CIFAR10 CNN
- Simple Dense Network
- LSTM Text Classifier
- VGG-style Network
- ResNet-style Block
- Autoencoder
- Transformer Encoder

**Functions**:
- `get_example()`: Retrieve by name
- `list_examples()`: Get all names
- `get_random_example()`: Random selection

## Data Flow

### 1. Compilation Flow

```
User writes DSL → Parse DSL → Validate → Configure backend →
Click Compile → Generate code → Save to temp file → Update UI
```

**Steps**:
1. User enters DSL in editor
2. Clicks "Parse DSL" for validation
3. Selects backend and dataset
4. Clicks "Compile"
5. `compile_model` callback triggered
6. Neural code generator produces Python code
7. Script saved to temporary location
8. Console shows compilation logs
9. Status updated to "Compiled"
10. Run/Export buttons enabled

### 2. Execution Flow

```
Compiled script → Configure training → Click Run → 
Start process → Stream output → Parse metrics → Display results
```

**Steps**:
1. User clicks "Run"
2. `run_model` callback triggered
3. ExecutionManager starts subprocess
4. Output streaming thread created
5. Log lines queued for UI update
6. `update_logs` callback polls queue
7. Metrics extracted and queued
8. Console displays real-time output
9. Process completes or stopped
10. Final status displayed

### 3. Export Flow

```
Compiled script → Click Export → Configure export →
Copy file → Save metadata → Show confirmation
```

**Steps**:
1. User clicks "Export Script"
2. Modal opens with configuration
3. User enters filename and location
4. Clicks "Export"
5. Script copied to destination
6. Metadata saved (optional)
7. Success notification shown

## Key Design Decisions

### 1. Separate Process Execution
- **Reason**: Prevents blocking Dash callback thread
- **Implementation**: Threading with subprocess
- **Benefit**: Responsive UI during training

### 2. Queue-Based Output Streaming
- **Reason**: Thread-safe communication
- **Implementation**: Python Queue with interval updates
- **Benefit**: Real-time log display without race conditions

### 3. Backend-Specific Script Generation
- **Reason**: Different training patterns per backend
- **Implementation**: Template-based generation
- **Benefit**: Complete, runnable scripts for each backend

### 4. Modular Component Design
- **Reason**: Maintainability and extensibility
- **Implementation**: Separate files for concerns
- **Benefit**: Easy to add features or modify behavior

### 5. Configuration Centralization
- **Reason**: Single source of truth
- **Implementation**: config.py module
- **Benefit**: Easy deployment configuration

## Integration Points

### 1. Neural DSL Parser
- **Location**: `neural.parser.parser`
- **Usage**: DSL validation and AST generation
- **Interface**: `create_parser()`, `ModelTransformer()`

### 2. Neural Code Generator
- **Location**: `neural.code_generation.code_generator`
- **Usage**: Backend code generation
- **Interface**: `generate_code(model_data, backend, ...)`

### 3. Neural CLI
- **Future Integration**: Add `aquarium` command
- **Usage**: `neural aquarium [--port PORT]`
- **Benefit**: Unified interface

### 4. NeuralDbg
- **Status**: Placeholder integration
- **Future**: Real-time debugging panel
- **Features**: Gradient flow, dead neurons, anomalies

## Extension Points

### 1. Adding New Backends

**Location**: `script_generator.py`

```python
def _generate_BACKEND_script(model_code, dataset, ...):
    # Implement backend-specific template
    pass
```

### 2. Adding New Datasets

**Location**: `config.py` and `execution_manager.py`

```python
BUILTIN_DATASETS["NewDataset"] = {
    "shape": (H, W, C),
    "classes": N,
    "type": "image"
}
```

### 3. Adding UI Components

**Location**: `src/components/` directory

Create new panel module similar to `runner/`

### 4. Custom Metrics Visualization

**Location**: `runner_panel.py`

Connect `runner-metrics-graph` to real-time data

## Performance Considerations

### 1. Output Buffering
- Limit console lines to prevent memory issues
- Truncate when exceeding threshold
- Configurable via `config.CONSOLE_MAX_LINES`

### 2. Process Timeouts
- Maximum execution time limit
- Configurable via `config.PROCESS_TIMEOUT`
- Prevents hanging processes

### 3. Queue Management
- Non-blocking queue operations
- Maximum item limits
- Periodic cleanup

### 4. File System
- Temporary file cleanup
- Cache management
- Path validation

## Security Considerations

### 1. Code Execution
- Scripts run in same environment
- No sandboxing (local development tool)
- User responsible for code review

### 2. File Operations
- Path validation for exports
- Directory traversal prevention
- Permission checks

### 3. Input Validation
- DSL syntax validation
- Parameter range checking
- Type validation

## Future Enhancements

### High Priority
1. Real-time metrics visualization
2. Syntax highlighting in editor
3. Code completion
4. Enhanced error messages

### Medium Priority
5. Model comparison tools
6. Experiment tracking
7. Custom dataset UI
8. Batch execution

### Low Priority
9. Cloud execution
10. Collaborative editing
11. Plugin system
12. Theme customization

## Testing Strategy

### Unit Tests
- Test execution manager functions
- Test script generation
- Test utility functions
- Mock subprocess calls

### Integration Tests
- Test compilation flow
- Test execution flow
- Test export functionality
- Test callback chains

### UI Tests
- Test component rendering
- Test user interactions
- Test state management
- Test error handling

## Deployment

### Local Development
```bash
python -m neural.aquarium.aquarium --debug
```

### Production
```bash
python -m neural.aquarium.aquarium --port 8052
```

### Docker (Future)
```dockerfile
FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install -e ".[full]"
EXPOSE 8052
CMD ["python", "-m", "neural.aquarium.aquarium"]
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change port with `--port` flag
2. **Dependencies missing**: Run `pip install -e ".[full]"`
3. **Process won't stop**: Check for zombie processes
4. **Output not streaming**: Check queue polling interval
5. **Script export fails**: Check directory permissions

## Contributing

See main repository CONTRIBUTING.md for guidelines.

## License

Same as Neural DSL package.
