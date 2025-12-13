# Aquarium IDE Architecture Overview

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Diagram](#component-diagram)
3. [Technology Stack](#technology-stack)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Backend Integration](#backend-integration)
7. [Plugin System](#plugin-system)
8. [Performance Considerations](#performance-considerations)
9. [Security Architecture](#security-architecture)

## System Architecture

Aquarium IDE follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               Web Browser (Client)                         │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │  Editor  │  │  Runner  │  │ Debugger │  │   Viz    │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Dash Framework                           │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │              Callback System                         │ │ │
│  │  │  • Parse DSL • Compile • Execute • Export           │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────────┐
│                       Business Layer                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │  Execution │  │   Script   │  │   Model    │               │
│  │  Manager   │  │  Generator │  │  Parser    │               │
│  └────────────┘  └────────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────────┐
│                       Integration Layer                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │   Neural   │  │  TensorFlow│  │  PyTorch   │               │
│  │  DSL Core  │  │   Backend  │  │  Backend   │               │
│  └────────────┘  └────────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────────┐
│                       Infrastructure Layer                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │  File I/O  │  │  Process   │  │  Config    │               │
│  │            │  │  Manager   │  │  Manager   │               │
│  └────────────┘  └────────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

## Component Diagram

### High-Level Components

```
┌──────────────────────────────────────────────────────────┐
│                    Aquarium IDE                          │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │                Main Application                    │  │
│  │              (aquarium.py)                        │  │
│  └───────────────────────────────────────────────────┘  │
│              │         │         │         │            │
│              ▼         ▼         ▼         ▼            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Editor  │ │  Runner  │ │ Debugger │ │   Viz    │  │
│  │  Panel   │ │  Panel   │ │  Panel   │ │  Panel   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│              │                    │                     │
│              ▼                    ▼                     │
│  ┌──────────────────┐   ┌──────────────────┐          │
│  │   Model Parser   │   │ Execution Manager│          │
│  │  (parser.py)     │   │ (execution_mgr)  │          │
│  └──────────────────┘   └──────────────────┘          │
│              │                    │                     │
│              ▼                    ▼                     │
│  ┌──────────────────────────────────────────┐          │
│  │         Neural DSL Core                  │          │
│  │   ┌──────────┐  ┌──────────┐            │          │
│  │   │  Parser  │  │   Code   │            │          │
│  │   │          │  │Generator │            │          │
│  │   └──────────┘  └──────────┘            │          │
│  └──────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

### Component Dependencies

```
aquarium.py
├── dash (Web framework)
├── dash_bootstrap_components (UI components)
├── runner_panel.py
│   ├── execution_manager.py
│   │   ├── script_generator.py
│   │   └── subprocess (Process management)
│   └── dataset_manager.py
├── neural.parser.parser (DSL parsing)
├── neural.code_generation (Code generation)
└── config.py (Configuration management)
```

## Technology Stack

### Frontend

**Framework**: Dash + Plotly
- **Dash**: Python web framework for building analytics applications
- **Plotly**: Interactive data visualization library
- **Dash Bootstrap Components**: Pre-built UI components

**UI Components**:
- Cards, buttons, dropdowns, inputs
- Tabs for navigation
- Modals for dialogs
- Alerts for notifications

**Styling**:
- Bootstrap Darkly theme
- Font Awesome icons
- Custom CSS for specific components

### Backend

**Core Framework**: Python 3.8+
- **Dash Server**: Built on Flask
- **Threading**: For non-blocking execution
- **Subprocess**: For running training scripts
- **Queue**: For inter-thread communication

**Neural DSL Integration**:
- Lark parser for DSL syntax
- Code generators for TF/PyTorch/ONNX
- Shape propagation engine

**Data Processing**:
- NumPy for array operations
- Pandas for data manipulation (optional)

### Infrastructure

**File System**:
- Compiled scripts: `~/.neural/aquarium/compiled/`
- Exported scripts: `~/.neural/aquarium/exported/`
- Temp files: `~/.neural/aquarium/temp/`

**Configuration**:
- YAML configuration files
- Environment variables
- Runtime settings

**Process Management**:
- Subprocess for training execution
- Process monitoring and control
- Output streaming

## Core Components

### 1. Main Application (aquarium.py)

**Responsibilities**:
- Initialize Dash application
- Create application layout
- Register callbacks
- Manage application lifecycle

**Key Features**:
- Multi-tab interface
- Responsive layout
- Dark theme
- Icon integration

```python
# Core structure
app = dash.Dash(__name__, ...)
runner_panel = RunnerPanel(app)
app.layout = html.Div([...])

@app.callback(...)
def parse_dsl(n_clicks, dsl_content):
    # Parse and validate DSL
    pass

def main(port, debug):
    app.run_server(...)
```

### 2. Runner Panel (runner_panel.py)

**Responsibilities**:
- Provide compilation UI
- Manage execution controls
- Display console output
- Handle user interactions

**Key Features**:
- Backend selection (TF/PyTorch/ONNX)
- Dataset configuration
- Training parameters
- Action buttons (Compile, Run, Stop, Export)
- Real-time console output
- Status indicators

```python
class RunnerPanel:
    def __init__(self, app):
        self.app = app
        self.execution_manager = ExecutionManager()
        
    def create_layout(self):
        # Build UI components
        pass
        
    def _register_callbacks(self):
        # Register Dash callbacks
        pass
```

### 3. Execution Manager (execution_manager.py)

**Responsibilities**:
- Compile models to Python code
- Execute training scripts
- Manage process lifecycle
- Stream output to UI
- Parse training metrics

**Key Features**:
- Non-blocking execution (threading)
- Process control (start/stop)
- Output capture and streaming
- Metrics extraction
- Error handling

```python
class ExecutionManager:
    def compile_model(self, dsl_content, backend):
        # Generate backend-specific code
        pass
        
    def run_script(self, script_path, **kwargs):
        # Execute in subprocess
        pass
        
    def stop_execution(self):
        # Terminate running process
        pass
        
    def get_console_output(self):
        # Return buffered output
        pass
```

### 4. Script Generator (script_generator.py)

**Responsibilities**:
- Generate complete training scripts
- Add dataset loading code
- Create training loops
- Include evaluation code

**Key Features**:
- Backend-specific templates
- Dataset integration
- Training configuration
- Model saving logic

```python
class ScriptGenerator:
    def generate_tensorflow_script(self, model_code, dataset, config):
        # Generate TF training script
        pass
        
    def generate_pytorch_script(self, model_code, dataset, config):
        # Generate PyTorch training script
        pass
        
    def _add_dataset_loader(self, backend, dataset):
        # Generate dataset loading code
        pass
```

### 5. Model Parser Integration

**Responsibilities**:
- Parse Neural DSL syntax
- Extract model structure
- Validate model definition
- Transform to internal representation

**Integration**:
```python
from neural.parser.parser import create_parser, ModelTransformer

parser = create_parser(start_rule='network')
tree = parser.parse(dsl_content)
model_data = ModelTransformer().transform(tree)
```

### 6. Configuration Manager (config.py)

**Responsibilities**:
- Centralized configuration
- Application settings
- Path management
- Backend configuration
- Dataset registry

**Structure**:
```python
class Config:
    # Application settings
    APP_NAME = "Neural Aquarium IDE"
    VERSION = "0.1.0"
    
    # Paths
    BASE_DIR = Path.home() / ".neural" / "aquarium"
    COMPILED_DIR = BASE_DIR / "compiled"
    EXPORTED_DIR = BASE_DIR / "exported"
    
    # Backends
    SUPPORTED_BACKENDS = ["tensorflow", "pytorch", "onnx"]
    DEFAULT_BACKEND = "tensorflow"
    
    # Datasets
    DATASETS = {
        "mnist": {...},
        "cifar10": {...},
        ...
    }
```

## Data Flow

### 1. Model Parsing Flow

```
User Input (DSL) → Parse Button Click
    ↓
Dash Callback Triggered
    ↓
create_parser() → Parse DSL Text
    ↓
Syntax Validation
    ↓
ModelTransformer() → Convert to Dict
    ↓
Update Model Info Panel
    ↓
Display Success/Error Alert
```

### 2. Compilation Flow

```
Parsed Model Data → Compile Button Click
    ↓
Dash Callback Triggered
    ↓
execution_manager.compile_model()
    ↓
Select Code Generator (TF/PyTorch/ONNX)
    ↓
neural.code_generation.generate()
    ↓
script_generator.generate_script()
    ↓
Add Dataset Loading Code
    ↓
Add Training Loop
    ↓
Write to File (temp or compiled dir)
    ↓
Update Status → "Compiled"
    ↓
Log to Console
```

### 3. Execution Flow

```
Compiled Script → Run Button Click
    ↓
Dash Callback Triggered
    ↓
execution_manager.run_script()
    ↓
Create subprocess.Popen()
    ↓
Start Thread for Output Streaming
    ↓
Stream stdout/stderr to Queue
    ↓
Interval Callback Reads Queue
    ↓
Update Console Output
    ↓
Parse Metrics (loss, accuracy)
    ↓
Update Metrics Graph
    ↓
Process Completes
    ↓
Update Status → "Idle"
```

### 4. Export Flow

```
Compiled Script → Export Button Click
    ↓
Open Export Modal
    ↓
User Enters Filename & Location
    ↓
Export Button in Modal
    ↓
Copy Script to Destination
    ↓
Save Metadata (optional)
    ↓
Display Success Notification
    ↓
Close Modal
```

## Backend Integration

### TensorFlow Backend

```python
# Code generation
from neural.code_generation.tensorflow_generator import TensorFlowGenerator

generator = TensorFlowGenerator()
code = generator.generate(model_data)

# Generated code structure
"""
import tensorflow as tf
from tensorflow import keras

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        ...
    ])
    return model

model = create_model()
model.compile(loss='...', optimizer='...', metrics=['...'])
...
"""
```

### PyTorch Backend

```python
# Code generation
from neural.code_generation.pytorch_generator import PyTorchGenerator

generator = PyTorchGenerator()
code = generator.generate(model_data)

# Generated code structure
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(...)
        ...
        
    def forward(self, x):
        x = self.conv1(x)
        ...
        return x
"""
```

### ONNX Backend

```python
# Code generation
from neural.code_generation.onnx_generator import ONNXGenerator

generator = ONNXGenerator()
code = generator.generate(model_data)
```

## Plugin System

### Plugin Architecture

```
Plugin Manager
├── Plugin Discovery
│   ├── Scan plugin directories
│   └── Load plugin metadata
├── Plugin Loading
│   ├── Import plugin modules
│   └── Instantiate plugin classes
├── Plugin Registration
│   ├── Register backends
│   ├── Register datasets
│   ├── Register UI components
│   └── Register exporters
└── Plugin Lifecycle
    ├── Initialize
    ├── Activate
    ├── Deactivate
    └── Cleanup
```

### Plugin Types

1. **Backend Plugins**: Add ML framework support
2. **Dataset Plugins**: Add dataset loaders
3. **UI Plugins**: Add UI components/panels
4. **Exporter Plugins**: Add export formats
5. **Debugger Plugins**: Add debugging tools
6. **Theme Plugins**: Add UI themes

### Plugin Registration

```python
# Backend plugin
from neural.aquarium.src.components.registry import BackendRegistry
BackendRegistry.register('mybackend', MyBackend)

# Dataset plugin
from neural.aquarium.src.components.registry import DatasetRegistry
DatasetRegistry.register('mydataset', MyDataset)
```

## Performance Considerations

### 1. Non-Blocking Execution

**Problem**: Training blocks UI
**Solution**: Execute in separate thread/process

```python
import threading

def run_training():
    # Long-running training
    pass

thread = threading.Thread(target=run_training)
thread.start()
```

### 2. Output Buffering

**Problem**: Too many console updates slow UI
**Solution**: Buffer output and batch updates

```python
import queue
output_queue = queue.Queue()

# Producer (subprocess)
for line in process.stdout:
    output_queue.put(line)
    
# Consumer (UI update every 100ms)
@app.callback(...)
def update_console():
    lines = []
    while not output_queue.empty():
        lines.append(output_queue.get())
    return '\n'.join(lines[-100:])  # Last 100 lines
```

### 3. Lazy Loading

**Problem**: Slow startup with all backends
**Solution**: Load backends on demand

```python
class BackendLoader:
    _backends = {}
    
    def get_backend(self, name):
        if name not in self._backends:
            self._backends[name] = self._load_backend(name)
        return self._backends[name]
```

### 4. Caching

**Problem**: Re-parsing DSL on every compile
**Solution**: Cache parsed models

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def parse_model(dsl_code):
    parser = create_parser()
    return parser.parse(dsl_code)
```

## Security Architecture

### Input Validation

```python
def validate_dsl_input(dsl_code):
    """Validate DSL input for security"""
    # Check length
    if len(dsl_code) > 1_000_000:
        raise ValueError("DSL code too large")
        
    # Check for suspicious patterns
    forbidden = ['__import__', 'eval', 'exec', 'os.system']
    for pattern in forbidden:
        if pattern in dsl_code:
            raise ValueError(f"Forbidden pattern: {pattern}")
```

### Path Sanitization

```python
def sanitize_path(path):
    """Sanitize file paths"""
    path = Path(path).resolve()
    
    # Ensure within allowed directories
    allowed_dirs = [Config.COMPILED_DIR, Config.EXPORTED_DIR]
    if not any(path.is_relative_to(d) for d in allowed_dirs):
        raise ValueError("Path outside allowed directories")
        
    return path
```

### Process Isolation

```python
def run_subprocess(script_path):
    """Run subprocess with resource limits"""
    process = subprocess.Popen(
        ['python', script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=SAFE_DIRECTORY,
        env={'PYTHONPATH': SAFE_PYTHONPATH}
    )
    return process
```

## Deployment Architecture

### Development Mode

```
Local Machine
├── Python Virtual Environment
├── Dash Dev Server (debug=True)
├── Hot Reload Enabled
└── Port: 8052 (localhost only)
```

### Production Mode

```
Production Server
├── WSGI Server (Gunicorn/uWSGI)
├── Reverse Proxy (Nginx)
├── SSL/TLS Termination
├── Load Balancer (optional)
└── Port: 80/443 (public)
```

### Docker Deployment

```
Docker Container
├── Python Base Image
├── Aquarium + Dependencies
├── Dash Production Server
└── Volume Mounts for Persistence
```

## Monitoring & Logging

### Application Logging

```python
import logging

logger = logging.getLogger('aquarium')
logger.setLevel(logging.INFO)

# Log important events
logger.info("Compiling model...")
logger.error("Compilation failed: %s", error)
```

### Metrics Collection

- Compilation time
- Execution time
- Error rates
- User actions
- Resource usage

## Future Architecture Considerations

### 1. WebSocket Support

Real-time bidirectional communication for:
- Live training metrics streaming
- Collaborative editing
- Real-time debugging

### 2. Microservices

Split into services:
- Frontend service (UI)
- Compilation service (model → code)
- Execution service (training)
- Storage service (files, models)

### 3. Cloud Integration

- Remote execution on cloud GPUs
- Distributed training
- Model serving integration
- Cloud storage

### 4. Scalability

- Horizontal scaling with load balancer
- Database for model history
- Caching layer (Redis)
- Message queue (RabbitMQ/Kafka)

## References

- [Dash Documentation](https://dash.plotly.com/)
- [Neural DSL Docs](../../docs/dsl.md)
- [Plugin Development Guide](plugin-development.md)
- [User Manual](user-manual.md)

---

**Version**: 1.0  
**Last Updated**: December 2024  
**License**: MIT
