# Aquarium IDE - Complete API Reference

**Version**: 1.0.0 | **Last Updated**: December 2024

---

## Table of Contents

1. [Python API](#1-python-api)
2. [REST API](#2-rest-api)
3. [Plugin API](#3-plugin-api)
4. [Component API](#4-component-api)
5. [TypeScript API](#5-typescript-api)
6. [CLI API](#6-cli-api)

---

## 1. Python API

### 1.1 ExecutionManager

**Import:**
```python
from neural.aquarium.src.components.runner.execution_manager import ExecutionManager
```

**Class: ExecutionManager**

Manages model compilation and execution.

#### Constructor

```python
manager = ExecutionManager()
```

**Parameters:** None

**Returns:** ExecutionManager instance

#### Methods

##### compile_model()

Compile Neural DSL code to backend-specific Python script.

```python
manager.compile_model(
    dsl_code: str,
    backend: str = 'tensorflow',
    output_path: str = None,
    config: dict = None
) -> str
```

**Parameters:**
- `dsl_code` (str): Neural DSL source code
- `backend` (str): Target backend ('tensorflow', 'pytorch', 'onnx')
- `output_path` (str, optional): Path for generated script
- `config` (dict, optional): Compilation configuration

**Returns:** str - Path to compiled script

**Raises:**
- `ValueError`: Invalid DSL syntax
- `CompilationError`: Backend code generation failed

**Example:**
```python
dsl_code = """
network MNISTNet {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
"""

manager = ExecutionManager()
script_path = manager.compile_model(
    dsl_code=dsl_code,
    backend='tensorflow',
    output_path='/tmp/mnist_model.py'
)
print(f"Compiled to: {script_path}")
```

##### run_script()

Execute compiled training script.

```python
manager.run_script(
    script_path: str,
    dataset: str = 'mnist',
    epochs: int = 10,
    batch_size: int = 32,
    validation_split: float = 0.2,
    verbose: bool = True,
    save_weights: bool = True,
    callback: callable = None
) -> subprocess.Popen
```

**Parameters:**
- `script_path` (str): Path to compiled script
- `dataset` (str): Dataset name or path
- `epochs` (int): Number of training epochs
- `batch_size` (int): Training batch size
- `validation_split` (float): Validation data split ratio
- `verbose` (bool): Enable verbose output
- `save_weights` (bool): Save model weights after training
- `callback` (callable): Function called with output lines

**Returns:** subprocess.Popen - Running process handle

**Example:**
```python
def output_callback(line):
    print(f"Training: {line}")

process = manager.run_script(
    script_path='/tmp/mnist_model.py',
    dataset='mnist',
    epochs=10,
    batch_size=32,
    callback=output_callback
)

# Wait for completion
process.wait()
```

##### stop_execution()

Terminate currently running training process.

```python
manager.stop_execution() -> bool
```

**Returns:** bool - True if process was stopped

**Example:**
```python
manager.stop_execution()
```

##### get_status()

Get current execution status.

```python
manager.get_status() -> str
```

**Returns:** str - Status ('idle', 'compiling', 'running', 'stopped', 'error')

**Example:**
```python
status = manager.get_status()
print(f"Current status: {status}")
```

##### get_output()

Get accumulated console output.

```python
manager.get_output() -> List[str]
```

**Returns:** List[str] - List of output lines

**Example:**
```python
output_lines = manager.get_output()
for line in output_lines:
    print(line)
```

##### parse_metrics()

Extract training metrics from output.

```python
manager.parse_metrics(output: str) -> dict
```

**Parameters:**
- `output` (str): Console output line

**Returns:** dict - Parsed metrics (loss, accuracy, val_loss, val_accuracy)

**Example:**
```python
line = "loss: 0.3245 - accuracy: 0.8921 - val_loss: 0.2156"
metrics = manager.parse_metrics(line)
print(metrics)
# {'loss': 0.3245, 'accuracy': 0.8921, 'val_loss': 0.2156}
```

---

### 1.2 ScriptGenerator

**Import:**
```python
from neural.aquarium.src.components.runner.script_generator import ScriptGenerator
```

**Class: ScriptGenerator**

Generates executable Python scripts from compiled DSL.

#### Constructor

```python
generator = ScriptGenerator(backend: str = 'tensorflow')
```

**Parameters:**
- `backend` (str): Target backend

#### Methods

##### generate()

Generate complete training script.

```python
generator.generate(
    model_code: str,
    dataset: str,
    config: dict
) -> str
```

**Parameters:**
- `model_code` (str): Generated model code
- `dataset` (str): Dataset name
- `config` (dict): Training configuration

**Returns:** str - Complete Python script

**Example:**
```python
generator = ScriptGenerator(backend='tensorflow')
script = generator.generate(
    model_code=compiled_model,
    dataset='mnist',
    config={
        'epochs': 10,
        'batch_size': 32,
        'validation_split': 0.2
    }
)
print(script)
```

---

### 1.3 PluginManager

**Import:**
```python
from neural.aquarium.src.plugins.plugin_manager import PluginManager
```

**Class: PluginManager**

Manages plugin lifecycle and coordination (Singleton).

#### Constructor

```python
manager = PluginManager()  # Returns singleton instance
```

#### Methods

##### list_plugins()

List all available plugins.

```python
manager.list_plugins() -> List[PluginMetadata]
```

**Returns:** List of plugin metadata

**Example:**
```python
plugins = manager.list_plugins()
for plugin in plugins:
    print(f"{plugin.name} v{plugin.version}")
```

##### enable_plugin()

Enable and activate a plugin.

```python
manager.enable_plugin(plugin_id: str) -> bool
```

**Parameters:**
- `plugin_id` (str): Plugin identifier

**Returns:** bool - Success status

**Example:**
```python
success = manager.enable_plugin('github-copilot-integration')
if success:
    print("Plugin enabled")
```

##### disable_plugin()

Disable and deactivate a plugin.

```python
manager.disable_plugin(plugin_id: str) -> bool
```

**Parameters:**
- `plugin_id` (str): Plugin identifier

**Returns:** bool - Success status

##### get_panels()

Get all registered panel plugins.

```python
manager.get_panels() -> List[dict]
```

**Returns:** List of panel configurations

##### get_themes()

Get all registered theme plugins.

```python
manager.get_themes() -> List[dict]
```

**Returns:** List of theme configurations

##### execute_command()

Execute a plugin command.

```python
manager.execute_command(
    command: str,
    args: dict = None
) -> Any
```

**Parameters:**
- `command` (str): Command name
- `args` (dict): Command arguments

**Returns:** Command result

##### register_hook()

Register event hook callback.

```python
manager.register_hook(
    event: str,
    callback: callable
) -> None
```

**Parameters:**
- `event` (str): Event name
- `callback` (callable): Callback function

**Example:**
```python
def on_compile(model_data):
    print(f"Model compiled: {model_data['name']}")

manager.register_hook('model_compiled', on_compile)
```

---

## 2. REST API

### 2.1 Base URL

```
http://localhost:8052/api
```

### 2.2 Compilation Endpoints

#### POST /compile

Compile Neural DSL code.

**Request:**
```json
{
  "dsl_code": "network MyModel {...}",
  "backend": "tensorflow",
  "config": {
    "auto_flatten": true,
    "verbose": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "script_path": "/tmp/aquarium_model_abc123.py",
  "model_info": {
    "name": "MyModel",
    "input_shape": [null, 28, 28, 1],
    "num_layers": 8,
    "total_params": 1199882
  }
}
```

**Errors:**
- 400: Invalid DSL syntax
- 500: Compilation failed

---

### 2.3 Execution Endpoints

#### POST /run

Execute compiled script.

**Request:**
```json
{
  "script_path": "/tmp/aquarium_model_abc123.py",
  "dataset": "mnist",
  "epochs": 10,
  "batch_size": 32,
  "validation_split": 0.2
}
```

**Response:**
```json
{
  "status": "started",
  "process_id": 12345
}
```

#### POST /stop

Stop running execution.

**Request:**
```json
{
  "process_id": 12345
}
```

**Response:**
```json
{
  "status": "stopped"
}
```

#### GET /status

Get execution status.

**Response:**
```json
{
  "status": "running",
  "process_id": 12345,
  "elapsed_time": 125.5
}
```

#### GET /output

Get console output.

**Response:**
```json
{
  "output": [
    "[RUN] Starting execution...",
    "Epoch 1/10",
    "loss: 0.3245 - accuracy: 0.8921"
  ]
}
```

---

### 2.4 Example Endpoints

#### GET /examples/list

List available examples.

**Response:**
```json
{
  "examples": [
    {
      "path": "mnist_cnn.neural",
      "name": "MNIST CNN",
      "category": "computer_vision",
      "complexity": "beginner",
      "description": "Simple CNN for digit recognition"
    }
  ]
}
```

#### GET /examples/load

Load example content.

**Query Parameters:**
- `path` (string): Example file path

**Response:**
```json
{
  "code": "network MNISTNet {...}",
  "path": "mnist_cnn.neural",
  "name": "MNIST CNN"
}
```

---

### 2.5 Plugin Endpoints

#### GET /plugins/list

List all plugins.

**Response:**
```json
{
  "plugins": [
    {
      "id": "github-copilot-integration",
      "name": "GitHub Copilot Integration",
      "version": "1.0.0",
      "enabled": true,
      "capabilities": ["code_completion"]
    }
  ]
}
```

#### POST /plugins/enable

Enable plugin.

**Request:**
```json
{
  "plugin_id": "github-copilot-integration"
}
```

**Response:**
```json
{
  "status": "enabled"
}
```

#### POST /plugins/install

Install plugin from npm/PyPI.

**Request:**
```json
{
  "source": "npm",
  "plugin_name": "@neural/my-plugin",
  "version": "1.0.0"
}
```

**Response:**
```json
{
  "status": "installed",
  "plugin_id": "my-plugin"
}
```

---

## 3. Plugin API

### 3.1 Plugin Base Class

**Import:**
```python
from neural.aquarium.src.plugins.plugin_base import PluginBase
```

#### Class: PluginBase

Base class for all plugins.

```python
class MyPlugin(PluginBase):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.name = metadata.id
        self.version = metadata.version
    
    def initialize(self) -> None:
        """Initialize plugin resources"""
        pass
    
    def activate(self) -> None:
        """Activate plugin functionality"""
        self._enabled = True
    
    def deactivate(self) -> None:
        """Deactivate plugin"""
        self._enabled = False
    
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
```

### 3.2 Plugin Types

#### PanelPlugin

Create custom UI panels.

```python
from neural.aquarium.src.plugins.plugin_base import PanelPlugin

class MyPanelPlugin(PanelPlugin):
    def get_panel_component(self) -> str:
        """Return component name"""
        return "MyPanelComponent"
    
    def get_panel_config(self) -> dict:
        """Return panel configuration"""
        return {
            'title': 'My Panel',
            'position': 'right',
            'width': 400,
            'height': 600,
            'collapsible': True
        }
```

#### ThemePlugin

Create custom color themes.

```python
from neural.aquarium.src.plugins.plugin_base import ThemePlugin

class MyThemePlugin(ThemePlugin):
    def get_theme_colors(self) -> dict:
        """Return theme color scheme"""
        return {
            'primary': '#61dafb',
            'background': '#1e1e1e',
            'text': '#ffffff',
            'border': '#333333'
        }
    
    def get_theme_css(self) -> str:
        """Return additional CSS"""
        return """
        .custom-class {
            background: var(--primary);
        }
        """
```

#### CommandPlugin

Register custom commands.

```python
from neural.aquarium.src.plugins.plugin_base import CommandPlugin

class MyCommandPlugin(CommandPlugin):
    def get_commands(self) -> List[dict]:
        """Return list of commands"""
        return [
            {
                'id': 'my-command',
                'name': 'My Command',
                'description': 'Does something useful',
                'shortcut': 'Ctrl+Alt+M',
                'handler': self.execute_command
            }
        ]
    
    def execute_command(self, args: dict) -> Any:
        """Execute command"""
        print(f"Executing with args: {args}")
        return {'success': True}
```

### 3.3 Plugin Manifest

**plugin.json:**
```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "email": "your.email@example.com",
  "description": "Plugin description",
  "homepage": "https://github.com/yourname/my-plugin",
  "license": "MIT",
  "capabilities": ["panel", "theme", "command"],
  "dependencies": {
    "neural-dsl": ">=0.3.0"
  },
  "keywords": ["neural", "aquarium", "plugin"],
  "min_aquarium_version": "0.3.0",
  "configuration": {
    "api_key": {
      "type": "string",
      "description": "API key for service",
      "required": true,
      "secret": true
    },
    "enabled": {
      "type": "boolean",
      "default": true
    }
  }
}
```

---

## 4. Component API

### 4.1 Parser API

**Import:**
```python
from neural.parser import parse
```

#### parse()

Parse Neural DSL code to AST.

```python
parse(dsl_code: str) -> AST
```

**Parameters:**
- `dsl_code` (str): Neural DSL source code

**Returns:** AST object

**Example:**
```python
from neural.parser import parse

dsl_code = """
network MyModel {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        Flatten()
        Dense(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
"""

ast = parse(dsl_code)
print(f"Network: {ast.network_name}")
print(f"Input: {ast.input_shape}")
print(f"Layers: {len(ast.layers)}")
```

### 4.2 Code Generation API

**Import:**
```python
from neural.code_generation import TensorFlowGenerator, PyTorchGenerator, ONNXGenerator
```

#### TensorFlowGenerator

```python
generator = TensorFlowGenerator()
code = generator.generate(ast)
```

#### PyTorchGenerator

```python
generator = PyTorchGenerator()
code = generator.generate(ast)
```

#### ONNXGenerator

```python
generator = ONNXGenerator()
code = generator.generate(ast)
```

**Example:**
```python
from neural.parser import parse
from neural.code_generation import TensorFlowGenerator

dsl_code = "network MyModel {...}"
ast = parse(dsl_code)

generator = TensorFlowGenerator()
tf_code = generator.generate(ast)

print(tf_code)
```

### 4.3 Shape Propagation API

**Import:**
```python
from neural.shape_propagation import propagate_shapes
```

#### propagate_shapes()

Calculate output shapes for all layers.

```python
propagate_shapes(ast: AST) -> dict
```

**Returns:** dict - Layer names to output shapes

**Example:**
```python
from neural.parser import parse
from neural.shape_propagation import propagate_shapes

ast = parse(dsl_code)
shapes = propagate_shapes(ast)

for layer_name, shape in shapes.items():
    print(f"{layer_name}: {shape}")
```

---

## 5. TypeScript API

### 5.1 PluginService

**Import:**
```typescript
import { pluginService } from './services/PluginService';
```

#### Methods

##### listPlugins()

```typescript
async listPlugins(): Promise<Plugin[]>
```

**Example:**
```typescript
const plugins = await pluginService.listPlugins();
plugins.forEach(plugin => {
    console.log(`${plugin.name} v${plugin.version}`);
});
```

##### enablePlugin()

```typescript
async enablePlugin(pluginId: string): Promise<boolean>
```

##### getThemes()

```typescript
async getThemes(): Promise<Theme[]>
```

##### executeCommand()

```typescript
async executeCommand(command: string, args: any): Promise<any>
```

### 5.2 Type Definitions

**types/plugins.ts:**

```typescript
export interface Plugin {
    id: string;
    name: string;
    version: string;
    author: string;
    description: string;
    enabled: boolean;
    capabilities: string[];
}

export interface Theme {
    id: string;
    name: string;
    colors: {
        primary: string;
        background: string;
        text: string;
        border: string;
    };
}

export interface Command {
    id: string;
    name: string;
    description: string;
    shortcut?: string;
}
```

---

## 6. CLI API

### 6.1 Aquarium Commands

#### Launch

```bash
python -m neural.aquarium.aquarium [OPTIONS]
```

**Options:**
- `--port PORT`: Server port (default: 8052)
- `--host HOST`: Server host (default: localhost)
- `--debug`: Enable debug mode
- `--no-browser`: Don't open browser

**Examples:**
```bash
# Default launch
python -m neural.aquarium.aquarium

# Custom port
python -m neural.aquarium.aquarium --port 8053

# Debug mode
python -m neural.aquarium.aquarium --debug

# Headless mode
python -m neural.aquarium.aquarium --no-browser
```

#### Compile (Headless)

```bash
python -m neural.aquarium.aquarium --headless --compile MODEL.neural
```

**Options:**
- `--headless`: Run without UI
- `--compile FILE`: Compile DSL file
- `--backend BACKEND`: Target backend
- `--output FILE`: Output file

**Example:**
```bash
python -m neural.aquarium.aquarium --headless \
    --compile mnist.neural \
    --backend tensorflow \
    --output mnist_tf.py
```

---

## Error Codes

### Compilation Errors

| Code | Name | Description |
|------|------|-------------|
| C001 | ParseError | Invalid DSL syntax |
| C002 | BackendNotFound | Backend not available |
| C003 | CodeGenError | Code generation failed |
| C004 | ShapeError | Shape propagation failed |

### Execution Errors

| Code | Name | Description |
|------|------|-------------|
| E001 | ScriptNotFound | Compiled script missing |
| E002 | DatasetError | Dataset loading failed |
| E003 | RuntimeError | Training error |
| E004 | ResourceError | Out of memory/resources |

### Plugin Errors

| Code | Name | Description |
|------|------|-------------|
| P001 | PluginNotFound | Plugin doesn't exist |
| P002 | LoadError | Plugin loading failed |
| P003 | DependencyError | Missing dependencies |
| P004 | VersionError | Incompatible version |

---

## Rate Limits

**REST API:**
- 100 requests per minute per IP
- 1000 requests per hour per IP

**WebSocket:**
- 10 connections per IP
- 1000 messages per minute per connection

---

## Versioning

**API Version**: 1.0.0

**Version Format**: MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

**Deprecation Policy:**
- Deprecated features marked 6 months before removal
- Old API versions supported for 1 year

---

## Support

**Issues**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)  
**Discussions**: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)  
**Email**: Lemniscate_zero@proton.me

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**License**: MIT
