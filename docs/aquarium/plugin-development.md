# Aquarium IDE Plugin Development Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Plugin Architecture](#plugin-architecture)
3. [Getting Started](#getting-started)
4. [Plugin Types](#plugin-types)
5. [API Reference](#api-reference)
6. [Example Plugins](#example-plugins)
7. [Testing](#testing)
8. [Distribution](#distribution)
9. [Best Practices](#best-practices)

## Introduction

Aquarium IDE supports a flexible plugin system that allows developers to extend functionality without modifying core code. Plugins can add new backends, datasets, UI components, exporters, and more.

### What You Can Build

- **Backend Plugins**: Support for new ML frameworks
- **Dataset Plugins**: Custom dataset loaders
- **UI Plugins**: New panels and visualizations
- **Exporter Plugins**: New export formats
- **Debugger Plugins**: Custom debugging tools
- **Theme Plugins**: Custom color schemes and layouts
- **Language Plugins**: DSL syntax extensions

### Prerequisites

- Python 3.8+
- Neural DSL installed
- Basic understanding of Dash framework
- Familiarity with Python packaging

## Plugin Architecture

### Plugin Structure

```
my_aquarium_plugin/
├── __init__.py              # Plugin entry point
├── plugin.yaml              # Plugin metadata
├── backend.py               # Backend implementation (optional)
├── dataset.py               # Dataset loader (optional)
├── ui_component.py          # UI component (optional)
├── exporter.py              # Exporter (optional)
├── tests/                   # Unit tests
│   ├── test_backend.py
│   └── test_dataset.py
├── examples/                # Example usage
│   └── example.py
├── README.md                # Documentation
└── setup.py                 # Installation script
```

### Plugin Lifecycle

```
1. Discovery    → Aquarium scans ~/.neural/plugins/
2. Loading      → Import plugin modules
3. Registration → Register plugin components
4. Initialization → Initialize plugin resources
5. Activation   → Enable plugin features
6. Usage        → Plugin functionality available
7. Deactivation → Disable plugin (optional)
8. Cleanup      → Release resources
```

## Getting Started

### Step 1: Create Plugin Template

```bash
# Create plugin directory
mkdir my_aquarium_plugin
cd my_aquarium_plugin

# Create basic structure
touch __init__.py plugin.yaml README.md setup.py
mkdir tests examples
```

### Step 2: Define Plugin Metadata

Create `plugin.yaml`:

```yaml
name: my-aquarium-plugin
display_name: My Aquarium Plugin
version: 1.0.0
author: Your Name
email: your.email@example.com
description: A sample Aquarium IDE plugin
homepage: https://github.com/yourname/my-aquarium-plugin
license: MIT

# Plugin capabilities
capabilities:
  - backend          # Provides ML backend
  - dataset          # Provides dataset loader
  - ui_component     # Provides UI component
  - exporter         # Provides export functionality

# Dependencies
requires:
  neural-dsl: ">=0.3.0"
  dash: ">=2.0.0"

# Optional dependencies
extras_require:
  tensorflow: ["tensorflow>=2.13.0"]
  pytorch: ["torch>=2.0.0"]

# Configuration
config:
  enabled: true
  load_on_startup: true
  priority: 100      # Higher = loaded first
```

### Step 3: Implement Plugin Entry Point

Create `__init__.py`:

```python
"""My Aquarium Plugin - Example plugin for Aquarium IDE"""

from typing import Dict, Any
from neural.aquarium.src.components.plugin_base import PluginBase


class MyPlugin(PluginBase):
    """Main plugin class"""
    
    def __init__(self):
        super().__init__()
        self.name = "my-aquarium-plugin"
        self.version = "1.0.0"
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration"""
        print(f"Initializing {self.name} v{self.version}")
        self.config = config
        
    def activate(self) -> None:
        """Activate plugin functionality"""
        print(f"Activating {self.name}")
        # Register components, backends, datasets, etc.
        
    def deactivate(self) -> None:
        """Deactivate plugin"""
        print(f"Deactivating {self.name}")
        # Cleanup resources
        
    def get_info(self) -> Dict[str, Any]:
        """Return plugin information"""
        return {
            "name": self.name,
            "version": self.version,
            "author": "Your Name",
            "description": "A sample Aquarium plugin",
            "capabilities": ["backend", "dataset"]
        }


# Plugin entry point
def get_plugin():
    """Factory function to create plugin instance"""
    return MyPlugin()
```

### Step 4: Install Plugin

```bash
# Development installation
pip install -e .

# Or copy to plugins directory
cp -r . ~/.neural/plugins/my-aquarium-plugin/
```

### Step 5: Test Plugin

```python
# Test in Python
from my_aquarium_plugin import get_plugin

plugin = get_plugin()
plugin.initialize({})
plugin.activate()
print(plugin.get_info())
```

## Plugin Types

### Backend Plugin

Add support for a new ML framework:

```python
# backend.py
from neural.aquarium.src.components.backend_base import BackendBase
from typing import Dict, Any, Optional


class MyBackend(BackendBase):
    """Custom ML backend implementation"""
    
    def __init__(self):
        super().__init__()
        self.name = "myframework"
        self.display_name = "MyFramework"
        
    def compile_model(self, dsl_code: str, output_path: str) -> str:
        """Compile DSL to MyFramework code"""
        # Parse DSL
        model_data = self.parse_dsl(dsl_code)
        
        # Generate code
        code = self._generate_code(model_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(code)
            
        return output_path
        
    def _generate_code(self, model_data: Dict[str, Any]) -> str:
        """Generate MyFramework code from model data"""
        code_lines = []
        
        # Imports
        code_lines.append("import myframework as mf")
        code_lines.append("")
        
        # Model definition
        code_lines.append("def create_model():")
        code_lines.append("    model = mf.Model()")
        
        # Layers
        for layer in model_data['layers']:
            layer_code = self._generate_layer(layer)
            code_lines.append(f"    {layer_code}")
            
        code_lines.append("    return model")
        
        return "\n".join(code_lines)
        
    def _generate_layer(self, layer: Dict[str, Any]) -> str:
        """Generate code for a single layer"""
        layer_type = layer['type']
        params = layer.get('params', {})
        
        if layer_type == 'Dense':
            units = params.get('units', 128)
            activation = params.get('activation', 'relu')
            return f"model.add_dense({units}, activation='{activation}')"
            
        elif layer_type == 'Conv2D':
            filters = params.get('filters', 32)
            kernel = params.get('kernel_size', (3, 3))
            return f"model.add_conv2d({filters}, kernel_size={kernel})"
            
        # Add more layer types...
        
        return f"# Unsupported layer: {layer_type}"
        
    def run_training(self, script_path: str, dataset: str, 
                    epochs: int, batch_size: int) -> None:
        """Execute training script"""
        import subprocess
        subprocess.run(['python', script_path], check=True)


# Register backend
def register_backend():
    """Register MyBackend with Aquarium"""
    from neural.aquarium.src.components.registry import BackendRegistry
    BackendRegistry.register('myframework', MyBackend)
```

### Dataset Plugin

Add support for custom datasets:

```python
# dataset.py
from neural.aquarium.src.components.dataset_base import DatasetBase
from typing import Tuple, Any
import numpy as np


class MyDataset(DatasetBase):
    """Custom dataset loader"""
    
    def __init__(self):
        super().__init__()
        self.name = "mydataset"
        self.display_name = "My Custom Dataset"
        
    def load_data(self) -> Tuple[Any, Any, Any, Any]:
        """Load and return train/test data"""
        # Load from custom source
        X_train, y_train = self._load_train_data()
        X_test, y_test = self._load_test_data()
        
        return X_train, y_train, X_test, y_test
        
    def _load_train_data(self):
        """Load training data"""
        # Implement custom loading logic
        X = np.load('data/train_X.npy')
        y = np.load('data/train_y.npy')
        return X, y
        
    def _load_test_data(self):
        """Load test data"""
        X = np.load('data/test_X.npy')
        y = np.load('data/test_y.npy')
        return X, y
        
    def get_info(self) -> dict:
        """Return dataset information"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "shape": (28, 28, 1),
            "classes": 10,
            "train_samples": 50000,
            "test_samples": 10000,
            "description": "My custom dataset"
        }
        
    def preprocess(self, X, y):
        """Preprocess data"""
        # Normalize
        X = X.astype('float32') / 255.0
        
        # One-hot encode labels
        from tensorflow.keras.utils import to_categorical
        y = to_categorical(y, 10)
        
        return X, y


# Register dataset
def register_dataset():
    """Register MyDataset with Aquarium"""
    from neural.aquarium.src.components.registry import DatasetRegistry
    DatasetRegistry.register('mydataset', MyDataset)
```

### UI Component Plugin

Add new UI panels or features:

```python
# ui_component.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State


class MyUIComponent:
    """Custom UI component for Aquarium"""
    
    def __init__(self, app: dash.Dash):
        self.app = app
        self.component_id = "my-ui-component"
        self._register_callbacks()
        
    def create_layout(self):
        """Create component layout"""
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-star me-2"),
                "My Custom Feature"
            ]),
            dbc.CardBody([
                html.P("This is a custom UI component"),
                dbc.Button(
                    "Click Me",
                    id=f"{self.component_id}-btn",
                    color="primary"
                ),
                html.Div(id=f"{self.component_id}-output")
            ])
        ])
        
    def _register_callbacks(self):
        """Register Dash callbacks"""
        @self.app.callback(
            Output(f"{self.component_id}-output", "children"),
            Input(f"{self.component_id}-btn", "n_clicks")
        )
        def handle_click(n_clicks):
            if not n_clicks:
                return ""
            return f"Button clicked {n_clicks} times!"


# Register UI component
def register_ui_component(app):
    """Register MyUIComponent with Aquarium"""
    from neural.aquarium.src.components.registry import UIRegistry
    component = MyUIComponent(app)
    UIRegistry.register('my-feature', component)
    return component
```

### Exporter Plugin

Add new export formats:

```python
# exporter.py
from neural.aquarium.src.components.exporter_base import ExporterBase
from typing import Dict, Any
import json


class MyExporter(ExporterBase):
    """Custom model exporter"""
    
    def __init__(self):
        super().__init__()
        self.name = "myformat"
        self.display_name = "My Format"
        self.extension = ".myformat"
        
    def export(self, model_data: Dict[str, Any], 
              output_path: str) -> str:
        """Export model to custom format"""
        # Convert model data to custom format
        custom_format = self._convert_to_custom_format(model_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(custom_format, f, indent=2)
            
        return output_path
        
    def _convert_to_custom_format(self, model_data: Dict[str, Any]) -> Dict:
        """Convert Neural DSL model to custom format"""
        custom = {
            "version": "1.0",
            "architecture": {
                "input": model_data['input'],
                "layers": []
            }
        }
        
        # Convert layers
        for layer in model_data['layers']:
            custom_layer = {
                "type": layer['type'],
                "config": layer.get('params', {})
            }
            custom["architecture"]["layers"].append(custom_layer)
            
        return custom


# Register exporter
def register_exporter():
    """Register MyExporter with Aquarium"""
    from neural.aquarium.src.components.registry import ExporterRegistry
    ExporterRegistry.register('myformat', MyExporter)
```

## API Reference

### Plugin Base Class

```python
class PluginBase:
    """Base class for all plugins"""
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration"""
        pass
        
    def activate(self) -> None:
        """Activate plugin functionality"""
        pass
        
    def deactivate(self) -> None:
        """Deactivate plugin"""
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Return plugin information"""
        pass
```

### Backend Base Class

```python
class BackendBase:
    """Base class for backend plugins"""
    
    def compile_model(self, dsl_code: str, output_path: str) -> str:
        """Compile DSL to backend code"""
        pass
        
    def run_training(self, script_path: str, **kwargs) -> None:
        """Execute training script"""
        pass
```

### Dataset Base Class

```python
class DatasetBase:
    """Base class for dataset plugins"""
    
    def load_data(self) -> Tuple[Any, Any, Any, Any]:
        """Load and return train/test data"""
        pass
        
    def preprocess(self, X, y) -> Tuple[Any, Any]:
        """Preprocess data"""
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Return dataset information"""
        pass
```

### Registry API

```python
# Register backend
from neural.aquarium.src.components.registry import BackendRegistry
BackendRegistry.register('backend_name', BackendClass)

# Register dataset
from neural.aquarium.src.components.registry import DatasetRegistry
DatasetRegistry.register('dataset_name', DatasetClass)

# Register UI component
from neural.aquarium.src.components.registry import UIRegistry
UIRegistry.register('component_name', ComponentInstance)

# Register exporter
from neural.aquarium.src.components.registry import ExporterRegistry
ExporterRegistry.register('format_name', ExporterClass)
```

## Example Plugins

### Example 1: JAX Backend

```python
# jax_backend.py
from neural.aquarium.src.components.backend_base import BackendBase


class JAXBackend(BackendBase):
    """JAX backend implementation"""
    
    def compile_model(self, dsl_code: str, output_path: str) -> str:
        model_data = self.parse_dsl(dsl_code)
        
        code = f"""
import jax
import jax.numpy as jnp
from flax import linen as nn

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
"""
        for layer in model_data['layers']:
            code += f"        x = {self._generate_jax_layer(layer)}\n"
            
        code += "        return x\n"
        
        with open(output_path, 'w') as f:
            f.write(code)
            
        return output_path
```

### Example 2: HuggingFace Dataset

```python
# huggingface_dataset.py
from neural.aquarium.src.components.dataset_base import DatasetBase
from datasets import load_dataset


class HuggingFaceDataset(DatasetBase):
    """HuggingFace dataset loader"""
    
    def __init__(self, dataset_name: str):
        super().__init__()
        self.dataset_name = dataset_name
        
    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        
        X_train = dataset['train']['image']
        y_train = dataset['train']['label']
        X_test = dataset['test']['image']
        y_test = dataset['test']['label']
        
        return X_train, y_train, X_test, y_test
```

### Example 3: Model Comparison Panel

```python
# comparison_panel.py
import dash_bootstrap_components as dbc
from dash import html, dcc


class ComparisonPanel:
    """Compare multiple models side-by-side"""
    
    def create_layout(self):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model A"),
                    dbc.CardBody([
                        html.P("Accuracy: 95%"),
                        dcc.Graph(id="model-a-metrics")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model B"),
                    dbc.CardBody([
                        html.P("Accuracy: 92%"),
                        dcc.Graph(id="model-b-metrics")
                    ])
                ])
            ], width=6)
        ])
```

## Testing

### Unit Tests

```python
# tests/test_backend.py
import unittest
from my_aquarium_plugin import MyBackend


class TestMyBackend(unittest.TestCase):
    
    def setUp(self):
        self.backend = MyBackend()
        
    def test_compilation(self):
        dsl_code = """
        network SimpleModel {
            input: (None, 784)
            layers:
                Dense(units=128, activation=relu)
                Output(units=10, activation=softmax)
            loss: categorical_crossentropy
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        output_path = self.backend.compile_model(dsl_code, "/tmp/test.py")
        self.assertTrue(os.path.exists(output_path))
        
    def test_layer_generation(self):
        layer = {"type": "Dense", "params": {"units": 128}}
        code = self.backend._generate_layer(layer)
        self.assertIn("128", code)
```

### Integration Tests

```python
# tests/test_integration.py
def test_plugin_loading():
    """Test that plugin loads correctly"""
    from my_aquarium_plugin import get_plugin
    
    plugin = get_plugin()
    assert plugin is not None
    assert plugin.name == "my-aquarium-plugin"
    
def test_plugin_activation():
    """Test plugin activation"""
    from my_aquarium_plugin import get_plugin
    
    plugin = get_plugin()
    plugin.initialize({})
    plugin.activate()
    
    # Verify plugin is active
    # Check registered components
```

## Distribution

### PyPI Package

Create `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="my-aquarium-plugin",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A sample Aquarium IDE plugin",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourname/my-aquarium-plugin",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "neural-dsl>=0.3.0",
        "dash>=2.0.0",
    ],
    entry_points={
        "neural.aquarium.plugins": [
            "my-plugin = my_aquarium_plugin:get_plugin",
        ],
    },
)
```

Publish to PyPI:

```bash
# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

### GitHub Release

```bash
# Tag release
git tag v1.0.0
git push origin v1.0.0

# Create GitHub release
gh release create v1.0.0 --title "v1.0.0" --notes "Initial release"
```

## Best Practices

### 1. Follow Naming Conventions

- Plugin names: `aquarium-plugin-name`
- Python modules: `aquarium_plugin_name`
- Classes: `PluginNameBackend`, `PluginNameDataset`

### 2. Handle Errors Gracefully

```python
def compile_model(self, dsl_code: str, output_path: str) -> str:
    try:
        model_data = self.parse_dsl(dsl_code)
        code = self._generate_code(model_data)
        with open(output_path, 'w') as f:
            f.write(code)
        return output_path
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        raise PluginError(f"Failed to compile model: {e}")
```

### 3. Document Thoroughly

- Add docstrings to all classes and methods
- Include usage examples
- Provide README with installation instructions
- Document configuration options

### 4. Version Compatibility

- Specify minimum Neural DSL version
- Test with multiple versions
- Use semantic versioning
- Maintain changelog

### 5. Performance

- Lazy load heavy dependencies
- Cache compiled code
- Use async operations where possible
- Profile performance

### 6. Security

- Validate all inputs
- Sanitize file paths
- Don't execute arbitrary code
- Follow security best practices

## Resources

### Documentation
- [Aquarium Architecture](architecture.md)
- [API Reference](../../neural/aquarium/backend/README.md)
- [Neural DSL Docs](../../docs/dsl.md)

### Examples
- [Official Plugins](https://github.com/Lemniscate-world/Neural-Plugins)
- [Community Plugins](https://github.com/topics/aquarium-plugin)

### Support
- [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
- [Discord](https://discord.gg/KFku4KvS)
- Email: Lemniscate_zero@proton.me

---

**Version**: 1.0  
**Last Updated**: December 2024  
**License**: MIT
