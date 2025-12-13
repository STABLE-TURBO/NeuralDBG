#!/usr/bin/env python3
"""
Part 4: Documentation, integration, and final setup files
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
PLUGINS_DIR = BASE_DIR / "src" / "plugins"
EXAMPLES_DIR = PLUGINS_DIR / "examples"

files = {}

# ============================================================================
# PLUGIN SYSTEM DOCUMENTATION
# ============================================================================

files[PLUGINS_DIR / "README.md"] = '''# Neural Aquarium Plugin System

A comprehensive plugin system for extending Neural Aquarium with custom panels, themes, commands, visualizations, and integrations.

## Overview

The plugin system provides a flexible, extensible architecture for adding new functionality to Neural Aquarium without modifying the core codebase.

## Plugin Types

### Panel Plugins
Add custom UI panels to the Aquarium interface.

```python
from plugin_base import PanelPlugin, PluginMetadata, PluginCapability

class MyPanelPlugin(PanelPlugin):
    def get_panel_component(self) -> str:
        return "MyCustomPanel"
    
    def get_panel_config(self) -> dict:
        return {
            'title': 'My Panel',
            'position': 'right',
            'width': 400
        }
```

### Theme Plugins
Create custom color schemes and styling.

```python
from plugin_base import ThemePlugin

class MyTheme(ThemePlugin):
    def get_theme_definition(self) -> dict:
        return {
            'name': 'My Theme',
            'type': 'dark',
            'colors': {
                'primary': '#0066cc',
                'background': '#1a1a1a',
                # ... more colors
            }
        }
```

### Command Plugins
Add custom commands to the command palette.

```python
from plugin_base import CommandPlugin

class MyCommandPlugin(CommandPlugin):
    def get_commands(self) -> list:
        return [
            {
                'id': 'my-command',
                'name': 'My Command',
                'description': 'Does something cool',
                'keybinding': 'Ctrl+Shift+M'
            }
        ]
    
    def execute_command(self, command_id: str, args: dict):
        # Execute the command
        pass
```

### Visualization Plugins
Add custom visualization types.

```python
from plugin_base import VisualizationPlugin

class MyVizPlugin(VisualizationPlugin):
    def get_visualization_types(self) -> list:
        return ['3d-graph', 'heatmap']
    
    def render_visualization(self, vis_type: str, data):
        # Render the visualization
        return {'type': vis_type, 'data': data}
```

### Integration Plugins
Connect to external services.

```python
from plugin_base import IntegrationPlugin

class MyIntegration(IntegrationPlugin):
    def get_integration_name(self) -> str:
        return "My Service"
    
    def connect(self, credentials: dict) -> bool:
        # Connect to service
        return True
    
    def disconnect(self) -> None:
        # Disconnect from service
        pass
```

## Creating a Plugin

### 1. Plugin Structure

```
my-plugin/
├── plugin.json          # Manifest file
├── main.py             # Main plugin code
├── README.md           # Documentation
└── assets/             # Optional assets
    ├── icon.png
    └── styles.css
```

### 2. Plugin Manifest (plugin.json)

```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "A description of what your plugin does",
  "capabilities": ["panel", "command"],
  "homepage": "https://github.com/you/my-plugin",
  "repository": "https://github.com/you/my-plugin",
  "keywords": ["tag1", "tag2"],
  "license": "MIT",
  "dependencies": [],
  "python_dependencies": ["requests>=2.28.0"],
  "npm_dependencies": {
    "react": "^18.0.0"
  },
  "min_aquarium_version": "0.3.0",
  "icon": "🔌"
}
```

### 3. Main Plugin Code (main.py)

```python
from plugin_base import Plugin, PluginMetadata

class MyPlugin(Plugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
    
    def initialize(self) -> None:
        print(f"Initializing {self.metadata.name}...")
    
    def activate(self) -> None:
        print(f"Activating {self.metadata.name}...")
        self._enabled = True
    
    def deactivate(self) -> None:
        print(f"Deactivating {self.metadata.name}...")
        self._enabled = False

def create_plugin(metadata: PluginMetadata) -> MyPlugin:
    """Factory function required by plugin loader."""
    return MyPlugin(metadata)
```

## Installing Plugins

### From Local Directory

```python
from plugins.plugin_loader import PluginLoader

loader = PluginLoader()
plugin = loader.load_plugin('my-plugin')
```

### From npm

```bash
# In your plugin directory
npm install @neural-aquarium/my-plugin
```

Or via API:
```python
manager.install_plugin('npm', '@neural-aquarium/my-plugin', '1.0.0')
```

### From PyPI

```bash
pip install neural-aquarium-my-plugin
```

Or via API:
```python
manager.install_plugin('pypi', 'neural-aquarium-my-plugin', '1.0.0')
```

## Using the Plugin Manager

```python
from plugins.plugin_manager import PluginManager

# Get singleton instance
manager = PluginManager()

# List all plugins
plugins = manager.list_plugins()

# Enable a plugin
manager.enable_plugin('my-plugin')

# Disable a plugin
manager.disable_plugin('my-plugin')

# Get plugin metadata
metadata = manager.get_plugin_metadata('my-plugin')

# Search plugins
results = manager.search_plugins('visualization')

# Get panels from all plugins
panels = manager.get_panels()

# Get themes from all plugins
themes = manager.get_themes()

# Execute a command
result = manager.execute_command('my-command', {'arg': 'value'})
```

## Plugin Hooks

Plugins can register hooks to respond to events:

```python
def on_file_opened(file_path):
    print(f"File opened: {file_path}")

manager.register_hook('file_opened', on_file_opened)
```

Available hooks:
- `file_opened` - When a file is opened
- `file_saved` - When a file is saved
- `model_compiled` - When a model is compiled
- `model_executed` - When a model is executed
- `theme_changed` - When the theme changes
- `plugin_enabled` - When a plugin is enabled
- `plugin_disabled` - When a plugin is disabled

## API Endpoints

### GET /api/plugins/list
List all available plugins.

### GET /api/plugins/enabled
List enabled plugins.

### GET /api/plugins/details/<plugin_id>
Get detailed information about a plugin.

### POST /api/plugins/enable
Enable a plugin.

### POST /api/plugins/disable
Disable a plugin.

### POST /api/plugins/install
Install a plugin from npm or PyPI.

### POST /api/plugins/uninstall
Uninstall a plugin.

### GET /api/plugins/search?q=<query>
Search for plugins.

### GET /api/plugins/panels
Get all panel plugins.

### GET /api/plugins/themes
Get all theme plugins.

### GET /api/plugins/commands
Get all available commands.

### POST /api/plugins/command/execute
Execute a plugin command.

## Publishing Plugins

### To npm

1. Create package.json with neuralAquariumPlugin field:

```json
{
  "name": "@your-org/neural-aquarium-my-plugin",
  "version": "1.0.0",
  "neuralAquariumPlugin": {
    "displayName": "My Plugin",
    "capabilities": ["panel"],
    "minAquariumVersion": "0.3.0"
  }
}
```

2. Publish:
```bash
npm publish
```

### To PyPI

1. Create setup.py:

```python
from setuptools import setup

setup(
    name='neural-aquarium-my-plugin',
    version='1.0.0',
    # ... other metadata
)
```

2. Publish:
```bash
python setup.py sdist
twine upload dist/*
```

## Examples

See the `examples/` directory for complete working examples:

- **copilot_plugin**: GitHub Copilot integration
- **viz_plugin**: Custom visualizations
- **dark_ocean_theme**: Custom dark theme

## Best Practices

1. **Follow naming conventions**: Use kebab-case for plugin IDs
2. **Version your plugins**: Use semantic versioning
3. **Document thoroughly**: Include clear README files
4. **Handle errors gracefully**: Don't crash the host application
5. **Clean up resources**: Properly deactivate in deactivate()
6. **Test thoroughly**: Test enable/disable cycles
7. **Respect user settings**: Don't override without permission
8. **Be performant**: Minimize initialization time
9. **Stay compatible**: Test with multiple Aquarium versions
10. **Provide examples**: Include usage examples in README

## Troubleshooting

### Plugin not loading
- Check plugin.json is valid JSON
- Ensure main.py has create_plugin function
- Verify dependencies are installed

### Plugin crashes on enable
- Check initialize() and activate() methods
- Look for missing imports or dependencies
- Review error logs

### Plugin not appearing in marketplace
- Verify plugin.json has all required fields
- Check plugin is in correct directory
- Ensure manifest is properly formatted

## Contributing

To contribute to the plugin system:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

The Neural Aquarium plugin system is licensed under the MIT License.
'''

files[PLUGINS_DIR / "PLUGIN_API.md"] = '''# Plugin API Reference

Complete API reference for Neural Aquarium plugins.

## Base Classes

### Plugin

Base class for all plugins.

```python
class Plugin(ABC):
    def __init__(self, metadata: PluginMetadata)
    
    @abstractmethod
    def initialize(self) -> None
    
    @abstractmethod
    def activate(self) -> None
    
    @abstractmethod
    def deactivate(self) -> None
    
    def configure(self, config: Dict[str, Any]) -> None
    
    def get_config_schema(self) -> Dict[str, Any]
    
    @property
    def enabled(self) -> bool
    
    @property
    def initialized(self) -> bool
```

### PanelPlugin

Plugin that provides custom UI panels.

```python
class PanelPlugin(Plugin):
    @abstractmethod
    def get_panel_component(self) -> str
    
    @abstractmethod
    def get_panel_config(self) -> Dict[str, Any]
```

### ThemePlugin

Plugin that provides custom themes.

```python
class ThemePlugin(Plugin):
    @abstractmethod
    def get_theme_definition(self) -> Dict[str, Any]
    
    def get_editor_theme(self) -> Optional[Dict[str, Any]]
```

### CommandPlugin

Plugin that provides custom commands.

```python
class CommandPlugin(Plugin):
    @abstractmethod
    def get_commands(self) -> List[Dict[str, Any]]
    
    @abstractmethod
    def execute_command(self, command_id: str, args: Dict[str, Any]) -> Any
```

### VisualizationPlugin

Plugin that provides custom visualizations.

```python
class VisualizationPlugin(Plugin):
    @abstractmethod
    def get_visualization_types(self) -> List[str]
    
    @abstractmethod
    def render_visualization(self, vis_type: str, data: Any) -> Dict[str, Any]
```

### IntegrationPlugin

Plugin that provides external integrations.

```python
class IntegrationPlugin(Plugin):
    @abstractmethod
    def get_integration_name(self) -> str
    
    @abstractmethod
    def connect(self, credentials: Dict[str, Any]) -> bool
    
    @abstractmethod
    def disconnect(self) -> None
```

## Data Classes

### PluginMetadata

```python
@dataclass
class PluginMetadata:
    id: str
    name: str
    version: str
    author: str
    description: str
    capabilities: List[PluginCapability]
    homepage: Optional[str] = None
    repository: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    license: str = "MIT"
    dependencies: List[str] = field(default_factory=list)
    python_dependencies: List[str] = field(default_factory=list)
    npm_dependencies: Dict[str, str] = field(default_factory=dict)
    min_aquarium_version: str = "0.3.0"
    icon: Optional[str] = None
    rating: float = 0.0
    downloads: int = 0
    
    def to_dict(self) -> Dict[str, Any]
```

### PluginCapability

```python
class PluginCapability(Enum):
    PANEL = "panel"
    THEME = "theme"
    COMMAND = "command"
    VISUALIZATION = "visualization"
    INTEGRATION = "integration"
    LANGUAGE_SUPPORT = "language_support"
    CODE_COMPLETION = "code_completion"
    LINTER = "linter"
    FORMATTER = "formatter"
```

## Plugin Manager

### PluginManager

Central plugin manager singleton.

```python
class PluginManager:
    def __init__(self)
    
    def list_plugins(self) -> List[PluginMetadata]
    
    def list_enabled_plugins(self) -> List[PluginMetadata]
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]
    
    def get_plugin_metadata(self, plugin_id: str) -> Optional[PluginMetadata]
    
    def enable_plugin(self, plugin_id: str) -> bool
    
    def disable_plugin(self, plugin_id: str) -> bool
    
    def install_plugin(
        self,
        source: str,
        plugin_name: str,
        version: Optional[str] = None
    ) -> bool
    
    def uninstall_plugin(self, plugin_id: str) -> bool
    
    def get_panels(self) -> List[Dict[str, Any]]
    
    def get_themes(self) -> List[Dict[str, Any]]
    
    def get_commands(self) -> List[Dict[str, Any]]
    
    def execute_command(self, command_id: str, args: Dict[str, Any]) -> Any
    
    def search_plugins(self, query: str) -> List[PluginMetadata]
    
    def register_hook(self, hook_name: str, callback: Callable) -> None
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]
```

## Plugin Loader

### PluginLoader

Handles plugin discovery and loading.

```python
class PluginLoader:
    def __init__(self, plugin_dir: Optional[Path] = None)
    
    def discover_plugins(self) -> List[PluginMetadata]
    
    def load_plugin(self, plugin_id: str) -> Optional[Plugin]
    
    def install_from_npm(
        self,
        package_name: str,
        version: Optional[str] = None
    ) -> bool
    
    def install_from_pypi(
        self,
        package_name: str,
        version: Optional[str] = None
    ) -> bool
    
    def uninstall_plugin(self, plugin_id: str) -> bool
```

## Plugin Registry

### PluginRegistry

Central registry for all plugins.

```python
class PluginRegistry:
    def __init__(self, registry_path: Optional[Path] = None)
    
    def register(self, plugin: Plugin) -> None
    
    def unregister(self, plugin_id: str) -> None
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]
    
    def get_metadata(self, plugin_id: str) -> Optional[PluginMetadata]
    
    def list_plugins(self) -> List[str]
    
    def list_enabled_plugins(self) -> List[str]
    
    def list_by_capability(
        self,
        capability: PluginCapability
    ) -> List[Plugin]
    
    def enable_plugin(self, plugin_id: str) -> bool
    
    def disable_plugin(self, plugin_id: str) -> bool
    
    def search(self, query: str) -> List[PluginMetadata]
```

## Hook Registry

### PluginHookRegistry

Registry for plugin hooks.

```python
class PluginHookRegistry:
    def __init__(self)
    
    def register_hook(self, hook_name: str, callback: Callable) -> None
    
    def unregister_hook(self, hook_name: str, callback: Callable) -> None
    
    def execute_hook(
        self,
        hook_name: str,
        *args,
        **kwargs
    ) -> List[Any]
```

## Configuration Schemas

Plugin configuration schemas use JSON Schema format:

```python
{
    'property_name': {
        'type': 'string',  # or 'number', 'boolean', 'array', 'object'
        'description': 'Property description',
        'default': 'default_value',
        'required': True,
        'sensitive': False,  # If true, value is masked in UI
        'enum': ['option1', 'option2'],  # Optional list of valid values
        'minimum': 0,  # For numbers
        'maximum': 100,  # For numbers
        'pattern': '^[a-z]+$',  # For strings (regex)
    }
}
```

## Panel Configuration

Panel plugins return configuration:

```python
{
    'title': 'Panel Title',
    'position': 'left',  # 'left', 'right', 'bottom', 'top'
    'width': 400,  # Optional, pixels
    'height': 300,  # Optional, pixels
    'resizable': True,
    'closeable': True,
    'icon': '📊',
}
```

## Theme Definition

Theme plugins return theme definition:

```python
{
    'name': 'Theme Name',
    'type': 'dark',  # 'dark' or 'light'
    'colors': {
        'primary': '#0066cc',
        'secondary': '#6c757d',
        'background': '#1a1a1a',
        'surface': '#2a2a2a',
        'error': '#ff4444',
        'warning': '#ffaa00',
        'success': '#00cc66',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc',
        'border': '#444444',
    },
    'fonts': {
        'primary': 'Inter, sans-serif',
        'monospace': 'Fira Code, monospace',
    },
    'spacing': {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px',
    }
}
```

## Command Definition

Command plugins return command definitions:

```python
[
    {
        'id': 'unique-command-id',
        'name': 'Command Name',
        'description': 'What the command does',
        'category': 'Edit',  # Optional category
        'keybinding': 'Ctrl+Shift+C',  # Optional keybinding
        'icon': '⚡',  # Optional icon
        'when': 'editorFocus',  # Optional context condition
    }
]
```

## Visualization Data Format

Visualization plugins return render data:

```python
{
    'type': 'plotly',  # or '3d', 'canvas', etc.
    'component': 'ComponentName',
    'data': {
        # Component-specific data
    },
    'options': {
        # Component-specific options
    }
}
```
'''

files[PLUGINS_DIR / "QUICKSTART.md"] = '''# Plugin System Quick Start

Get started with Neural Aquarium plugins in 5 minutes.

## Installation

The plugin system is built into Neural Aquarium. No additional installation required!

## Using the Marketplace

1. Open Neural Aquarium
2. Navigate to the Plugin Marketplace (Menu → Plugins → Marketplace)
3. Browse available plugins
4. Click "Install" on any plugin you like
5. Enable the plugin to activate it

## Creating Your First Plugin

### 1. Create Directory Structure

```bash
mkdir my-first-plugin
cd my-first-plugin
```

### 2. Create plugin.json

```json
{
  "id": "my-first-plugin",
  "name": "My First Plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "My first Neural Aquarium plugin",
  "capabilities": ["command"],
  "license": "MIT",
  "min_aquarium_version": "0.3.0",
  "icon": "🔌"
}
```

### 3. Create main.py

```python
from plugin_base import CommandPlugin, PluginMetadata

class MyFirstPlugin(CommandPlugin):
    def __init__(self, metadata):
        super().__init__(metadata)
    
    def initialize(self):
        print("Plugin initialized!")
    
    def activate(self):
        print("Plugin activated!")
        self._enabled = True
    
    def deactivate(self):
        print("Plugin deactivated!")
        self._enabled = False
    
    def get_commands(self):
        return [{
            'id': 'my-first-command',
            'name': 'My First Command',
            'description': 'Says hello!',
        }]
    
    def execute_command(self, command_id, args):
        return "Hello from my first plugin!"

def create_plugin(metadata):
    return MyFirstPlugin(metadata)
```

### 4. Test Your Plugin

```python
from plugins.plugin_loader import PluginLoader
from plugins.plugin_manager import PluginManager

# Load plugin
loader = PluginLoader()
plugin = loader.load_plugin('my-first-plugin')

# Register and enable
manager = PluginManager()
manager.registry.register(plugin)
manager.enable_plugin('my-first-plugin')

# Execute command
result = manager.execute_command('my-first-command', {})
print(result)  # "Hello from my first plugin!"
```

## Next Steps

- Read the full [README.md](README.md)
- Check out [example plugins](examples/)
- Review the [API Reference](PLUGIN_API.md)
- Join the community forums

## Getting Help

- GitHub Issues: [github.com/neural-dsl/neural-dsl](https://github.com/neural-dsl/neural-dsl)
- Discord: [discord.gg/neural-dsl](https://discord.gg/neural-dsl)
- Documentation: [docs.neural-dsl.org](https://docs.neural-dsl.org)
'''

# ============================================================================
# INTEGRATION WITH BACKEND
# ============================================================================

files[BASE_DIR / "backend" / "api_updated.py"] = '''"""
Updated Flask API with plugin endpoints integrated
"""

from flask import Flask
from flask_cors import CORS
import logging

from api import app as base_app
from plugin_api import plugin_bp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.register_blueprint(plugin_bp)

for rule in base_app.url_map.iter_rules():
    if rule.endpoint != 'static':
        app.add_url_rule(
            rule.rule,
            endpoint=rule.endpoint,
            view_func=base_app.view_functions[rule.endpoint],
            methods=rule.methods
        )

@app.route('/health', methods=['GET'])
def health_check():
    from flask import jsonify
    return jsonify({
        'status': 'healthy',
        'service': 'neural-aquarium-api',
        'plugins_enabled': True
    }), 200


if __name__ == '__main__':
    logger.info("Starting Neural Aquarium API with plugin support...")
    logger.info("Plugin endpoints available at /api/plugins/*")
    app.run(host='0.0.0.0', port=5000, debug=True)
'''

# Write all files
for filepath, content in files.items():
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\\n✅ Documentation and integration files created successfully!")
print("\\n🎉 Plugin system implementation complete!")
print("\\n📚 Next steps:")
print("   1. Run setup_plugins.py to create core plugin files")
print("   2. Run setup_plugins_part2.py to create example plugins")
print("   3. Run setup_plugins_part3.py to create marketplace UI")
print("   4. Run this script (setup_plugins_part4.py) for docs")
print("\\n   Or run all at once:")
print("   python neural/aquarium/create_plugin_system.py")
