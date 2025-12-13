# Neural Aquarium Plugin System

## Overview

The Neural Aquarium plugin system provides a comprehensive, extensible architecture for adding custom functionality to the IDE without modifying core code. It supports plugins written in Python and JavaScript/TypeScript, with automatic discovery from npm and PyPI.

## Architecture

```
neural/aquarium/
├── src/
│   ├── plugins/                      # Core plugin system (Python)
│   │   ├── __init__.py
│   │   ├── plugin_base.py           # Base classes and interfaces
│   │   ├── plugin_manager.py        # Central plugin coordinator
│   │   ├── plugin_loader.py         # Plugin discovery and loading
│   │   ├── plugin_registry.py       # Plugin registration and state
│   │   ├── examples/                # Example plugins
│   │   │   ├── copilot_plugin/      # GitHub Copilot integration
│   │   │   ├── viz_plugin/          # Custom visualizations
│   │   │   └── dark_ocean_theme/    # Dark Ocean theme
│   │   ├── README.md                # Plugin development guide
│   │   ├── PLUGIN_API.md            # Complete API reference
│   │   └── QUICKSTART.md            # Quick start guide
│   ├── components/
│   │   └── marketplace/             # React UI components
│   │       ├── PluginMarketplace.tsx
│   │       ├── PluginMarketplace.css
│   │       └── index.ts
│   ├── services/
│   │   └── PluginService.ts         # Frontend plugin API client
│   └── types/
│       └── plugins.ts               # TypeScript type definitions
└── backend/
    ├── plugin_api.py                # Flask API endpoints
    └── api_updated.py               # Integrated API server
```

## Features

### Plugin Types

1. **Panel Plugins** - Custom UI panels with configurable layout
2. **Theme Plugins** - Complete color schemes and styling
3. **Command Plugins** - Custom commands with keybindings
4. **Visualization Plugins** - Custom visualization types
5. **Integration Plugins** - External service integrations
6. **Language Support Plugins** - Additional DSL language features
7. **Code Completion Plugins** - AI-powered completions
8. **Linter Plugins** - Custom code linting
9. **Formatter Plugins** - Code formatting rules

### Key Capabilities

- **Automatic Discovery**: Plugins are automatically discovered from:
  - Local plugin directory (`~/.neural_aquarium/plugins`)
  - npm packages with `neuralAquariumPlugin` field
  - PyPI packages with plugin manifest
  
- **Dependency Management**: Automatic installation of:
  - Python dependencies via pip
  - npm dependencies via npm
  - Plugin-to-plugin dependencies

- **Hot Reloading**: Enable/disable plugins without restart

- **Configuration System**: JSON Schema-based configuration with UI generation

- **Hook System**: Event-driven architecture for plugin communication

- **Marketplace UI**: Beautiful, searchable plugin browser with ratings and reviews

## Installation

The plugin system is built into Neural Aquarium. To access it:

1. **Via UI**: Navigate to Menu → Plugins → Marketplace
2. **Via Code**: Import and use the PluginManager

## Usage

### Python API

```python
from neural.aquarium.src.plugins import PluginManager

# Get singleton instance
manager = PluginManager()

# List all plugins
plugins = manager.list_plugins()

# Enable a plugin
manager.enable_plugin('github-copilot-integration')

# Get panels from plugins
panels = manager.get_panels()

# Execute a command
result = manager.execute_command('my-command', {'arg': 'value'})

# Search plugins
results = manager.search_plugins('visualization')
```

### TypeScript/React API

```typescript
import { pluginService } from './services/PluginService';

// List plugins
const plugins = await pluginService.listPlugins();

// Enable plugin
await pluginService.enablePlugin('github-copilot-integration');

// Get themes
const themes = await pluginService.getThemes();

// Execute command
const result = await pluginService.executeCommand('my-command', { arg: 'value' });
```

### REST API

```bash
# List plugins
GET /api/plugins/list

# Enable plugin
POST /api/plugins/enable
{
  "plugin_id": "github-copilot-integration"
}

# Search plugins
GET /api/plugins/search?q=visualization

# Install from npm
POST /api/plugins/install
{
  "source": "npm",
  "plugin_name": "@neural/my-plugin",
  "version": "1.0.0"
}
```

## Creating Plugins

### Minimal Plugin Structure

```
my-plugin/
├── plugin.json          # Manifest
├── main.py             # Main code
└── README.md           # Documentation
```

### plugin.json

```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "Plugin description",
  "capabilities": ["panel"],
  "license": "MIT",
  "min_aquarium_version": "0.3.0"
}
```

### main.py

```python
from plugin_base import PanelPlugin, PluginMetadata

class MyPlugin(PanelPlugin):
    def initialize(self):
        print("Initializing...")
    
    def activate(self):
        self._enabled = True
    
    def deactivate(self):
        self._enabled = False
    
    def get_panel_component(self):
        return "MyPanel"
    
    def get_panel_config(self):
        return {
            'title': 'My Panel',
            'position': 'right',
            'width': 400
        }

def create_plugin(metadata):
    return MyPlugin(metadata)
```

## Example Plugins

### 1. GitHub Copilot Integration

AI-powered code completion for Neural DSL:
- Real-time suggestions
- Context-aware completions
- Layer recommendations

Location: `neural/aquarium/src/plugins/examples/copilot_plugin/`

### 2. Custom Visualizations

Advanced visualization types:
- 3D architecture models
- Interactive flow diagrams
- Heatmaps
- Sankey diagrams
- Circular graphs

Location: `neural/aquarium/src/plugins/examples/viz_plugin/`

### 3. Dark Ocean Theme

Beautiful dark theme with ocean colors:
- Deep blue color scheme
- Excellent contrast
- Custom editor theme
- Reduced eye strain

Location: `neural/aquarium/src/plugins/examples/dark_ocean_theme/`

## Marketplace UI

The plugin marketplace provides:
- **Search**: Full-text search across names, descriptions, keywords
- **Filters**: Filter by capability type
- **Sorting**: By rating, downloads, or name
- **Ratings**: 5-star rating system with reviews
- **Download Stats**: Track plugin popularity
- **One-Click Install**: Install directly from UI
- **Plugin Details**: Modal with complete plugin information

## API Endpoints

### Plugin Management
- `GET /api/plugins/list` - List all plugins
- `GET /api/plugins/enabled` - List enabled plugins
- `GET /api/plugins/details/<id>` - Get plugin details
- `POST /api/plugins/enable` - Enable plugin
- `POST /api/plugins/disable` - Disable plugin
- `POST /api/plugins/install` - Install from npm/PyPI
- `POST /api/plugins/uninstall` - Uninstall plugin
- `GET /api/plugins/search?q=<query>` - Search plugins

### Plugin Capabilities
- `GET /api/plugins/panels` - Get all panel plugins
- `GET /api/plugins/themes` - Get all theme plugins
- `GET /api/plugins/commands` - Get all commands
- `POST /api/plugins/command/execute` - Execute command

## Plugin Hooks

Plugins can register for events:

```python
def on_file_saved(file_path):
    print(f"File saved: {file_path}")

manager.register_hook('file_saved', on_file_saved)
```

Available hooks:
- `file_opened` - File opened
- `file_saved` - File saved
- `model_compiled` - Model compiled
- `model_executed` - Model executed
- `theme_changed` - Theme changed
- `plugin_enabled` - Plugin enabled
- `plugin_disabled` - Plugin disabled

## Publishing Plugins

### To npm

1. Add to package.json:
```json
{
  "name": "@yourorg/neural-aquarium-plugin",
  "neuralAquariumPlugin": {
    "displayName": "Plugin Name",
    "capabilities": ["panel"]
  }
}
```

2. Publish: `npm publish`

### To PyPI

1. Add plugin.json to package
2. Publish: `python setup.py sdist && twine upload dist/*`

## Documentation

- **README.md**: Complete development guide
- **PLUGIN_API.md**: Full API reference
- **QUICKSTART.md**: 5-minute getting started guide
- **Example plugins**: Working implementations

## Setup Scripts

Run these scripts to create the plugin system:

```bash
# Create all plugin system files at once
python neural/aquarium/create_complete_plugin_system.py

# Or run individually:
python neural/aquarium/setup_plugins.py          # Core system
python neural/aquarium/setup_plugins_part2.py    # Example plugins
python neural/aquarium/setup_plugins_part3.py    # Marketplace UI
python neural/aquarium/setup_plugins_part4.py    # Documentation
```

## Best Practices

1. **Follow conventions**: Use kebab-case for IDs
2. **Version properly**: Use semantic versioning
3. **Document thoroughly**: Include README with examples
4. **Handle errors**: Don't crash the host app
5. **Clean up**: Properly deactivate in deactivate()
6. **Test thoroughly**: Test enable/disable cycles
7. **Be performant**: Minimize initialization time
8. **Stay compatible**: Test with multiple versions

## Security

- API keys stored securely
- Sensitive configs masked in UI
- Plugin sandboxing (planned)
- Code signing (planned)

## Future Enhancements

- [ ] Plugin sandboxing for security
- [ ] Plugin marketplace backend service
- [ ] Automated testing framework
- [ ] Plugin templates generator
- [ ] Version compatibility checker
- [ ] Dependency conflict resolution
- [ ] Plugin performance profiling
- [ ] Community plugin ratings/reviews
- [ ] Official plugin certification
- [ ] Plugin update notifications

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE.md for details.
