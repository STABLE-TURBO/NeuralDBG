# Neural Aquarium Plugin System - Complete Implementation

This directory contains a fully-implemented plugin system for Neural Aquarium that enables extending the IDE with custom panels, themes, commands, visualizations, and integrations.

## 🎯 What's Implemented

### Core Plugin System (Python)
- **Plugin Base Classes**: Abstract base classes for all plugin types
- **Plugin Manager**: Singleton manager coordinating all plugins
- **Plugin Loader**: Automatic discovery from local/npm/PyPI
- **Plugin Registry**: Central registration and state management
- **Hook System**: Event-driven plugin communication

### Plugin Types Supported
1. **Panel Plugins** - Custom UI panels
2. **Theme Plugins** - Color schemes and styling
3. **Command Plugins** - Custom commands with keybindings
4. **Visualization Plugins** - Custom visualization types
5. **Integration Plugins** - External service connectors
6. **Language Support** - Additional DSL features
7. **Code Completion** - AI-powered completions
8. **Linters** - Custom code linting
9. **Formatters** - Code formatting

### Frontend (React/TypeScript)
- **Marketplace UI**: Beautiful plugin browser with search, filters, ratings
- **Plugin Service**: TypeScript API client
- **Type Definitions**: Complete TypeScript types
- **Example Components**: React panel components

### Backend (Flask API)
- **REST API**: Complete plugin management endpoints
- **Installation**: Install from npm/PyPI
- **Configuration**: Plugin config management
- **Command Execution**: Execute plugin commands

### Example Plugins
1. **GitHub Copilot Integration** - AI code completion
2. **Custom Visualizations** - 3D, interactive graphs
3. **Dark Ocean Theme** - Beautiful dark theme
4. **NPM Panel Example** - Complete npm package example

### Documentation
- **README.md**: Development guide
- **PLUGIN_API.md**: Complete API reference
- **QUICKSTART.md**: 5-minute getting started
- **PLUGIN_SYSTEM.md**: Architecture overview

## 🚀 Quick Start

### Setup

Run the master setup script to create all files:

```bash
python neural/aquarium/create_complete_plugin_system.py
```

This creates:
- Core plugin system in `src/plugins/`
- Example plugins in `src/plugins/examples/`
- Marketplace UI in `src/components/marketplace/`
- Backend API in `backend/plugin_api.py`
- Complete documentation

### Using the Plugin System

**Python:**
```python
from neural.aquarium.src.plugins import PluginManager

manager = PluginManager()
plugins = manager.list_plugins()
manager.enable_plugin('github-copilot-integration')
```

**TypeScript:**
```typescript
import { pluginService } from './services/PluginService';

const plugins = await pluginService.listPlugins();
await pluginService.enablePlugin('github-copilot-integration');
```

**REST API:**
```bash
curl http://localhost:5000/api/plugins/list
curl -X POST http://localhost:5000/api/plugins/enable \
  -H "Content-Type: application/json" \
  -d '{"plugin_id": "github-copilot-integration"}'
```

## 📦 File Structure

```
neural/aquarium/
├── src/
│   ├── plugins/                           # Python plugin system
│   │   ├── __init__.py                   # Main exports
│   │   ├── plugin_base.py                # Base classes
│   │   ├── plugin_manager.py             # Central coordinator
│   │   ├── plugin_loader.py              # Discovery & loading
│   │   ├── plugin_registry.py            # Registration & state
│   │   ├── examples/                     # Example plugins
│   │   │   ├── copilot_plugin/          # GitHub Copilot
│   │   │   ├── viz_plugin/              # Visualizations
│   │   │   ├── dark_ocean_theme/        # Theme
│   │   │   └── npm_plugin_example/      # NPM package
│   │   ├── README.md                     # Dev guide
│   │   ├── PLUGIN_API.md                 # API reference
│   │   └── QUICKSTART.md                 # Quick start
│   ├── components/
│   │   └── marketplace/                  # React components
│   │       ├── PluginMarketplace.tsx     # Main UI
│   │       ├── PluginMarketplace.css     # Styles
│   │       └── index.ts                  # Exports
│   ├── services/
│   │   └── PluginService.ts              # API client
│   └── types/
│       └── plugins.ts                    # TypeScript types
├── backend/
│   ├── plugin_api.py                     # Flask endpoints
│   └── api_updated.py                    # Integrated server
├── PLUGIN_SYSTEM.md                      # Architecture doc
└── PLUGINS_README.md                     # This file
```

## 🔧 Setup Scripts

All setup scripts are in `neural/aquarium/`:

1. **create_complete_plugin_system.py** - Master script (runs all)
2. **setup_plugins.py** - Core Python system
3. **setup_plugins_part2.py** - Example plugins
4. **setup_plugins_part3.py** - Marketplace UI & API
5. **setup_plugins_part4.py** - Documentation
6. **setup_npm_plugin_example.py** - NPM package example

Run them all at once:
```bash
python neural/aquarium/create_complete_plugin_system.py
```

Or individually for specific parts.

## 📚 Documentation

- **[PLUGIN_SYSTEM.md](PLUGIN_SYSTEM.md)** - Complete architecture overview
- **[src/plugins/README.md](src/plugins/README.md)** - Plugin development guide
- **[src/plugins/PLUGIN_API.md](src/plugins/PLUGIN_API.md)** - Full API reference
- **[src/plugins/QUICKSTART.md](src/plugins/QUICKSTART.md)** - 5-minute tutorial

## 🎨 Example Plugins

### 1. GitHub Copilot Integration
AI-powered code completion for Neural DSL.

**Features:**
- Real-time suggestions
- Context-aware completions
- Layer recommendations

**Location:** `src/plugins/examples/copilot_plugin/`

### 2. Custom Visualizations
Advanced visualization capabilities.

**Types:**
- 3D architecture models
- Interactive flow diagrams
- Heatmaps
- Sankey diagrams
- Circular graphs

**Location:** `src/plugins/examples/viz_plugin/`

### 3. Dark Ocean Theme
Beautiful dark theme with ocean colors.

**Features:**
- Deep blue color scheme
- Excellent contrast
- Custom editor theme
- Reduced eye strain

**Location:** `src/plugins/examples/dark_ocean_theme/`

### 4. NPM Panel Example
Complete npm package demonstrating best practices.

**Includes:**
- TypeScript + React
- Full package structure
- Publishing guide
- CI/CD workflow

**Location:** `src/plugins/examples/npm_plugin_example/`

## 🌐 API Endpoints

### Plugin Management
- `GET /api/plugins/list` - List all plugins
- `GET /api/plugins/enabled` - List enabled plugins
- `GET /api/plugins/details/<id>` - Get details
- `POST /api/plugins/enable` - Enable plugin
- `POST /api/plugins/disable` - Disable plugin
- `POST /api/plugins/install` - Install from npm/PyPI
- `POST /api/plugins/uninstall` - Uninstall
- `GET /api/plugins/search?q=<query>` - Search

### Plugin Capabilities
- `GET /api/plugins/panels` - Get panels
- `GET /api/plugins/themes` - Get themes
- `GET /api/plugins/commands` - Get commands
- `POST /api/plugins/command/execute` - Execute command

## 💡 Creating Your Own Plugin

### 1. Create Structure
```bash
mkdir my-plugin
cd my-plugin
```

### 2. Create plugin.json
```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "What it does",
  "capabilities": ["panel"],
  "license": "MIT"
}
```

### 3. Create main.py
```python
from plugin_base import PanelPlugin, PluginMetadata

class MyPlugin(PanelPlugin):
    def initialize(self):
        pass
    
    def activate(self):
        self._enabled = True
    
    def deactivate(self):
        self._enabled = False
    
    def get_panel_component(self):
        return "MyPanel"
    
    def get_panel_config(self):
        return {
            'title': 'My Panel',
            'position': 'right'
        }

def create_plugin(metadata):
    return MyPlugin(metadata)
```

### 4. Test It
```python
from plugins.plugin_loader import PluginLoader

loader = PluginLoader()
plugin = loader.load_plugin('my-plugin')
```

## 🔌 Publishing Plugins

### To npm
1. Add `neuralAquariumPlugin` field to package.json
2. Build: `npm run build`
3. Publish: `npm publish --access public`

### To PyPI
1. Include plugin.json in package
2. Build: `python setup.py sdist`
3. Publish: `twine upload dist/*`

See [Publishing Guide](src/plugins/examples/npm_plugin_example/PUBLISHING.md) for details.

## 🎯 Features

### ✅ Implemented
- [x] Core plugin system (Python)
- [x] Plugin base classes (Panel, Theme, Command, Viz, Integration)
- [x] Plugin manager with singleton pattern
- [x] Plugin loader with npm/PyPI discovery
- [x] Plugin registry with persistence
- [x] Hook system for events
- [x] REST API endpoints
- [x] Marketplace UI (React)
- [x] TypeScript types and service
- [x] Example plugins (4 complete examples)
- [x] Comprehensive documentation
- [x] Configuration system
- [x] Dependency management

### 🚧 Future Enhancements
- [ ] Plugin sandboxing for security
- [ ] Marketplace backend service
- [ ] Automated testing framework
- [ ] Plugin templates CLI
- [ ] Version compatibility checks
- [ ] Conflict resolution
- [ ] Performance profiling
- [ ] Community ratings/reviews
- [ ] Update notifications

## 🛠️ Development

### Running Tests
```bash
# Python tests
pytest neural/aquarium/tests/

# TypeScript tests
cd neural/aquarium
npm test
```

### Adding New Plugin Types
1. Add capability to `PluginCapability` enum
2. Create new base class in `plugin_base.py`
3. Add methods to `PluginManager`
4. Update documentation

## 📖 Best Practices

1. **Use semantic versioning** - Follow semver for versions
2. **Document thoroughly** - Include README and examples
3. **Handle errors gracefully** - Don't crash the host
4. **Clean up resources** - Implement deactivate() properly
5. **Test thoroughly** - Test enable/disable cycles
6. **Be performant** - Minimize initialization time
7. **Follow conventions** - Use kebab-case for IDs
8. **Stay compatible** - Test with multiple versions

## 🤝 Contributing

Contributions welcome! Areas needing help:
- Additional example plugins
- Documentation improvements
- Testing coverage
- UI/UX enhancements
- Performance optimization

## 📄 License

MIT License - See LICENSE.md

## 🙏 Credits

Created for Neural Aquarium by the Neural DSL Team.

## 📞 Support

- GitHub Issues: [github.com/neural-dsl/neural-dsl](https://github.com/neural-dsl/neural-dsl)
- Documentation: [docs.neural-dsl.org](https://docs.neural-dsl.org)
- Discord: [discord.gg/neural-dsl](https://discord.gg/neural-dsl)
