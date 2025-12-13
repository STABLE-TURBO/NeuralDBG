#!/usr/bin/env python3
"""
Comprehensive script to create the entire plugin system for Neural Aquarium.
This creates all Python files, TypeScript/React components, and example plugins.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
PLUGINS_DIR = BASE_DIR / "src" / "plugins"
EXAMPLES_DIR = PLUGINS_DIR / "examples"
MARKETPLACE_DIR = BASE_DIR / "src" / "components" / "marketplace"

# Create all directories
for directory in [PLUGINS_DIR, EXAMPLES_DIR, MARKETPLACE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Define all files and their content
files = {}

# ============================================================================
# CORE PLUGIN SYSTEM FILES
# ============================================================================

files[PLUGINS_DIR / "__init__.py"] = '''"""
Neural Aquarium Plugin System

Provides extension API for custom panels, themes, commands, and more.
"""

from .plugin_manager import PluginManager
from .plugin_base import Plugin, PluginMetadata, PluginCapability
from .plugin_loader import PluginLoader
from .plugin_registry import PluginRegistry

__all__ = [
    'PluginManager',
    'Plugin',
    'PluginMetadata',
    'PluginCapability',
    'PluginLoader',
    'PluginRegistry',
]
'''

files[PLUGINS_DIR / "plugin_base.py"] = '''"""
Base plugin classes and interfaces for Neural Aquarium plugin system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class PluginCapability(Enum):
    """Plugin capability types."""
    PANEL = "panel"
    THEME = "theme"
    COMMAND = "command"
    VISUALIZATION = "visualization"
    INTEGRATION = "integration"
    LANGUAGE_SUPPORT = "language_support"
    CODE_COMPLETION = "code_completion"
    LINTER = "linter"
    FORMATTER = "formatter"


@dataclass
class PluginMetadata:
    """Plugin metadata."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'capabilities': [c.value for c in self.capabilities],
            'homepage': self.homepage,
            'repository': self.repository,
            'keywords': self.keywords,
            'license': self.license,
            'dependencies': self.dependencies,
            'python_dependencies': self.python_dependencies,
            'npm_dependencies': self.npm_dependencies,
            'min_aquarium_version': self.min_aquarium_version,
            'icon': self.icon,
            'rating': self.rating,
            'downloads': self.downloads,
        }


class Plugin(ABC):
    """Base plugin class."""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self._enabled = False
        self._initialized = False
        self.config: Dict[str, Any] = {}
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def activate(self) -> None:
        """Activate the plugin."""
        pass
    
    @abstractmethod
    def deactivate(self) -> None:
        """Deactivate the plugin."""
        pass
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin."""
        self.config.update(config)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this plugin."""
        return {}
    
    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled
    
    @property
    def initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized


class PanelPlugin(Plugin):
    """Plugin that provides custom panels."""
    
    @abstractmethod
    def get_panel_component(self) -> str:
        """Get the React component path for the panel."""
        pass
    
    @abstractmethod
    def get_panel_config(self) -> Dict[str, Any]:
        """Get panel configuration."""
        pass


class ThemePlugin(Plugin):
    """Plugin that provides custom themes."""
    
    @abstractmethod
    def get_theme_definition(self) -> Dict[str, Any]:
        """Get theme definition (colors, fonts, etc.)."""
        pass
    
    def get_editor_theme(self) -> Optional[Dict[str, Any]]:
        """Get editor-specific theme configuration."""
        return None


class CommandPlugin(Plugin):
    """Plugin that provides custom commands."""
    
    @abstractmethod
    def get_commands(self) -> List[Dict[str, Any]]:
        """Get list of commands provided by this plugin."""
        pass
    
    @abstractmethod
    def execute_command(self, command_id: str, args: Dict[str, Any]) -> Any:
        """Execute a command."""
        pass


class VisualizationPlugin(Plugin):
    """Plugin that provides custom visualizations."""
    
    @abstractmethod
    def get_visualization_types(self) -> List[str]:
        """Get list of visualization types provided."""
        pass
    
    @abstractmethod
    def render_visualization(self, vis_type: str, data: Any) -> Dict[str, Any]:
        """Render a visualization."""
        pass


class IntegrationPlugin(Plugin):
    """Plugin that provides external integrations."""
    
    @abstractmethod
    def get_integration_name(self) -> str:
        """Get integration name."""
        pass
    
    @abstractmethod
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to external service."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from external service."""
        pass


class PluginHookRegistry:
    """Registry for plugin hooks."""
    
    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {}
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def unregister_hook(self, hook_name: str, callback: Callable) -> None:
        """Unregister a hook callback."""
        if hook_name in self._hooks and callback in self._hooks[hook_name]:
            self._hooks[hook_name].remove(callback)
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all callbacks for a hook."""
        results = []
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"Error executing hook {hook_name}: {e}")
        return results
'''

files[PLUGINS_DIR / "plugin_registry.py"] = '''"""
Plugin registry for managing installed plugins.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
from pathlib import Path
import json
import logging

from .plugin_base import Plugin, PluginMetadata, PluginCapability

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central registry for all plugins."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        self._plugins: Dict[str, Plugin] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        self._enabled_plugins: Set[str] = set()
        self.registry_path = registry_path or Path.home() / ".neural_aquarium" / "plugins"
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._load_registry()
    
    def register(self, plugin: Plugin) -> None:
        """Register a plugin."""
        plugin_id = plugin.metadata.id
        if plugin_id in self._plugins:
            logger.warning(f"Plugin {plugin_id} is already registered")
            return
        
        self._plugins[plugin_id] = plugin
        self._metadata[plugin_id] = plugin.metadata
        logger.info(f"Registered plugin: {plugin_id}")
        self._save_registry()
    
    def unregister(self, plugin_id: str) -> None:
        """Unregister a plugin."""
        if plugin_id in self._plugins:
            plugin = self._plugins[plugin_id]
            if plugin.enabled:
                plugin.deactivate()
            del self._plugins[plugin_id]
            del self._metadata[plugin_id]
            self._enabled_plugins.discard(plugin_id)
            logger.info(f"Unregistered plugin: {plugin_id}")
            self._save_registry()
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)
    
    def get_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by ID."""
        return self._metadata.get(plugin_id)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin IDs."""
        return list(self._plugins.keys())
    
    def list_enabled_plugins(self) -> List[str]:
        """List enabled plugin IDs."""
        return list(self._enabled_plugins)
    
    def list_by_capability(self, capability: PluginCapability) -> List[Plugin]:
        """List plugins by capability."""
        return [
            plugin for plugin in self._plugins.values()
            if capability in plugin.metadata.capabilities
        ]
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        try:
            if not plugin.initialized:
                plugin.initialize()
                plugin._initialized = True
            
            plugin.activate()
            plugin._enabled = True
            self._enabled_plugins.add(plugin_id)
            logger.info(f"Enabled plugin: {plugin_id}")
            self._save_registry()
            return True
        except Exception as e:
            logger.error(f"Failed to enable plugin {plugin_id}: {e}")
            return False
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        try:
            plugin.deactivate()
            plugin._enabled = False
            self._enabled_plugins.discard(plugin_id)
            logger.info(f"Disabled plugin: {plugin_id}")
            self._save_registry()
            return True
        except Exception as e:
            logger.error(f"Failed to disable plugin {plugin_id}: {e}")
            return False
    
    def _load_registry(self) -> None:
        """Load plugin registry from disk."""
        registry_file = self.registry_path / "registry.json"
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)
                self._enabled_plugins = set(data.get('enabled', []))
                logger.info(f"Loaded plugin registry from {registry_file}")
        except Exception as e:
            logger.error(f"Failed to load plugin registry: {e}")
    
    def _save_registry(self) -> None:
        """Save plugin registry to disk."""
        registry_file = self.registry_path / "registry.json"
        try:
            data = {
                'enabled': list(self._enabled_plugins),
                'plugins': [meta.to_dict() for meta in self._metadata.values()]
            }
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved plugin registry to {registry_file}")
        except Exception as e:
            logger.error(f"Failed to save plugin registry: {e}")
    
    def search(self, query: str) -> List[PluginMetadata]:
        """Search plugins by name, description, or keywords."""
        query_lower = query.lower()
        results = []
        
        for metadata in self._metadata.values():
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in keyword.lower() for keyword in metadata.keywords)):
                results.append(metadata)
        
        return results
'''

files[PLUGINS_DIR / "plugin_loader.py"] = '''"""
Plugin loader for discovering and loading plugins from various sources.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pathlib import Path
import importlib
import importlib.util
import json
import logging
import sys
import subprocess
import shutil

from .plugin_base import Plugin, PluginMetadata, PluginCapability

logger = logging.getLogger(__name__)


class PluginLoader:
    """Loads plugins from filesystem and package managers."""
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        self.plugin_dir = plugin_dir or Path.home() / ".neural_aquarium" / "plugins"
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.npm_dir = self.plugin_dir / "npm_plugins"
        self.pypi_dir = self.plugin_dir / "pypi_plugins"
        self.npm_dir.mkdir(exist_ok=True)
        self.pypi_dir.mkdir(exist_ok=True)
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover all available plugins."""
        plugins = []
        
        plugins.extend(self._discover_local_plugins())
        plugins.extend(self._discover_npm_plugins())
        plugins.extend(self._discover_pypi_plugins())
        
        return plugins
    
    def _discover_local_plugins(self) -> List[PluginMetadata]:
        """Discover plugins in local plugin directory."""
        plugins = []
        
        for plugin_path in self.plugin_dir.iterdir():
            if not plugin_path.is_dir() or plugin_path.name.startswith('.'):
                continue
            
            manifest_file = plugin_path / "plugin.json"
            if manifest_file.exists():
                try:
                    metadata = self._load_manifest(manifest_file)
                    plugins.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to load plugin manifest from {manifest_file}: {e}")
        
        return plugins
    
    def _discover_npm_plugins(self) -> List[PluginMetadata]:
        """Discover plugins installed via npm."""
        plugins = []
        
        for plugin_path in self.npm_dir.iterdir():
            if not plugin_path.is_dir():
                continue
            
            package_json = plugin_path / "package.json"
            if package_json.exists():
                try:
                    with open(package_json, 'r') as f:
                        pkg_data = json.load(f)
                    
                    if 'neuralAquariumPlugin' in pkg_data:
                        metadata = self._parse_npm_plugin_metadata(pkg_data)
                        plugins.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to load npm plugin from {plugin_path}: {e}")
        
        return plugins
    
    def _discover_pypi_plugins(self) -> List[PluginMetadata]:
        """Discover plugins installed via PyPI."""
        plugins = []
        
        for plugin_path in self.pypi_dir.iterdir():
            if not plugin_path.is_dir():
                continue
            
            manifest = plugin_path / "plugin.json"
            if manifest.exists():
                try:
                    metadata = self._load_manifest(manifest)
                    plugins.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to load PyPI plugin from {plugin_path}: {e}")
        
        return plugins
    
    def load_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Load a plugin by ID."""
        plugin_path = self.plugin_dir / plugin_id
        
        if not plugin_path.exists():
            plugin_path = self.npm_dir / plugin_id
        if not plugin_path.exists():
            plugin_path = self.pypi_dir / plugin_id
        if not plugin_path.exists():
            logger.error(f"Plugin not found: {plugin_id}")
            return None
        
        manifest_file = plugin_path / "plugin.json"
        if not manifest_file.exists():
            logger.error(f"Plugin manifest not found: {manifest_file}")
            return None
        
        try:
            metadata = self._load_manifest(manifest_file)
            plugin = self._load_plugin_module(plugin_path, metadata)
            return plugin
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            return None
    
    def install_from_npm(self, package_name: str, version: Optional[str] = None) -> bool:
        """Install a plugin from npm."""
        try:
            install_cmd = f"npm install {package_name}"
            if version:
                install_cmd += f"@{version}"
            
            install_dir = self.npm_dir / package_name.split('/')[-1]
            install_dir.mkdir(exist_ok=True)
            
            result = subprocess.run(
                install_cmd,
                shell=True,
                cwd=str(install_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully installed npm plugin: {package_name}")
                return True
            else:
                logger.error(f"Failed to install npm plugin: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error installing npm plugin {package_name}: {e}")
            return False
    
    def install_from_pypi(self, package_name: str, version: Optional[str] = None) -> bool:
        """Install a plugin from PyPI."""
        try:
            install_cmd = [sys.executable, "-m", "pip", "install"]
            if version:
                install_cmd.append(f"{package_name}=={version}")
            else:
                install_cmd.append(package_name)
            
            install_cmd.extend(["--target", str(self.pypi_dir / package_name)])
            
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully installed PyPI plugin: {package_name}")
                return True
            else:
                logger.error(f"Failed to install PyPI plugin: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error installing PyPI plugin {package_name}: {e}")
            return False
    
    def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin."""
        plugin_path = self.plugin_dir / plugin_id
        
        if not plugin_path.exists():
            plugin_path = self.npm_dir / plugin_id
        if not plugin_path.exists():
            plugin_path = self.pypi_dir / plugin_id
        
        if not plugin_path.exists():
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        try:
            shutil.rmtree(plugin_path)
            logger.info(f"Successfully uninstalled plugin: {plugin_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to uninstall plugin {plugin_id}: {e}")
            return False
    
    def _load_manifest(self, manifest_file: Path) -> PluginMetadata:
        """Load plugin manifest from JSON file."""
        with open(manifest_file, 'r') as f:
            data = json.load(f)
        
        capabilities = [PluginCapability(c) for c in data.get('capabilities', [])]
        
        return PluginMetadata(
            id=data['id'],
            name=data['name'],
            version=data['version'],
            author=data['author'],
            description=data['description'],
            capabilities=capabilities,
            homepage=data.get('homepage'),
            repository=data.get('repository'),
            keywords=data.get('keywords', []),
            license=data.get('license', 'MIT'),
            dependencies=data.get('dependencies', []),
            python_dependencies=data.get('python_dependencies', []),
            npm_dependencies=data.get('npm_dependencies', {}),
            min_aquarium_version=data.get('min_aquarium_version', '0.3.0'),
            icon=data.get('icon'),
            rating=data.get('rating', 0.0),
            downloads=data.get('downloads', 0)
        )
    
    def _parse_npm_plugin_metadata(self, package_data: Dict[str, Any]) -> PluginMetadata:
        """Parse plugin metadata from package.json."""
        plugin_data = package_data['neuralAquariumPlugin']
        capabilities = [PluginCapability(c) for c in plugin_data.get('capabilities', [])]
        
        return PluginMetadata(
            id=package_data['name'],
            name=plugin_data.get('displayName', package_data['name']),
            version=package_data['version'],
            author=package_data.get('author', 'Unknown'),
            description=package_data.get('description', ''),
            capabilities=capabilities,
            homepage=package_data.get('homepage'),
            repository=package_data.get('repository', {}).get('url'),
            keywords=package_data.get('keywords', []),
            license=package_data.get('license', 'MIT'),
            dependencies=plugin_data.get('dependencies', []),
            npm_dependencies=package_data.get('dependencies', {}),
            min_aquarium_version=plugin_data.get('minAquariumVersion', '0.3.0'),
            icon=plugin_data.get('icon')
        )
    
    def _load_plugin_module(self, plugin_path: Path, metadata: PluginMetadata) -> Plugin:
        """Load plugin Python module."""
        main_file = plugin_path / "main.py"
        if not main_file.exists():
            raise FileNotFoundError(f"Plugin main.py not found: {main_file}")
        
        spec = importlib.util.spec_from_file_location(metadata.id, main_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load plugin module: {metadata.id}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[metadata.id] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'create_plugin'):
            raise ImportError(f"Plugin module missing create_plugin function: {metadata.id}")
        
        plugin = module.create_plugin(metadata)
        return plugin
'''

files[PLUGINS_DIR / "plugin_manager.py"] = '''"""
Plugin manager for Neural Aquarium - coordinates all plugin operations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .plugin_base import Plugin, PluginMetadata, PluginCapability, PluginHookRegistry
from .plugin_loader import PluginLoader
from .plugin_registry import PluginRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """Central plugin manager for Neural Aquarium."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.registry = PluginRegistry()
        self.loader = PluginLoader()
        self.hooks = PluginHookRegistry()
        self._auto_load_plugins()
    
    def _auto_load_plugins(self) -> None:
        """Automatically load and enable plugins from registry."""
        plugins = self.loader.discover_plugins()
        
        for metadata in plugins:
            try:
                plugin = self.loader.load_plugin(metadata.id)
                if plugin:
                    self.registry.register(plugin)
                    
                    if metadata.id in self.registry.list_enabled_plugins():
                        self.enable_plugin(metadata.id)
            except Exception as e:
                logger.error(f"Failed to auto-load plugin {metadata.id}: {e}")
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all available plugins."""
        return [
            self.registry.get_metadata(plugin_id)
            for plugin_id in self.registry.list_plugins()
        ]
    
    def list_enabled_plugins(self) -> List[PluginMetadata]:
        """List enabled plugins."""
        return [
            self.registry.get_metadata(plugin_id)
            for plugin_id in self.registry.list_enabled_plugins()
        ]
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID."""
        return self.registry.get_plugin(plugin_id)
    
    def get_plugin_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get plugin metadata."""
        return self.registry.get_metadata(plugin_id)
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin."""
        return self.registry.enable_plugin(plugin_id)
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin."""
        return self.registry.disable_plugin(plugin_id)
    
    def install_plugin(self, source: str, plugin_name: str, version: Optional[str] = None) -> bool:
        """Install a plugin from npm or PyPI."""
        if source == 'npm':
            success = self.loader.install_from_npm(plugin_name, version)
        elif source == 'pypi':
            success = self.loader.install_from_pypi(plugin_name, version)
        else:
            logger.error(f"Unknown plugin source: {source}")
            return False
        
        if success:
            plugin = self.loader.load_plugin(plugin_name.split('/')[-1])
            if plugin:
                self.registry.register(plugin)
                return True
        
        return False
    
    def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall a plugin."""
        self.registry.unregister(plugin_id)
        return self.loader.uninstall_plugin(plugin_id)
    
    def get_panels(self) -> List[Dict[str, Any]]:
        """Get all panel plugins."""
        from .plugin_base import PanelPlugin
        panels = []
        
        for plugin in self.registry.list_by_capability(PluginCapability.PANEL):
            if isinstance(plugin, PanelPlugin) and plugin.enabled:
                panels.append({
                    'id': plugin.metadata.id,
                    'name': plugin.metadata.name,
                    'component': plugin.get_panel_component(),
                    'config': plugin.get_panel_config()
                })
        
        return panels
    
    def get_themes(self) -> List[Dict[str, Any]]:
        """Get all theme plugins."""
        from .plugin_base import ThemePlugin
        themes = []
        
        for plugin in self.registry.list_by_capability(PluginCapability.THEME):
            if isinstance(plugin, ThemePlugin) and plugin.enabled:
                themes.append({
                    'id': plugin.metadata.id,
                    'name': plugin.metadata.name,
                    'definition': plugin.get_theme_definition(),
                    'editor_theme': plugin.get_editor_theme()
                })
        
        return themes
    
    def get_commands(self) -> List[Dict[str, Any]]:
        """Get all command plugins."""
        from .plugin_base import CommandPlugin
        commands = []
        
        for plugin in self.registry.list_by_capability(PluginCapability.COMMAND):
            if isinstance(plugin, CommandPlugin) and plugin.enabled:
                commands.extend(plugin.get_commands())
        
        return commands
    
    def execute_command(self, command_id: str, args: Dict[str, Any]) -> Any:
        """Execute a command from a plugin."""
        from .plugin_base import CommandPlugin
        
        for plugin in self.registry.list_by_capability(PluginCapability.COMMAND):
            if isinstance(plugin, CommandPlugin) and plugin.enabled:
                for cmd in plugin.get_commands():
                    if cmd['id'] == command_id:
                        return plugin.execute_command(command_id, args)
        
        logger.error(f"Command not found: {command_id}")
        return None
    
    def search_plugins(self, query: str) -> List[PluginMetadata]:
        """Search for plugins."""
        return self.registry.search(query)
    
    def register_hook(self, hook_name: str, callback) -> None:
        """Register a plugin hook."""
        self.hooks.register_hook(hook_name, callback)
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute a plugin hook."""
        return self.hooks.execute_hook(hook_name, *args, **kwargs)
'''

# Write all files
for filepath, content in files.items():
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\\n✅ Core plugin system created successfully!")
print(f"   Base directory: {PLUGINS_DIR}")
print(f"   Examples directory: {EXAMPLES_DIR}")
print(f"   Marketplace directory: {MARKETPLACE_DIR}")
