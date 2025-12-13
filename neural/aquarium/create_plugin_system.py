#!/usr/bin/env python3
"""
Script to create the plugin system structure for Neural Aquarium.
Run this script to set up all plugin-related directories and files.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent / "src" / "plugins"
EXAMPLES_DIR = BASE_DIR / "examples"
MARKETPLACE_DIR = Path(__file__).parent / "src" / "components" / "marketplace"

BASE_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
MARKETPLACE_DIR.mkdir(parents=True, exist_ok=True)

# Plugin system core files
files = {
    BASE_DIR / "__init__.py": '''"""
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
''',
    
    BASE_DIR / "plugin_base.py": '''"""
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
''',
}

# Write all files
for filepath, content in files.items():
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\\nPlugin system structure created successfully!")
print(f"Base directory: {BASE_DIR}")
print(f"Examples directory: {EXAMPLES_DIR}")
print(f"Marketplace directory: {MARKETPLACE_DIR}")
