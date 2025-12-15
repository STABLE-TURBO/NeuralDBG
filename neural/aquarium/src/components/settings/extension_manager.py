from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class Extension:
    """Represents an Aquarium IDE extension"""
    
    def __init__(self, id: str, name: str, version: str, enabled: bool = True):
        self.id = id
        self.name = name
        self.version = version
        self.enabled = enabled
        self.metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert extension to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Extension:
        """Create extension from dictionary"""
        ext = cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            version=data.get('version', '0.0.0'),
            enabled=data.get('enabled', True)
        )
        ext.metadata = data.get('metadata', {})
        return ext


class ExtensionManager:
    """Manages extensions and plugins for Aquarium IDE"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.extensions_dir = config_dir / 'extensions'
        self.extensions_dir.mkdir(parents=True, exist_ok=True)
        
        self.extensions: Dict[str, Extension] = {}
        self._load_extensions()
    
    def _load_extensions(self):
        """Load all installed extensions"""
        extensions_file = self.config_dir / 'extensions.json'
        if extensions_file.exists():
            try:
                with open(extensions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for ext_data in data.get('extensions', []):
                        ext = Extension.from_dict(ext_data)
                        self.extensions[ext.id] = ext
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading extensions: {e}")
    
    def _save_extensions(self):
        """Save extensions configuration"""
        extensions_file = self.config_dir / 'extensions.json'
        try:
            data = {
                'extensions': [ext.to_dict() for ext in self.extensions.values()]
            }
            with open(extensions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Error saving extensions: {e}")
    
    def install_extension(self, extension_path: str) -> bool:
        """Install an extension from a file"""
        try:
            with open(extension_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            ext = Extension(
                id=manifest.get('id', ''),
                name=manifest.get('name', ''),
                version=manifest.get('version', '0.0.0'),
                enabled=True
            )
            ext.metadata = manifest
            
            self.extensions[ext.id] = ext
            self._save_extensions()
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error installing extension: {e}")
            return False
    
    def uninstall_extension(self, extension_id: str) -> bool:
        """Uninstall an extension"""
        if extension_id in self.extensions:
            del self.extensions[extension_id]
            self._save_extensions()
            return True
        return False
    
    def enable_extension(self, extension_id: str) -> bool:
        """Enable an extension"""
        if extension_id in self.extensions:
            self.extensions[extension_id].enabled = True
            self._save_extensions()
            return True
        return False
    
    def disable_extension(self, extension_id: str) -> bool:
        """Disable an extension"""
        if extension_id in self.extensions:
            self.extensions[extension_id].enabled = False
            self._save_extensions()
            return True
        return False
    
    def get_extension(self, extension_id: str) -> Extension | None:
        """Get an extension by ID"""
        return self.extensions.get(extension_id)
    
    def get_enabled_extensions(self) -> list[Extension]:
        """Get all enabled extensions"""
        return [ext for ext in self.extensions.values() if ext.enabled]
    
    def get_disabled_extensions(self) -> list[Extension]:
        """Get all disabled extensions"""
        return [ext for ext in self.extensions.values() if not ext.enabled]
    
    def get_all_extensions(self) -> list[Extension]:
        """Get all installed extensions"""
        return list(self.extensions.values())
    
    def check_updates(self) -> list[dict[str, Any]]:
        """Check for extension updates"""
        available_updates = []
        
        for ext in self.extensions.values():
            # In a real implementation, this would query an extensions registry
            # For now, return empty list
            pass
        
        return available_updates
    
    def update_extension(self, extension_id: str) -> bool:
        """Update an extension to the latest version"""
        # In a real implementation, this would download and install the update
        return False
