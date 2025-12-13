from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json


class WorkspaceConfig:
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.config: Dict[str, Any] = self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            "neural": {
                "compiler": {
                    "backend": "tensorflow",
                    "optimize": True,
                    "output_dir": "build",
                },
                "linter": {
                    "enabled": True,
                    "check_shapes": True,
                    "strict_mode": False,
                },
                "formatter": {
                    "indent_size": 2,
                    "max_line_length": 80,
                },
            },
            "editor": {
                "rulers": [80, 100],
                "bracket_pair_colorization": True,
                "minimap": True,
            },
            "files": {
                "exclude": [
                    "**/__pycache__",
                    "**/.git",
                    "**/.venv",
                    "**/venv",
                    "**/node_modules",
                    "**/*.pyc",
                ],
                "associations": {
                    "*.neural": "neural-dsl",
                },
            },
            "search": {
                "exclude": [
                    "**/build",
                    "**/dist",
                    "**/.venv",
                ],
            },
        }
    
    def get(self, *keys: str, default: Any = None) -> Any:
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys: str, value: Any) -> None:
        if len(keys) == 0:
            return
        
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get_compiler_backend(self) -> str:
        return self.get("neural", "compiler", "backend", default="tensorflow")
    
    def set_compiler_backend(self, backend: str) -> None:
        self.set("neural", "compiler", "backend", value=backend)
    
    def is_linter_enabled(self) -> bool:
        return self.get("neural", "linter", "enabled", default=True)
    
    def get_excluded_patterns(self) -> list[str]:
        return self.get("files", "exclude", default=[])
    
    def add_excluded_pattern(self, pattern: str) -> None:
        excluded = self.get_excluded_patterns()
        if pattern not in excluded:
            excluded.append(pattern)
            self.set("files", "exclude", value=excluded)
    
    def remove_excluded_pattern(self, pattern: str) -> None:
        excluded = self.get_excluded_patterns()
        if pattern in excluded:
            excluded.remove(pattern)
            self.set("files", "exclude", value=excluded)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        self.config = data
    
    def merge(self, other: Dict[str, Any]) -> None:
        self._deep_merge(self.config, other)
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
