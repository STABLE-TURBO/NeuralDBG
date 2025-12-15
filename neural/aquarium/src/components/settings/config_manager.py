from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ConfigManager:
    """Manages persistent configuration storage for Aquarium IDE"""
    
    def __init__(self):
        self.config_dir = Path.home() / '.aquarium'
        self.config_file = self.config_dir / 'config.json'
        self._config = self._load_config()
    
    def _ensure_config_dir(self):
        """Ensure the configuration directory exists"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_config(self) -> dict[str, Any]:
        """Return default configuration settings"""
        return {
            'editor': {
                'theme': 'dark',
                'custom_theme': {
                    'background': '#1e1e1e',
                    'foreground': '#d4d4d4',
                    'selection': '#264f78',
                    'comment': '#6a9955',
                    'keyword': '#569cd6',
                    'string': '#ce9178',
                    'number': '#b5cea8',
                    'function': '#dcdcaa',
                    'operator': '#d4d4d4',
                    'cursor': '#ffffff'
                },
                'font_size': 14,
                'font_family': 'Consolas, Monaco, monospace',
                'line_numbers': True,
                'word_wrap': False,
                'tab_size': 4,
                'insert_spaces': True,
                'auto_indent': True,
                'bracket_matching': True,
                'highlight_active_line': True
            },
            'keybindings': {
                'save': 'Ctrl+S',
                'save_as': 'Ctrl+Shift+S',
                'open': 'Ctrl+O',
                'new_file': 'Ctrl+N',
                'close_tab': 'Ctrl+W',
                'find': 'Ctrl+F',
                'replace': 'Ctrl+H',
                'goto_line': 'Ctrl+G',
                'comment_line': 'Ctrl+/',
                'indent': 'Tab',
                'outdent': 'Shift+Tab',
                'duplicate_line': 'Ctrl+D',
                'delete_line': 'Ctrl+Shift+K',
                'move_line_up': 'Alt+Up',
                'move_line_down': 'Alt+Down',
                'toggle_terminal': 'Ctrl+`',
                'toggle_sidebar': 'Ctrl+B',
                'command_palette': 'Ctrl+Shift+P',
                'run_model': 'F5',
                'debug_model': 'Shift+F5',
                'compile_model': 'Ctrl+F5'
            },
            'python': {
                'interpreter_path': '',
                'default_interpreter': 'python',
                'virtual_env_path': '',
                'conda_env': '',
                'use_system_python': True
            },
            'backend': {
                'default': 'tensorflow',
                'auto_detect': True,
                'preferences': {
                    'tensorflow': True,
                    'pytorch': True,
                    'onnx': True
                }
            },
            'autosave': {
                'enabled': True,
                'interval': 30,
                'on_focus_lost': True,
                'on_window_change': True
            },
            'extensions': {
                'enabled': [],
                'disabled': [],
                'auto_update': True,
                'check_updates_on_startup': True
            },
            'plugins': {
                'installed': [],
                'auto_install_dependencies': True
            },
            'ui': {
                'sidebar_width': 250,
                'panel_height': 200,
                'show_minimap': True,
                'show_breadcrumbs': True,
                'show_status_bar': True,
                'show_activity_bar': True
            },
            'terminal': {
                'shell': 'powershell' if os.name == 'nt' else 'bash',
                'font_size': 12,
                'cursor_style': 'block'
            },
            'neuraldbg': {
                'auto_launch': False,
                'port': 8050,
                'host': 'localhost'
            }
        }
    
    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    default_config = self._get_default_config()
                    return self._merge_configs(default_config, loaded_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}. Using default configuration.")
                return self._get_default_config()
        return self._get_default_config()
    
    def _merge_configs(self, default: dict, loaded: dict) -> dict:
        """Recursively merge loaded config with defaults"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self):
        """Save current configuration to file"""
        self._ensure_config_dir()
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"Error saving config: {e}")
    
    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        """Get a configuration value"""
        if section not in self._config:
            return default
        
        if key is None:
            return self._config[section]
        
        return self._config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Set a configuration value"""
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section][key] = value
        self.save_config()
    
    def update_section(self, section: str, values: dict[str, Any]):
        """Update an entire configuration section"""
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section].update(values)
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self._config = self._get_default_config()
        self.save_config()
    
    def get_all(self) -> dict[str, Any]:
        """Get entire configuration"""
        return self._config.copy()
    
    def export_config(self, export_path: str):
        """Export configuration to a specific file"""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            print(f"Error exporting config: {e}")
    
    def import_config(self, import_path: str):
        """Import configuration from a specific file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
                default_config = self._get_default_config()
                self._config = self._merge_configs(default_config, imported_config)
                self.save_config()
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error importing config: {e}")
