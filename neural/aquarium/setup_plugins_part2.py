#!/usr/bin/env python3
"""
Part 2: Example plugins and marketplace UI components
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
PLUGINS_DIR = BASE_DIR / "src" / "plugins"
EXAMPLES_DIR = PLUGINS_DIR / "examples"
MARKETPLACE_DIR = BASE_DIR / "src" / "components" / "marketplace"

files = {}

# ============================================================================
# EXAMPLE PLUGIN 1: GitHub Copilot Integration
# ============================================================================

copilot_dir = EXAMPLES_DIR / "copilot_plugin"
copilot_dir.mkdir(parents=True, exist_ok=True)

files[copilot_dir / "plugin.json"] = '''{
  "id": "github-copilot-integration",
  "name": "GitHub Copilot Integration",
  "version": "1.0.0",
  "author": "Neural DSL Team",
  "description": "Integrates GitHub Copilot AI assistance for Neural DSL code completion and suggestions",
  "capabilities": ["integration", "code_completion"],
  "homepage": "https://github.com/neural-dsl/plugins/copilot",
  "repository": "https://github.com/neural-dsl/plugins",
  "keywords": ["copilot", "ai", "completion", "github"],
  "license": "MIT",
  "dependencies": [],
  "python_dependencies": ["requests>=2.28.0"],
  "npm_dependencies": {},
  "min_aquarium_version": "0.3.0",
  "icon": "🤖",
  "rating": 4.8,
  "downloads": 15420
}
'''

files[copilot_dir / "main.py"] = '''"""
GitHub Copilot Integration Plugin for Neural Aquarium
"""

from __future__ import annotations
import sys
import os
from typing import Dict, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugin_base import IntegrationPlugin, PluginMetadata


class CopilotPlugin(IntegrationPlugin):
    """GitHub Copilot integration plugin."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.api_key = None
        self.connected = False
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        print(f"Initializing {self.metadata.name}...")
    
    def activate(self) -> None:
        """Activate the plugin."""
        print(f"Activating {self.metadata.name}...")
        self._enabled = True
    
    def deactivate(self) -> None:
        """Deactivate the plugin."""
        print(f"Deactivating {self.metadata.name}...")
        if self.connected:
            self.disconnect()
        self._enabled = False
    
    def get_integration_name(self) -> str:
        """Get integration name."""
        return "GitHub Copilot"
    
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to GitHub Copilot."""
        self.api_key = credentials.get('api_key')
        if not self.api_key:
            return False
        
        self.connected = True
        print(f"Connected to GitHub Copilot")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from GitHub Copilot."""
        self.api_key = None
        self.connected = False
        print("Disconnected from GitHub Copilot")
    
    def get_completions(self, code: str, cursor_position: int) -> list:
        """Get code completions from Copilot."""
        if not self.connected:
            return []
        
        return [
            {
                'text': 'Conv2D(filters=64, kernel_size=3, activation="relu")',
                'type': 'layer',
                'confidence': 0.95
            },
            {
                'text': 'MaxPooling2D(pool_size=2)',
                'type': 'layer',
                'confidence': 0.88
            }
        ]
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            'api_key': {
                'type': 'string',
                'description': 'GitHub Copilot API key',
                'required': True,
                'sensitive': True
            },
            'auto_suggest': {
                'type': 'boolean',
                'description': 'Enable automatic suggestions',
                'default': True
            }
        }


def create_plugin(metadata: PluginMetadata) -> CopilotPlugin:
    """Factory function to create plugin instance."""
    return CopilotPlugin(metadata)
'''

files[copilot_dir / "README.md"] = '''# GitHub Copilot Integration Plugin

AI-powered code completion for Neural DSL using GitHub Copilot.

## Features

- Real-time code suggestions as you type
- Context-aware Neural DSL completions
- Layer and architecture recommendations
- Pattern recognition from existing models

## Configuration

1. Get a GitHub Copilot API key
2. Configure the plugin with your API key
3. Enable auto-suggestions in settings

## Usage

Simply start typing in the Neural DSL editor and Copilot will provide intelligent suggestions based on your context.
'''

# ============================================================================
# EXAMPLE PLUGIN 2: Custom Visualizations
# ============================================================================

viz_dir = EXAMPLES_DIR / "viz_plugin"
viz_dir.mkdir(parents=True, exist_ok=True)

files[viz_dir / "plugin.json"] = '''{
  "id": "custom-visualizations",
  "name": "Custom Visualizations",
  "version": "1.0.0",
  "author": "Neural DSL Team",
  "description": "Advanced custom visualizations for neural network architectures including 3D models and interactive graphs",
  "capabilities": ["visualization", "panel"],
  "homepage": "https://github.com/neural-dsl/plugins/visualizations",
  "repository": "https://github.com/neural-dsl/plugins",
  "keywords": ["visualization", "3d", "graphs", "charts"],
  "license": "MIT",
  "dependencies": [],
  "python_dependencies": ["plotly>=5.0.0", "networkx>=2.8.0"],
  "npm_dependencies": {
    "plotly.js": "^2.18.0",
    "three": "^0.150.0"
  },
  "min_aquarium_version": "0.3.0",
  "icon": "📊",
  "rating": 4.6,
  "downloads": 8932
}
'''

files[viz_dir / "main.py"] = '''"""
Custom Visualizations Plugin for Neural Aquarium
"""

from __future__ import annotations
import sys
from typing import Dict, Any, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugin_base import VisualizationPlugin, PluginMetadata


class CustomVisualizationsPlugin(VisualizationPlugin):
    """Custom visualizations plugin."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        print(f"Initializing {self.metadata.name}...")
    
    def activate(self) -> None:
        """Activate the plugin."""
        print(f"Activating {self.metadata.name}...")
        self._enabled = True
    
    def deactivate(self) -> None:
        """Deactivate the plugin."""
        print(f"Deactivating {self.metadata.name}...")
        self._enabled = False
    
    def get_visualization_types(self) -> List[str]:
        """Get list of visualization types provided."""
        return [
            '3d-architecture',
            'interactive-flow',
            'heatmap',
            'sankey-diagram',
            'circular-graph'
        ]
    
    def render_visualization(self, vis_type: str, data: Any) -> Dict[str, Any]:
        """Render a visualization."""
        if vis_type == '3d-architecture':
            return self._render_3d_architecture(data)
        elif vis_type == 'interactive-flow':
            return self._render_interactive_flow(data)
        elif vis_type == 'heatmap':
            return self._render_heatmap(data)
        elif vis_type == 'sankey-diagram':
            return self._render_sankey(data)
        elif vis_type == 'circular-graph':
            return self._render_circular_graph(data)
        else:
            return {'error': f'Unknown visualization type: {vis_type}'}
    
    def _render_3d_architecture(self, data: Any) -> Dict[str, Any]:
        """Render 3D architecture visualization."""
        return {
            'type': '3d',
            'component': 'Three3DArchitecture',
            'data': {
                'layers': data.get('layers', []),
                'connections': data.get('connections', []),
                'camera_position': [10, 10, 10]
            }
        }
    
    def _render_interactive_flow(self, data: Any) -> Dict[str, Any]:
        """Render interactive flow diagram."""
        return {
            'type': 'flow',
            'component': 'InteractiveFlowDiagram',
            'data': {
                'nodes': data.get('nodes', []),
                'edges': data.get('edges', []),
                'interactive': True
            }
        }
    
    def _render_heatmap(self, data: Any) -> Dict[str, Any]:
        """Render heatmap visualization."""
        return {
            'type': 'plotly',
            'component': 'PlotlyHeatmap',
            'data': {
                'z': data.get('values', []),
                'x': data.get('x_labels', []),
                'y': data.get('y_labels', []),
                'colorscale': 'Viridis'
            }
        }
    
    def _render_sankey(self, data: Any) -> Dict[str, Any]:
        """Render Sankey diagram."""
        return {
            'type': 'plotly',
            'component': 'PlotlySankey',
            'data': {
                'nodes': data.get('nodes', []),
                'links': data.get('links', [])
            }
        }
    
    def _render_circular_graph(self, data: Any) -> Dict[str, Any]:
        """Render circular graph."""
        return {
            'type': 'networkx',
            'component': 'CircularNetworkGraph',
            'data': {
                'nodes': data.get('nodes', []),
                'edges': data.get('edges', []),
                'layout': 'circular'
            }
        }


def create_plugin(metadata: PluginMetadata) -> CustomVisualizationsPlugin:
    """Factory function to create plugin instance."""
    return CustomVisualizationsPlugin(metadata)
'''

files[viz_dir / "README.md"] = '''# Custom Visualizations Plugin

Advanced visualization capabilities for Neural Aquarium.

## Visualization Types

### 3D Architecture
Interactive 3D model of your neural network architecture using Three.js.

### Interactive Flow
Dynamic flow diagrams showing data flow through the network.

### Heatmap
Layer activation heatmaps for debugging and analysis.

### Sankey Diagram
Visualize data flow and transformations as a Sankey diagram.

### Circular Graph
Network topology in a circular layout for better overview.

## Usage

Select a visualization type from the dropdown in the Visualization panel to see your model rendered in different ways.
'''

# ============================================================================
# EXAMPLE PLUGIN 3: Dark Ocean Theme
# ============================================================================

theme_dir = EXAMPLES_DIR / "dark_ocean_theme"
theme_dir.mkdir(parents=True, exist_ok=True)

files[theme_dir / "plugin.json"] = '''{
  "id": "dark-ocean-theme",
  "name": "Dark Ocean Theme",
  "version": "1.0.0",
  "author": "Neural DSL Team",
  "description": "A beautiful dark theme inspired by deep ocean colors with excellent contrast and readability",
  "capabilities": ["theme"],
  "homepage": "https://github.com/neural-dsl/plugins/themes",
  "repository": "https://github.com/neural-dsl/plugins",
  "keywords": ["theme", "dark", "ocean", "blue"],
  "license": "MIT",
  "dependencies": [],
  "python_dependencies": [],
  "npm_dependencies": {},
  "min_aquarium_version": "0.3.0",
  "icon": "🌊",
  "rating": 4.9,
  "downloads": 23156
}
'''

files[theme_dir / "main.py"] = '''"""
Dark Ocean Theme Plugin for Neural Aquarium
"""

from __future__ import annotations
import sys
from typing import Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugin_base import ThemePlugin, PluginMetadata


class DarkOceanTheme(ThemePlugin):
    """Dark Ocean theme plugin."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        print(f"Initializing {self.metadata.name}...")
    
    def activate(self) -> None:
        """Activate the plugin."""
        print(f"Activating {self.metadata.name}...")
        self._enabled = True
    
    def deactivate(self) -> None:
        """Deactivate the plugin."""
        print(f"Deactivating {self.metadata.name}...")
        self._enabled = False
    
    def get_theme_definition(self) -> Dict[str, Any]:
        """Get theme definition."""
        return {
            'name': 'Dark Ocean',
            'type': 'dark',
            'colors': {
                'primary': '#0A7EA4',
                'secondary': '#1E5F74',
                'background': '#0D1B2A',
                'surface': '#1B263B',
                'surface_variant': '#273549',
                'error': '#FF6B6B',
                'warning': '#FFD93D',
                'success': '#6BCF7F',
                'info': '#4ECDC4',
                'text_primary': '#E0E1DD',
                'text_secondary': '#B0B8C1',
                'text_disabled': '#6C757D',
                'border': '#415A77',
                'hover': '#2A4159',
                'active': '#35516D',
                'shadow': 'rgba(0, 0, 0, 0.4)',
            },
            'fonts': {
                'primary': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                'monospace': '"Fira Code", "Consolas", "Monaco", monospace',
            },
            'spacing': {
                'xs': '4px',
                'sm': '8px',
                'md': '16px',
                'lg': '24px',
                'xl': '32px',
            },
            'borderRadius': {
                'sm': '4px',
                'md': '8px',
                'lg': '12px',
                'full': '9999px',
            },
            'shadows': {
                'sm': '0 1px 3px rgba(0, 0, 0, 0.4)',
                'md': '0 4px 6px rgba(0, 0, 0, 0.4)',
                'lg': '0 10px 15px rgba(0, 0, 0, 0.4)',
            }
        }
    
    def get_editor_theme(self) -> Optional[Dict[str, Any]]:
        """Get editor-specific theme configuration."""
        return {
            'base': 'vs-dark',
            'inherit': True,
            'rules': [
                {'token': 'keyword', 'foreground': '4ECDC4', 'fontStyle': 'bold'},
                {'token': 'string', 'foreground': 'FFD93D'},
                {'token': 'number', 'foreground': '6BCF7F'},
                {'token': 'comment', 'foreground': '6C757D', 'fontStyle': 'italic'},
                {'token': 'type', 'foreground': '0A7EA4'},
                {'token': 'variable', 'foreground': 'E0E1DD'},
                {'token': 'operator', 'foreground': 'FF6B6B'},
            ],
            'colors': {
                'editor.background': '#0D1B2A',
                'editor.foreground': '#E0E1DD',
                'editor.lineHighlightBackground': '#1B263B',
                'editor.selectionBackground': '#2A4159',
                'editorCursor.foreground': '#0A7EA4',
                'editorWhitespace.foreground': '#415A77',
            }
        }


def create_plugin(metadata: PluginMetadata) -> DarkOceanTheme:
    """Factory function to create plugin instance."""
    return DarkOceanTheme(metadata)
'''

files[theme_dir / "README.md"] = '''# Dark Ocean Theme

A beautiful dark theme inspired by the deep ocean.

## Features

- Deep blue color scheme
- Excellent contrast for long coding sessions
- Reduced eye strain
- Beautiful syntax highlighting
- Consistent across all UI elements

## Colors

- Primary: Deep teal (#0A7EA4)
- Background: Navy (#0D1B2A)
- Accent: Turquoise (#4ECDC4)
- Text: Light gray (#E0E1DD)

Perfect for late-night coding sessions!
'''

# Write all files
for filepath, content in files.items():
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\\n✅ Example plugins created successfully!")
