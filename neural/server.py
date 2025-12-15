"""
Unified Neural DSL Server

Consolidates all web interfaces and services into a single entry point:
- Debug Dashboard (NeuralDbg)
- No-Code Interface
- Monitoring Dashboard
- REST API endpoints
- WebSocket services

Run with: neural server start
Or: python -m neural.server
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from flask import Flask

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """Registry for optional features that can be enabled/disabled."""
    
    def __init__(self):
        self._features: Dict[str, bool] = {
            'debug': True,
            'nocode': True,
            'monitoring': True,
            'api': True,
            'collaboration': False,
        }
        self._load_from_env()
    
    def _load_from_env(self):
        """Load feature flags from environment variables."""
        for feature in self._features.keys():
            env_var = f"NEURAL_FEATURE_{feature.upper()}"
            if env_var in os.environ:
                self._features[feature] = os.environ[env_var].lower() in ('true', '1', 'yes')
    
    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self._features.get(feature, False)
    
    def enable(self, feature: str):
        """Enable a feature."""
        if feature in self._features:
            self._features[feature] = True
    
    def disable(self, feature: str):
        """Disable a feature."""
        if feature in self._features:
            self._features[feature] = False
    
    def list_enabled(self) -> list:
        """List all enabled features."""
        return [f for f, enabled in self._features.items() if enabled]


registry = FeatureRegistry()


def create_unified_app(port: int = 8050) -> dash.Dash:
    """
    Create unified Dash application with all features.
    
    Parameters
    ----------
    port : int
        Port to run the server on
        
    Returns
    -------
    dash.Dash
        Unified Dash application
    """
    server = Flask(__name__)
    
    app = dash.Dash(
        __name__,
        server=server,
        title="Neural DSL - Unified Interface",
        external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True
    )
    
    # Create navigation bar
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Debug", href="/debug", id="nav-debug")),
            dbc.NavItem(dbc.NavLink("Build", href="/build", id="nav-build")),
            dbc.NavItem(dbc.NavLink("Monitor", href="/monitor", id="nav-monitor")),
            dbc.NavItem(dbc.NavLink("Docs", href="/docs", id="nav-docs")),
        ],
        brand="Neural DSL",
        brand_href="/",
        color="dark",
        dark=True,
        className="mb-2"
    )
    
    # Create tabs for different features
    tabs = dbc.Tabs([
        dbc.Tab(
            label="🐛 Debug",
            tab_id="debug",
            disabled=not registry.is_enabled('debug')
        ),
        dbc.Tab(
            label="🏗️ Build",
            tab_id="build",
            disabled=not registry.is_enabled('nocode')
        ),
        dbc.Tab(
            label="📊 Monitor",
            tab_id="monitor",
            disabled=not registry.is_enabled('monitoring')
        ),
        dbc.Tab(
            label="⚙️ Settings",
            tab_id="settings"
        ),
    ], id="main-tabs", active_tab="debug")
    
    # Main layout
    app.layout = html.Div([
        navbar,
        dbc.Container([
            tabs,
            html.Div(id="tab-content", className="mt-4")
        ], fluid=True)
    ])
    
    # Tab content callback
    @app.callback(
        dash.dependencies.Output("tab-content", "children"),
        [dash.dependencies.Input("main-tabs", "active_tab")]
    )
    def render_tab_content(active_tab):
        if active_tab == "debug" and registry.is_enabled('debug'):
            return get_debug_layout()
        elif active_tab == "build" and registry.is_enabled('nocode'):
            return get_build_layout()
        elif active_tab == "monitor" and registry.is_enabled('monitoring'):
            return get_monitor_layout()
        elif active_tab == "settings":
            return get_settings_layout()
        else:
            return html.Div([
                html.H3("Feature Not Available"),
                html.P(f"The {active_tab} feature is not enabled."),
                html.P("Enable it in your configuration or contact your administrator.")
            ])
    
    return app


def get_debug_layout():
    """Get layout for debug dashboard (NeuralDbg)."""
    try:
        from neural.dashboard.dashboard import (
            trace_data, model_data,
            create_progress_component
        )
        
        return html.Div([
            html.H2("NeuralDbg - Real-Time Execution Monitoring"),
            html.P("Monitor neural network execution in real-time with detailed layer analysis."),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Model Architecture"),
                    dcc.Graph(id="debug-architecture"),
                ], width=6),
                dbc.Col([
                    html.H4("Layer Performance"),
                    dcc.Graph(id="debug-performance"),
                ], width=6),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Resource Monitoring"),
                    dcc.Graph(id="debug-resources"),
                ], width=12),
            ]),
            
            dcc.Interval(id="debug-interval", interval=1000, n_intervals=0),
        ])
    except ImportError as e:
        logger.warning(f"Debug dashboard components not available: {e}")
        return html.Div([
            html.H3("Debug Dashboard"),
            html.P("Debug dashboard requires additional dependencies."),
            html.Pre("pip install -e \".[dashboard]\"")
        ])


def get_build_layout():
    """Get layout for no-code builder."""
    try:
        return html.Div([
            html.H2("Model Builder - No-Code Interface"),
            html.P("Build neural networks visually without writing code."),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Layer Palette"),
                    html.Div(id="layer-palette", children=[
                        dbc.Button("Conv2D", color="primary", className="m-1"),
                        dbc.Button("MaxPooling2D", color="secondary", className="m-1"),
                        dbc.Button("Dense", color="info", className="m-1"),
                        dbc.Button("Dropout", color="warning", className="m-1"),
                    ])
                ], width=3),
                dbc.Col([
                    html.H4("Model Canvas"),
                    html.Div(id="model-canvas", style={
                        'border': '2px dashed #666',
                        'minHeight': '400px',
                        'padding': '20px'
                    }, children=[
                        html.P("Drag layers here to build your model", 
                              style={'color': '#888', 'textAlign': 'center'})
                    ])
                ], width=6),
                dbc.Col([
                    html.H4("Layer Properties"),
                    html.Div(id="layer-properties")
                ], width=3),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Generate Code", color="success", className="m-2"),
                    dbc.Button("Save Model", color="primary", className="m-2"),
                    dbc.Button("Load Model", color="info", className="m-2"),
                ])
            ], className="mt-3"),
        ])
    except Exception as e:
        logger.warning(f"Build interface not available: {e}")
        return html.Div([
            html.H3("Model Builder"),
            html.P("No-code interface requires additional setup.")
        ])


def get_monitor_layout():
    """Get layout for production monitoring."""
    try:
        return html.Div([
            html.H2("Production Monitoring"),
            html.P("Monitor deployed models for drift, quality, and performance."),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Predictions", className="card-title"),
                            html.H2("1,234,567", id="monitor-predictions"),
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Error Rate", className="card-title"),
                            html.H2("0.12%", id="monitor-error-rate"),
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Drift Score", className="card-title"),
                            html.H2("0.045", id="monitor-drift"),
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Latency P95", className="card-title"),
                            html.H2("45ms", id="monitor-latency"),
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="monitor-drift-chart"),
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="monitor-quality-chart"),
                ], width=6),
            ]),
            
            dcc.Interval(id="monitor-interval", interval=5000, n_intervals=0),
        ])
    except Exception as e:
        logger.warning(f"Monitoring dashboard not available: {e}")
        return html.Div([
            html.H3("Monitoring Dashboard"),
            html.P("Monitoring requires additional dependencies.")
        ])


def get_settings_layout():
    """Get layout for settings and configuration."""
    enabled_features = registry.list_enabled()
    
    feature_toggles = []
    for feature in ['debug', 'nocode', 'monitoring', 'api', 'collaboration']:
        feature_toggles.append(
            dbc.Row([
                dbc.Col(html.Label(feature.title()), width=3),
                dbc.Col(
                    dbc.Switch(
                        id=f"feature-{feature}",
                        value=registry.is_enabled(feature),
                        disabled=True  # Requires restart
                    ),
                    width=2
                ),
                dbc.Col(html.Small("(Requires restart)", style={'color': '#888'}), width=7),
            ], className="mb-2")
        )
    
    return html.Div([
        html.H2("Settings"),
        
        dbc.Card([
            dbc.CardHeader("Feature Toggles"),
            dbc.CardBody([
                html.P("Enable or disable optional features. Changes require server restart."),
                html.Div(feature_toggles),
                html.Hr(),
                html.P([
                    html.Strong("Note: "),
                    "Set environment variables or use configuration file for persistence."
                ], style={'color': '#888'})
            ])
        ], className="mb-3"),
        
        dbc.Card([
            dbc.CardHeader("Server Information"),
            dbc.CardBody([
                html.Dl([
                    html.Dt("Version"),
                    html.Dd("0.3.0"),
                    html.Dt("Active Features"),
                    html.Dd(", ".join(enabled_features)),
                    html.Dt("Server Port"),
                    html.Dd("8050"),
                ])
            ])
        ]),
    ])


def start_server(
    host: str = 'localhost',
    port: int = 8050,
    debug: bool = False,
    features: Optional[list] = None
):
    """
    Start the unified Neural DSL server.
    
    Parameters
    ----------
    host : str
        Host address
    port : int
        Port number
    debug : bool
        Debug mode
    features : Optional[list]
        List of features to enable
    """
    if features:
        # Enable only specified features
        for feature in registry._features.keys():
            registry.disable(feature)
        for feature in features:
            registry.enable(feature)
    
    logger.info("=" * 60)
    logger.info("Neural DSL - Unified Server")
    logger.info("=" * 60)
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Enabled features: {', '.join(registry.list_enabled())}")
    logger.info(f"Open browser: http://{host}:{port}")
    logger.info("=" * 60)
    
    app = create_unified_app(port)
    app.run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural DSL Unified Server")
    parser.add_argument('--host', default='localhost', help='Host address')
    parser.add_argument('--port', type=int, default=8050, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--features', nargs='+', help='Features to enable')
    
    args = parser.parse_args()
    start_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
        features=args.features
    )
