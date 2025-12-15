#!/usr/bin/env python
"""
Neural Aquarium IDE - Integrated Development Environment for Neural DSL
A modern IDE with model compilation, execution, and debugging capabilities
"""

import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from flask import jsonify

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from neural.aquarium.src.components.runner import RunnerPanel
except ImportError as e:
    print(f"Warning: Could not import RunnerPanel: {e}")
    RunnerPanel = None

try:
    from neural.aquarium.examples import get_random_example
except ImportError as e:
    print(f"Warning: Could not import examples: {e}")
    def get_random_example():
        return """network DefaultModel {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}"""

try:
    from neural.parser.parser import ModelTransformer, create_parser
except ImportError as e:
    print(f"Warning: Could not import parser: {e}")
    ModelTransformer = None
    create_parser = None


NEURAL_ASCII = r"""
    _   __                      __    ___                            _                 
   / | / /__  __  ___________  / /   /   | ____ ___  ______ ______(_)_  ______ ___   
  /  |/ / _ \/ / / / ___/ __ \/ /   / /| |/ __ `/ / / / __ `/ ___/ / / / / __ `__ \  
 / /|  /  __/ /_/ / /  / /_/ / /   / ___ / /_/ / /_/ / /_/ / /  / / /_/ / / / / / /  
/_/ |_/\___/\__,_/_/   \____/_/   /_/  |_\__, /\__,_/\__,_/_/  /_/\__,_/_/ /_/ /_/   
                                            /_/                                        
"""

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.FONT_AWESOME
    ],
    suppress_callback_exceptions=True,
    title="Neural Aquarium IDE"
)

# Get Flask server for health check endpoints
server = app.server

# Add health check endpoints
try:
    from neural.config.health import HealthChecker
    
    @server.route('/health')
    def health_check():
        """Health check endpoint for Aquarium service."""
        return jsonify({
            "status": "healthy",
            "service": "aquarium",
            "version": "0.3.0"
        }), 200
    
    @server.route('/health/live')
    def liveness_probe():
        """Kubernetes liveness probe."""
        health_checker = HealthChecker()
        if health_checker.get_liveness_status():
            return jsonify({"status": "alive"}), 200
        return jsonify({"status": "dead"}), 503
    
    @server.route('/health/ready')
    def readiness_probe():
        """Kubernetes readiness probe."""
        health_checker = HealthChecker()
        if health_checker.get_readiness_status(['aquarium']):
            return jsonify({"status": "ready"}), 200
        return jsonify({"status": "not ready"}), 503
except ImportError:
    # Health checker not available, add basic health endpoint
    @server.route('/health')
    def health_check():
        """Basic health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "aquarium",
            "version": "0.3.0"
        }), 200

# Initialize runner panel if available
runner_panel = RunnerPanel(app) if RunnerPanel else None

app.layout = html.Div([
    html.Div([
        html.Pre(
            NEURAL_ASCII,
            style={
                'color': '#00BFFF',
                'fontSize': '10px',
                'fontFamily': 'monospace',
                'whiteSpace': 'pre',
                'margin': '10px 20px',
                'lineHeight': '1.2'
            }
        ),
        html.Div([
            html.H4("Neural Aquarium IDE", style={'color': 'white', 'margin': '0'}),
            html.P(
                "Integrated Development Environment for Neural DSL",
                style={'color': '#888', 'margin': '0', 'fontSize': '0.9em'}
            )
        ], style={'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '20px'}),
        
        html.Div([
            dbc.ButtonGroup([
                dbc.Button([html.I(className="fas fa-file me-1"), "New"], color="primary", size="sm"),
                dbc.Button([html.I(className="fas fa-folder-open me-1"), "Open"], color="secondary", size="sm"),
                dbc.Button([html.I(className="fas fa-save me-1"), "Save"], color="success", size="sm"),
            ], className="me-2"),
            dbc.ButtonGroup([
                dbc.Button([html.I(className="fas fa-question-circle me-1"), "Help"], color="info", size="sm"),
                dbc.Button([html.I(className="fas fa-cog me-1"), "Settings"], color="secondary", size="sm"),
            ])
        ], style={'float': 'right', 'marginTop': '20px', 'marginRight': '20px'})
    ], style={
        'backgroundColor': '#1a1a1a',
        'padding': '10px 0',
        'borderBottom': '2px solid #00BFFF'
    }),
    
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-code me-2"),
                        "DSL Editor"
                    ]),
                    dbc.CardBody([
                        dcc.Textarea(
                            id="dsl-editor",
                            placeholder="""network MyModel {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}""",
                            style={
                                'width': '100%',
                                'height': '400px',
                                'fontFamily': 'Consolas, Monaco, monospace',
                                'fontSize': '0.9em',
                                'backgroundColor': '#1e1e1e',
                                'color': '#d4d4d4',
                                'border': '1px solid #444',
                                'padding': '10px',
                                'resize': 'vertical'
                            }
                        ),
                        html.Div([
                            dbc.Button(
                                [html.I(className="fas fa-check me-2"), "Parse DSL"],
                                id="parse-dsl-btn",
                                color="primary",
                                className="mt-2 me-2"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-eye me-2"), "Visualize"],
                                id="visualize-btn",
                                color="info",
                                className="mt-2 me-2"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-file-import me-2"), "Load Example"],
                                id="load-example-btn",
                                color="secondary",
                                className="mt-2"
                            )
                        ]),
                        html.Div(id="parse-status", className="mt-3")
                    ])
                ], className="mb-3"),
                
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-info-circle me-2"),
                        "Model Information"
                    ]),
                    dbc.CardBody([
                        html.Div(id="model-info", children=[
                            html.P("Parse a DSL model to see information here.", className="text-muted")
                        ])
                    ])
                ])
            ], width=4),
            
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(
                        label="Runner",
                        children=[
                            runner_panel.create_layout() if runner_panel else html.Div([
                                html.P("Runner panel unavailable. Please check installation.", 
                                       className="text-warning p-3")
                            ])
                        ],
                        tab_id="tab-runner"
                    ),
                    dbc.Tab(
                        label="Debugger",
                        children=[
                            html.Div([
                                html.H5("NeuralDbg Integration", className="mt-3"),
                                html.P("Debug your models with real-time visualization and metrics."),
                                dbc.Button(
                                    [html.I(className="fas fa-bug me-2"), "Launch NeuralDbg"],
                                    color="danger",
                                    className="mt-2"
                                )
                            ], className="p-3")
                        ],
                        tab_id="tab-debugger"
                    ),
                    dbc.Tab(
                        label="Visualization",
                        children=[
                            html.Div([
                                html.H5("Model Architecture", className="mt-3"),
                                dcc.Graph(
                                    id="architecture-graph",
                                    style={'height': '400px'}
                                )
                            ], className="p-3")
                        ],
                        tab_id="tab-visualization"
                    ),
                    dbc.Tab(
                        label="Documentation",
                        children=[
                            html.Div([
                                html.H5("Neural DSL Documentation", className="mt-3"),
                                html.P("Quick reference for Neural DSL syntax and features."),
                                html.Hr(),
                                html.H6("Basic Structure"),
                                html.Pre("""network ModelName {
    input: (batch, height, width, channels)
    layers:
        LayerType(param=value, ...)
        ...
    loss: loss_function
    optimizer: OptimizerName(param=value, ...)
}""", style={'backgroundColor': '#2a2a2a', 'padding': '10px', 'borderRadius': '5px'}),
                                html.Hr(),
                                html.H6("Available Layers"),
                                html.Ul([
                                    html.Li("Convolutional: Conv1D, Conv2D, Conv3D, SeparableConv2D"),
                                    html.Li("Pooling: MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D"),
                                    html.Li("Core: Dense, Flatten, Dropout, BatchNormalization"),
                                    html.Li("Recurrent: LSTM, GRU, SimpleRNN"),
                                    html.Li("Attention: MultiHeadAttention"),
                                ])
                            ], className="p-3")
                        ],
                        tab_id="tab-docs"
                    )
                ], id="main-tabs", active_tab="tab-runner")
            ], width=8)
        ], className="mt-3")
    ], fluid=True),
    
    dcc.Store(id="parsed-model-data"),
    dcc.Store(id="dsl-content-store")
])


@app.callback(
    [Output("parse-status", "children"),
     Output("model-info", "children"),
     Output("parsed-model-data", "data"),
     Output("dsl-content-store", "data"),
     Output("runner-model-data", "data"),
     Output("runner-dsl-content", "data")],
    Input("parse-dsl-btn", "n_clicks"),
    State("dsl-editor", "value")
)
def parse_dsl(n_clicks, dsl_content):
    if not n_clicks or not dsl_content:
        return (
            None,
            html.P("Parse a DSL model to see information here.", className="text-muted"),
            None,
            None,
            None,
            None
        )
    
    if not create_parser or not ModelTransformer:
        status = dbc.Alert(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Parser not available. Please check installation."
            ],
            color="warning",
            dismissable=True
        )
        return (
            status,
            html.P("Parser module not available.", className="text-warning"),
            None,
            None,
            None,
            None
        )
    
    try:
        parser = create_parser(start_rule='network')
        tree = parser.parse(dsl_content)
        model_data = ModelTransformer().transform(tree)
        
        status = dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"), "DSL parsed successfully!"],
            color="success",
            dismissable=True
        )
        
        input_shape = model_data.get('input', {}).get('shape', 'Not specified')
        layers = model_data.get('layers', [])
        loss = model_data.get('loss', {}).get('value', 'Not specified')
        optimizer = model_data.get('optimizer', {})
        
        info = html.Div([
            html.H6("Model Details", className="mb-3"),
            html.P([html.Strong("Input Shape: "), str(input_shape)]),
            html.P([html.Strong("Number of Layers: "), str(len(layers))]),
            html.P([html.Strong("Loss Function: "), str(loss)]),
            html.P([html.Strong("Optimizer: "), str(optimizer.get('type', 'Not specified'))]),
            html.Hr(),
            html.H6("Layer Summary", className="mb-2"),
            html.Ul([
                html.Li(f"{i+1}. {layer.get('type', 'Unknown')}")
                for i, layer in enumerate(layers)
            ], style={'maxHeight': '200px', 'overflowY': 'auto'})
        ])
        
        return status, info, model_data, dsl_content, model_data, dsl_content
        
    except Exception as e:
        status = dbc.Alert(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Parse error: {str(e)}"
            ],
            color="danger",
            dismissable=True
        )
        
        return (
            status,
            html.P("Failed to parse DSL. Check syntax.", className="text-danger"),
            None,
            None,
            None,
            None
        )


@app.callback(
    Output("dsl-editor", "value"),
    Input("load-example-btn", "n_clicks"),
    prevent_initial_call=True
)
def load_example_callback(n_clicks):
    """Load a random example into the editor."""
    if not n_clicks:
        return dash.no_update
    
    try:
        example_code = get_random_example()
        if example_code:
            return example_code
        else:
            print("Warning: No example code returned")
            return dash.no_update
    except Exception as e:
        print(f"Error loading example: {e}")
        import traceback
        traceback.print_exc()
        return dash.no_update


def main(port=8052, debug=False, host="0.0.0.0"):
    """
    Start the Neural Aquarium IDE.
    
    Parameters
    ----------
    port : int
        Port to run the server on (default: 8052)
    debug : bool
        Enable debug mode (default: False)
    host : str
        Host to bind to (default: "0.0.0.0")
    """
    print("="*70)
    print(NEURAL_ASCII)
    print("="*70)
    print(f"\n🚀 Starting Neural Aquarium IDE on http://localhost:{port}")
    print(f"   Backend: Dash + Plotly")
    print(f"   Debug Mode: {debug}")
    print(f"   Host: {host}")
    print("\n📝 Features:")
    print("   • DSL Editor with syntax validation")
    print("   • Model Compilation (TensorFlow, PyTorch, ONNX)")
    print("   • Execution Panel with live logs")
    print("   • Dataset Selection (MNIST, CIFAR10, CIFAR100, ImageNet)")
    print("   • Export and IDE Integration")
    print(f"\n🌐 Open your browser to: http://localhost:{port}")
    print(f"   Health endpoint: http://localhost:{port}/health")
    print("   Press Ctrl+C to stop the server\n")
    print("="*70)
    
    try:
        app.run_server(debug=debug, host=host, port=port)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Aquarium IDE")
    parser.add_argument("--port", type=int, default=8052, help="Port to run the server on (default: 8052)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    main(port=args.port, debug=args.debug, host=args.host)
