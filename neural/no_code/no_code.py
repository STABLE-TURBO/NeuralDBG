from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
import dash
from dash_bootstrap_components import themes, Navbar, NavItem, NavLink, Container, Row, Col, Card, Button, Spinner, Tooltip
import dash_bootstrap_components as dbc
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import uuid
import time
from neural.parser.parser import create_parser, ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.code_generation.code_generator import generate_code
from neural.visualization.static_visualizer.visualizer import NeuralVisualizer

# ASCII Art for welcome message
NEURAL_ASCII = r"""
    _   __                      __   ____  _____ __
   / | / /__  __  ___________  / /  / __ \/ ___// /
  /  |/ / _ \/ / / / ___/ __ \/ /  / / / /\__ \/ /
 / /|  /  __/ /_/ / /  / /_/ / /  / /_/ /___/ / /___
/_/ |_/\___/\__,_/_/   \____/_/   \____//____/_____/

"""

# Create a directory for saving models if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_models'), exist_ok=True)

# Initialize the Dash app with the dark theme
app = Dash(
    __name__,
    external_stylesheets=[themes.DARKLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

# Define all available layer types with their parameters
LAYER_TYPES = {
    "Convolutional": [
        {"label": "Conv1D", "value": "Conv1D"},
        {"label": "Conv2D", "value": "Conv2D"},
        {"label": "Conv3D", "value": "Conv3D"},
        {"label": "SeparableConv2D", "value": "SeparableConv2D"},
        {"label": "DepthwiseConv2D", "value": "DepthwiseConv2D"},
        {"label": "TransposedConv2D", "value": "TransposedConv2D"},
    ],
    "Pooling": [
        {"label": "MaxPooling1D", "value": "MaxPooling1D"},
        {"label": "MaxPooling2D", "value": "MaxPooling2D"},
        {"label": "AveragePooling1D", "value": "AveragePooling1D"},
        {"label": "AveragePooling2D", "value": "AveragePooling2D"},
        {"label": "GlobalMaxPooling1D", "value": "GlobalMaxPooling1D"},
        {"label": "GlobalMaxPooling2D", "value": "GlobalMaxPooling2D"},
        {"label": "GlobalAveragePooling1D", "value": "GlobalAveragePooling1D"},
        {"label": "GlobalAveragePooling2D", "value": "GlobalAveragePooling2D"},
    ],
    "Core": [
        {"label": "Dense", "value": "Dense"},
        {"label": "Flatten", "value": "Flatten"},
        {"label": "Reshape", "value": "Reshape"},
        {"label": "Permute", "value": "Permute"},
        {"label": "RepeatVector", "value": "RepeatVector"},
        {"label": "Lambda", "value": "Lambda"},
    ],
    "Normalization": [
        {"label": "BatchNormalization", "value": "BatchNormalization"},
        {"label": "LayerNormalization", "value": "LayerNormalization"},
        {"label": "GroupNormalization", "value": "GroupNormalization"},
    ],
    "Regularization": [
        {"label": "Dropout", "value": "Dropout"},
        {"label": "SpatialDropout1D", "value": "SpatialDropout1D"},
        {"label": "SpatialDropout2D", "value": "SpatialDropout2D"},
        {"label": "GaussianNoise", "value": "GaussianNoise"},
        {"label": "GaussianDropout", "value": "GaussianDropout"},
        {"label": "ActivityRegularization", "value": "ActivityRegularization"},
    ],
    "Recurrent": [
        {"label": "LSTM", "value": "LSTM"},
        {"label": "GRU", "value": "GRU"},
        {"label": "SimpleRNN", "value": "SimpleRNN"},
        {"label": "Bidirectional", "value": "Bidirectional"},
        {"label": "ConvLSTM2D", "value": "ConvLSTM2D"},
    ],
    "Attention": [
        {"label": "MultiHeadAttention", "value": "MultiHeadAttention"},
        {"label": "Attention", "value": "Attention"},
    ],
    "Embedding": [
        {"label": "Embedding", "value": "Embedding"},
    ],
    "Activation": [
        {"label": "ReLU", "value": "ReLU"},
        {"label": "LeakyReLU", "value": "LeakyReLU"},
        {"label": "PReLU", "value": "PReLU"},
        {"label": "ELU", "value": "ELU"},
        {"label": "ThresholdedReLU", "value": "ThresholdedReLU"},
        {"label": "Softmax", "value": "Softmax"},
        {"label": "Sigmoid", "value": "Sigmoid"},
        {"label": "Tanh", "value": "Tanh"},
    ],
    "Output": [
        {"label": "Output", "value": "Output"},
    ],
}

# Define default parameters for each layer type
DEFAULT_PARAMS = {
    "Conv1D": {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "valid"},
    "Conv2D": {"filters": 32, "kernel_size": (3, 3), "activation": "relu", "padding": "valid"},
    "Conv3D": {"filters": 32, "kernel_size": (3, 3, 3), "activation": "relu", "padding": "valid"},
    "SeparableConv2D": {"filters": 32, "kernel_size": (3, 3), "activation": "relu"},
    "DepthwiseConv2D": {"kernel_size": (3, 3), "depth_multiplier": 1, "activation": "relu"},
    "TransposedConv2D": {"filters": 32, "kernel_size": (3, 3), "strides": (1, 1), "padding": "valid"},
    "MaxPooling1D": {"pool_size": 2, "strides": None, "padding": "valid"},
    "MaxPooling2D": {"pool_size": (2, 2), "strides": None, "padding": "valid"},
    "AveragePooling1D": {"pool_size": 2, "strides": None, "padding": "valid"},
    "AveragePooling2D": {"pool_size": (2, 2), "strides": None, "padding": "valid"},
    "GlobalMaxPooling1D": {},
    "GlobalMaxPooling2D": {},
    "GlobalAveragePooling1D": {},
    "GlobalAveragePooling2D": {},
    "Dense": {"units": 128, "activation": "relu"},
    "Flatten": {},
    "Reshape": {"target_shape": (1, 1, -1)},
    "Permute": {"dims": (2, 1)},
    "RepeatVector": {"n": 3},
    "Lambda": {"function": "lambda x: x"},
    "BatchNormalization": {"axis": -1, "momentum": 0.99, "epsilon": 0.001},
    "LayerNormalization": {"axis": -1, "epsilon": 0.001},
    "GroupNormalization": {"groups": 32, "axis": -1, "epsilon": 0.001},
    "Dropout": {"rate": 0.5},
    "SpatialDropout1D": {"rate": 0.5},
    "SpatialDropout2D": {"rate": 0.5},
    "GaussianNoise": {"stddev": 0.1},
    "GaussianDropout": {"rate": 0.5},
    "ActivityRegularization": {"l1": 0.0, "l2": 0.0},
    "LSTM": {"units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "return_sequences": False},
    "GRU": {"units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "return_sequences": False},
    "SimpleRNN": {"units": 128, "activation": "tanh", "return_sequences": False},
    "Bidirectional": {"layer": "LSTM", "merge_mode": "concat"},
    "ConvLSTM2D": {"filters": 32, "kernel_size": (3, 3), "activation": "tanh", "recurrent_activation": "hard_sigmoid"},
    "MultiHeadAttention": {"num_heads": 8, "key_dim": 64},
    "Attention": {},
    "Embedding": {"input_dim": 1000, "output_dim": 64},
    "ReLU": {"max_value": None, "negative_slope": 0.0, "threshold": 0.0},
    "LeakyReLU": {"alpha": 0.3},
    "PReLU": {},
    "ELU": {"alpha": 1.0},
    "ThresholdedReLU": {"theta": 1.0},
    "Softmax": {"axis": -1},
    "Sigmoid": {},
    "Tanh": {},
    "Output": {"units": 10, "activation": "softmax"},
}

# Define optimizer options
OPTIMIZERS = [
    {"label": "Adam", "value": "Adam"},
    {"label": "SGD", "value": "SGD"},
    {"label": "RMSprop", "value": "RMSprop"},
    {"label": "Adagrad", "value": "Adagrad"},
    {"label": "Adadelta", "value": "Adadelta"},
    {"label": "Adamax", "value": "Adamax"},
    {"label": "Nadam", "value": "Nadam"},
]

# Define loss function options
LOSS_FUNCTIONS = [
    {"label": "Categorical Crossentropy", "value": "categorical_crossentropy"},
    {"label": "Binary Crossentropy", "value": "binary_crossentropy"},
    {"label": "Mean Squared Error", "value": "mse"},
    {"label": "Mean Absolute Error", "value": "mae"},
    {"label": "Sparse Categorical Crossentropy", "value": "sparse_categorical_crossentropy"},
    {"label": "Kullback-Leibler Divergence", "value": "kld"},
    {"label": "Huber Loss", "value": "huber_loss"},
    {"label": "Log Cosh", "value": "log_cosh"},
]

# Define default optimizer parameters
DEFAULT_OPTIMIZER_PARAMS = {
    "Adam": {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7},
    "SGD": {"learning_rate": 0.01, "momentum": 0.0, "nesterov": False},
    "RMSprop": {"learning_rate": 0.001, "rho": 0.9, "epsilon": 1e-7},
    "Adagrad": {"learning_rate": 0.01, "epsilon": 1e-7},
    "Adadelta": {"learning_rate": 1.0, "rho": 0.95, "epsilon": 1e-7},
    "Adamax": {"learning_rate": 0.002, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7},
    "Nadam": {"learning_rate": 0.002, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7},
}

# Define common input shapes
COMMON_INPUT_SHAPES = [
    {"label": "MNIST (28x28x1)", "value": "(None, 28, 28, 1)"},
    {"label": "CIFAR-10 (32x32x3)", "value": "(None, 32, 32, 3)"},
    {"label": "ImageNet (224x224x3)", "value": "(None, 224, 224, 3)"},
    {"label": "Custom", "value": "custom"},
]

# Define model templates
MODEL_TEMPLATES = [
    {"label": "Simple CNN for MNIST", "value": "mnist_cnn"},
    {"label": "VGG-like for CIFAR-10", "value": "cifar10_vgg"},
    {"label": "Simple LSTM for Text", "value": "text_lstm"},
    {"label": "Transformer Encoder", "value": "transformer_encoder"},
    {"label": "Empty Model", "value": "empty"},
]

# Define the app layout with a modern dashboard structure
app.layout = html.Div([
    # Header with logo and title
    html.Div([
        html.Pre(NEURAL_ASCII, style={'color': '#00BFFF', 'fontSize': '12px', 'fontFamily': 'monospace', 'whiteSpace': 'pre', 'margin': '0'}),
        html.H3("No-Code Interface", style={'color': 'white', 'display': 'inline-block', 'marginLeft': '20px'}),
        html.Div([
            dbc.Button("Save Model", id="save-model-btn", color="success", className="me-2"),
            dbc.Button("Load Model", id="load-model-btn", color="info", className="me-2"),
            dbc.Button("Export DSL", id="export-dsl-btn", color="warning", className="me-2"),
            dbc.Button("Help", id="help-btn", color="secondary", className="me-2"),
        ], style={'float': 'right', 'marginTop': '10px'}),
    ], style={'backgroundColor': '#222', 'padding': '10px', 'borderBottom': '1px solid #444'}),

    # Main content area with sidebar and workspace
    dbc.Container(fluid=True, children=[
        dbc.Row([
            # Sidebar for layer selection and model configuration
            dbc.Col([
                html.Div([
                    html.H4("Model Configuration", className="mt-3"),

                    # Model template selection
                    html.Label("Start with Template:"),
                    dcc.Dropdown(
                        id="model-template",
                        options=MODEL_TEMPLATES,
                        value="empty",
                        className="mb-3"
                    ),

                    # Input shape configuration
                    html.Label("Input Shape:"),
                    dcc.Dropdown(
                        id="input-shape-preset",
                        options=COMMON_INPUT_SHAPES,
                        value="(None, 28, 28, 1)",
                        className="mb-2"
                    ),
                    dcc.Input(
                        id="custom-input-shape",
                        type="text",
                        placeholder="e.g., (None, 224, 224, 3)",
                        className="mb-3",
                        style={'display': 'none', 'width': '100%'}
                    ),

                    # Layer selection
                    html.H4("Add Layers", className="mt-3"),

                    # Layer category tabs
                    dbc.Tabs(id="layer-category-tabs", children=[
                        dbc.Tab(label=category, children=[
                            dcc.Dropdown(
                                id=f"layer-type-{category.lower()}",
                                options=layers,
                                placeholder=f"Select {category} Layer",
                                className="mb-2"
                            ),
                        ]) for category, layers in LAYER_TYPES.items()
                    ]),

                    dbc.Button("Add Selected Layer", id="add-layer", color="primary", className="mt-3 mb-3 w-100"),

                    # Training configuration
                    html.H4("Training Configuration", className="mt-3"),

                    # Loss function selection
                    html.Label("Loss Function:"),
                    dcc.Dropdown(
                        id="loss-function",
                        options=LOSS_FUNCTIONS,
                        value="categorical_crossentropy",
                        className="mb-3"
                    ),

                    # Optimizer selection
                    html.Label("Optimizer:"),
                    dcc.Dropdown(
                        id="optimizer",
                        options=OPTIMIZERS,
                        value="Adam",
                        className="mb-3"
                    ),

                    # Optimizer parameters
                    html.Div(id="optimizer-params-container", children=[
                        dash_table.DataTable(
                            id="optimizer-params",
                            columns=[
                                {"name": "Parameter", "id": "param"},
                                {"name": "Value", "id": "value"}
                            ],
                            data=[
                                {"param": param, "value": str(value)}
                                for param, value in DEFAULT_OPTIMIZER_PARAMS["Adam"].items()
                            ],
                            editable=True,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                            style_header={'backgroundColor': '#444', 'color': 'white'},
                            style_data={'backgroundColor': '#333', 'color': 'white'},
                        )
                    ]),

                    # Compile and debug buttons
                    html.Div([
                        dbc.Button("Compile Model", id="compile-btn", color="success", className="mt-3 w-100"),
                        dbc.Button("Debug with NeuralDbg", id="debug-btn", color="warning", className="mt-2 w-100"),
                    ], className="mt-3 mb-3"),

                ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px', 'height': '100%'})
            ], width=3, style={'height': 'calc(100vh - 100px)', 'overflowY': 'auto'}),

            # Main workspace
            dbc.Col([
                dbc.Tabs([
                    # Model Builder Tab
                    dbc.Tab(label="Model Builder", children=[
                        html.Div([
                            # Layer list
                            html.Div([
                                html.H4("Model Layers", className="mt-3"),
                                html.Div(id="layer-list-container", children=[
                                    html.Div(id="layer-list", children=[], style={'minHeight': '200px'})
                                ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'})
                            ], style={'flex': '1', 'marginRight': '20px'}),

                            # Layer properties
                            html.Div([
                                html.H4("Layer Properties", className="mt-3"),
                                html.Div(id="layer-properties-container", children=[
                                    dash_table.DataTable(
                                        id="layer-params",
                                        columns=[
                                            {"name": "Parameter", "id": "param"},
                                            {"name": "Value", "id": "value"}
                                        ],
                                        data=[],
                                        editable=True,
                                        style_table={'overflowX': 'auto'},
                                        style_cell={'textAlign': 'left'},
                                        style_header={'backgroundColor': '#444', 'color': 'white'},
                                        style_data={'backgroundColor': '#333', 'color': 'white'},
                                    )
                                ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px'})
                            ], style={'flex': '1'})
                        ], style={'display': 'flex', 'flexDirection': 'row'})
                    ]),

                    # Visualization Tab
                    dbc.Tab(label="Visualization", children=[
                        html.Div([
                            html.H4("Model Architecture", className="mt-3"),
                            html.Div(id="visualization-container", children=[
                                dcc.Loading(
                                    id="loading-visualization",
                                    type="circle",
                                    children=[
                                        html.Div(id="architecture-visualization", style={'height': '400px'})
                                    ]
                                )
                            ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'}),

                            html.H4("Shape Propagation", className="mt-3"),
                            html.Div(id="shape-propagation-container", children=[
                                dcc.Loading(
                                    id="loading-shape-propagation",
                                    type="circle",
                                    children=[
                                        dcc.Graph(id="shape-propagation-graph", style={'height': '300px'})
                                    ]
                                )
                            ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px'})
                        ])
                    ]),

                    # Code Generation Tab
                    dbc.Tab(label="Generated Code", children=[
                        html.Div([
                            html.H4("Neural DSL Code", className="mt-3"),
                            html.Div(id="dsl-code-container", children=[
                                dcc.Loading(
                                    id="loading-dsl-code",
                                    type="circle",
                                    children=[
                                        html.Pre(id="dsl-code", style={'backgroundColor': '#222', 'color': '#00FF00', 'padding': '15px', 'borderRadius': '5px', 'whiteSpace': 'pre-wrap'})
                                    ]
                                )
                            ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'}),

                            dbc.Tabs([
                                dbc.Tab(label="TensorFlow Code", children=[
                                    html.Div(id="tensorflow-code-container", children=[
                                        dcc.Loading(
                                            id="loading-tensorflow-code",
                                            type="circle",
                                            children=[
                                                html.Pre(id="tensorflow-code", style={'backgroundColor': '#222', 'color': '#00FF00', 'padding': '15px', 'borderRadius': '5px', 'whiteSpace': 'pre-wrap'})
                                            ]
                                        )
                                    ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px'})
                                ]),
                                dbc.Tab(label="PyTorch Code", children=[
                                    html.Div(id="pytorch-code-container", children=[
                                        dcc.Loading(
                                            id="loading-pytorch-code",
                                            type="circle",
                                            children=[
                                                html.Pre(id="pytorch-code", style={'backgroundColor': '#222', 'color': '#00FF00', 'padding': '15px', 'borderRadius': '5px', 'whiteSpace': 'pre-wrap'})
                                            ]
                                        )
                                    ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px'})
                                ])
                            ])
                        ])
                    ]),

                    # Debug Tab
                    dbc.Tab(label="Debug", children=[
                        html.Div([
                            html.H4("NeuralDbg Integration", className="mt-3"),
                            html.P("Connect to NeuralDbg to debug your model in real-time."),
                            dbc.Button("Launch NeuralDbg", id="launch-neuraldbg-btn", color="danger", className="mt-2"),
                            html.Div(id="neuraldbg-status", className="mt-3")
                        ], style={'backgroundColor': '#333', 'padding': '15px', 'borderRadius': '5px'})
                    ])
                ])
            ], width=9, style={'height': 'calc(100vh - 100px)', 'overflowY': 'auto'})
        ])
    ]),

    # Store components for maintaining state
    dcc.Store(id="model-layers", data=[]),
    dcc.Store(id="selected-layer-index", data=None),
    dcc.Store(id="input-shape", data="(None, 28, 28, 1)"),

    # Modals for various operations
    dbc.Modal([
        dbc.ModalHeader("Save Model"),
        dbc.ModalBody([
            dbc.Input(id="save-model-name", placeholder="Enter model name", type="text"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="save-model-cancel", className="ml-auto"),
            dbc.Button("Save", id="save-model-confirm", color="primary"),
        ]),
    ], id="save-model-modal"),

    dbc.Modal([
        dbc.ModalHeader("Load Model"),
        dbc.ModalBody([
            dcc.Dropdown(id="load-model-dropdown"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="load-model-cancel", className="ml-auto"),
            dbc.Button("Load", id="load-model-confirm", color="primary"),
        ]),
    ], id="load-model-modal"),

    dbc.Modal([
        dbc.ModalHeader("Help"),
        dbc.ModalBody([
            html.H5("Neural No-Code Interface"),
            html.P("This interface allows you to build neural networks without writing code."),
            html.H6("Getting Started:"),
            html.Ol([
                html.Li("Select an input shape or template from the sidebar"),
                html.Li("Add layers to your model using the layer selection dropdown"),
                html.Li("Configure layer parameters in the Layer Properties panel"),
                html.Li("Visualize your model in the Visualization tab"),
                html.Li("Generate code in the Code Generation tab"),
                html.Li("Debug your model with NeuralDbg in the Debug tab"),
            ]),
            html.H6("Keyboard Shortcuts:"),
            html.Ul([
                html.Li("Ctrl+S: Save model"),
                html.Li("Ctrl+O: Load model"),
                html.Li("Ctrl+E: Export DSL code"),
                html.Li("Ctrl+H: Show this help"),
            ]),
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="help-close", className="ml-auto"),
        ]),
    ], id="help-modal"),

    # Toast notifications
    html.Div(id="notifications-container")
])

# Helper functions
def get_selected_layer_type():
    """Get the currently selected layer type from any of the category dropdowns"""
    ctx = callback_context
    if not ctx.triggered:
        return None

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Check if the trigger is from a layer type dropdown
    if trigger_id.startswith('layer-type-'):
        category = trigger_id.replace('layer-type-', '')
        return ctx.triggered[0]['value']

    return None

def create_layer_card(layer_index, layer_type, layer_params):
    """Create a card component for a layer in the layer list"""
    param_str = ", ".join([f"{k}={v}" for k, v in layer_params.items() if v is not None])
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H5(f"{layer_index + 1}. {layer_type}", className="card-title"),
                html.Div([
                    dbc.Button("Edit", id={"type": "edit-layer-btn", "index": layer_index},
                              color="primary", size="sm", className="me-1"),
                    dbc.Button("Delete", id={"type": "delete-layer-btn", "index": layer_index},
                              color="danger", size="sm"),
                ], style={"float": "right"})
            ]),
            html.P(param_str, className="card-text text-muted"),
        ]),
        className="mb-2",
        style={"backgroundColor": "#2a2a2a", "border": "1px solid #444"}
    )

def generate_dsl_code(model_data):
    """Generate Neural DSL code from model data"""
    input_shape = model_data.get("input", {}).get("shape", "(None, 28, 28, 1)")
    layers = model_data.get("layers", [])
    loss = model_data.get("loss", {}).get("value", "categorical_crossentropy")
    optimizer_type = model_data.get("optimizer", {}).get("type", "Adam")
    optimizer_params = model_data.get("optimizer", {}).get("params", {})

    # Format optimizer parameters
    optimizer_params_str = ", ".join([f"{k}={v}" for k, v in optimizer_params.items()])
    if optimizer_params_str:
        optimizer_str = f"{optimizer_type}({optimizer_params_str})"
    else:
        optimizer_str = optimizer_type

    # Format layers
    layers_str = ""
    for layer in layers:
        layer_type = layer.get("type", "")
        layer_params = layer.get("params", {})
        params_str = ", ".join([f"{k}={v}" for k, v in layer_params.items() if v is not None])
        layers_str += f"        {layer_type}({params_str})\n"

    # Generate the DSL code
    dsl_code = f"""network MyModel {{
    input: {input_shape}
    layers:
{layers_str}
    loss: {loss}
    optimizer: {optimizer_str}
}}
"""
    return dsl_code

def get_model_templates():
    """Return predefined model templates"""
    templates = {
        "mnist_cnn": {
            "input": {"shape": "(None, 28, 28, 1)"},
            "layers": [
                {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "activation": "relu"}},
                {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
                {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "activation": "relu"}},
                {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
                {"type": "Flatten", "params": {}},
                {"type": "Dense", "params": {"units": 128, "activation": "relu"}},
                {"type": "Dropout", "params": {"rate": 0.5}},
                {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
            ]
        },
        "cifar10_vgg": {
            "input": {"shape": "(None, 32, 32, 3)"},
            "layers": [
                {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}},
                {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}},
                {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
                {"type": "Dropout", "params": {"rate": 0.25}},
                {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}},
                {"type": "Conv2D", "params": {"filters": 64, "kernel_size": (3, 3), "activation": "relu", "padding": "same"}},
                {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
                {"type": "Dropout", "params": {"rate": 0.25}},
                {"type": "Flatten", "params": {}},
                {"type": "Dense", "params": {"units": 512, "activation": "relu"}},
                {"type": "Dropout", "params": {"rate": 0.5}},
                {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
            ]
        },
        "text_lstm": {
            "input": {"shape": "(None, 100)"},
            "layers": [
                {"type": "Embedding", "params": {"input_dim": 10000, "output_dim": 128}},
                {"type": "LSTM", "params": {"units": 64, "return_sequences": True}},
                {"type": "LSTM", "params": {"units": 64}},
                {"type": "Dense", "params": {"units": 64, "activation": "relu"}},
                {"type": "Dropout", "params": {"rate": 0.5}},
                {"type": "Output", "params": {"units": 1, "activation": "sigmoid"}}
            ]
        },
        "transformer_encoder": {
            "input": {"shape": "(None, 512)"},
            "layers": [
                {"type": "Embedding", "params": {"input_dim": 10000, "output_dim": 256}},
                {"type": "MultiHeadAttention", "params": {"num_heads": 8, "key_dim": 32}},
                {"type": "LayerNormalization", "params": {}},
                {"type": "Dense", "params": {"units": 512, "activation": "relu"}},
                {"type": "Dense", "params": {"units": 256}},
                {"type": "LayerNormalization", "params": {}},
                {"type": "GlobalAveragePooling1D", "params": {}},
                {"type": "Dense", "params": {"units": 64, "activation": "relu"}},
                {"type": "Output", "params": {"units": 2, "activation": "softmax"}}
            ]
        },
        "empty": {
            "input": {"shape": "(None, 28, 28, 1)"},
            "layers": []
        }
    }
    return templates

# Callback for showing/hiding custom input shape field
@app.callback(
    Output("custom-input-shape", "style"),
    Input("input-shape-preset", "value")
)
def toggle_custom_input_shape(selected_shape):
    if selected_shape == "custom":
        return {'display': 'block', 'width': '100%'}
    return {'display': 'none', 'width': '100%'}

# Callback for updating the input shape store
@app.callback(
    Output("input-shape", "data"),
    [Input("input-shape-preset", "value"),
     Input("custom-input-shape", "value")]
)
def update_input_shape(preset, custom):
    if preset == "custom" and custom:
        return custom
    return preset

# Callback for loading a model template
@app.callback(
    [Output("model-layers", "data"),
     Output("input-shape-preset", "value")],
    Input("model-template", "value")
)
def load_template(template_name):
    if not template_name or template_name == "empty":
        return [], "(None, 28, 28, 1)"

    templates = get_model_templates()
    if template_name in templates:
        template = templates[template_name]
        input_shape = str(template["input"]["shape"])

        # Find the matching preset or set to custom
        for preset in COMMON_INPUT_SHAPES:
            if preset["value"] == input_shape:
                return template["layers"], preset["value"]

        return template["layers"], "custom"

    return [], "(None, 28, 28, 1)"

# Callback for updating layer parameters when a layer type is selected
@app.callback(
    Output("layer-params", "data"),
    [Input(f"layer-type-{category.lower()}", "value") for category in LAYER_TYPES.keys()]
)
def update_layer_params(*layer_types):
    ctx = callback_context
    if not ctx.triggered:
        return []

    layer_type = get_selected_layer_type()
    if not layer_type:
        return []

    # Get default parameters for the selected layer type
    if layer_type in DEFAULT_PARAMS:
        params = DEFAULT_PARAMS[layer_type]
        return [{"param": k, "value": str(v)} for k, v in params.items()]

    return []

# Callback for adding a layer to the model
@app.callback(
    [Output("model-layers", "data"),
     Output("layer-list", "children")],
    Input("add-layer", "n_clicks"),
    [State("model-layers", "data"),
     State("layer-params", "data")] +
    [State(f"layer-type-{category.lower()}", "value") for category in LAYER_TYPES.keys()]
)
def add_layer(n_clicks, current_layers, params_data, *layer_types):
    if not n_clicks:
        # Initial load - just render existing layers if any
        if current_layers:
            layer_cards = [
                create_layer_card(i, layer["type"], layer["params"])
                for i, layer in enumerate(current_layers)
            ]
            return current_layers, layer_cards
        return [], []

    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'add-layer.n_clicks':
        if current_layers:
            layer_cards = [
                create_layer_card(i, layer["type"], layer["params"])
                for i, layer in enumerate(current_layers)
            ]
            return current_layers, layer_cards
        return [], []

    # Get the selected layer type
    layer_type = get_selected_layer_type()
    if not layer_type:
        # No layer type selected
        if current_layers:
            layer_cards = [
                create_layer_card(i, layer["type"], layer["params"])
                for i, layer in enumerate(current_layers)
            ]
        else:
            layer_cards = []
        return current_layers, layer_cards

    # Convert parameters from table format to dictionary
    layer_params = {}
    for row in params_data:
        param_name = row["param"]
        param_value = row["value"]

        # Try to evaluate the parameter value if it's not a string
        try:
            if param_value.lower() == "none":
                param_value = None
            elif param_value.lower() == "true":
                param_value = True
            elif param_value.lower() == "false":
                param_value = False
            elif param_value.replace(".", "").isdigit():
                if "." in param_value:
                    param_value = float(param_value)
                else:
                    param_value = int(param_value)
            elif param_value.startswith("(") and param_value.endswith(")"):
                # Try to evaluate as a tuple
                param_value = eval(param_value)
        except:
            # Keep as string if evaluation fails
            pass

        layer_params[param_name] = param_value

    # Create the new layer
    new_layer = {"type": layer_type, "params": layer_params}

    # Add to the current layers
    updated_layers = current_layers + [new_layer]

    # Create layer cards for display
    layer_cards = [
        create_layer_card(i, layer["type"], layer["params"])
        for i, layer in enumerate(updated_layers)
    ]

    return updated_layers, layer_cards

# Callback for selecting a layer to edit
@app.callback(
    [Output("selected-layer-index", "data"),
     Output("layer-params", "data", allow_duplicate=True)],
    [Input({"type": "edit-layer-btn", "index": dash.ALL}, "n_clicks")],
    [State("model-layers", "data")],
    prevent_initial_call=True
)
def select_layer_to_edit(edit_clicks, layers):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    # Get the index of the clicked layer
    trigger_id = ctx.triggered[0]['prop_id']
    layer_index = json.loads(trigger_id.split('.')[0])["index"]

    # Get the layer data
    if 0 <= layer_index < len(layers):
        layer = layers[layer_index]
        params_data = [{"param": k, "value": str(v)} for k, v in layer["params"].items()]
        return layer_index, params_data

    return dash.no_update, dash.no_update

# Callback for updating a layer after editing
@app.callback(
    [Output("model-layers", "data", allow_duplicate=True),
     Output("layer-list", "children", allow_duplicate=True),
     Output("selected-layer-index", "data", allow_duplicate=True)],
    Input("layer-params", "data"),
    [State("selected-layer-index", "data"),
     State("model-layers", "data")],
    prevent_initial_call=True
)
def update_layer_after_edit(params_data, selected_index, layers):
    if selected_index is None or selected_index < 0 or selected_index >= len(layers):
        return dash.no_update, dash.no_update, dash.no_update

    # Convert parameters from table format to dictionary
    layer_params = {}
    for row in params_data:
        param_name = row["param"]
        param_value = row["value"]

        # Try to evaluate the parameter value if it's not a string
        try:
            if param_value.lower() == "none":
                param_value = None
            elif param_value.lower() == "true":
                param_value = True
            elif param_value.lower() == "false":
                param_value = False
            elif param_value.replace(".", "").isdigit():
                if "." in param_value:
                    param_value = float(param_value)
                else:
                    param_value = int(param_value)
            elif param_value.startswith("(") and param_value.endswith(")"):
                # Try to evaluate as a tuple
                param_value = eval(param_value)
        except:
            # Keep as string if evaluation fails
            pass

        layer_params[param_name] = param_value

    # Update the layer
    updated_layers = layers.copy()
    updated_layers[selected_index]["params"] = layer_params

    # Create layer cards for display
    layer_cards = [
        create_layer_card(i, layer["type"], layer["params"])
        for i, layer in enumerate(updated_layers)
    ]

    return updated_layers, layer_cards, None

# Callback for deleting a layer
@app.callback(
    [Output("model-layers", "data", allow_duplicate=True),
     Output("layer-list", "children", allow_duplicate=True)],
    [Input({"type": "delete-layer-btn", "index": dash.ALL}, "n_clicks")],
    [State("model-layers", "data")],
    prevent_initial_call=True
)
def delete_layer(delete_clicks, layers):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    # Get the index of the clicked layer
    trigger_id = ctx.triggered[0]['prop_id']
    layer_index = json.loads(trigger_id.split('.')[0])["index"]

    # Delete the layer
    if 0 <= layer_index < len(layers):
        updated_layers = layers.copy()
        updated_layers.pop(layer_index)

        # Create layer cards for display
        layer_cards = [
            create_layer_card(i, layer["type"], layer["params"])
            for i, layer in enumerate(updated_layers)
        ]

        return updated_layers, layer_cards

    return dash.no_update, dash.no_update

# Callback for updating optimizer parameters when optimizer is changed
@app.callback(
    Output("optimizer-params", "data"),
    Input("optimizer", "value")
)
def update_optimizer_params(optimizer):
    if optimizer in DEFAULT_OPTIMIZER_PARAMS:
        params = DEFAULT_OPTIMIZER_PARAMS[optimizer]
        return [{"param": k, "value": str(v)} for k, v in params.items()]
    return []

# Callback for generating shape propagation visualization
@app.callback(
    Output("shape-propagation-graph", "figure"),
    [Input("model-layers", "data"),
     Input("input-shape", "data")]
)
def update_shape_propagation(layers, input_shape_str):
    if not layers:
        # Create an empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Shape Propagation",
            xaxis_title="Layer",
            yaxis_title="Tensor Size",
            template="plotly_dark",
            plot_bgcolor='rgba(50, 50, 50, 0.8)',
            paper_bgcolor='rgba(50, 50, 50, 0.8)',
            font=dict(color='white')
        )
        return fig

    # Parse input shape
    try:
        input_shape = eval(input_shape_str)
    except:
        input_shape = (None, 28, 28, 1)  # Default to MNIST shape

    # Propagate shapes through the model
    propagator = ShapePropagator()
    current_shape = input_shape
    shape_history = [{"layer": "Input", "output_shape": current_shape}]

    for layer in layers:
        try:
            current_shape = propagator.propagate(current_shape, layer, "tensorflow")
            shape_history.append({"layer": layer["type"], "output_shape": current_shape})
        except Exception as e:
            # If shape propagation fails, add the layer with None shape
            shape_history.append({"layer": layer["type"], "output_shape": None})
            break

    # Create the visualization
    layer_names = [s["layer"] for s in shape_history]

    # Calculate tensor sizes (product of dimensions, excluding batch dimension)
    tensor_sizes = []
    for s in shape_history:
        if s["output_shape"] is None:
            tensor_sizes.append(0)
        else:
            # Skip the batch dimension (None)
            dims = s["output_shape"][1:]
            tensor_sizes.append(np.prod(dims))

    # Create the figure
    fig = go.Figure()

    # Add bar chart for tensor sizes
    fig.add_trace(go.Bar(
        x=layer_names,
        y=tensor_sizes,
        marker_color='rgba(0, 191, 255, 0.8)',
        name='Tensor Size'
    ))

    # Add shape annotations
    shape_annotations = []
    for i, s in enumerate(shape_history):
        if s["output_shape"] is not None:
            shape_annotations.append(
                dict(
                    x=i,
                    y=tensor_sizes[i],
                    text=str(s["output_shape"]),
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            )

    fig.update_layout(
        title="Shape Propagation",
        xaxis_title="Layer",
        yaxis_title="Tensor Size (elements)",
        annotations=shape_annotations,
        template="plotly_dark",
        plot_bgcolor='rgba(50, 50, 50, 0.8)',
        paper_bgcolor='rgba(50, 50, 50, 0.8)',
        font=dict(color='white')
    )

    return fig

# Callback for generating code
@app.callback(
    [Output("dsl-code", "children"),
     Output("tensorflow-code", "children"),
     Output("pytorch-code", "children")],
    [Input("compile-btn", "n_clicks")],
    [State("model-layers", "data"),
     State("input-shape", "data"),
     State("optimizer", "value"),
     State("optimizer-params", "data"),
     State("loss-function", "value")]
)
def compile_model(n_clicks, layers, input_shape, optimizer_type, optimizer_params, loss_function):
    if not n_clicks or not layers:
        return "# No model defined yet", "# No model defined yet", "# No model defined yet"

    # Convert optimizer parameters from table format to dictionary
    optimizer_params_dict = {}
    for row in optimizer_params:
        param_name = row["param"]
        param_value = row["value"]

        # Try to evaluate the parameter value
        try:
            if param_value.lower() == "none":
                param_value = None
            elif param_value.lower() == "true":
                param_value = True
            elif param_value.lower() == "false":
                param_value = False
            elif param_value.replace(".", "").isdigit():
                if "." in param_value:
                    param_value = float(param_value)
                else:
                    param_value = int(param_value)
        except:
            # Keep as string if evaluation fails
            pass

        optimizer_params_dict[param_name] = param_value

    # Create the model data
    model_data = {
        "type": "model",
        "input": {"type": "Input", "shape": input_shape},
        "layers": layers,
        "loss": {"value": f'"{loss_function}"'},
        "optimizer": {"type": optimizer_type, "params": optimizer_params_dict}
    }

    # Generate DSL code
    dsl_code = generate_dsl_code(model_data)

    # Generate backend code
    try:
        code_tf = generate_code(model_data, "tensorflow")
    except Exception as e:
        code_tf = f"# Error generating TensorFlow code: {str(e)}"

    try:
        code_torch = generate_code(model_data, "pytorch")
    except Exception as e:
        code_torch = f"# Error generating PyTorch code: {str(e)}"

    return dsl_code, code_tf, code_torch

# Callback for saving a model
@app.callback(
    Output("save-model-modal", "is_open"),
    [Input("save-model-btn", "n_clicks"),
     Input("save-model-confirm", "n_clicks"),
     Input("save-model-cancel", "n_clicks")],
    [State("save-model-modal", "is_open")]
)
def toggle_save_model_modal(open_clicks, confirm_clicks, cancel_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "save-model-btn" and open_clicks:
        return True
    elif (trigger_id == "save-model-confirm" or trigger_id == "save-model-cancel") and is_open:
        return False

    return is_open

# Callback for actually saving the model
@app.callback(
    Output("notifications-container", "children"),
    [Input("save-model-confirm", "n_clicks")],
    [State("save-model-name", "value"),
     State("model-layers", "data"),
     State("input-shape", "data"),
     State("optimizer", "value"),
     State("optimizer-params", "data"),
     State("loss-function", "value")]
)
def save_model(n_clicks, model_name, layers, input_shape, optimizer_type, optimizer_params, loss_function):
    if not n_clicks or not model_name:
        return dash.no_update

    # Convert optimizer parameters from table format to dictionary
    optimizer_params_dict = {}
    for row in optimizer_params:
        param_name = row["param"]
        param_value = row["value"]
        optimizer_params_dict[param_name] = param_value

    # Create the model data
    model_data = {
        "input": {"shape": input_shape},
        "layers": layers,
        "loss": loss_function,
        "optimizer": {"type": optimizer_type, "params": optimizer_params_dict}
    }

    # Save the model to a JSON file
    save_path = os.path.join(os.path.dirname(__file__), 'saved_models', f"{model_name}.json")
    with open(save_path, 'w') as f:
        json.dump(model_data, f, indent=2)

    # Return a notification
    return dbc.Alert(
        f"Model '{model_name}' saved successfully!",
        color="success",
        dismissable=True,
        duration=4000
    )

# Callback for loading model list
@app.callback(
    Output("load-model-dropdown", "options"),
    Input("load-model-btn", "n_clicks")
)
def load_model_list(n_clicks):
    if not n_clicks:
        return []

    # Get list of saved models
    saved_models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    model_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.json')]

    # Create options for dropdown
    options = [{"label": f.replace('.json', ''), "value": f.replace('.json', '')} for f in model_files]

    return options

# Callback for toggling load model modal
@app.callback(
    Output("load-model-modal", "is_open"),
    [Input("load-model-btn", "n_clicks"),
     Input("load-model-confirm", "n_clicks"),
     Input("load-model-cancel", "n_clicks")],
    [State("load-model-modal", "is_open")]
)
def toggle_load_model_modal(open_clicks, confirm_clicks, cancel_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "load-model-btn" and open_clicks:
        return True
    elif (trigger_id == "load-model-confirm" or trigger_id == "load-model-cancel") and is_open:
        return False

    return is_open

# Callback for actually loading the model
@app.callback(
    [Output("model-layers", "data", allow_duplicate=True),
     Output("input-shape", "data", allow_duplicate=True),
     Output("optimizer", "value"),
     Output("loss-function", "value"),
     Output("layer-list", "children", allow_duplicate=True),
     Output("notifications-container", "children", allow_duplicate=True)],
    [Input("load-model-confirm", "n_clicks")],
    [State("load-model-dropdown", "value")],
    prevent_initial_call=True
)
def load_model(n_clicks, model_name):
    if not n_clicks or not model_name:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Load the model from JSON file
    load_path = os.path.join(os.path.dirname(__file__), 'saved_models', f"{model_name}.json")
    try:
        with open(load_path, 'r') as f:
            model_data = json.load(f)

        # Extract model components
        layers = model_data.get("layers", [])
        input_shape = model_data.get("input", {}).get("shape", "(None, 28, 28, 1)")
        optimizer_type = model_data.get("optimizer", {}).get("type", "Adam")
        loss_function = model_data.get("loss", "categorical_crossentropy")

        # Create layer cards for display
        layer_cards = [
            create_layer_card(i, layer["type"], layer["params"])
            for i, layer in enumerate(layers)
        ]

        # Return a notification
        notification = dbc.Alert(
            f"Model '{model_name}' loaded successfully!",
            color="success",
            dismissable=True,
            duration=4000
        )

        return layers, input_shape, optimizer_type, loss_function, layer_cards, notification

    except Exception as e:
        # Return an error notification
        notification = dbc.Alert(
            f"Error loading model: {str(e)}",
            color="danger",
            dismissable=True,
            duration=4000
        )

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, notification

# Callback for toggling help modal
@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-btn", "n_clicks"),
     Input("help-close", "n_clicks")],
    [State("help-modal", "is_open")]
)
def toggle_help_modal(open_clicks, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "help-btn" and open_clicks:
        return True
    elif trigger_id == "help-close" and is_open:
        return False

    return is_open

# Callback for exporting DSL code
@app.callback(
    Output("notifications-container", "children", allow_duplicate=True),
    [Input("export-dsl-btn", "n_clicks")],
    [State("model-layers", "data"),
     State("input-shape", "data"),
     State("optimizer", "value"),
     State("optimizer-params", "data"),
     State("loss-function", "value")],
    prevent_initial_call=True
)
def export_dsl(n_clicks, layers, input_shape, optimizer_type, optimizer_params, loss_function):
    if not n_clicks or not layers:
        return dash.no_update

    # Convert optimizer parameters from table format to dictionary
    optimizer_params_dict = {}
    for row in optimizer_params:
        param_name = row["param"]
        param_value = row["value"]
        optimizer_params_dict[param_name] = param_value

    # Create the model data
    model_data = {
        "input": {"shape": input_shape},
        "layers": layers,
        "loss": {"value": f'"{loss_function}"'},
        "optimizer": {"type": optimizer_type, "params": optimizer_params_dict}
    }

    # Generate DSL code
    dsl_code = generate_dsl_code(model_data)

    # Save the DSL code to a file
    export_path = os.path.join(os.path.dirname(__file__), 'exported_models')
    os.makedirs(export_path, exist_ok=True)

    filename = f"model_{int(time.time())}.neural"
    with open(os.path.join(export_path, filename), 'w') as f:
        f.write(dsl_code)

    # Return a notification
    return dbc.Alert(
        [
            html.Span(f"DSL code exported to {filename}. "),
            html.A("Download", href=f"/download/{filename}", target="_blank")
        ],
        color="success",
        dismissable=True,
        duration=4000
    )

# Callback for visualizing the model architecture
@app.callback(
    Output("architecture-visualization", "children"),
    [Input("model-layers", "data"),
     Input("input-shape", "data")]
)
def visualize_architecture(layers, input_shape_str):
    if not layers:
        return html.Div("No model defined yet. Add layers to visualize the architecture.")

    try:
        # Create a model data structure for the visualizer
        model_data = {
            "type": "model",
            "name": "MyModel",
            "input": {"type": "Input", "shape": eval(input_shape_str)},
            "layers": layers
        }

        # Create a visualizer instance
        visualizer = NeuralVisualizer(model_data)

        # Generate a visualization
        try:
            # Try to use the built-in visualization methods
            fig = visualizer.create_architecture_diagram()
            return dcc.Graph(figure=fig)
        except:
            # Fallback to a simpler visualization
            d3_data = visualizer.model_to_d3_json()

            # Create a simple network diagram
            nodes = d3_data.get("nodes", [])
            links = d3_data.get("links", [])

            # Create a Plotly figure
            fig = go.Figure()

            # Add nodes as scatter points
            node_x = []
            node_y = []
            node_text = []

            for i, node in enumerate(nodes):
                node_x.append(i)
                node_y.append(0)

                # Create node text with parameters
                if "params" in node and node["params"]:
                    params_str = ", ".join([f"{k}={v}" for k, v in node["params"].items()])
                    node_text.append(f"{node['type']}<br>{params_str}")
                else:
                    node_text.append(node['type'])

            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=30,
                    color='rgba(0, 191, 255, 0.8)',
                    line=dict(width=2, color='rgb(0, 0, 0)')
                ),
                text=node_text,
                textposition="top center",
                hoverinfo="text"
            ))

            # Add links as lines
            for link in links:
                source_idx = link.get("source")
                target_idx = link.get("target")

                if source_idx is not None and target_idx is not None:
                    fig.add_trace(go.Scatter(
                        x=[node_x[source_idx], node_x[target_idx]],
                        y=[node_y[source_idx], node_y[target_idx]],
                        mode="lines",
                        line=dict(width=2, color='rgb(210, 210, 210)'),
                        hoverinfo="none"
                    ))

            # Update layout
            fig.update_layout(
                title="Model Architecture",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark",
                plot_bgcolor='rgba(50, 50, 50, 0.8)',
                paper_bgcolor='rgba(50, 50, 50, 0.8)',
                font=dict(color='white')
            )

            return dcc.Graph(figure=fig)

    except Exception as e:
        return html.Div(f"Error visualizing architecture: {str(e)}")

# Callback for launching NeuralDbg
@app.callback(
    Output("neuraldbg-status", "children"),
    [Input("launch-neuraldbg-btn", "n_clicks")],
    [State("model-layers", "data"),
     State("input-shape", "data"),
     State("optimizer", "value"),
     State("optimizer-params", "data"),
     State("loss-function", "value")]
)
def launch_neuraldbg(n_clicks, layers, input_shape, optimizer_type, optimizer_params, loss_function):
    if not n_clicks or not layers:
        return html.Div("Click 'Launch NeuralDbg' to debug your model.")

    # Generate DSL code
    optimizer_params_dict = {}
    for row in optimizer_params:
        param_name = row["param"]
        param_value = row["value"]
        optimizer_params_dict[param_name] = param_value

    # Create the model data
    model_data = {
        "input": {"shape": input_shape},
        "layers": layers,
        "loss": {"value": f'"{loss_function}"'},
        "optimizer": {"type": optimizer_type, "params": optimizer_params_dict}
    }

    # Generate DSL code
    dsl_code = generate_dsl_code(model_data)

    # Save the DSL code to a temporary file
    temp_file = os.path.join(os.path.dirname(__file__), 'temp_model.neural')
    with open(temp_file, 'w') as f:
        f.write(dsl_code)

    # Launch NeuralDbg in a new process
    try:
        import subprocess
        subprocess.Popen(["neural", "debug", temp_file])

        return html.Div([
            html.P("NeuralDbg launched successfully! Open your browser to http://localhost:8050 to access the dashboard."),
            html.A("Open NeuralDbg Dashboard", href="http://localhost:8050", target="_blank",
                  className="btn btn-primary mt-2")
        ])
    except Exception as e:
        return html.Div(f"Error launching NeuralDbg: {str(e)}")

if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
