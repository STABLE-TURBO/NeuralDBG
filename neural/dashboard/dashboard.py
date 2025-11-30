import dash
from dash import Dash, dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import sys
import os
import numpy as np
import pysnooper
import plotly.graph_objects as go
from flask import Flask
from numpy import random
import json
import requests
import time
# Make flask_socketio optional - allows tests to run without it
try:
    from flask_socketio import SocketIO
    SOCKETIO_AVAILABLE = True
except ImportError:
    SocketIO = None
    SOCKETIO_AVAILABLE = False
import threading
from dash_bootstrap_components import themes

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.dashboard.tensor_flow import (
    create_animated_network,
    create_progress_component,
    create_layer_computation_timeline
)



# Flask app for WebSocket integration (if needed later)
server = Flask(__name__)

# Dash app
app = dash.Dash(
    __name__,
    server=server,
    title="NeuralDbg: Real-Time Execution Monitoring",
    external_stylesheets=[themes.DARKLY]
)

# Initialize WebSocket Connection (only if flask_socketio is available)
if SOCKETIO_AVAILABLE and SocketIO is not None:
    socketio = SocketIO(server, cors_allowed_origins=["http://localhost:8050"])
else:
    socketio = None

# Configuration (load from config.yaml or set defaults)
try:
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    config = {}

UPDATE_INTERVAL = config.get("websocket_interval", 1000) # Use default if config file not found


# Store Execution Trace Data and Model Data
# Use a shared list so TRACE_DATA and trace_data always reference the same object
_trace_data_list = []
trace_data = _trace_data_list
TRACE_DATA = _trace_data_list  # Export for test compatibility - both point to same list
model_data = None
backend = 'tensorflow'
shape_history = []

def get_trace_data():
    """Get trace data, checking TRACE_DATA first (for test compatibility)."""
    # Check if TRACE_DATA has been reassigned in the module namespace
    import neural.dashboard.dashboard as dashboard_module
    if hasattr(dashboard_module, 'TRACE_DATA') and dashboard_module.TRACE_DATA is not None:
        return dashboard_module.TRACE_DATA
    return trace_data

# Function to print data for debugging
def print_dashboard_data():
    global trace_data, model_data
    print("\n=== DASHBOARD DATA ===")
    print(f"Model data: {model_data is not None}")
    if model_data:
        print(f"  Input: {model_data.get('input', 'None')}")
        print(f"  Layers: {len(model_data.get('layers', []))} layers")

    print(f"Trace data: {len(trace_data) if trace_data else 0} entries")
    if trace_data and len(trace_data) > 0:
        print(f"  First entry: {trace_data[0]}")
    print("=====================\n")

# Function to update dashboard data
def update_dashboard_data(new_model_data=None, new_trace_data=None, new_backend=None):
    """Update the dashboard data with new data from the CLI."""
    global model_data, trace_data, backend, shape_history, _trace_data_list

    if new_model_data is not None:
        model_data = new_model_data

    if new_trace_data is not None:
        # Convert numpy values to Python native types for JSON serialization
        processed_trace_data = []
        for entry in new_trace_data:
            processed_entry = {}
            for key, value in entry.items():
                if hasattr(value, 'item') and callable(getattr(value, 'item')):
                    # Convert numpy values to Python native types
                    processed_entry[key] = value.item()
                else:
                    processed_entry[key] = value
            processed_trace_data.append(processed_entry)
        # Update the shared list to maintain reference sharing
        # Clear and extend to preserve the reference that trace_data and TRACE_DATA share
        _trace_data_list.clear()
        _trace_data_list.extend(processed_trace_data)

    if new_backend is not None:
        backend = new_backend

    # Clear shape history to force recalculation
    shape_history = []

    # Print the updated data
    print_dashboard_data()

# Print initial data when module is loaded
print_dashboard_data()

### Interval Updates ####
@app.callback(
    [Output("interval_component", "interval")],
    [Input("update_interval", "value")]
)

def update_interval(new_interval):
    """Update the interval dynamically based on slider value."""
    return [new_interval]

# Start WebSocket in a Separate Thread (only if socketio is available)
propagator = ShapePropagator()
if socketio is not None:
    threading.Thread(target=socketio.run, args=(server,), kwargs={"host": "localhost", "port": 5001, "allow_unsafe_werkzeug": True}, daemon=True).start()

####################################################
#### Layers Execution Trace Graph & Its Subplots ###
####################################################

@app.callback(
    [Output("trace_graph", "figure")],
    [Input("interval_component", "n_intervals"), Input("viz_type", "value"), Input("layer_filter", "value")]
)
def update_trace_graph(n, viz_type, selected_layers=None):
    """Update execution trace graph with various visualization types."""
    global trace_data
    # Use get_trace_data() to get the current trace data (handles test reassignments)
    data_source = get_trace_data()

    ### ***Errors Handling*** ###
    if not data_source or any(not isinstance(entry["execution_time"], (int, float)) for entry in data_source):
        return [go.Figure()]  # Return empty figure for invalid data

    # Filter data based on selected layers (if any)
    if selected_layers:
        filtered_data = [entry for entry in data_source if entry["layer"] in selected_layers]
    else:
        filtered_data = data_source

    if not filtered_data:
        return [go.Figure()]

    layers = [entry["layer"] for entry in filtered_data]
    execution_times = [entry["execution_time"] for entry in filtered_data]

    # Simulate compute_time and transfer_time for stacked bar (you can extend ShapePropagator to include these)
    compute_times = []
    transfer_times = []
    for t in execution_times:
        # Ensure t is a number
        if isinstance(t, (int, float)):
            compute_times.append(t * 0.7)  # 70% of execution time for compute
            transfer_times.append(t * 0.3)  # 30% for data transfer
        else:
            # Default values if t is not a number
            compute_times.append(0.1)
            transfer_times.append(0.05)

    fig = go.Figure()

    if viz_type == "basic":
        ### Basic Bar Chart ###
        fig = go.Figure([go.Bar(x=layers, y=execution_times, name="Execution Time (s)")])
        fig.update_layout(
            title="Layer Execution Time",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    elif viz_type == "stacked":
        # Stacked Bar Chart
        fig = go.Figure([
            go.Bar(x=layers, y=compute_times, name="Compute Time"),
            go.Bar(x=layers, y=transfer_times, name="Data Transfer"),
        ])
        fig.update_layout(
            barmode="stack",
            title="Layer Execution Time Breakdown",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    elif viz_type == "horizontal":
        # Horizontal Bar Chart with Sorting
        sorted_data = sorted(filtered_data, key=lambda x: x["execution_time"], reverse=True)
        sorted_layers = [entry["layer"] for entry in sorted_data]
        sorted_times = [entry["execution_time"] for entry in sorted_data]
        fig = go.Figure([go.Bar(x=sorted_times, y=sorted_layers, orientation="h", name="Execution Time")])
        fig.update_layout(
            title="Layer Execution Time (Sorted)",
            xaxis_title="Time (s)",
            yaxis_title="Layers",
            template="plotly_white"
        )

    elif viz_type == "box":
        ### Box Plots for Variability ###
        # Use unique layers from filtered_data, maintaining original order
        unique_layers = list(dict.fromkeys(entry["layer"] for entry in filtered_data))  # Preserves order, removes duplicates
        times_by_layer = {layer: [entry["execution_time"] for entry in filtered_data if entry["layer"] == layer] for layer in unique_layers}
        fig = go.Figure([go.Box(x=unique_layers, y=[times_by_layer[layer] for layer in unique_layers], name="Execution Variability")])
        fig.update_layout(
            title="Layer Execution Time Variability",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    elif viz_type == "gantt":
        # Gantt Chart for Timeline
        for i, entry in enumerate(filtered_data):
            fig.add_trace(go.Bar(x=[i, i], y=[0, entry["execution_time"]], orientation="v", name=entry["layer"]))
        fig.update_layout(
            title="Layer Execution Timeline",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            showlegend=True,
            template="plotly_white"
        )

    elif viz_type == "heatmap":
        # Ensure TRACE_DATA has multiple iterations or simulate them
        iterations = 5
        heatmap_data = np.random.rand(len(layers), iterations)
        fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=[f"Iteration {i+1}" for i in range(iterations)], y=layers))
        fig.update_layout(title="Execution Time Heatmap", xaxis_title="Iterations", yaxis_title="Layers")



    elif viz_type == "thresholds":
        # Bar Chart with Annotations and Thresholds
        marker_colors = []
        for t in execution_times:
            if isinstance(t, (int, float)) and t > 0.003:
                marker_colors.append("red")
            else:
                marker_colors.append("blue")

        fig = go.Figure([go.Bar(x=layers, y=execution_times, name="Execution Time", marker_color=marker_colors)])

        for i, t in enumerate(execution_times):
            if isinstance(t, (int, float)) and t > 0.003:
                fig.add_annotation(
                    x=layers[i], y=t, text=f"High: {t}s", showarrow=True, arrowhead=2,
                    font=dict(size=10), align="center"
                )
        fig.update_layout(
            title="Layer Execution Time with Thresholds",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    # Add common layout enhancements
    fig.update_layout(
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1000
    )

    return [fig]

############################
#### FLOPS Memory Chart ####
############################

@app.callback(
    Output("flops_memory_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_flops_memory_chart(n):
    """Update FLOPs and memory usage visualization."""
    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    flops = [entry["flops"] for entry in trace_data]
    memory = [entry["memory"] for entry in trace_data]

    # Create Dual Bar Graph (FLOPs & Memory)
    fig = go.Figure([
        go.Bar(x=layers, y=flops, name="FLOPs"),
        go.Bar(x=layers, y=memory, name="Memory Usage (MB)")
    ])
    fig.update_layout(title="FLOPs & Memory Usage", xaxis_title="Layers", yaxis_title="Values", barmode="group")

    return [fig]

##################
### Loss Graph ###
##################

@app.callback(
    Output("loss_graph", "figure"),
    Input("interval_component", "n_intervals")
)

def update_loss(n):
    loss_data = [random.uniform(0.1, 1.0) for _ in range(n)]  # Simulated loss data
    fig = go.Figure(data=[go.Scatter(y=loss_data, mode="lines+markers")])
    fig.update_layout(title="Loss Over Time")
    return fig

# This layout is replaced by the principal layout below
# app.layout = html.Div([
#     html.H1("Compare Architectures"),
#     dcc.Dropdown(id="architecture_selector", options=[
#         {"label": "Model A", "value": "A"},
#         {"label": "Model B", "value": "B"},
#     ], value="A"),
#     dcc.Graph(id="architecture_graph"),
# ])

##########################
### Architecture Graph ###
##########################


@app.callback(
    Output("architecture_graph", "figure"),
    Input("architecture_selector", "value")
)
def update_graph(arch):
    """Update the architecture graph visualization."""
    global model_data, backend, trace_data

    # Print debug information
    print(f"Updating architecture graph with model_data: {model_data is not None}")

    # Create a figure
    fig = go.Figure()

    # If we have model data, use it
    if model_data and isinstance(model_data, dict):
        # Get input shape from model data
        if 'input' in model_data and 'shape' in model_data['input']:
            input_shape = model_data['input']['shape']
            print(f"Input shape: {input_shape}")

            # Get layers from model data
            if 'layers' in model_data and isinstance(model_data['layers'], list):
                layers = model_data['layers']
                print(f"Layers: {len(layers)}")

                # Extract layer types for visualization
                layer_types = []
                for layer in layers:
                    if isinstance(layer, dict) and 'type' in layer:
                        layer_types.append(layer['type'])

                # Create a simple network visualization
                x_positions = [0]  # Input node
                y_positions = [0]
                node_labels = ["Input"]
                node_colors = ["blue"]

                # Add layer nodes
                for i, layer_type in enumerate(layer_types):
                    x_positions.append(i + 1)
                    y_positions.append(0 if i % 2 == 0 else 1)  # Alternate y positions
                    node_labels.append(layer_type)

                    # Color based on layer type
                    if "Conv" in layer_type:
                        node_colors.append("red")
                    elif "Pool" in layer_type:
                        node_colors.append("green")
                    elif "Dense" in layer_type:
                        node_colors.append("purple")
                    elif "Dropout" in layer_type:
                        node_colors.append("orange")
                    else:
                        node_colors.append("gray")

                # Add output node
                x_positions.append(len(layer_types) + 1)
                y_positions.append(0)
                node_labels.append("Output")
                node_colors.append("blue")

                # Add nodes to the figure
                fig.add_trace(go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode="markers+text",
                    marker=dict(size=30, color=node_colors),
                    text=node_labels,
                    textposition="bottom center"
                ))

                # Add edges (connections between nodes)
                edge_x = []
                edge_y = []

                for i in range(len(x_positions) - 1):
                    edge_x.extend([x_positions[i], x_positions[i+1], None])
                    edge_y.extend([y_positions[i], y_positions[i+1], None])

                fig.add_trace(go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=2, color="gray"),
                    hoverinfo="none"
                ))

                # Update layout
                fig.update_layout(
                    title="Network Architecture",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=400
                )

                return fig
            else:
                print("Model data does not contain a valid 'layers' list")
        else:
            print("Model data does not contain a valid 'input' with 'shape'")
    else:
        print("No valid model data available")

    # Fallback to default behavior if no model data is available
    if arch == "A":
        # Simple architecture
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3],
            y=[0, 1, 0, 1],
            mode="markers+text",
            marker=dict(size=30, color=["blue", "red", "green", "purple"]),
            text=["Input", "Conv", "Pool", "Output"],
            textposition="bottom center"
        ))

        # Add edges
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 2, 2, 3],
            y=[0, 1, 1, 0, 0, 1],
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none"
        ))
    else:
        # More complex architecture
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3, 4, 5],
            y=[0, 1, 0, 1, 0, 1],
            mode="markers+text",
            marker=dict(size=30, color=["blue", "red", "green", "red", "purple", "blue"]),
            text=["Input", "Conv1", "Pool", "Conv2", "Dense", "Output"],
            textposition="bottom center"
        ))

        # Add edges
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            y=[0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none"
        ))

    # Update layout
    fig.update_layout(
        title=f"Network Architecture {arch}",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )

    return fig


###########################
### Gradient Flow Panel ###
###########################
@app.callback(
    Output("gradient_flow_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_gradient_chart(n):
    """Visualizes gradient flow per layer."""
    global trace_data

    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    grad_norms = [entry.get("grad_norm", 0) for entry in trace_data]

    fig = go.Figure([go.Bar(x=layers, y=grad_norms, name="Gradient Magnitude")])
    fig.update_layout(title="Gradient Flow", xaxis_title="Layers", yaxis_title="Gradient Magnitude")

    return fig

#########################
### Dead Neuron Panel ###
#########################
@app.callback(
    Output("dead_neuron_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_dead_neurons(n):
    """Displays percentage of dead neurons per layer."""
    global trace_data

    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    dead_ratios = [entry.get("dead_ratio", 0) for entry in trace_data]

    fig = go.Figure([go.Bar(x=layers, y=dead_ratios, name="Dead Neurons (%)")])
    fig.update_layout(title="Dead Neuron Detection", xaxis_title="Layers", yaxis_title="Dead Ratio", yaxis_range=[0, 1])

    return fig

##############################
### Anomaly Detection Panel###
##############################
@app.callback(
    Output("anomaly_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_anomaly_chart(n):
    """Visualizes unusual activations per layer."""
    global trace_data

    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    activations = [entry.get("mean_activation", 0) for entry in trace_data]
    anomalies = [1 if entry.get("anomaly", False) else 0 for entry in trace_data]

    fig = go.Figure([
        go.Bar(x=layers, y=activations, name="Mean Activation"),
        go.Bar(x=layers, y=anomalies, name="Anomaly Detected", marker_color="red")
    ])
    fig.update_layout(title="Activation Anomalies", xaxis_title="Layers", yaxis_title="Activation Magnitude")

    return fig

###########################
### Step Debugger Button###
###########################
@app.callback(
    Output("step_debug_output", "children"),
    Input("step_debug_button", "n_clicks")
)
def trigger_step_debug(n):
    """Manually pauses execution at a layer."""
    if n:
        requests.get("http://localhost:5001/trigger_step_debug")
        return "Paused. Check terminal for tensor inspection."
    return "Click to pause execution."

####################################
### Resource Monitoring Callback ###
####################################

@app.callback(
    Output("resource_graph", "figure"),
    Input("interval_component", "n_intervals")
)
def update_resource_graph(n):
    """Visualize CPU/GPU usage, memory, and I/O bottlenecks."""
    try:
        import psutil

        # Get CPU and memory usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        # Try to get GPU usage if available
        gpu_memory = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3) * 100  # Convert to percentage
        except (ImportError, Exception):
            # If torch is not available or there's an error, just use 0
            pass

        fig = go.Figure([
            go.Bar(x=["CPU", "Memory", "GPU"], y=[cpu_usage, memory_usage, gpu_memory], name="Resource Usage (%)"),
        ])
        fig.update_layout(
            title="Resource Monitoring",
            xaxis_title="Resource",
            yaxis_title="Usage (%)",
            template="plotly_dark",
            height=400
        )
    except Exception as e:
        # If there's an error, return an empty figure
        print(f"Error in resource monitoring: {e}")
        fig = go.Figure()
        fig.update_layout(
            title="Resource Monitoring (Error)",
            height=400
        )

    return fig

#################################
### Tensor Flow Visualization ###
#################################
@app.callback(
    Output("tensor_flow_graph", "figure"),
    Input("interval_component", "n_intervals")
)
def update_tensor_flow(n):
    """Visualize tensor flow through the network."""
    global shape_history, model_data, backend

    # If we have shape_history, use it
    if shape_history:
        return create_animated_network(shape_history)

    # If we have a global propagator with shape_history, use it
    if 'propagator' in globals() and hasattr(propagator, 'shape_history'):
        return create_animated_network(propagator.shape_history)

    # If we have model_data, create a new shape history
    if model_data and isinstance(model_data, dict):
        if 'input' in model_data and 'shape' in model_data['input'] and 'layers' in model_data:
            input_shape = model_data['input']['shape']
            layers = model_data['layers']

            # Check if layers is a list
            if isinstance(layers, list) and layers:
                # Create a new propagator
                local_propagator = ShapePropagator()

                # Propagate shapes through the network
                for layer in layers:
                    try:
                        input_shape = local_propagator.propagate(input_shape, layer, backend)
                    except Exception as e:
                        print(f"Error propagating shape for layer {layer.get('type', 'unknown')}: {e}")

                # Store the shape history for future use
                shape_history = local_propagator.shape_history

                # Return the animated network
                return create_animated_network(shape_history)

    # If all else fails, return an empty figure
    return go.Figure()


# Custom Theme
app = Dash(__name__, external_stylesheets=[themes.DARKLY])  # Darkly theme for Dash Bootstrap

# Custom CSS for additional styling
app.css.append_css({
    "external_url": "https://custom-theme.com/neural.css"  # Create this file or use inline CSS
})


########################
### Principal Layout ###
########################

app.layout = html.Div([
    html.H1("NeuralDbg: Neural Network Debugger", style={"textAlign": "center", "marginBottom": "30px"}),

    # Main container with two columns
    html.Div([
        # Left column - Model Structure
        html.Div([
            html.H2("Model Structure", style={"textAlign": "center"}),
            html.Div([
                html.Button("Visualize Model", id="generate-viz-button", n_clicks=1,
                           style={"marginBottom": "10px", "width": "100%", "padding": "10px",
                                  "backgroundColor": "#4CAF50", "color": "white", "border": "none"}),
                dcc.Loading(
                    id="loading-network-viz",
                    type="circle",
                    children=[
                        html.Div(id="network-viz-container", children=[
                            dcc.Graph(id="architecture_viz_graph"),
                            create_progress_component(),
                        ])
                    ]
                ),
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),

            html.H2("Layer Performance", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                dcc.Graph(id="flops_memory_chart"),
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

        # Right column - Analysis
        html.Div([
            html.H2("Gradient Flow Analysis", style={"textAlign": "center"}),
            html.Div([
                dcc.Graph(id="gradient_flow_chart"),
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),

            html.H2("Dead Neuron Detection", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                dcc.Graph(id="dead_neuron_chart"),
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),

            html.H2("Anomaly Detection", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                dcc.Graph(id="anomaly_chart"),
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "4%"}),
    ]),

    # Bottom row - Resource monitoring
    html.Div([
        html.H2("Resource Monitoring", style={"textAlign": "center", "marginTop": "20px"}),
        html.Div([
            dcc.Graph(id="resource_graph"),
        ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
    ]),

    # Interval for updates
    dcc.Interval(id="interval_component", interval=UPDATE_INTERVAL, n_intervals=0),

    # Hidden div for storing progress data
    html.Div(id="progress-store", style={"display": "none"})
])

# Add callbacks for the visualization and progress updates
@app.callback(
    [Output("architecture_viz_graph", "figure"),
     Output("progress-store", "children")],
    [Input("generate-viz-button", "n_clicks")],
    [State("progress-store", "children")]
)
def update_network_visualization(n_clicks, _):
    """Generate a visualization of the neural network architecture."""
    global model_data, backend, shape_history

    # Print debug information
    print(f"Updating network visualization with n_clicks={n_clicks}")
    print(f"Model data available: {model_data is not None}")
    if model_data:
        print(f"Model data keys: {model_data.keys() if isinstance(model_data, dict) else 'Not a dict'}")

    # Create a default figure
    fig = go.Figure()

    # Always generate visualization regardless of n_clicks
    # Start with progress updates
    progress = 10
    details = "Starting visualization..."

    # If we have model_data, use it to create a visualization
    if model_data and isinstance(model_data, dict):
        if 'input' in model_data and 'layers' in model_data and isinstance(model_data['layers'], list):
            progress = 20
            details = "Processing model data..."

            # Extract layer types for visualization
            layers = model_data['layers']
            layer_types = []
            for layer in layers:
                if isinstance(layer, dict) and 'type' in layer:
                    layer_type = layer['type']
                    layer_types.append(layer_type)
                    print(f"Found layer: {layer_type}")

            # Create a simple network visualization
            x_positions = [0]  # Input node
            y_positions = [0]
            node_labels = ["Input"]
            node_colors = ["blue"]

            # Add layer nodes
            for i, layer_type in enumerate(layer_types):
                x_positions.append(i + 1)
                y_positions.append(0 if i % 2 == 0 else 1)  # Alternate y positions
                node_labels.append(layer_type)

                # Color based on layer type
                if "Conv" in layer_type:
                    node_colors.append("red")
                elif "Pool" in layer_type:
                    node_colors.append("green")
                elif "Dense" in layer_type:
                    node_colors.append("purple")
                elif "Dropout" in layer_type:
                    node_colors.append("orange")
                else:
                    node_colors.append("gray")

            # Add output node
            x_positions.append(len(layer_types) + 1)
            y_positions.append(0)
            node_labels.append("Output")
            node_colors.append("blue")

            # Add nodes to the figure
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=y_positions,
                mode="markers+text",
                marker=dict(size=30, color=node_colors),
                text=node_labels,
                textposition="bottom center"
            ))

            # Add edges (connections between nodes)
            edge_x = []
            edge_y = []

            for i in range(len(x_positions) - 1):
                edge_x.extend([x_positions[i], x_positions[i+1], None])
                edge_y.extend([y_positions[i], y_positions[i+1], None])

            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=2, color="gray"),
                hoverinfo="none"
            ))

            # Update layout
            fig.update_layout(
                title="Network Architecture",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )

            progress = 100
            details = "Visualization complete!"

            return fig, json.dumps({"progress": progress, "details": details})

    # Fallback to default visualization if we don't have model_data
    progress = 50
    details = "Using default visualization..."

    # Create a default visualization
    # Simple architecture
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4, 5],
        y=[0, 1, 0, 1, 0, 1],
        mode="markers+text",
        marker=dict(size=30, color=["blue", "red", "green", "red", "purple", "blue"]),
        text=["Input", "Conv2D", "MaxPool", "Conv2D", "Dense", "Output"],
        textposition="bottom center"
    ))

    # Add edges
    fig.add_trace(go.Scatter(
        x=[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        y=[0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        mode="lines",
        line=dict(width=2, color="gray"),
        hoverinfo="none"
    ))

    # Update layout
    fig.update_layout(
        title="Network Architecture (Default)",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    # Return the figure and final progress state
    return fig, json.dumps({"progress": 100, "details": "Default visualization complete"})

# Update progress bar
@app.callback(
    [Output("progress-bar", "style"),
     Output("progress-text", "children"),
     Output("progress-details", "children")],
    [Input("progress-store", "children")]
)
def update_progress_display(progress_json):
    if not progress_json:
        raise PreventUpdate

    progress_data = json.loads(progress_json)
    progress = progress_data.get("progress", 0)
    details = progress_data.get("details", "")

    # Update progress bar style
    bar_style = {
        "width": f"{progress}%",
        "backgroundColor": "#4CAF50",
        "height": "30px"
    }

    return bar_style, f"{progress:.1f}%", details

# Add computation timeline
@app.callback(
    Output("computation-timeline", "figure"),
    [Input("interval_component", "n_intervals")]
)
def update_computation_timeline(n_intervals):
    """Create a Gantt chart showing layer execution times."""
    global trace_data

    # Print debug information
    print(f"Updating computation timeline with trace_data: {len(trace_data) if trace_data else 0} entries")

    # Create a figure
    fig = go.Figure()

    if trace_data and len(trace_data) > 0:
        # Extract layer names and execution times
        layers = [entry.get("layer", "Unknown") for entry in trace_data]
        execution_times = [entry.get("execution_time", 0) for entry in trace_data]

        # Calculate cumulative times for Gantt chart
        start_times = [0]
        for i in range(1, len(execution_times)):
            start_times.append(start_times[i-1] + execution_times[i-1])

        # Create Gantt chart
        for i, layer in enumerate(layers):
            fig.add_trace(go.Bar(
                x=[execution_times[i]],
                y=[layer],
                orientation='h',
                base=start_times[i],
                marker=dict(color='rgb(55, 83, 109)'),
                name=layer
            ))

        fig.update_layout(
            title="Layer Execution Timeline",
            xaxis_title="Time (s)",
            yaxis_title="Layer",
            height=400,
            showlegend=False
        )
    else:
        # Use default data if no trace data is available
        layer_data = [
            {"layer": "Input", "execution_time": 0.1},
            {"layer": "Conv2D", "execution_time": 0.8},
            {"layer": "MaxPooling2D", "execution_time": 0.3},
            {"layer": "Flatten", "execution_time": 0.1},
            {"layer": "Dense", "execution_time": 0.5},
            {"layer": "Output", "execution_time": 0.2}
        ]

        # Extract layer names and execution times
        layers = [entry["layer"] for entry in layer_data]
        execution_times = [entry["execution_time"] for entry in layer_data]

        # Calculate cumulative times for Gantt chart
        start_times = [0]
        for i in range(1, len(execution_times)):
            start_times.append(start_times[i-1] + execution_times[i-1])

        # Create Gantt chart
        for i, layer in enumerate(layers):
            fig.add_trace(go.Bar(
                x=[execution_times[i]],
                y=[layer],
                orientation='h',
                base=start_times[i],
                marker=dict(color='rgb(55, 83, 109)'),
                name=layer
            ))

        fig.update_layout(
            title="Layer Execution Timeline (Default Data)",
            xaxis_title="Time (s)",
            yaxis_title="Layer",
            height=400,
            showlegend=False
        )

    return fig

if __name__ == "__main__":
    app.run_server(debug=False, use_reloader=False)
