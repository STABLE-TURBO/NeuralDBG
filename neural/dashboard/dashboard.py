import dash
from dash import Dash, dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import sys
import os
import numpy as np
import pysnooper
import plotly.graph_objects as go
from flask import Flask, request, jsonify
from numpy import random
import json
import requests
import time
from collections import deque
from scipy import stats
from sklearn.ensemble import IsolationForest
import threading
from dash_bootstrap_components import themes

try:
    from flask_socketio import SocketIO
    SOCKETIO_AVAILABLE = True
except ImportError:
    SocketIO = None
    SOCKETIO_AVAILABLE = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.dashboard.tensor_flow import (
    create_animated_network,
    create_progress_component,
    create_layer_computation_timeline
)


server = Flask(__name__)

app = dash.Dash(
    __name__,
    server=server,
    title="NeuralDbg: Real-Time Execution Monitoring",
    external_stylesheets=[themes.DARKLY]
)

if SOCKETIO_AVAILABLE and SocketIO is not None:
    socketio = SocketIO(server, cors_allowed_origins=["http://localhost:8050"])
else:
    socketio = None

try:
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    config = {}

UPDATE_INTERVAL = config.get("websocket_interval", 1000)


_trace_data_list = []
trace_data = _trace_data_list
TRACE_DATA = _trace_data_list
model_data = None
backend = 'tensorflow'
shape_history = []

breakpoints = {}
breakpoint_state = {"paused": False, "current_layer": None, "layer_data": {}}

layer_inspection_history = deque(maxlen=100)
performance_profile = {"layers": [], "execution_times": [], "memory_usage": [], "timestamps": []}
anomaly_detector = None
anomaly_history = deque(maxlen=1000)


class BreakpointManager:
    def __init__(self):
        self.breakpoints = {}
        self.paused = False
        self.current_layer = None
        self.condition_cache = {}
        self.hit_counts = {}
    
    def add_breakpoint(self, layer_name, condition=None, enabled=True):
        self.breakpoints[layer_name] = {
            "enabled": enabled,
            "condition": condition,
            "hit_count": 0
        }
    
    def remove_breakpoint(self, layer_name):
        if layer_name in self.breakpoints:
            del self.breakpoints[layer_name]
    
    def toggle_breakpoint(self, layer_name):
        if layer_name in self.breakpoints:
            self.breakpoints[layer_name]["enabled"] = not self.breakpoints[layer_name]["enabled"]
    
    def should_break(self, layer_name, layer_data):
        if layer_name not in self.breakpoints:
            return False
        
        bp = self.breakpoints[layer_name]
        if not bp["enabled"]:
            return False
        
        bp["hit_count"] += 1
        
        if bp["condition"]:
            try:
                return eval(bp["condition"], {"layer_data": layer_data, "np": np})
            except Exception as e:
                print(f"Error evaluating breakpoint condition: {e}")
                return False
        
        return True
    
    def get_breakpoint_info(self):
        return {name: {"enabled": bp["enabled"], "hit_count": bp["hit_count"]} 
                for name, bp in self.breakpoints.items()}


class AnomalyDetector:
    def __init__(self, sensitivity=0.95):
        self.sensitivity = sensitivity
        self.history = deque(maxlen=100)
        self.layer_stats = {}
        self.isolation_forest = IsolationForest(contamination=1 - sensitivity, random_state=42)
        self.fitted = False
    
    def add_sample(self, layer_name, metrics):
        if layer_name not in self.layer_stats:
            self.layer_stats[layer_name] = {
                "execution_times": deque(maxlen=50),
                "memory_usage": deque(maxlen=50),
                "activations": deque(maxlen=50),
                "gradients": deque(maxlen=50)
            }
        
        stats = self.layer_stats[layer_name]
        stats["execution_times"].append(metrics.get("execution_time", 0))
        stats["memory_usage"].append(metrics.get("memory", 0))
        stats["activations"].append(metrics.get("mean_activation", 0))
        stats["gradients"].append(metrics.get("grad_norm", 0))
        
        self.history.append({
            "layer": layer_name,
            "metrics": metrics,
            "timestamp": time.time()
        })
    
    def detect_anomalies(self, layer_name, current_metrics):
        if layer_name not in self.layer_stats:
            return {"is_anomaly": False, "scores": {}}
        
        stats = self.layer_stats[layer_name]
        anomaly_scores = {}
        is_anomaly = False
        
        for metric_name, values in stats.items():
            if len(values) < 5:
                continue
            
            current_value = current_metrics.get(metric_name.replace("_", ""), 0)
            if current_value == 0:
                continue
            
            values_array = np.array(list(values))
            mean = np.mean(values_array)
            std = np.std(values_array)
            
            if std > 0:
                z_score = abs((current_value - mean) / std)
                anomaly_scores[metric_name] = z_score
                
                if z_score > 3:
                    is_anomaly = True
        
        iqr_anomaly = self._detect_iqr_anomaly(layer_name, current_metrics)
        if iqr_anomaly:
            is_anomaly = True
            anomaly_scores["iqr_based"] = iqr_anomaly
        
        return {"is_anomaly": is_anomaly, "scores": anomaly_scores}
    
    def _detect_iqr_anomaly(self, layer_name, current_metrics):
        if layer_name not in self.layer_stats:
            return False
        
        stats = self.layer_stats[layer_name]
        exec_times = list(stats["execution_times"])
        
        if len(exec_times) < 10:
            return False
        
        q1 = np.percentile(exec_times, 25)
        q3 = np.percentile(exec_times, 75)
        iqr = q3 - q1
        
        current_time = current_metrics.get("execution_time", 0)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return current_time < lower_bound or current_time > upper_bound
    
    def get_layer_statistics(self, layer_name):
        if layer_name not in self.layer_stats:
            return {}
        
        stats = self.layer_stats[layer_name]
        result = {}
        
        for metric_name, values in stats.items():
            if len(values) > 0:
                values_array = np.array(list(values))
                result[metric_name] = {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "median": float(np.median(values_array))
                }
        
        return result


class PerformanceProfiler:
    def __init__(self):
        self.call_stack = []
        self.flame_data = []
        self.layer_profiles = {}
        self.start_time = time.time()
    
    def start_layer(self, layer_name):
        self.call_stack.append({
            "name": layer_name,
            "start": time.time(),
            "children": []
        })
    
    def end_layer(self, layer_name):
        if not self.call_stack or self.call_stack[-1]["name"] != layer_name:
            return
        
        layer_profile = self.call_stack.pop()
        layer_profile["end"] = time.time()
        layer_profile["duration"] = layer_profile["end"] - layer_profile["start"]
        
        if layer_name not in self.layer_profiles:
            self.layer_profiles[layer_name] = {
                "total_time": 0,
                "call_count": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "avg_time": 0
            }
        
        profile = self.layer_profiles[layer_name]
        profile["total_time"] += layer_profile["duration"]
        profile["call_count"] += 1
        profile["min_time"] = min(profile["min_time"], layer_profile["duration"])
        profile["max_time"] = max(profile["max_time"], layer_profile["duration"])
        profile["avg_time"] = profile["total_time"] / profile["call_count"]
        
        self.flame_data.append(layer_profile)
        
        if self.call_stack:
            self.call_stack[-1]["children"].append(layer_profile)
    
    def get_flame_graph_data(self):
        flame_rows = []
        
        def process_node(node, depth=0, parent_start=0):
            start = node["start"] - self.start_time
            duration = node.get("duration", 0)
            
            flame_rows.append({
                "name": node["name"],
                "start": start,
                "end": start + duration,
                "duration": duration,
                "depth": depth
            })
            
            for child in node.get("children", []):
                process_node(child, depth + 1, start)
        
        for root_node in self.flame_data:
            if "duration" in root_node:
                process_node(root_node)
        
        return flame_rows
    
    def get_summary(self):
        return self.layer_profiles


breakpoint_manager = BreakpointManager()
anomaly_detector = AnomalyDetector()
profiler = PerformanceProfiler()


@server.route('/api/breakpoint', methods=['POST', 'GET', 'DELETE'])
def manage_breakpoint():
    if request.method == 'POST':
        data = request.json
        layer_name = data.get('layer_name')
        condition = data.get('condition')
        breakpoint_manager.add_breakpoint(layer_name, condition)
        return jsonify({"status": "success", "message": f"Breakpoint added for {layer_name}"})
    
    elif request.method == 'GET':
        return jsonify(breakpoint_manager.get_breakpoint_info())
    
    elif request.method == 'DELETE':
        data = request.json
        layer_name = data.get('layer_name')
        breakpoint_manager.remove_breakpoint(layer_name)
        return jsonify({"status": "success", "message": f"Breakpoint removed for {layer_name}"})


@server.route('/api/layer_inspect/<layer_name>', methods=['GET'])
def inspect_layer(layer_name):
    layer_info = {
        "statistics": anomaly_detector.get_layer_statistics(layer_name),
        "breakpoint": breakpoint_manager.breakpoints.get(layer_name, {}),
        "profile": profiler.layer_profiles.get(layer_name, {})
    }
    return jsonify(layer_info)


@server.route('/api/continue', methods=['POST'])
def continue_execution():
    breakpoint_manager.paused = False
    return jsonify({"status": "success", "message": "Execution continued"})


def get_trace_data():
    import neural.dashboard.dashboard as dashboard_module
    if hasattr(dashboard_module, 'TRACE_DATA') and dashboard_module.TRACE_DATA is not None:
        return dashboard_module.TRACE_DATA
    return trace_data

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Function to log data for debugging
def print_dashboard_data():
    global trace_data, model_data
    logger.debug("=== DASHBOARD DATA ===")
    logger.debug("Model data: %s", model_data is not None)
    if model_data:
        logger.debug("  Input: %s", model_data.get('input', 'None'))
        logger.debug("  Layers: %s layers", len(model_data.get('layers', [])))

    logger.debug("Trace data: %s entries", len(trace_data) if trace_data else 0)
    if trace_data and len(trace_data) > 0:
        logger.debug("  First entry: %s", trace_data[0])
    logger.debug("=====================")


def update_dashboard_data(new_model_data=None, new_trace_data=None, new_backend=None):
    global model_data, trace_data, backend, shape_history, _trace_data_list

    if new_model_data is not None:
        model_data = new_model_data

    if new_trace_data is not None:
        processed_trace_data = []
        for entry in new_trace_data:
            processed_entry = {}
            for key, value in entry.items():
                if hasattr(value, 'item') and callable(getattr(value, 'item')):
                    processed_entry[key] = value.item()
                else:
                    processed_entry[key] = value
            processed_trace_data.append(processed_entry)
            
            layer_name = processed_entry.get("layer", "Unknown")
            anomaly_detector.add_sample(layer_name, processed_entry)
            
            layer_inspection_history.append({
                "layer": layer_name,
                "data": processed_entry,
                "timestamp": time.time()
            })
        
        _trace_data_list.clear()
        _trace_data_list.extend(processed_trace_data)

    if new_backend is not None:
        backend = new_backend

    shape_history = []
    print_dashboard_data()


print_dashboard_data()


@app.callback(
    [Output("interval_component", "interval")],
    [Input("update_interval", "value")]
)
def update_interval(new_interval):
    return [new_interval]


propagator = ShapePropagator()
if socketio is not None:
    threading.Thread(target=socketio.run, args=(server,), kwargs={"host": "localhost", "port": 5001, "allow_unsafe_werkzeug": True}, daemon=True).start()


@app.callback(
    [Output("trace_graph", "figure")],
    [Input("interval_component", "n_intervals"), Input("viz_type", "value"), Input("layer_filter", "value")]
)
def update_trace_graph(n, viz_type, selected_layers=None):
    global trace_data
    data_source = get_trace_data()

    if not data_source or any(not isinstance(entry["execution_time"], (int, float)) for entry in data_source):
        return [go.Figure()]

    if selected_layers:
        filtered_data = [entry for entry in data_source if entry["layer"] in selected_layers]
    else:
        filtered_data = data_source

    if not filtered_data:
        return [go.Figure()]

    layers = [entry["layer"] for entry in filtered_data]
    execution_times = [entry["execution_time"] for entry in filtered_data]

    compute_times = []
    transfer_times = []
    for t in execution_times:
        if isinstance(t, (int, float)):
            compute_times.append(t * 0.7)
            transfer_times.append(t * 0.3)
        else:
            compute_times.append(0.1)
            transfer_times.append(0.05)

    fig = go.Figure()

    if viz_type == "basic":
        fig = go.Figure([go.Bar(x=layers, y=execution_times, name="Execution Time (s)")])
        fig.update_layout(
            title="Layer Execution Time",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    elif viz_type == "stacked":
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
        unique_layers = list(dict.fromkeys(entry["layer"] for entry in filtered_data))
        times_by_layer = {layer: [entry["execution_time"] for entry in filtered_data if entry["layer"] == layer] for layer in unique_layers}
        fig = go.Figure([go.Box(x=unique_layers, y=[times_by_layer[layer] for layer in unique_layers], name="Execution Variability")])
        fig.update_layout(
            title="Layer Execution Time Variability",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    elif viz_type == "gantt":
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
        iterations = 5
        heatmap_data = np.random.rand(len(layers), iterations)
        fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=[f"Iteration {i+1}" for i in range(iterations)], y=layers))
        fig.update_layout(title="Execution Time Heatmap", xaxis_title="Iterations", yaxis_title="Layers")

    elif viz_type == "thresholds":
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

    fig.update_layout(
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1000
    )

    return [fig]


@app.callback(
    Output("flops_memory_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_flops_memory_chart(n):
    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    flops = [entry["flops"] for entry in trace_data]
    memory = [entry["memory"] for entry in trace_data]

    fig = go.Figure([
        go.Bar(x=layers, y=flops, name="FLOPs"),
        go.Bar(x=layers, y=memory, name="Memory Usage (MB)")
    ])
    fig.update_layout(title="FLOPs & Memory Usage", xaxis_title="Layers", yaxis_title="Values", barmode="group")

    return [fig]


@app.callback(
    Output("loss_graph", "figure"),
    Input("interval_component", "n_intervals")
)
def update_loss(n):
    loss_data = [random.uniform(0.1, 1.0) for _ in range(n)]
    fig = go.Figure(data=[go.Scatter(y=loss_data, mode="lines+markers")])
    fig.update_layout(title="Loss Over Time")
    return fig


@app.callback(
    Output("architecture_graph", "figure"),
    Input("architecture_selector", "value")
)
def update_graph(arch):
    global model_data, backend, trace_data

    print(f"Updating architecture graph with model_data: {model_data is not None}")

    fig = go.Figure()

    if model_data and isinstance(model_data, dict):
        if 'input' in model_data and 'shape' in model_data['input']:
            input_shape = model_data['input']['shape']
            print(f"Input shape: {input_shape}")

            if 'layers' in model_data and isinstance(model_data['layers'], list):
                layers = model_data['layers']
                print(f"Layers: {len(layers)}")

                layer_types = []
                for layer in layers:
                    if isinstance(layer, dict) and 'type' in layer:
                        layer_types.append(layer['type'])

                x_positions = [0]
                y_positions = [0]
                node_labels = ["Input"]
                node_colors = ["blue"]

                for i, layer_type in enumerate(layer_types):
                    x_positions.append(i + 1)
                    y_positions.append(0 if i % 2 == 0 else 1)
                    node_labels.append(layer_type)

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

                x_positions.append(len(layer_types) + 1)
                y_positions.append(0)
                node_labels.append("Output")
                node_colors.append("blue")

                fig.add_trace(go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode="markers+text",
                    marker=dict(size=30, color=node_colors),
                    text=node_labels,
                    textposition="bottom center"
                ))

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

    if arch == "A":
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3],
            y=[0, 1, 0, 1],
            mode="markers+text",
            marker=dict(size=30, color=["blue", "red", "green", "purple"]),
            text=["Input", "Conv", "Pool", "Output"],
            textposition="bottom center"
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 2, 2, 3],
            y=[0, 1, 1, 0, 0, 1],
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3, 4, 5],
            y=[0, 1, 0, 1, 0, 1],
            mode="markers+text",
            marker=dict(size=30, color=["blue", "red", "green", "red", "purple", "blue"]),
            text=["Input", "Conv1", "Pool", "Conv2", "Dense", "Output"],
            textposition="bottom center"
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            y=[0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none"
        ))

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


@app.callback(
    Output("gradient_flow_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_gradient_chart(n):
    global trace_data

    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    grad_norms = [entry.get("grad_norm", 0) for entry in trace_data]

    fig = go.Figure([go.Bar(x=layers, y=grad_norms, name="Gradient Magnitude")])
    fig.update_layout(title="Gradient Flow", xaxis_title="Layers", yaxis_title="Gradient Magnitude")

    return fig


@app.callback(
    Output("dead_neuron_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_dead_neurons(n):
    global trace_data

    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    dead_ratios = [entry.get("dead_ratio", 0) for entry in trace_data]

    fig = go.Figure([go.Bar(x=layers, y=dead_ratios, name="Dead Neurons (%)")])
    fig.update_layout(title="Dead Neuron Detection", xaxis_title="Layers", yaxis_title="Dead Ratio", yaxis_range=[0, 1])

    return fig


@app.callback(
    Output("anomaly_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_anomaly_chart(n):
    global trace_data

    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    activations = [entry.get("mean_activation", 0) for entry in trace_data]
    
    anomalies_list = []
    anomaly_scores = []
    for entry in trace_data:
        layer_name = entry["layer"]
        result = anomaly_detector.detect_anomalies(layer_name, entry)
        anomalies_list.append(1 if result["is_anomaly"] else 0)
        
        if result["scores"]:
            max_score = max(result["scores"].values())
            anomaly_scores.append(max_score)
        else:
            anomaly_scores.append(0)

    fig = go.Figure([
        go.Bar(x=layers, y=activations, name="Mean Activation", marker_color="lightblue"),
        go.Bar(x=layers, y=anomalies_list, name="Anomaly Detected", marker_color="red"),
        go.Scatter(x=layers, y=anomaly_scores, name="Anomaly Score", mode="lines+markers", 
                  line=dict(color="orange", width=2), yaxis="y2")
    ])
    
    fig.update_layout(
        title="Activation Anomalies with Z-Score Detection",
        xaxis_title="Layers",
        yaxis_title="Activation Magnitude",
        yaxis2=dict(title="Anomaly Score (Z-Score)", overlaying="y", side="right"),
        template="plotly_white"
    )

    return fig


@app.callback(
    Output("step_debug_output", "children"),
    Input("step_debug_button", "n_clicks")
)
def trigger_step_debug(n):
    if n:
        requests.get("http://localhost:5001/trigger_step_debug")
        return "Paused. Check terminal for tensor inspection."
    return "Click to pause execution."


@app.callback(
    Output("resource_graph", "figure"),
    Input("interval_component", "n_intervals")
)
def update_resource_graph(n):
    try:
        import psutil

        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        gpu_memory = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3) * 100
        except (ImportError, Exception):
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
        print(f"Error in resource monitoring: {e}")
        fig = go.Figure()
        fig.update_layout(
            title="Resource Monitoring (Error)",
            height=400
        )

    return fig


@app.callback(
    Output("tensor_flow_graph", "figure"),
    Input("interval_component", "n_intervals")
)
def update_tensor_flow(n):
    global shape_history, model_data, backend

    if shape_history:
        return create_animated_network(shape_history)

    if 'propagator' in globals() and hasattr(propagator, 'shape_history'):
        return create_animated_network(propagator.shape_history)

    if model_data and isinstance(model_data, dict):
        if 'input' in model_data and 'shape' in model_data['input'] and 'layers' in model_data:
            input_shape = model_data['input']['shape']
            layers = model_data['layers']

            if isinstance(layers, list) and layers:
                local_propagator = ShapePropagator()

                for layer in layers:
                    try:
                        input_shape = local_propagator.propagate(input_shape, layer, backend)
                    except Exception as e:
                        print(f"Error propagating shape for layer {layer.get('type', 'unknown')}: {e}")

                shape_history = local_propagator.shape_history

                return create_animated_network(shape_history)

    return go.Figure()


@app.callback(
    Output("layer_inspector", "children"),
    [Input("layer_selector", "value")]
)
def update_layer_inspector(selected_layer):
    if not selected_layer or not trace_data:
        return html.Div("Select a layer to inspect")
    
    layer_entries = [entry for entry in trace_data if entry.get("layer") == selected_layer]
    if not layer_entries:
        return html.Div(f"No data available for {selected_layer}")
    
    latest_entry = layer_entries[-1]
    stats = anomaly_detector.get_layer_statistics(selected_layer)
    
    details = [
        html.H4(f"Layer: {selected_layer}"),
        html.Hr(),
        html.H5("Current Metrics:"),
        html.P(f"Execution Time: {latest_entry.get('execution_time', 0):.6f}s"),
        html.P(f"Memory Usage: {latest_entry.get('memory', 0):.2f} MB"),
        html.P(f"FLOPs: {latest_entry.get('flops', 0):,}"),
        html.P(f"Mean Activation: {latest_entry.get('mean_activation', 0):.4f}"),
        html.P(f"Gradient Norm: {latest_entry.get('grad_norm', 0):.4f}"),
        html.Hr(),
        html.H5("Statistical Summary:"),
    ]
    
    if stats:
        for metric_name, metric_stats in stats.items():
            details.append(html.H6(f"{metric_name.replace('_', ' ').title()}:"))
            details.append(html.P(f"  Mean: {metric_stats['mean']:.6f}, Std: {metric_stats['std']:.6f}"))
            details.append(html.P(f"  Min: {metric_stats['min']:.6f}, Max: {metric_stats['max']:.6f}"))
    else:
        details.append(html.P("Insufficient data for statistics"))
    
    return html.Div(details)


@app.callback(
    Output("breakpoint_list", "children"),
    [Input("interval_component", "n_intervals")]
)
def update_breakpoint_list(n):
    bp_info = breakpoint_manager.get_breakpoint_info()
    
    if not bp_info:
        return html.Div("No breakpoints set", style={"padding": "10px"})
    
    bp_items = []
    for layer_name, info in bp_info.items():
        status = "✓" if info["enabled"] else "✗"
        bp_items.append(
            html.Div([
                html.Span(f"{status} {layer_name}", style={"fontWeight": "bold"}),
                html.Span(f" (hits: {info['hit_count']})", style={"marginLeft": "10px", "color": "gray"}),
                html.Button("Toggle", id={"type": "toggle-bp", "index": layer_name}, 
                           style={"marginLeft": "10px", "fontSize": "10px"}),
                html.Button("Remove", id={"type": "remove-bp", "index": layer_name}, 
                           style={"marginLeft": "5px", "fontSize": "10px", "backgroundColor": "red"})
            ], style={"padding": "5px", "borderBottom": "1px solid #ddd"})
        )
    
    return html.Div(bp_items)


@app.callback(
    Output("flame_graph", "figure"),
    [Input("interval_component", "n_intervals")]
)
def update_flame_graph(n):
    flame_data = profiler.get_flame_graph_data()
    
    if not flame_data:
        return go.Figure().update_layout(title="Flame Graph (No Data Available)")
    
    fig = go.Figure()
    
    max_depth = max([item["depth"] for item in flame_data]) if flame_data else 0
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    
    for item in flame_data:
        color = colors[item["depth"] % len(colors)]
        fig.add_trace(go.Bar(
            x=[item["duration"]],
            y=[item["depth"]],
            orientation='h',
            base=item["start"],
            marker=dict(color=color, line=dict(color='white', width=1)),
            name=item["name"],
            text=f"{item['name']}<br>{item['duration']:.4f}s",
            textposition="inside",
            hovertemplate=f"<b>{item['name']}</b><br>Duration: {item['duration']:.6f}s<br>Start: {item['start']:.6f}s<extra></extra>"
        ))
    
    fig.update_layout(
        title="Performance Flame Graph",
        xaxis_title="Time (seconds)",
        yaxis_title="Call Depth",
        showlegend=False,
        height=600,
        barmode='overlay',
        template="plotly_white"
    )
    
    return fig


@app.callback(
    Output("performance_summary", "children"),
    [Input("interval_component", "n_intervals")]
)
def update_performance_summary(n):
    summary = profiler.get_summary()
    
    if not summary:
        return html.Div("No performance data available")
    
    sorted_layers = sorted(summary.items(), key=lambda x: x[1]["total_time"], reverse=True)
    
    rows = [
        html.Tr([
            html.Th("Layer"),
            html.Th("Calls"),
            html.Th("Total Time"),
            html.Th("Avg Time"),
            html.Th("Min Time"),
            html.Th("Max Time")
        ])
    ]
    
    for layer_name, stats in sorted_layers[:10]:
        rows.append(html.Tr([
            html.Td(layer_name),
            html.Td(str(stats["call_count"])),
            html.Td(f"{stats['total_time']:.6f}s"),
            html.Td(f"{stats['avg_time']:.6f}s"),
            html.Td(f"{stats['min_time']:.6f}s"),
            html.Td(f"{stats['max_time']:.6f}s")
        ]))
    
    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


app = Dash(__name__, external_stylesheets=[themes.DARKLY])

app.css.append_css({
    "external_url": "https://custom-theme.com/neural.css"
})


app.layout = html.Div([
    html.H1("NeuralDbg: Advanced Neural Network Debugger", 
            style={"textAlign": "center", "marginBottom": "30px"}),

    html.Div([
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

        html.Div([
            html.H2("Gradient Flow Analysis", style={"textAlign": "center"}),
            html.Div([
                dcc.Graph(id="gradient_flow_chart"),
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),

            html.H2("Dead Neuron Detection", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                dcc.Graph(id="dead_neuron_chart"),
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),

            html.H2("Enhanced Anomaly Detection", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                dcc.Graph(id="anomaly_chart"),
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "4%"}),
    ]),

    html.Div([
        html.Div([
            html.H2("Layer-by-Layer Inspector", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                dcc.Dropdown(
                    id="layer_selector",
                    options=[],
                    placeholder="Select a layer to inspect",
                    style={"marginBottom": "10px"}
                ),
                html.Div(id="layer_inspector", style={"padding": "10px", "minHeight": "200px"})
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            html.H2("Breakpoint Manager", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                html.Div([
                    dcc.Input(id="bp_layer_name", placeholder="Layer name", 
                             style={"width": "40%", "marginRight": "5px"}),
                    dcc.Input(id="bp_condition", placeholder="Condition (optional)", 
                             style={"width": "40%", "marginRight": "5px"}),
                    html.Button("Add Breakpoint", id="add_bp_button", 
                               style={"width": "15%", "backgroundColor": "#2196F3", "color": "white"})
                ], style={"marginBottom": "10px"}),
                html.Div(id="breakpoint_list", style={"maxHeight": "150px", "overflowY": "auto"})
            ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "4%"}),
    ]),

    html.Div([
        html.H2("Performance Flame Graph", style={"textAlign": "center", "marginTop": "20px"}),
        html.Div([
            dcc.Graph(id="flame_graph"),
        ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
    ]),

    html.Div([
        html.H2("Performance Summary", style={"textAlign": "center", "marginTop": "20px"}),
        html.Div([
            html.Div(id="performance_summary", style={"padding": "10px"})
        ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
    ]),

    html.Div([
        html.H2("Resource Monitoring", style={"textAlign": "center", "marginTop": "20px"}),
        html.Div([
            dcc.Graph(id="resource_graph"),
        ], style={"border": "1px solid #ddd", "padding": "15px", "borderRadius": "5px"}),
    ]),

    dcc.Interval(id="interval_component", interval=UPDATE_INTERVAL, n_intervals=0),
    html.Div(id="progress-store", style={"display": "none"})
])


@app.callback(
    [Output("architecture_viz_graph", "figure"),
     Output("progress-store", "children")],
    [Input("generate-viz-button", "n_clicks")],
    [State("progress-store", "children")]
)
def update_network_visualization(n_clicks, _):
    global model_data, backend, shape_history

    print(f"Updating network visualization with n_clicks={n_clicks}")
    print(f"Model data available: {model_data is not None}")
    if model_data:
        print(f"Model data keys: {model_data.keys() if isinstance(model_data, dict) else 'Not a dict'}")

    fig = go.Figure()

    progress = 10
    details = "Starting visualization..."

    if model_data and isinstance(model_data, dict):
        if 'input' in model_data and 'layers' in model_data and isinstance(model_data['layers'], list):
            progress = 20
            details = "Processing model data..."

            layers = model_data['layers']
            layer_types = []
            for layer in layers:
                if isinstance(layer, dict) and 'type' in layer:
                    layer_type = layer['type']
                    layer_types.append(layer_type)
                    print(f"Found layer: {layer_type}")

            x_positions = [0]
            y_positions = [0]
            node_labels = ["Input"]
            node_colors = ["blue"]

            for i, layer_type in enumerate(layer_types):
                x_positions.append(i + 1)
                y_positions.append(0 if i % 2 == 0 else 1)
                node_labels.append(layer_type)

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

            x_positions.append(len(layer_types) + 1)
            y_positions.append(0)
            node_labels.append("Output")
            node_colors.append("blue")

            fig.add_trace(go.Scatter(
                x=x_positions,
                y=y_positions,
                mode="markers+text",
                marker=dict(size=30, color=node_colors),
                text=node_labels,
                textposition="bottom center"
            ))

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

    progress = 50
    details = "Using default visualization..."

    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4, 5],
        y=[0, 1, 0, 1, 0, 1],
        mode="markers+text",
        marker=dict(size=30, color=["blue", "red", "green", "red", "purple", "blue"]),
        text=["Input", "Conv2D", "MaxPool", "Conv2D", "Dense", "Output"],
        textposition="bottom center"
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        y=[0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        mode="lines",
        line=dict(width=2, color="gray"),
        hoverinfo="none"
    ))

    fig.update_layout(
        title="Network Architecture (Default)",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    return fig, json.dumps({"progress": 100, "details": "Default visualization complete"})


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

    bar_style = {
        "width": f"{progress}%",
        "backgroundColor": "#4CAF50",
        "height": "30px"
    }

    return bar_style, f"{progress:.1f}%", details


@app.callback(
    Output("computation-timeline", "figure"),
    [Input("interval_component", "n_intervals")]
)
def update_computation_timeline(n_intervals):
    global trace_data

    print(f"Updating computation timeline with trace_data: {len(trace_data) if trace_data else 0} entries")

    fig = go.Figure()

    if trace_data and len(trace_data) > 0:
        layers = [entry.get("layer", "Unknown") for entry in trace_data]
        execution_times = [entry.get("execution_time", 0) for entry in trace_data]

        start_times = [0]
        for i in range(1, len(execution_times)):
            start_times.append(start_times[i-1] + execution_times[i-1])

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
        layer_data = [
            {"layer": "Input", "execution_time": 0.1},
            {"layer": "Conv2D", "execution_time": 0.8},
            {"layer": "MaxPooling2D", "execution_time": 0.3},
            {"layer": "Flatten", "execution_time": 0.1},
            {"layer": "Dense", "execution_time": 0.5},
            {"layer": "Output", "execution_time": 0.2}
        ]

        layers = [entry["layer"] for entry in layer_data]
        execution_times = [entry["execution_time"] for entry in layer_data]

        start_times = [0]
        for i in range(1, len(execution_times)):
            start_times.append(start_times[i-1] + execution_times[i-1])

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


@app.callback(
    Output("layer_selector", "options"),
    [Input("interval_component", "n_intervals")]
)
def update_layer_options(n):
    if not trace_data:
        return []
    
    unique_layers = list(dict.fromkeys([entry.get("layer", "Unknown") for entry in trace_data]))
    return [{"label": layer, "value": layer} for layer in unique_layers]


@app.callback(
    Output("bp_layer_name", "value"),
    [Input("add_bp_button", "n_clicks")],
    [State("bp_layer_name", "value"), State("bp_condition", "value")]
)
def add_breakpoint(n_clicks, layer_name, condition):
    if n_clicks and layer_name:
        breakpoint_manager.add_breakpoint(layer_name, condition)
        return ""
    return layer_name or ""


if __name__ == "__main__":
    app.run_server(debug=False, use_reloader=False)
