import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html

from neural.code_generation.code_generator import generate_code
from neural.parser.parser import ModelTransformer, create_parser


class RunnerPanel:
    def __init__(self, app):
        self.app = app
        self.process = None
        self.output_queue = Queue()
        self.setup_callbacks()

    def create_layout(self):
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Model Compilation & Execution", className="mb-3"),
                    
                    dbc.Card([
                        dbc.CardHeader("Backend Configuration"),
                        dbc.CardBody([
                            html.Label("Select Backend:", className="form-label"),
                            dcc.Dropdown(
                                id="runner-backend-select",
                                options=[
                                    {"label": "TensorFlow", "value": "tensorflow"},
                                    {"label": "PyTorch", "value": "pytorch"},
                                    {"label": "ONNX", "value": "onnx"}
                                ],
                                value="tensorflow",
                                className="mb-3"
                            ),
                            
                            html.Label("Dataset:", className="form-label"),
                            dcc.Dropdown(
                                id="runner-dataset-select",
                                options=[
                                    {"label": "MNIST", "value": "MNIST"},
                                    {"label": "CIFAR10", "value": "CIFAR10"},
                                    {"label": "CIFAR100", "value": "CIFAR100"},
                                    {"label": "ImageNet", "value": "ImageNet"},
                                    {"label": "Custom", "value": "custom"}
                                ],
                                value="MNIST",
                                className="mb-3"
                            ),
                            
                            html.Div([
                                html.Label("Custom Dataset Path:", className="form-label"),
                                dbc.Input(
                                    id="runner-custom-dataset-path",
                                    type="text",
                                    placeholder="/path/to/dataset",
                                    disabled=True
                                )
                            ], id="custom-dataset-container", className="mb-3"),
                            
                            html.Label("Training Configuration:", className="form-label"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Epochs:", style={"fontSize": "0.9em"}),
                                    dbc.Input(
                                        id="runner-epochs",
                                        type="number",
                                        value=10,
                                        min=1,
                                        max=1000
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Batch Size:", style={"fontSize": "0.9em"}),
                                    dbc.Input(
                                        id="runner-batch-size",
                                        type="number",
                                        value=32,
                                        min=1,
                                        max=1024
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Validation Split:", style={"fontSize": "0.9em"}),
                                    dbc.Input(
                                        id="runner-val-split",
                                        type="number",
                                        value=0.2,
                                        min=0,
                                        max=1,
                                        step=0.05
                                    )
                                ], width=4)
                            ], className="mb-3"),
                            
                            dbc.Checklist(
                                id="runner-options",
                                options=[
                                    {"label": "Auto-flatten output", "value": "auto_flatten"},
                                    {"label": "Enable HPO", "value": "hpo"},
                                    {"label": "Verbose output", "value": "verbose"},
                                    {"label": "Save model weights", "value": "save_weights"}
                                ],
                                value=["verbose"],
                                className="mb-3"
                            )
                        ])
                    ], className="mb-3"),
                    
                    dbc.Card([
                        dbc.CardHeader("Actions"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        [html.I(className="fas fa-cog me-2"), "Compile"],
                                        id="runner-compile-btn",
                                        color="primary",
                                        className="w-100 mb-2"
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-play me-2"), "Run"],
                                        id="runner-run-btn",
                                        color="success",
                                        className="w-100 mb-2",
                                        disabled=True
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-stop me-2"), "Stop"],
                                        id="runner-stop-btn",
                                        color="danger",
                                        className="w-100 mb-2",
                                        disabled=True
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Button(
                                        [html.I(className="fas fa-download me-2"), "Export Script"],
                                        id="runner-export-btn",
                                        color="info",
                                        className="w-100 mb-2",
                                        disabled=True
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-folder-open me-2"), "Open in IDE"],
                                        id="runner-open-ide-btn",
                                        color="secondary",
                                        className="w-100 mb-2",
                                        disabled=True
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-trash me-2"), "Clear"],
                                        id="runner-clear-btn",
                                        color="warning",
                                        className="w-100 mb-2"
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("Compilation & Execution Logs", className="me-3"),
                            dbc.Badge("Idle", id="runner-status-badge", color="secondary")
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="runner-loading",
                                type="default",
                                children=[
                                    html.Div([
                                        html.Pre(
                                            id="runner-console-output",
                                            children="Ready to compile and run models...\n",
                                            style={
                                                "backgroundColor": "#1e1e1e",
                                                "color": "#d4d4d4",
                                                "padding": "15px",
                                                "borderRadius": "5px",
                                                "minHeight": "500px",
                                                "maxHeight": "70vh",
                                                "overflowY": "auto",
                                                "fontFamily": "Consolas, Monaco, 'Courier New', monospace",
                                                "fontSize": "0.85em",
                                                "whiteSpace": "pre-wrap",
                                                "wordBreak": "break-word"
                                            }
                                        )
                                    ])
                                ]
                            )
                        ])
                    ], className="mb-3"),
                    
                    dbc.Card([
                        dbc.CardHeader("Training Metrics"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="runner-metrics-graph",
                                style={"height": "250px"},
                                config={"displayModeBar": False}
                            )
                        ])
                    ])
                ], width=8)
            ]),
            
            dcc.Store(id="runner-compiled-script-path", data=None),
            dcc.Store(id="runner-dsl-content", data=None),
            dcc.Store(id="runner-model-data", data=None),
            dcc.Interval(id="runner-log-interval", interval=500, disabled=True),
            
            dbc.Modal([
                dbc.ModalHeader("Export Script"),
                dbc.ModalBody([
                    html.Label("Script Name:"),
                    dbc.Input(
                        id="runner-export-filename",
                        type="text",
                        placeholder="model_script.py",
                        value=""
                    ),
                    html.Label("Export Location:", className="mt-3"),
                    dbc.Input(
                        id="runner-export-location",
                        type="text",
                        placeholder="./exported_scripts",
                        value="./exported_scripts"
                    )
                ]),
                dbc.ModalFooter([
                    dbc.Button("Cancel", id="runner-export-cancel", className="ms-auto"),
                    dbc.Button("Export", id="runner-export-confirm", color="primary")
                ])
            ], id="runner-export-modal"),
            
            html.Div(id="runner-notifications")
        ], className="p-3")

    def setup_callbacks(self):
        @self.app.callback(
            [Output("custom-dataset-container", "style"),
             Output("runner-custom-dataset-path", "disabled")],
            Input("runner-dataset-select", "value")
        )
        def toggle_custom_dataset(dataset):
            if dataset == "custom":
                return {"display": "block"}, False
            return {"display": "none"}, True

        @self.app.callback(
            [Output("runner-console-output", "children"),
             Output("runner-status-badge", "children"),
             Output("runner-status-badge", "color"),
             Output("runner-compiled-script-path", "data"),
             Output("runner-run-btn", "disabled"),
             Output("runner-export-btn", "disabled"),
             Output("runner-open-ide-btn", "disabled")],
            Input("runner-compile-btn", "n_clicks"),
            [State("runner-backend-select", "value"),
             State("runner-dataset-select", "value"),
             State("runner-dsl-content", "data"),
             State("runner-model-data", "data"),
             State("runner-options", "value"),
             State("runner-console-output", "children")]
        )
        def compile_model(n_clicks, backend, dataset, dsl_content, model_data, options, current_output):
            if not n_clicks or not model_data:
                return (current_output, "Idle", "secondary", None, True, True, True)
            
            try:
                output_lines = [current_output, "\n" + "="*60 + "\n"]
                output_lines.append(f"[COMPILE] Starting compilation for {backend} backend...\n")
                
                auto_flatten = "auto_flatten" in (options or [])
                
                output_lines.append(f"[COMPILE] Backend: {backend}\n")
                output_lines.append(f"[COMPILE] Dataset: {dataset}\n")
                output_lines.append(f"[COMPILE] Auto-flatten: {auto_flatten}\n")
                output_lines.append("[COMPILE] Generating code...\n")
                
                generated_code = generate_code(model_data, backend, auto_flatten_output=auto_flatten)
                
                output_lines.append(f"[COMPILE] Generated {len(generated_code)} bytes of code\n")
                
                temp_dir = Path(tempfile.gettempdir()) / "neural_aquarium"
                temp_dir.mkdir(exist_ok=True)
                
                script_path = temp_dir / f"model_{backend}.py"
                with open(script_path, "w") as f:
                    f.write(generated_code)
                
                output_lines.append(f"[COMPILE] Script saved to: {script_path}\n")
                output_lines.append("[COMPILE] ✓ Compilation successful!\n")
                output_lines.append("="*60 + "\n")
                
                return (
                    "".join(output_lines),
                    "Compiled",
                    "success",
                    str(script_path),
                    False,
                    False,
                    False
                )
                
            except Exception as e:
                output_lines.append(f"[ERROR] Compilation failed: {str(e)}\n")
                output_lines.append("="*60 + "\n")
                return (
                    "".join(output_lines),
                    "Error",
                    "danger",
                    None,
                    True,
                    True,
                    True
                )

        @self.app.callback(
            [Output("runner-console-output", "children", allow_duplicate=True),
             Output("runner-status-badge", "children", allow_duplicate=True),
             Output("runner-status-badge", "color", allow_duplicate=True),
             Output("runner-stop-btn", "disabled"),
             Output("runner-log-interval", "disabled")],
            Input("runner-run-btn", "n_clicks"),
            [State("runner-compiled-script-path", "data"),
             State("runner-epochs", "value"),
             State("runner-batch-size", "value"),
             State("runner-val-split", "value"),
             State("runner-options", "value"),
             State("runner-console-output", "children")],
            prevent_initial_call=True
        )
        def run_model(n_clicks, script_path, epochs, batch_size, val_split, options, current_output):
            if not n_clicks or not script_path or not os.path.exists(script_path):
                return current_output, "Compiled", "success", True, True
            
            try:
                output_lines = [current_output, "\n" + "="*60 + "\n"]
                output_lines.append("[RUN] Starting model execution...\n")
                output_lines.append(f"[RUN] Script: {script_path}\n")
                output_lines.append(f"[RUN] Epochs: {epochs}\n")
                output_lines.append(f"[RUN] Batch size: {batch_size}\n")
                output_lines.append(f"[RUN] Validation split: {val_split}\n")
                output_lines.append("="*60 + "\n\n")
                
                self._start_execution(script_path)
                
                return (
                    "".join(output_lines),
                    "Running",
                    "info",
                    False,
                    False
                )
                
            except Exception as e:
                output_lines.append(f"[ERROR] Execution failed: {str(e)}\n")
                return (
                    "".join(output_lines),
                    "Error",
                    "danger",
                    True,
                    True
                )

        @self.app.callback(
            [Output("runner-console-output", "children", allow_duplicate=True),
             Output("runner-status-badge", "children", allow_duplicate=True),
             Output("runner-status-badge", "color", allow_duplicate=True),
             Output("runner-stop-btn", "disabled", allow_duplicate=True),
             Output("runner-log-interval", "disabled", allow_duplicate=True)],
            Input("runner-stop-btn", "n_clicks"),
            State("runner-console-output", "children"),
            prevent_initial_call=True
        )
        def stop_execution(n_clicks, current_output):
            if not n_clicks:
                return current_output, "Running", "info", False, False
            
            if self.process:
                self.process.terminate()
                self.process = None
                
            output_lines = [current_output, "\n[STOP] Execution stopped by user\n"]
            
            return (
                "".join(output_lines),
                "Stopped",
                "warning",
                True,
                True
            )

        @self.app.callback(
            Output("runner-console-output", "children", allow_duplicate=True),
            Input("runner-log-interval", "n_intervals"),
            State("runner-console-output", "children"),
            prevent_initial_call=True
        )
        def update_logs(n_intervals, current_output):
            if not self.output_queue.empty():
                new_lines = []
                try:
                    while not self.output_queue.empty():
                        new_lines.append(self.output_queue.get_nowait())
                except Empty:
                    pass
                
                if new_lines:
                    return current_output + "".join(new_lines)
            
            return current_output

        @self.app.callback(
            Output("runner-console-output", "children", allow_duplicate=True),
            Input("runner-clear-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def clear_console(n_clicks):
            if n_clicks:
                return "Console cleared.\n"
            return "Ready to compile and run models...\n"

        @self.app.callback(
            Output("runner-export-modal", "is_open"),
            [Input("runner-export-btn", "n_clicks"),
             Input("runner-export-confirm", "n_clicks"),
             Input("runner-export-cancel", "n_clicks")],
            State("runner-export-modal", "is_open")
        )
        def toggle_export_modal(export_click, confirm_click, cancel_click, is_open):
            ctx = callback_context
            if not ctx.triggered:
                return is_open
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger_id == "runner-export-btn" and export_click:
                return True
            elif trigger_id in ["runner-export-confirm", "runner-export-cancel"]:
                return False
            
            return is_open

        @self.app.callback(
            Output("runner-notifications", "children"),
            Input("runner-export-confirm", "n_clicks"),
            [State("runner-compiled-script-path", "data"),
             State("runner-export-filename", "value"),
             State("runner-export-location", "value"),
             State("runner-backend-select", "value")],
            prevent_initial_call=True
        )
        def export_script(n_clicks, script_path, filename, location, backend):
            if not n_clicks or not script_path:
                return None
            
            try:
                if not filename:
                    filename = f"model_{backend}.py"
                
                if not filename.endswith(".py"):
                    filename += ".py"
                
                export_dir = Path(location)
                export_dir.mkdir(parents=True, exist_ok=True)
                
                export_path = export_dir / filename
                
                with open(script_path, "r") as src:
                    with open(export_path, "w") as dst:
                        dst.write(src.read())
                
                return dbc.Alert(
                    f"Script exported successfully to: {export_path}",
                    color="success",
                    dismissable=True,
                    duration=4000
                )
                
            except Exception as e:
                return dbc.Alert(
                    f"Export failed: {str(e)}",
                    color="danger",
                    dismissable=True,
                    duration=4000
                )

        @self.app.callback(
            Output("runner-notifications", "children", allow_duplicate=True),
            Input("runner-open-ide-btn", "n_clicks"),
            State("runner-compiled-script-path", "data"),
            prevent_initial_call=True
        )
        def open_in_ide(n_clicks, script_path):
            if not n_clicks or not script_path:
                return None
            
            try:
                if sys.platform == "win32":
                    os.startfile(script_path)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", script_path])
                else:
                    subprocess.Popen(["xdg-open", script_path])
                
                return dbc.Alert(
                    f"Opening script in default editor: {script_path}",
                    color="info",
                    dismissable=True,
                    duration=3000
                )
                
            except Exception as e:
                return dbc.Alert(
                    f"Failed to open IDE: {str(e)}",
                    color="danger",
                    dismissable=True,
                    duration=4000
                )

    def _start_execution(self, script_path: str):
        def run_process():
            try:
                self.process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                for line in self.process.stdout:
                    self.output_queue.put(line)
                
                self.process.wait()
                
                if self.process.returncode == 0:
                    self.output_queue.put("\n[RUN] ✓ Execution completed successfully!\n")
                else:
                    self.output_queue.put(f"\n[RUN] ✗ Execution failed with code {self.process.returncode}\n")
                    
            except Exception as e:
                self.output_queue.put(f"\n[ERROR] {str(e)}\n")
            finally:
                self.process = None
        
        thread = threading.Thread(target=run_process, daemon=True)
        thread.start()

    def update_model_data(self, model_data, dsl_content):
        return model_data, dsl_content
