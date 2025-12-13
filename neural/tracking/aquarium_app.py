"""
Aquarium: Comprehensive experiment tracking dashboard for Neural.

This module provides a web-based interface for viewing, comparing, and exporting
experiments tracked with Neural's experiment tracking system.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html
from dash.exceptions import PreventUpdate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.tracking.experiment_tracker import ExperimentManager, ExperimentTracker
from neural.tracking.integrations import (
    MLflowIntegration,
    TensorBoardIntegration,
    WandbIntegration,
)


class AquariumDashboard:
    """Main dashboard application for experiment tracking."""

    def __init__(self, base_dir: str = "neural_experiments", port: int = 8053):
        """
        Initialize the Aquarium dashboard.

        Args:
            base_dir: Base directory containing experiments
            port: Port to run the dashboard on
        """
        self.base_dir = base_dir
        self.port = port
        self.manager = ExperimentManager(base_dir=base_dir)

        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True,
        )

        self.app.title = "Aquarium - Neural Experiment Tracking"
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = dbc.Container(
            [
                dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
                dcc.Store(id="selected-experiments", data=[]),
                dcc.Store(id="current-view", data="list"),
                self._create_header(),
                html.Hr(),
                dbc.Tabs(
                    id="main-tabs",
                    active_tab="experiments",
                    children=[
                        dbc.Tab(label="Experiments", tab_id="experiments"),
                        dbc.Tab(label="Compare", tab_id="compare"),
                        dbc.Tab(label="Export", tab_id="export"),
                    ],
                ),
                html.Div(id="tab-content", style={"marginTop": "20px"}),
            ],
            fluid=True,
            style={"padding": "20px"},
        )

    def _create_header(self):
        """Create the dashboard header."""
        return dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            [
                                html.I(className="fas fa-fish me-2"),
                                "Aquarium",
                            ],
                            className="text-primary",
                        ),
                        html.P(
                            "Neural Experiment Tracking Dashboard",
                            className="text-muted",
                        ),
                    ],
                    width=8,
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            [html.I(className="fas fa-sync-alt me-2"), "Refresh"],
                            id="refresh-button",
                            color="primary",
                            className="float-end",
                        ),
                    ],
                    width=4,
                ),
            ]
        )

    def _create_experiment_list(self):
        """Create the experiment list view."""
        experiments = self.manager.list_experiments()

        if not experiments:
            return dbc.Alert(
                [
                    html.I(className="fas fa-info-circle me-2"),
                    "No experiments found. Start tracking experiments with Neural!",
                ],
                color="info",
            )

        table_data = []
        for exp in experiments:
            metrics = {}
            if exp.get("summary") and exp["summary"].get("metrics"):
                metrics = exp["summary"]["metrics"].get("latest", {})

            table_data.append(
                {
                    "Select": exp["experiment_id"],
                    "Name": exp["experiment_name"],
                    "ID": exp["experiment_id"][:8],
                    "Status": exp["status"],
                    "Start Time": exp["start_time"][:19] if exp["start_time"] != "unknown" else "N/A",
                    "Metrics": ", ".join(
                        [f"{k}: {v:.4f}" for k, v in list(metrics.items())[:3]]
                    )
                    if metrics
                    else "N/A",
                }
            )

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("Experiments"),
                                html.P(
                                    f"Total: {len(experiments)} experiments",
                                    className="text-muted",
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Select for Comparison",
                                    id="select-compare-button",
                                    color="info",
                                    className="float-end",
                                ),
                            ],
                            width=4,
                        ),
                    ]
                ),
                html.Hr(),
                dash_table.DataTable(
                    id="experiment-table",
                    columns=[
                        {"name": "Select", "id": "Select", "presentation": "input"},
                        {"name": "Name", "id": "Name"},
                        {"name": "ID", "id": "ID"},
                        {"name": "Status", "id": "Status"},
                        {"name": "Start Time", "id": "Start Time"},
                        {"name": "Metrics", "id": "Metrics"},
                    ],
                    data=table_data,
                    row_selectable="multi",
                    selected_rows=[],
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "10px",
                        "backgroundColor": "#2b3e50",
                        "color": "white",
                    },
                    style_header={
                        "backgroundColor": "#1a252f",
                        "fontWeight": "bold",
                        "color": "white",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#374855",
                        },
                        {
                            "if": {"state": "selected"},
                            "backgroundColor": "#375a7f",
                            "border": "1px solid #5b8cb8",
                        },
                    ],
                    page_size=20,
                ),
                html.Div(id="experiment-details", style={"marginTop": "20px"}),
            ],
            fluid=True,
        )

    def _create_comparison_view(self, experiment_ids: List[str]):
        """Create the comparison view for selected experiments."""
        if not experiment_ids:
            return dbc.Alert(
                [
                    html.I(className="fas fa-info-circle me-2"),
                    "Select experiments from the Experiments tab to compare.",
                ],
                color="info",
            )

        experiments = [
            self.manager.get_experiment(exp_id)
            for exp_id in experiment_ids
            if self.manager.get_experiment(exp_id)
        ]

        if not experiments:
            return dbc.Alert("No valid experiments selected.", color="warning")

        return dbc.Container(
            [
                html.H3("Experiment Comparison"),
                html.P(
                    f"Comparing {len(experiments)} experiments", className="text-muted"
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [self._create_metrics_comparison_chart(experiments)],
                            width=12,
                        ),
                    ]
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [self._create_hyperparameter_comparison_table(experiments)],
                            width=12,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

    def _create_metrics_comparison_chart(self, experiments: List[ExperimentTracker]):
        """Create a chart comparing metrics across experiments."""
        all_metrics = set()
        for exp in experiments:
            for entry in exp.metrics_history:
                all_metrics.update(
                    [k for k in entry.keys() if k not in ["timestamp", "step"]]
                )

        if not all_metrics:
            return dbc.Alert("No metrics to display.", color="info")

        metric_name = list(all_metrics)[0]

        fig = go.Figure()

        for exp in experiments:
            steps = []
            values = []

            for entry in exp.metrics_history:
                if "step" in entry and entry["step"] is not None and metric_name in entry:
                    steps.append(entry["step"])
                    values.append(entry[metric_name])

            if steps and values:
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=values,
                        mode="lines+markers",
                        name=f"{exp.experiment_name} ({exp.experiment_id[:8]})",
                    )
                )

        fig.update_layout(
            title=f"Comparison of {metric_name}",
            xaxis_title="Step",
            yaxis_title=metric_name,
            template="plotly_dark",
            hovermode="x unified",
        )

        return dcc.Graph(figure=fig)

    def _create_hyperparameter_comparison_table(
        self, experiments: List[ExperimentTracker]
    ):
        """Create a table comparing hyperparameters across experiments."""
        all_hyperparams = set()
        for exp in experiments:
            all_hyperparams.update(exp.hyperparameters.keys())

        if not all_hyperparams:
            return dbc.Alert("No hyperparameters to display.", color="info")

        table_data = []
        for param in sorted(all_hyperparams):
            row = {"Hyperparameter": param}
            for exp in experiments:
                value = exp.hyperparameters.get(param, "N/A")
                row[f"{exp.experiment_name} ({exp.experiment_id[:8]})"] = str(value)
            table_data.append(row)

        columns = [{"name": "Hyperparameter", "id": "Hyperparameter"}]
        for exp in experiments:
            columns.append(
                {
                    "name": f"{exp.experiment_name} ({exp.experiment_id[:8]})",
                    "id": f"{exp.experiment_name} ({exp.experiment_id[:8]})",
                }
            )

        return html.Div(
            [
                html.H4("Hyperparameter Comparison"),
                dash_table.DataTable(
                    columns=columns,
                    data=table_data,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "10px",
                        "backgroundColor": "#2b3e50",
                        "color": "white",
                    },
                    style_header={
                        "backgroundColor": "#1a252f",
                        "fontWeight": "bold",
                        "color": "white",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#374855",
                        },
                    ],
                ),
            ]
        )

    def _create_export_controls(self):
        """Create the export controls view."""
        return dbc.Container(
            [
                html.H3("Export Experiments"),
                html.P(
                    "Export your experiments to external tracking platforms.",
                    className="text-muted",
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                [
                                                    html.I(className="fas fa-flask me-2"),
                                                    "MLflow",
                                                ]
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.P(
                                                    "Export experiments to MLflow tracking server."
                                                ),
                                                dbc.Input(
                                                    id="mlflow-uri",
                                                    placeholder="Tracking URI (optional)",
                                                    type="text",
                                                    className="mb-2",
                                                ),
                                                dbc.Input(
                                                    id="mlflow-experiment-ids",
                                                    placeholder="Experiment IDs (comma-separated)",
                                                    type="text",
                                                    className="mb-2",
                                                ),
                                                dbc.Button(
                                                    "Export to MLflow",
                                                    id="export-mlflow-button",
                                                    color="primary",
                                                    className="w-100",
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                )
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                [
                                                    html.I(className="fas fa-weight me-2"),
                                                    "Weights & Biases",
                                                ]
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.P(
                                                    "Export experiments to Weights & Biases."
                                                ),
                                                dbc.Input(
                                                    id="wandb-project",
                                                    placeholder="Project name",
                                                    type="text",
                                                    className="mb-2",
                                                ),
                                                dbc.Input(
                                                    id="wandb-experiment-ids",
                                                    placeholder="Experiment IDs (comma-separated)",
                                                    type="text",
                                                    className="mb-2",
                                                ),
                                                dbc.Button(
                                                    "Export to W&B",
                                                    id="export-wandb-button",
                                                    color="success",
                                                    className="w-100",
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                )
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                [
                                                    html.I(
                                                        className="fas fa-chart-line me-2"
                                                    ),
                                                    "TensorBoard",
                                                ]
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.P(
                                                    "Export experiments to TensorBoard logs."
                                                ),
                                                dbc.Input(
                                                    id="tensorboard-logdir",
                                                    placeholder="Log directory",
                                                    type="text",
                                                    value="runs/neural",
                                                    className="mb-2",
                                                ),
                                                dbc.Input(
                                                    id="tensorboard-experiment-ids",
                                                    placeholder="Experiment IDs (comma-separated)",
                                                    type="text",
                                                    className="mb-2",
                                                ),
                                                dbc.Button(
                                                    "Export to TensorBoard",
                                                    id="export-tensorboard-button",
                                                    color="warning",
                                                    className="w-100",
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                )
                            ],
                            width=4,
                        ),
                    ]
                ),
                html.Hr(),
                html.Div(id="export-status", style={"marginTop": "20px"}),
            ],
            fluid=True,
        )

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            Output("tab-content", "children"),
            [
                Input("main-tabs", "active_tab"),
                Input("refresh-button", "n_clicks"),
                Input("refresh-interval", "n_intervals"),
            ],
            [State("experiment-table", "selected_rows")],
        )
        def update_tab_content(active_tab, n_clicks, n_intervals, selected_rows):
            """Update the content based on the active tab."""
            if active_tab == "experiments":
                return self._create_experiment_list()
            elif active_tab == "compare":
                experiments = self.manager.list_experiments()
                if selected_rows:
                    experiment_ids = [
                        experiments[i]["experiment_id"] for i in selected_rows
                    ]
                    return self._create_comparison_view(experiment_ids)
                return self._create_comparison_view([])
            elif active_tab == "export":
                return self._create_export_controls()
            return html.Div()

        @self.app.callback(
            Output("experiment-details", "children"),
            [Input("experiment-table", "active_cell")],
            [State("experiment-table", "data")],
        )
        def show_experiment_details(active_cell, data):
            """Show details for the selected experiment."""
            if not active_cell or not data:
                raise PreventUpdate

            row = active_cell["row"]
            exp_id = data[row]["Select"]
            exp = self.manager.get_experiment(exp_id)

            if not exp:
                return dbc.Alert("Experiment not found.", color="danger")

            return dbc.Card(
                [
                    dbc.CardHeader(html.H5(f"Experiment: {exp.experiment_name}")),
                    dbc.CardBody(
                        [
                            html.H6("Metrics History"),
                            dcc.Graph(figure=exp.plot_metrics()),
                            html.Hr(),
                            html.H6("Hyperparameters"),
                            html.Pre(
                                json.dumps(exp.hyperparameters, indent=2),
                                style={"backgroundColor": "#1a252f", "padding": "10px"},
                            ),
                        ]
                    ),
                ],
                className="mt-3",
            )

        @self.app.callback(
            Output("export-status", "children"),
            [
                Input("export-mlflow-button", "n_clicks"),
                Input("export-wandb-button", "n_clicks"),
                Input("export-tensorboard-button", "n_clicks"),
            ],
            [
                State("mlflow-uri", "value"),
                State("mlflow-experiment-ids", "value"),
                State("wandb-project", "value"),
                State("wandb-experiment-ids", "value"),
                State("tensorboard-logdir", "value"),
                State("tensorboard-experiment-ids", "value"),
            ],
        )
        def handle_export(
            mlflow_clicks,
            wandb_clicks,
            tensorboard_clicks,
            mlflow_uri,
            mlflow_ids,
            wandb_project,
            wandb_ids,
            tensorboard_logdir,
            tensorboard_ids,
        ):
            """Handle export to external platforms."""
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            try:
                if button_id == "export-mlflow-button":
                    return self._export_to_mlflow(mlflow_uri, mlflow_ids)
                elif button_id == "export-wandb-button":
                    return self._export_to_wandb(wandb_project, wandb_ids)
                elif button_id == "export-tensorboard-button":
                    return self._export_to_tensorboard(tensorboard_logdir, tensorboard_ids)
            except Exception as e:
                return dbc.Alert(f"Export failed: {str(e)}", color="danger")

            raise PreventUpdate

    def _export_to_mlflow(self, tracking_uri: Optional[str], experiment_ids: str):
        """Export experiments to MLflow."""
        if not experiment_ids:
            return dbc.Alert("Please provide experiment IDs.", color="warning")

        ids = [id.strip() for id in experiment_ids.split(",")]
        exported = []

        for exp_id in ids:
            exp = self.manager.get_experiment(exp_id)
            if not exp:
                continue

            integration = MLflowIntegration(
                experiment_name=exp.experiment_name,
                tracking_uri=tracking_uri,
                run_name=f"{exp.experiment_name}_{exp.experiment_id}",
            )

            integration.log_hyperparameters(exp.hyperparameters)

            for entry in exp.metrics_history:
                step = entry.get("step")
                metrics = {
                    k: v for k, v in entry.items() if k not in ["timestamp", "step"]
                }
                integration.log_metrics(metrics, step=step)

            for artifact_name, artifact_info in exp.artifacts.items():
                integration.log_artifact(artifact_info["path"], artifact_name)

            integration.finish()
            exported.append(exp.experiment_name)

        return dbc.Alert(
            f"Successfully exported {len(exported)} experiments to MLflow: {', '.join(exported)}",
            color="success",
        )

    def _export_to_wandb(self, project_name: Optional[str], experiment_ids: str):
        """Export experiments to Weights & Biases."""
        if not experiment_ids:
            return dbc.Alert("Please provide experiment IDs.", color="warning")

        if not project_name:
            project_name = "neural"

        ids = [id.strip() for id in experiment_ids.split(",")]
        exported = []

        for exp_id in ids:
            exp = self.manager.get_experiment(exp_id)
            if not exp:
                continue

            integration = WandbIntegration(
                experiment_name=exp.experiment_name,
                project_name=project_name,
                config=exp.hyperparameters,
            )

            for entry in exp.metrics_history:
                step = entry.get("step")
                metrics = {
                    k: v for k, v in entry.items() if k not in ["timestamp", "step"]
                }
                integration.log_metrics(metrics, step=step)

            for artifact_name, artifact_info in exp.artifacts.items():
                integration.log_artifact(artifact_info["path"], artifact_name)

            integration.finish()
            exported.append(exp.experiment_name)

        return dbc.Alert(
            f"Successfully exported {len(exported)} experiments to W&B: {', '.join(exported)}",
            color="success",
        )

    def _export_to_tensorboard(self, log_dir: Optional[str], experiment_ids: str):
        """Export experiments to TensorBoard."""
        if not experiment_ids:
            return dbc.Alert("Please provide experiment IDs.", color="warning")

        if not log_dir:
            log_dir = "runs/neural"

        ids = [id.strip() for id in experiment_ids.split(",")]
        exported = []

        for exp_id in ids:
            exp = self.manager.get_experiment(exp_id)
            if not exp:
                continue

            integration = TensorBoardIntegration(
                experiment_name=exp.experiment_name, log_dir=log_dir
            )

            integration.log_hyperparameters(exp.hyperparameters)

            for entry in exp.metrics_history:
                step = entry.get("step")
                metrics = {
                    k: v for k, v in entry.items() if k not in ["timestamp", "step"]
                }
                integration.log_metrics(metrics, step=step)

            for artifact_name, artifact_info in exp.artifacts.items():
                if artifact_info.get("type") in ["image", "figure"]:
                    integration.log_artifact(artifact_info["path"], artifact_name)

            integration.finish()
            exported.append(exp.experiment_name)

        return dbc.Alert(
            f"Successfully exported {len(exported)} experiments to TensorBoard: {', '.join(exported)}",
            color="success",
        )

    def run(self, debug: bool = False, host: str = "127.0.0.1"):
        """Run the dashboard."""
        print(f"\nStarting Aquarium Dashboard...")
        print(f"Base directory: {self.base_dir}")
        print(f"URL: http://{host}:{self.port}")

        experiments = self.manager.list_experiments()
        print(f"Found {len(experiments)} experiments\n")

        self.app.run_server(debug=debug, host=host, port=self.port)


def main():
    """Main entry point for the Aquarium dashboard."""
    parser = argparse.ArgumentParser(
        description="Launch Neural Aquarium Experiment Tracking Dashboard"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="neural_experiments",
        help="Base directory containing experiments (default: neural_experiments)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8053,
        help="Port to run the dashboard on (default: 8053)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the dashboard on (default: 127.0.0.1)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    dashboard = AquariumDashboard(base_dir=args.base_dir, port=args.port)
    dashboard.run(debug=args.debug, host=args.host)


if __name__ == "__main__":
    main()
