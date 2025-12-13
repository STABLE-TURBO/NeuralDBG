"""
Comparison component for comparing multiple experiments side by side.
"""

from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from neural.tracking.experiment_tracker import ExperimentTracker


class ComparisonComponent:
    """Component for comparing multiple experiments."""

    def __init__(self, experiments: List[ExperimentTracker]):
        """
        Initialize the comparison component.

        Args:
            experiments: List of ExperimentTracker instances
        """
        self.experiments = experiments

    def render(self) -> html.Div:
        """
        Render the comparison view.

        Returns:
            Dash HTML component
        """
        if not self.experiments:
            return dbc.Alert(
                [
                    html.I(className="fas fa-info-circle me-2"),
                    "Select experiments from the Experiments tab to compare.",
                ],
                color="info",
            )

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3("Experiment Comparison"),
                                html.P(
                                    f"Comparing {len(self.experiments)} experiments",
                                    className="text-muted",
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    [html.I(className="fas fa-download me-2"), "Export Comparison"],
                                    id="export-comparison",
                                    color="primary",
                                    size="sm",
                                    className="float-end",
                                ),
                            ],
                            width=4,
                        ),
                    ]
                ),
                html.Hr(),
                self._create_summary_cards(),
                html.Hr(),
                html.H4("Metrics Comparison"),
                self._create_metrics_comparison(),
                html.Hr(),
                html.H4("Hyperparameter Comparison"),
                self._create_hyperparameter_comparison(),
                html.Hr(),
                html.H4("Performance Summary"),
                self._create_performance_summary(),
            ],
            fluid=True,
        )

    def _create_summary_cards(self) -> html.Div:
        """Create summary cards for each experiment."""
        cards = []

        for exp in self.experiments:
            metrics = {}
            if exp.metrics_history:
                last_entry = exp.metrics_history[-1]
                metrics = {k: v for k, v in last_entry.items() if k not in ["timestamp", "step"]}

            card = dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H5(
                                    f"{exp.experiment_name}",
                                    className="mb-0",
                                )
                            ),
                            dbc.CardBody(
                                [
                                    html.P(
                                        [
                                            html.Strong("ID: "),
                                            exp.experiment_id[:8],
                                        ],
                                        className="mb-1",
                                    ),
                                    html.P(
                                        [
                                            html.Strong("Status: "),
                                            dbc.Badge(
                                                exp.metadata.get("status", "unknown"),
                                                color=self._get_status_color(
                                                    exp.metadata.get("status", "unknown")
                                                ),
                                            ),
                                        ],
                                        className="mb-1",
                                    ),
                                    html.P(
                                        [
                                            html.Strong("Steps: "),
                                            len(exp.metrics_history),
                                        ],
                                        className="mb-1",
                                    ),
                                    html.Hr(className="my-2"),
                                    html.P("Latest Metrics:", className="mb-1 text-muted"),
                                ]
                                + [
                                    html.P(
                                        f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}",
                                        className="mb-0",
                                        style={"fontSize": "0.85em"},
                                    )
                                    for k, v in list(metrics.items())[:3]
                                ]
                            ),
                        ],
                        className="mb-3",
                    )
                ],
                width=12 // len(self.experiments) if len(self.experiments) <= 4 else 3,
            )
            cards.append(card)

        return dbc.Row(cards)

    def _create_metrics_comparison(self) -> html.Div:
        """Create metrics comparison charts."""
        all_metrics = set()
        for exp in self.experiments:
            for entry in exp.metrics_history:
                all_metrics.update([k for k in entry.keys() if k not in ["timestamp", "step"]])

        if not all_metrics:
            return dbc.Alert("No metrics to compare.", color="info")

        charts = []
        for metric_name in sorted(all_metrics)[:6]:
            fig = self._create_metric_chart(metric_name)
            charts.append(
                dbc.Col(
                    [dcc.Graph(figure=fig)],
                    width=6,
                )
            )

        return dbc.Row(charts)

    def _create_metric_chart(self, metric_name: str) -> go.Figure:
        """Create a chart for a specific metric."""
        fig = go.Figure()

        for exp in self.experiments:
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
                        name=f"{exp.experiment_name[:20]} ({exp.experiment_id[:6]})",
                        hovertemplate="Step: %{x}<br>Value: %{y:.6f}<extra></extra>",
                    )
                )

        fig.update_layout(
            title=f"{metric_name}",
            xaxis_title="Step",
            yaxis_title="Value",
            template="plotly_dark",
            hovermode="x unified",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
        )

        return fig

    def _create_hyperparameter_comparison(self) -> html.Div:
        """Create hyperparameter comparison table."""
        all_hyperparams = set()
        for exp in self.experiments:
            all_hyperparams.update(exp.hyperparameters.keys())

        if not all_hyperparams:
            return dbc.Alert("No hyperparameters to compare.", color="info")

        table_data = []
        for param in sorted(all_hyperparams):
            row = {"Hyperparameter": param}
            for exp in self.experiments:
                value = exp.hyperparameters.get(param, "N/A")
                col_name = f"{exp.experiment_name[:15]} ({exp.experiment_id[:6]})"
                row[col_name] = str(value)
            table_data.append(row)

        columns = [{"name": "Hyperparameter", "id": "Hyperparameter"}]
        for exp in self.experiments:
            col_name = f"{exp.experiment_name[:15]} ({exp.experiment_id[:6]})"
            columns.append({"name": col_name, "id": col_name})

        return dash_table.DataTable(
            columns=columns,
            data=table_data,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "10px",
                "backgroundColor": "#2b3e50",
                "color": "white",
                "minWidth": "150px",
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
        )

    def _create_performance_summary(self) -> html.Div:
        """Create a performance summary table."""
        all_metrics = set()
        for exp in self.experiments:
            for entry in exp.metrics_history:
                all_metrics.update([k for k in entry.keys() if k not in ["timestamp", "step"]])

        if not all_metrics:
            return dbc.Alert("No metrics to summarize.", color="info")

        summary_data = []
        for metric_name in sorted(all_metrics):
            row = {"Metric": metric_name}

            for exp in self.experiments:
                col_name = f"{exp.experiment_name[:15]} ({exp.experiment_id[:6]})"

                best_value, best_step = exp.get_best_metric(metric_name, mode="max")
                if best_value is not None:
                    row[col_name] = f"{best_value:.6f} (step {best_step})"
                else:
                    row[col_name] = "N/A"

            summary_data.append(row)

        columns = [{"name": "Metric", "id": "Metric"}]
        for exp in self.experiments:
            col_name = f"{exp.experiment_name[:15]} ({exp.experiment_id[:6]})"
            columns.append({"name": col_name, "id": col_name})

        return dash_table.DataTable(
            columns=columns,
            data=summary_data,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "10px",
                "backgroundColor": "#2b3e50",
                "color": "white",
                "minWidth": "150px",
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
        )

    def _get_status_color(self, status: str) -> str:
        """Get the color for a status badge."""
        status_colors = {
            "running": "warning",
            "completed": "success",
            "failed": "danger",
            "created": "info",
        }
        return status_colors.get(status, "secondary")
