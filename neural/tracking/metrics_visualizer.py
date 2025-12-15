"""
Advanced metrics visualization component.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from neural.tracking.experiment_tracker import ExperimentTracker


class MetricsVisualizerComponent:
    """Component for advanced metrics visualization."""

    def __init__(self, experiment: ExperimentTracker):
        """
        Initialize the metrics visualizer.

        Args:
            experiment: ExperimentTracker instance
        """
        self.experiment = experiment

    def create_training_curves(self, metric_names: Optional[List[str]] = None) -> go.Figure:
        """
        Create training curves for specified metrics.

        Args:
            metric_names: List of metric names to plot (plots all if None)

        Returns:
            Plotly figure
        """
        if not self.experiment.metrics_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        if metric_names is None:
            all_keys = set()
            for entry in self.experiment.metrics_history:
                all_keys.update(entry.keys())
            metric_names = [key for key in all_keys if key not in ["timestamp", "step"]]

        if not metric_names:
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics to plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        num_metrics = len(metric_names)
        num_cols = 2 if num_metrics > 1 else 1
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=metric_names,
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        for idx, metric_name in enumerate(metric_names):
            row = (idx // num_cols) + 1
            col = (idx % num_cols) + 1

            steps = []
            values = []

            for entry in self.experiment.metrics_history:
                if metric_name in entry:
                    step = entry.get("step", len(steps))
                    steps.append(step)
                    values.append(entry[metric_name])

            if steps and values:
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=values,
                        mode="lines+markers",
                        name=metric_name,
                        hovertemplate="Step: %{x}<br>Value: %{y:.6f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

                fig.update_xaxes(title_text="Step", row=row, col=col)
                fig.update_yaxes(title_text="Value", row=row, col=col)

        fig.update_layout(
            template="plotly_dark",
            height=400 * num_rows,
            showlegend=False,
            title_text="Training Curves",
        )

        return fig

    def create_metrics_heatmap(self, metric_names: Optional[List[str]] = None) -> go.Figure:
        """
        Create a heatmap of metrics over time.

        Args:
            metric_names: List of metric names to include

        Returns:
            Plotly figure
        """
        if not self.experiment.metrics_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        if metric_names is None:
            all_keys = set()
            for entry in self.experiment.metrics_history:
                all_keys.update(entry.keys())
            metric_names = [key for key in all_keys if key not in ["timestamp", "step"]]

        if not metric_names:
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics to plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        steps = []
        for entry in self.experiment.metrics_history:
            steps.append(entry.get("step", len(steps)))

        z_data = []
        for metric_name in metric_names:
            values = []
            for entry in self.experiment.metrics_history:
                values.append(entry.get(metric_name, np.nan))
            z_data.append(values)

        fig = go.Figure(
            data=go.Heatmap(
                z=z_data,
                x=steps,
                y=metric_names,
                colorscale="Viridis",
                hovertemplate="Step: %{x}<br>Metric: %{y}<br>Value: %{z:.6f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Metrics Heatmap",
            xaxis_title="Step",
            yaxis_title="Metric",
            template="plotly_dark",
            height=max(400, len(metric_names) * 50),
        )

        return fig
 
class MetricVisualizer:
    """Lightweight matplotlib-based visualizer for tests."""
    @staticmethod
    def create_metric_plots(metrics_history: List[Dict[str, Any]]) -> Dict[str, plt.Figure]:
        """
        Create simple line plots for each metric using matplotlib.
        Returns a dict mapping metric name to figure.
        """
        if not metrics_history:
            return {}
        # Determine metric names excluding timestamp and step
        metric_names = set()
        for entry in metrics_history:
            metric_names.update([k for k in entry.keys() if k not in ["timestamp", "step"]])
        plots: Dict[str, plt.Figure] = {}
        # Build steps
        steps = []
        for entry in metrics_history:
            s = entry.get("step")
            steps.append(s if s is not None else len(steps))
        for name in sorted(metric_names):
            values = [entry.get(name) for entry in metrics_history]
            valid_idx = [i for i, v in enumerate(values) if v is not None]
            if not valid_idx:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot([steps[i] for i in valid_idx], [values[i] for i in valid_idx], marker="o", linestyle="-")
            ax.set_title(f"{name} over time")
            ax.set_xlabel("Step")
            ax.set_ylabel(name)
            ax.grid(True)
            plots[name] = fig
        return plots

    @staticmethod
    def create_distribution_plots(metrics_history: List[Dict[str, Any]]) -> Dict[str, plt.Figure]:
        """
        Create histogram plots for each metric using matplotlib.
        Returns a dict mapping metric name to figure.
        """
        if not metrics_history:
            return {}
        metric_names = set()
        for entry in metrics_history:
            metric_names.update([k for k in entry.keys() if k not in ["timestamp", "step"]])
        plots: Dict[str, plt.Figure] = {}
        for name in sorted(metric_names):
            values = [entry.get(name) for entry in metrics_history if entry.get(name) is not None]
            if not values:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(values, bins=min(10, max(3, int(len(values) / 2))), color="steelblue", edgecolor="white")
            ax.set_title(f"{name} distribution")
            ax.set_xlabel(name)
            ax.set_ylabel("Frequency")
            ax.grid(True)
            plots[name] = fig
        return plots

    def create_distribution_plot(self, metric_name: str) -> go.Figure:
        """
        Create a distribution plot for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Plotly figure
        """
        values = []
        for entry in self.experiment.metrics_history:
            if metric_name in entry:
                values.append(entry[metric_name])

        if not values:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data for metric: {metric_name}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        fig = go.Figure()

        fig.add_trace(go.Histogram(x=values, name="Distribution", nbinsx=30))

        fig.add_vline(
            x=np.mean(values),
            line_dash="dash",
            line_color="red",
            annotation_text="Mean",
            annotation_position="top",
        )

        fig.add_vline(
            x=np.median(values),
            line_dash="dash",
            line_color="green",
            annotation_text="Median",
            annotation_position="top",
        )

        fig.update_layout(
            title=f"Distribution of {metric_name}",
            xaxis_title="Value",
            yaxis_title="Frequency",
            template="plotly_dark",
            showlegend=True,
        )

        return fig

    def create_smoothed_curves(
        self, metric_names: Optional[List[str]] = None, window_size: int = 10
    ) -> go.Figure:
        """
        Create smoothed training curves using moving average.

        Args:
            metric_names: List of metric names to plot
            window_size: Window size for moving average

        Returns:
            Plotly figure
        """
        if not self.experiment.metrics_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        if metric_names is None:
            all_keys = set()
            for entry in self.experiment.metrics_history:
                all_keys.update(entry.keys())
            metric_names = [key for key in all_keys if key not in ["timestamp", "step"]]

        if not metric_names:
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics to plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        fig = go.Figure()

        for metric_name in metric_names:
            steps = []
            values = []

            for entry in self.experiment.metrics_history:
                if metric_name in entry:
                    step = entry.get("step", len(steps))
                    steps.append(step)
                    values.append(entry[metric_name])

            if len(values) < window_size:
                continue

            smoothed_values = self._moving_average(values, window_size)
            smoothed_steps = steps[window_size - 1 :]

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    mode="lines",
                    name=f"{metric_name} (raw)",
                    opacity=0.3,
                    line=dict(width=1),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=smoothed_steps,
                    y=smoothed_values,
                    mode="lines",
                    name=f"{metric_name} (smoothed)",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title=f"Smoothed Training Curves (window={window_size})",
            xaxis_title="Step",
            yaxis_title="Value",
            template="plotly_dark",
            hovermode="x unified",
        )

        return fig

    def create_correlation_matrix(self) -> go.Figure:
        """
        Create a correlation matrix of all metrics.

        Returns:
            Plotly figure
        """
        if not self.experiment.metrics_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        all_keys = set()
        for entry in self.experiment.metrics_history:
            all_keys.update(entry.keys())
        metric_names = [key for key in all_keys if key not in ["timestamp", "step"]]

        if len(metric_names) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 metrics for correlation",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        data_dict = {metric: [] for metric in metric_names}
        for entry in self.experiment.metrics_history:
            for metric in metric_names:
                data_dict[metric].append(entry.get(metric, np.nan))

        import pandas as pd

        df = pd.DataFrame(data_dict)
        corr_matrix = df.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Metrics Correlation Matrix",
            template="plotly_dark",
            height=max(400, len(metric_names) * 50),
        )

        return fig

    def _moving_average(self, values: List[float], window_size: int) -> List[float]:
        """
        Calculate moving average.

        Args:
            values: List of values
            window_size: Window size

        Returns:
            List of smoothed values
        """
        if len(values) < window_size:
            return values

        result = []
        for i in range(window_size - 1, len(values)):
            window = values[i - window_size + 1 : i + 1]
            result.append(sum(window) / window_size)

        return result
