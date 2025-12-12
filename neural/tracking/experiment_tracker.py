"""
Experiment tracking for Neural models.

This module provides functionality to track experiments, including hyperparameters,
metrics, and artifacts.
"""

from __future__ import annotations

import hashlib
import os
import json
import logging
import time
import datetime
import uuid
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import dash
    from dash import Dash, dcc, html, Input, Output, State, callback
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not available. Install with 'pip install dash dash-bootstrap-components plotly'")


class ArtifactVersion:
    """Represents a versioned artifact."""

    def __init__(self, artifact_name: str, version: int, path: str, metadata: Dict[str, Any]):
        self.artifact_name = artifact_name
        self.version = version
        self.path = path
        self.metadata = metadata
        self.timestamp = metadata.get("timestamp", time.time())
        self.checksum = metadata.get("checksum", "")
        self.size = metadata.get("size", 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "version": self.version,
            "path": self.path,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "checksum": self.checksum,
            "size": self.size
        }


class MetricVisualizer:
    """Automatic metric visualization."""

    @staticmethod
    def create_metric_plots(metrics_history: List[Dict[str, Any]], 
                           figsize: Tuple[int, int] = (15, 10)) -> Dict[str, plt.Figure]:
        if not metrics_history:
            return {}

        all_keys = set()
        for entry in metrics_history:
            all_keys.update(entry.keys())
        metric_names = [key for key in all_keys if key not in ["timestamp", "step"]]

        if not metric_names:
            return {}

        plots = {}
        steps = []
        for entry in metrics_history:
            if "step" in entry and entry["step"] is not None:
                steps.append(entry["step"])
            else:
                steps.append(len(steps))

        fig_individual, axes = plt.subplots(
            len(metric_names), 1, figsize=(figsize[0], figsize[1] * len(metric_names) / 4)
        )
        if len(metric_names) == 1:
            axes = [axes]

        for idx, metric_name in enumerate(metric_names):
            values = [entry.get(metric_name, None) for entry in metrics_history]
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_steps = [steps[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]

            if valid_values:
                axes[idx].plot(valid_steps, valid_values, marker='o', linestyle='-', linewidth=2)
                axes[idx].set_xlabel("Step")
                axes[idx].set_ylabel(metric_name)
                axes[idx].set_title(f"{metric_name} over time")
                axes[idx].grid(True, alpha=0.3)

                if len(valid_values) > 1:
                    max_idx = valid_values.index(max(valid_values))
                    min_idx = valid_values.index(min(valid_values))
                    axes[idx].scatter([valid_steps[max_idx]], [valid_values[max_idx]], 
                                    color='green', s=100, zorder=5, label=f'Max: {valid_values[max_idx]:.4f}')
                    axes[idx].scatter([valid_steps[min_idx]], [valid_values[min_idx]], 
                                    color='red', s=100, zorder=5, label=f'Min: {valid_values[min_idx]:.4f}')
                    axes[idx].legend()

        plt.tight_layout()
        plots["all_metrics_individual"] = fig_individual

        fig_combined, ax = plt.subplots(figsize=figsize)
        for metric_name in metric_names:
            values = [entry.get(metric_name, None) for entry in metrics_history]
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_steps = [steps[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]

            if valid_values:
                ax.plot(valid_steps, valid_values, marker='o', linestyle='-', 
                       linewidth=2, label=metric_name)

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title("All Metrics Combined")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plots["all_metrics_combined"] = fig_combined

        if len(metric_names) >= 2:
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            metric_data = {}
            for metric_name in metric_names:
                values = [entry.get(metric_name, None) for entry in metrics_history]
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    metric_data[metric_name] = valid_values

            if len(metric_data) >= 2:
                import numpy as np
                min_len = min(len(v) for v in metric_data.values())
                matrix = np.array([v[:min_len] for v in metric_data.values()])
                corr = np.corrcoef(matrix)

                im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(np.arange(len(metric_names)))
                ax.set_yticks(np.arange(len(metric_names)))
                ax.set_xticklabels(metric_names, rotation=45, ha='right')
                ax.set_yticklabels(metric_names)

                for i in range(len(metric_names)):
                    for j in range(len(metric_names)):
                        text = ax.text(j, i, f'{corr[i, j]:.2f}',
                                     ha="center", va="center", color="black")

                ax.set_title("Metric Correlation Matrix")
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                plots["metric_correlation"] = fig_corr

        return plots

    @staticmethod
    def create_distribution_plots(metrics_history: List[Dict[str, Any]], 
                                 figsize: Tuple[int, int] = (15, 8)) -> Dict[str, plt.Figure]:
        if not metrics_history:
            return {}

        all_keys = set()
        for entry in metrics_history:
            all_keys.update(entry.keys())
        metric_names = [key for key in all_keys if key not in ["timestamp", "step"]]

        if not metric_names:
            return {}

        plots = {}
        fig, axes = plt.subplots(1, len(metric_names), figsize=figsize)
        if len(metric_names) == 1:
            axes = [axes]

        for idx, metric_name in enumerate(metric_names):
            values = [entry.get(metric_name, None) for entry in metrics_history 
                     if metric_name in entry and entry[metric_name] is not None]

            if values:
                axes[idx].hist(values, bins=20, edgecolor='black', alpha=0.7)
                axes[idx].set_xlabel(metric_name)
                axes[idx].set_ylabel("Frequency")
                axes[idx].set_title(f"Distribution of {metric_name}")
                axes[idx].grid(True, alpha=0.3)

                mean_val = np.mean(values)
                axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                linewidth=2, label=f'Mean: {mean_val:.4f}')
                axes[idx].legend()

        plt.tight_layout()
        plots["metric_distributions"] = fig

        return plots


class ExperimentTracker:
    """Tracks experiments, including hyperparameters, metrics, and artifacts."""

    def __init__(self, experiment_name: str = None, base_dir: str = "neural_experiments",
                 auto_visualize: bool = True, backend: Optional[str] = None):
        self.experiment_name = experiment_name or f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_id = str(uuid.uuid4())[:8]
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, f"{self.experiment_name}_{self.experiment_id}")
        self.metrics_history = []
        self.hyperparameters = {}
        self.metadata = {
            "start_time": datetime.datetime.now().isoformat(),
            "status": "created"
        }
        self.artifacts = {}
        self.artifact_versions = {}
        self.auto_visualize = auto_visualize
        self.backend_integration = None

        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "versions"), exist_ok=True)

        self._save_metadata()

        if backend:
            self._setup_backend(backend)

        logger.info(f"Initialized experiment tracker: {self.experiment_name} (ID: {self.experiment_id})")

    def _setup_backend(self, backend: str, **kwargs):
        try:
            from neural.tracking.integrations import create_integration
            self.backend_integration = create_integration(backend, self.experiment_name, **kwargs)
            if self.backend_integration:
                logger.info(f"Initialized {backend} backend integration")
        except Exception as e:
            logger.warning(f"Could not initialize {backend} backend: {str(e)}")

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        self.hyperparameters.update(hyperparameters)
        self._save_hyperparameters()
        if self.backend_integration:
            self.backend_integration.log_hyperparameters(hyperparameters)
        logger.debug(f"Logged hyperparameters: {hyperparameters}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        timestamp = time.time()
        metrics_entry = {
            "timestamp": timestamp,
            "step": step,
            **metrics
        }
        self.metrics_history.append(metrics_entry)
        self._save_metrics()
        
        if self.backend_integration:
            self.backend_integration.log_metrics(metrics, step)
        
        if self.auto_visualize and len(self.metrics_history) % 10 == 0:
            self._auto_visualize_metrics()
        
        logger.debug(f"Logged metrics at step {step}: {metrics}")

    def _auto_visualize_metrics(self):
        try:
            plots = MetricVisualizer.create_metric_plots(self.metrics_history)
            for plot_name, fig in plots.items():
                plot_path = os.path.join(self.experiment_dir, "plots", f"auto_{plot_name}.png")
                fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
            logger.debug("Auto-generated metric visualizations")
        except Exception as e:
            logger.warning(f"Could not auto-visualize metrics: {str(e)}")

    def _compute_checksum(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def log_artifact(self, artifact_path: str, artifact_name: str = None, version: bool = True):
        if not os.path.exists(artifact_path):
            logger.error(f"Artifact not found: {artifact_path}")
            return

        artifact_name = artifact_name or os.path.basename(artifact_path)
        
        if version and artifact_name in self.artifact_versions:
            next_version = len(self.artifact_versions[artifact_name]) + 1
        else:
            next_version = 1
            if artifact_name not in self.artifact_versions:
                self.artifact_versions[artifact_name] = []

        version_dir = os.path.join(self.experiment_dir, "versions", artifact_name)
        os.makedirs(version_dir, exist_ok=True)
        
        artifact_dest = os.path.join(version_dir, f"v{next_version}_{artifact_name}")
        shutil.copy2(artifact_path, artifact_dest)

        checksum = self._compute_checksum(artifact_path)
        
        artifact_metadata = {
            "path": artifact_dest,
            "type": self._get_artifact_type(artifact_path),
            "size": os.path.getsize(artifact_path),
            "timestamp": time.time(),
            "checksum": checksum
        }

        artifact_version = ArtifactVersion(
            artifact_name=artifact_name,
            version=next_version,
            path=artifact_dest,
            metadata=artifact_metadata
        )
        
        self.artifact_versions[artifact_name].append(artifact_version)
        self.artifacts[artifact_name] = artifact_metadata
        
        self._save_artifacts()
        
        if self.backend_integration:
            self.backend_integration.log_artifact(artifact_path, artifact_name)
        
        logger.debug(f"Logged artifact: {artifact_name} (version {next_version})")

    def get_artifact_version(self, artifact_name: str, version: int = -1) -> Optional[ArtifactVersion]:
        if artifact_name not in self.artifact_versions:
            return None
        
        versions = self.artifact_versions[artifact_name]
        if not versions:
            return None
        
        if version == -1 or version > len(versions):
            return versions[-1]
        
        return versions[version - 1] if version > 0 else None

    def list_artifact_versions(self, artifact_name: str) -> List[ArtifactVersion]:
        return self.artifact_versions.get(artifact_name, [])

    def log_model(self, model_path: str, framework: str = "unknown", version: bool = True):
        model_name = os.path.basename(model_path)
        
        if version and model_name in self.artifact_versions:
            next_version = len(self.artifact_versions[model_name]) + 1
        else:
            next_version = 1
            if model_name not in self.artifact_versions:
                self.artifact_versions[model_name] = []

        version_dir = os.path.join(self.experiment_dir, "versions", model_name)
        os.makedirs(version_dir, exist_ok=True)
        
        model_dest = os.path.join(version_dir, f"v{next_version}_{model_name}")
        shutil.copy2(model_path, model_dest)

        checksum = self._compute_checksum(model_path)
        
        model_metadata = {
            "path": model_dest,
            "type": "model",
            "framework": framework,
            "size": os.path.getsize(model_path),
            "timestamp": time.time(),
            "checksum": checksum
        }

        model_version = ArtifactVersion(
            artifact_name=model_name,
            version=next_version,
            path=model_dest,
            metadata=model_metadata
        )
        
        self.artifact_versions[model_name].append(model_version)
        self.artifacts[model_name] = model_metadata
        
        self._save_artifacts()
        
        if self.backend_integration:
            self.backend_integration.log_model(model_path, framework)
        
        logger.debug(f"Logged {framework} model: {model_name} (version {next_version})")

    def log_figure(self, figure: plt.Figure, figure_name: str):
        if not figure_name.endswith(('.png', '.jpg', '.jpeg', '.svg')):
            figure_name += '.png'

        figure_path = os.path.join(self.experiment_dir, "plots", figure_name)
        figure.savefig(figure_path, dpi=150, bbox_inches='tight')

        self.artifacts[figure_name] = {
            "path": figure_path,
            "type": "figure",
            "size": os.path.getsize(figure_path),
            "timestamp": time.time()
        }

        self._save_artifacts()
        
        if self.backend_integration:
            self.backend_integration.log_figure(figure, figure_name)
        
        logger.debug(f"Logged figure: {figure_name}")

    def set_status(self, status: str):
        self.metadata["status"] = status
        if status in ["completed", "failed"]:
            self.metadata["end_time"] = datetime.datetime.now().isoformat()
        self._save_metadata()
        logger.info(f"Set experiment status to: {status}")

    def get_metrics(self, metric_name: str = None) -> Union[List[Dict[str, Any]], List[float]]:
        if metric_name:
            return [entry.get(metric_name, None) for entry in self.metrics_history 
                   if metric_name in entry]
        return self.metrics_history

    def get_best_metric(self, metric_name: str, mode: str = "max") -> Tuple[float, int]:
        if not self.metrics_history:
            return None, None

        metric_values = [(entry.get(metric_name, None), entry.get("step", i))
                         for i, entry in enumerate(self.metrics_history)
                         if metric_name in entry]

        if not metric_values:
            return None, None

        if mode == "max":
            best_value, step = max(metric_values, 
                                  key=lambda x: x[0] if x[0] is not None else float('-inf'))
        else:
            best_value, step = min(metric_values, 
                                  key=lambda x: x[0] if x[0] is not None else float('inf'))

        return best_value, step

    def plot_metrics(self, metric_names: List[str] = None, 
                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        if not self.metrics_history:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No metrics data available", ha='center', va='center')
            return fig

        if metric_names is None:
            all_keys = set()
            for entry in self.metrics_history:
                all_keys.update(entry.keys())
            metric_names = [key for key in all_keys if key not in ["timestamp", "step"]]

        if not metric_names:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No metrics to plot", ha='center', va='center')
            return fig

        fig, ax = plt.subplots(figsize=figsize)

        steps = []
        for entry in self.metrics_history:
            if "step" in entry and entry["step"] is not None:
                steps.append(entry["step"])
            else:
                steps.append(len(steps))

        for metric_name in metric_names:
            values = [entry.get(metric_name, None) for entry in self.metrics_history]
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_steps = [steps[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]

            if valid_values:
                ax.plot(valid_steps, valid_values, marker='o', linestyle='-', label=metric_name)

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title("Metrics History")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        return fig

    def generate_visualizations(self) -> Dict[str, str]:
        if not self.metrics_history:
            logger.warning("No metrics to visualize")
            return {}

        saved_plots = {}

        plots = MetricVisualizer.create_metric_plots(self.metrics_history)
        for plot_name, fig in plots.items():
            plot_path = os.path.join(self.experiment_dir, "plots", f"{plot_name}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_plots[plot_name] = plot_path

        dist_plots = MetricVisualizer.create_distribution_plots(self.metrics_history)
        for plot_name, fig in dist_plots.items():
            plot_path = os.path.join(self.experiment_dir, "plots", f"{plot_name}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_plots[plot_name] = plot_path

        logger.info(f"Generated {len(saved_plots)} visualizations")
        return saved_plots

    def save_experiment_summary(self) -> str:
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
            "hyperparameters": self.hyperparameters,
            "metrics": {
                "latest": {k: v for entry in self.metrics_history[-1:] 
                          for k, v in entry.items() 
                          if k not in ["timestamp", "step"]} if self.metrics_history else {},
                "best": {}
            },
            "artifacts": list(self.artifacts.keys()),
            "artifact_versions": {name: len(versions) 
                                 for name, versions in self.artifact_versions.items()}
        }

        all_metrics = set()
        for entry in self.metrics_history:
            all_metrics.update([k for k in entry.keys() if k not in ["timestamp", "step"]])

        for metric in all_metrics:
            best_value, step = self.get_best_metric(metric, mode="max")
            if best_value is not None:
                summary["metrics"]["best"][metric] = {
                    "value": best_value,
                    "step": step,
                    "mode": "max"
                }

        summary_path = os.path.join(self.experiment_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def finish(self):
        self.set_status("completed")
        self.save_experiment_summary()
        
        if self.auto_visualize:
            self.generate_visualizations()
        
        if self.backend_integration:
            self.backend_integration.finish()
        
        logger.info(f"Finished experiment: {self.experiment_name} ({self.experiment_id})")

    def _save_metadata(self):
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _save_hyperparameters(self):
        hyperparameters_path = os.path.join(self.experiment_dir, "hyperparameters.json")
        with open(hyperparameters_path, 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)

    def _save_metrics(self):
        metrics_path = os.path.join(self.experiment_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def _save_artifacts(self):
        artifacts_path = os.path.join(self.experiment_dir, "artifacts.json")
        artifact_data = {
            "current": self.artifacts,
            "versions": {
                name: [v.to_dict() for v in versions]
                for name, versions in self.artifact_versions.items()
            }
        }
        with open(artifacts_path, 'w') as f:
            json.dump(artifact_data, f, indent=2)

    def _get_artifact_type(self, artifact_path: str) -> str:
        ext = os.path.splitext(artifact_path)[1].lower()

        if ext in ['.h5', '.pt', '.pth', '.onnx', '.pb', '.tflite']:
            return 'model'
        elif ext in ['.png', '.jpg', '.jpeg', '.svg', '.gif']:
            return 'image'
        elif ext in ['.csv', '.tsv']:
            return 'table'
        elif ext in ['.json', '.yaml', '.yml']:
            return 'config'
        elif ext in ['.txt', '.log', '.md']:
            return 'text'
        else:
            return 'other'


class ExperimentManager:
    """Manages multiple experiments."""

    def __init__(self, base_dir: str = "neural_experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_experiment(self, experiment_name: str = None, 
                         auto_visualize: bool = True, 
                         backend: Optional[str] = None) -> ExperimentTracker:
        return ExperimentTracker(
            experiment_name=experiment_name, 
            base_dir=self.base_dir,
            auto_visualize=auto_visualize,
            backend=backend
        )

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentTracker]:
        for item in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, item)) and item.endswith(f"_{experiment_id}"):
                experiment_name = item[:-(len(experiment_id) + 1)]

                tracker = ExperimentTracker(experiment_name=experiment_name, base_dir=self.base_dir)
                tracker.experiment_id = experiment_id
                tracker.experiment_dir = os.path.join(self.base_dir, item)

                metadata_path = os.path.join(tracker.experiment_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        tracker.metadata = json.load(f)

                hyperparameters_path = os.path.join(tracker.experiment_dir, "hyperparameters.json")
                if os.path.exists(hyperparameters_path):
                    with open(hyperparameters_path, 'r') as f:
                        tracker.hyperparameters = json.load(f)

                metrics_path = os.path.join(tracker.experiment_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        tracker.metrics_history = json.load(f)

                artifacts_path = os.path.join(tracker.experiment_dir, "artifacts.json")
                if os.path.exists(artifacts_path):
                    with open(artifacts_path, 'r') as f:
                        artifact_data = json.load(f)
                        tracker.artifacts = artifact_data.get("current", {})
                        
                        versions_data = artifact_data.get("versions", {})
                        for artifact_name, versions_list in versions_data.items():
                            tracker.artifact_versions[artifact_name] = [
                                ArtifactVersion(
                                    artifact_name=v["artifact_name"],
                                    version=v["version"],
                                    path=v["path"],
                                    metadata=v["metadata"]
                                )
                                for v in versions_list
                            ]

                return tracker

        return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        experiments = []

        for item in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, item)):
                metadata_path = os.path.join(self.base_dir, item, "metadata.json")
                if os.path.exists(metadata_path):
                    experiment_id = item.split('_')[-1]

                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    summary_path = os.path.join(self.base_dir, item, "summary.json")
                    summary = None
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)

                    experiment = {
                        "experiment_name": item[:-(len(experiment_id) + 1)],
                        "experiment_id": experiment_id,
                        "status": metadata.get("status", "unknown"),
                        "start_time": metadata.get("start_time", "unknown"),
                        "end_time": metadata.get("end_time", None),
                        "summary": summary
                    }

                    experiments.append(experiment)

        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        for item in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, item)) and item.endswith(f"_{experiment_id}"):
                shutil.rmtree(os.path.join(self.base_dir, item))
                return True

        return False

    def compare_experiments(self, experiment_ids: List[str], 
                          metric_names: List[str] = None) -> Dict[str, plt.Figure]:
        experiments = [self.get_experiment(exp_id) for exp_id in experiment_ids]
        experiments = [exp for exp in experiments if exp is not None]

        if not experiments:
            return {}

        if metric_names is None:
            all_metrics = set()
            for exp in experiments:
                for entry in exp.metrics_history:
                    all_metrics.update([k for k in entry.keys() 
                                      if k not in ["timestamp", "step"]])
            metric_names = sorted(list(all_metrics))

        if not metric_names:
            return {}

        plots = {}

        for metric_name in metric_names:
            fig, ax = plt.subplots(figsize=(12, 7))

            for exp in experiments:
                steps = []
                values = []

                for entry in exp.metrics_history:
                    if "step" in entry and entry["step"] is not None and metric_name in entry:
                        steps.append(entry["step"])
                        values.append(entry[metric_name])

                if steps and values:
                    ax.plot(steps, values, marker='o', linestyle='-', linewidth=2,
                           label=f"{exp.experiment_name} ({exp.experiment_id})")

            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f"Comparison of {metric_name}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plots[f"comparison_{metric_name}"] = fig

        if len(experiments) > 1:
            all_hyperparams = set()
            for exp in experiments:
                all_hyperparams.update(exp.hyperparameters.keys())

            if all_hyperparams:
                fig, ax = plt.subplots(figsize=(14, max(len(all_hyperparams) * 0.6, 4)))

                ax.axis('tight')
                ax.axis('off')

                table_data = []
                header = ["Hyperparameter"] + [f"{exp.experiment_name}\n({exp.experiment_id})" 
                                               for exp in experiments]
                table_data.append(header)

                for param in sorted(all_hyperparams):
                    row = [param]
                    for exp in experiments:
                        value = exp.hyperparameters.get(param, "N/A")
                        row.append(str(value))
                    table_data.append(row)

                table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                               loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.8)

                for i in range(len(table_data[0])):
                    table[(0, i)].set_facecolor('#40466e')
                    table[(0, i)].set_text_props(weight='bold', color='white')

                ax.set_title("Hyperparameter Comparison", fontsize=14, 
                           fontweight='bold', pad=20)
                plt.tight_layout()

                plots["hyperparameter_comparison"] = fig

        if len(experiments) > 1 and metric_names:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            x_labels = []
            for metric_name in metric_names[:5]:
                metric_data = []
                for exp in experiments:
                    best_val, _ = exp.get_best_metric(metric_name, mode="max")
                    if best_val is not None:
                        metric_data.append(best_val)
                    else:
                        metric_data.append(0)
                
                if any(v != 0 for v in metric_data):
                    x = np.arange(len(experiments))
                    width = 0.8 / len(metric_names[:5])
                    offset = width * metric_names[:5].index(metric_name)
                    
                    ax.bar(x + offset, metric_data, width, label=metric_name, alpha=0.8)
            
            ax.set_xlabel('Experiments', fontsize=12)
            ax.set_ylabel('Best Metric Value', fontsize=12)
            ax.set_title('Best Metrics Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width * len(metric_names[:5]) / 2)
            ax.set_xticklabels([f"{exp.experiment_name}\n({exp.experiment_id})" 
                               for exp in experiments], rotation=15, ha='right')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plots["best_metrics_comparison"] = fig

        return plots

    def export_comparison(self, experiment_ids: List[str], output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(self.base_dir, "comparisons", 
                                     f"comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        os.makedirs(output_dir, exist_ok=True)

        plots = self.compare_experiments(experiment_ids)
        for plot_name, fig in plots.items():
            plot_path = os.path.join(output_dir, f"{plot_name}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        experiments = [self.get_experiment(exp_id) for exp_id in experiment_ids]
        experiments = [exp for exp in experiments if exp is not None]

        comparison_summary = {
            "experiments": [
                {
                    "name": exp.experiment_name,
                    "id": exp.experiment_id,
                    "status": exp.metadata.get("status"),
                    "hyperparameters": exp.hyperparameters,
                    "best_metrics": {}
                }
                for exp in experiments
            ],
            "comparison_date": datetime.datetime.now().isoformat()
        }

        summary_path = os.path.join(output_dir, "comparison_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(comparison_summary, f, indent=2)

        logger.info(f"Exported comparison to {output_dir}")
        return output_dir


class ExperimentComparisonUI:
    """Interactive web UI for comparing experiments."""

    def __init__(self, manager: ExperimentManager, port: int = 8052):
        if not DASH_AVAILABLE:
            raise ImportError("Dash not available. Install with 'pip install dash dash-bootstrap-components plotly'")
        
        self.manager = manager
        self.port = port
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="Neural Experiment Comparison"
        )
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        experiments = self.manager.list_experiments()
        
        experiment_options = [
            {"label": f"{exp['experiment_name']} ({exp['experiment_id']}) - {exp['status']}", 
             "value": exp['experiment_id']}
            for exp in experiments
        ]

        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🧪 Neural Experiment Comparison", 
                           className="text-center mb-4 mt-4",
                           style={"color": "#00d4ff"})
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Select Experiments to Compare", className="mb-3"),
                    dcc.Dropdown(
                        id='experiment-selector',
                        options=experiment_options,
                        value=[],
                        multi=True,
                        placeholder="Select experiments...",
                        style={"color": "#000"}
                    ),
                    html.Br(),
                    dbc.Button("Compare Experiments", id="compare-btn", 
                              color="primary", className="mb-3", size="lg"),
                    dbc.Button("Refresh List", id="refresh-btn", 
                              color="secondary", className="mb-3 ms-2", size="lg"),
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id='comparison-status', className="mb-3")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Tabs(id='comparison-tabs', value='metrics', children=[
                        dcc.Tab(label='📊 Metrics Comparison', value='metrics',
                               style={"padding": "10px"}),
                        dcc.Tab(label='⚙️ Hyperparameters', value='hyperparameters',
                               style={"padding": "10px"}),
                        dcc.Tab(label='📈 Best Metrics', value='best-metrics',
                               style={"padding": "10px"}),
                        dcc.Tab(label='📋 Summary', value='summary',
                               style={"padding": "10px"}),
                    ])
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id='comparison-content', className="mt-4")
                ], width=12)
            ]),
            
            dcc.Store(id='experiments-data'),
            
        ], fluid=True, style={"padding": "20px"})

    def _setup_callbacks(self):
        @self.app.callback(
            [Output('experiment-selector', 'options'),
             Output('comparison-status', 'children')],
            Input('refresh-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def refresh_experiments(n_clicks):
            experiments = self.manager.list_experiments()
            options = [
                {"label": f"{exp['experiment_name']} ({exp['experiment_id']}) - {exp['status']}", 
                 "value": exp['experiment_id']}
                for exp in experiments
            ]
            return options, dbc.Alert("Experiment list refreshed!", color="success", duration=3000)

        @self.app.callback(
            Output('experiments-data', 'data'),
            Input('compare-btn', 'n_clicks'),
            State('experiment-selector', 'value'),
            prevent_initial_call=True
        )
        def load_experiments(n_clicks, selected_ids):
            if not selected_ids or len(selected_ids) == 0:
                return None
            
            experiments_data = []
            for exp_id in selected_ids:
                exp = self.manager.get_experiment(exp_id)
                if exp:
                    experiments_data.append({
                        'id': exp.experiment_id,
                        'name': exp.experiment_name,
                        'hyperparameters': exp.hyperparameters,
                        'metrics_history': exp.metrics_history,
                        'metadata': exp.metadata
                    })
            
            return experiments_data

        @self.app.callback(
            Output('comparison-content', 'children'),
            [Input('comparison-tabs', 'value'),
             Input('experiments-data', 'data')]
        )
        def update_comparison_content(tab, experiments_data):
            if not experiments_data or len(experiments_data) == 0:
                return dbc.Alert("Please select experiments and click 'Compare Experiments'", 
                               color="info")
            
            if tab == 'metrics':
                return self._create_metrics_comparison(experiments_data)
            elif tab == 'hyperparameters':
                return self._create_hyperparameters_table(experiments_data)
            elif tab == 'best-metrics':
                return self._create_best_metrics_comparison(experiments_data)
            elif tab == 'summary':
                return self._create_summary_view(experiments_data)
            
            return html.Div("Unknown tab")

    def _create_metrics_comparison(self, experiments_data):
        all_metrics = set()
        for exp in experiments_data:
            for entry in exp['metrics_history']:
                all_metrics.update([k for k in entry.keys() if k not in ['timestamp', 'step']])
        
        if not all_metrics:
            return dbc.Alert("No metrics found in selected experiments", color="warning")
        
        graphs = []
        for metric_name in sorted(all_metrics):
            fig = go.Figure()
            
            for exp in experiments_data:
                steps = []
                values = []
                
                for entry in exp['metrics_history']:
                    if 'step' in entry and entry['step'] is not None and metric_name in entry:
                        steps.append(entry['step'])
                        values.append(entry[metric_name])
                
                if steps and values:
                    fig.add_trace(go.Scatter(
                        x=steps,
                        y=values,
                        mode='lines+markers',
                        name=f"{exp['name']} ({exp['id']})",
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
            
            fig.update_layout(
                title=f"Comparison: {metric_name}",
                xaxis_title="Step",
                yaxis_title=metric_name,
                template="plotly_dark",
                hovermode='x unified',
                height=500
            )
            
            graphs.append(dcc.Graph(figure=fig, className="mb-4"))
        
        return html.Div(graphs)

    def _create_hyperparameters_table(self, experiments_data):
        all_params = set()
        for exp in experiments_data:
            all_params.update(exp['hyperparameters'].keys())
        
        if not all_params:
            return dbc.Alert("No hyperparameters found", color="warning")
        
        header = [html.Thead(html.Tr([
            html.Th("Hyperparameter", style={"width": "30%"})] + 
            [html.Th(f"{exp['name']}\n({exp['id']})", style={"width": f"{70//len(experiments_data)}%"}) 
             for exp in experiments_data]
        ))]
        
        rows = []
        for param in sorted(all_params):
            row = [html.Td(html.Strong(param))]
            for exp in experiments_data:
                value = exp['hyperparameters'].get(param, 'N/A')
                row.append(html.Td(str(value)))
            rows.append(html.Tr(row))
        
        body = [html.Tbody(rows)]
        
        table = dbc.Table(
            header + body,
            bordered=True,
            hover=True,
            responsive=True,
            striped=True,
            className="mb-4"
        )
        
        return html.Div([
            html.H4("Hyperparameter Comparison", className="mb-3"),
            table
        ])

    def _create_best_metrics_comparison(self, experiments_data):
        all_metrics = set()
        for exp in experiments_data:
            for entry in exp['metrics_history']:
                all_metrics.update([k for k in entry.keys() if k not in ['timestamp', 'step']])
        
        if not all_metrics:
            return dbc.Alert("No metrics found", color="warning")
        
        fig = go.Figure()
        
        for metric_name in sorted(list(all_metrics))[:10]:
            metric_values = []
            exp_labels = []
            
            for exp in experiments_data:
                values = [entry.get(metric_name) for entry in exp['metrics_history'] 
                         if metric_name in entry and entry[metric_name] is not None]
                if values:
                    best_value = max(values)
                    metric_values.append(best_value)
                    exp_labels.append(f"{exp['name'][:20]}")
                else:
                    metric_values.append(0)
                    exp_labels.append(f"{exp['name'][:20]}")
            
            if any(v != 0 for v in metric_values):
                fig.add_trace(go.Bar(
                    name=metric_name,
                    x=exp_labels,
                    y=metric_values,
                    text=[f'{v:.4f}' for v in metric_values],
                    textposition='auto',
                ))
        
        fig.update_layout(
            title="Best Metric Values Comparison",
            xaxis_title="Experiments",
            yaxis_title="Metric Value",
            template="plotly_dark",
            barmode='group',
            height=600,
            showlegend=True
        )
        
        return dcc.Graph(figure=fig)

    def _create_summary_view(self, experiments_data):
        cards = []
        
        for exp in experiments_data:
            metrics_count = len(exp['metrics_history'])
            param_count = len(exp['hyperparameters'])
            status = exp['metadata'].get('status', 'unknown')
            start_time = exp['metadata'].get('start_time', 'unknown')
            
            all_metrics = set()
            for entry in exp['metrics_history']:
                all_metrics.update([k for k in entry.keys() if k not in ['timestamp', 'step']])
            
            latest_metrics = {}
            if exp['metrics_history']:
                latest_entry = exp['metrics_history'][-1]
                latest_metrics = {k: v for k, v in latest_entry.items() 
                                if k not in ['timestamp', 'step']}
            
            card = dbc.Card([
                dbc.CardHeader([
                    html.H5(f"📊 {exp['name']}", className="mb-0"),
                    html.Small(f"ID: {exp['id']}", className="text-muted")
                ]),
                dbc.CardBody([
                    html.P([
                        html.Strong("Status: "),
                        dbc.Badge(status, color="success" if status == "completed" else "warning")
                    ]),
                    html.P([html.Strong("Started: "), start_time]),
                    html.P([html.Strong("Metrics Logged: "), str(metrics_count)]),
                    html.P([html.Strong("Hyperparameters: "), str(param_count)]),
                    html.Hr(),
                    html.H6("Latest Metrics:", className="mb-2"),
                    html.Ul([
                        html.Li(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
                        for k, v in list(latest_metrics.items())[:5]
                    ]) if latest_metrics else html.P("No metrics available", className="text-muted")
                ])
            ], className="mb-3")
            
            cards.append(dbc.Col(card, width=12, lg=6, xl=4))
        
        return dbc.Row(cards)

    def run(self, debug: bool = False, host: str = "127.0.0.1"):
        logger.info(f"Starting Experiment Comparison UI on http://{host}:{self.port}")
        self.app.run_server(debug=debug, host=host, port=self.port)
