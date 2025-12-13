"""
Export manager for exporting experiments to external platforms.
"""

import logging
from typing import Dict, List, Optional

from neural.tracking.experiment_tracker import ExperimentManager, ExperimentTracker
from neural.tracking.integrations import (
    MLflowIntegration,
    TensorBoardIntegration,
    WandbIntegration,
)


logger = logging.getLogger(__name__)


class ExportManager:
    """Manager for exporting experiments to external platforms."""

    def __init__(self, manager: ExperimentManager):
        """
        Initialize the export manager.

        Args:
            manager: ExperimentManager instance
        """
        self.manager = manager

    def export_to_mlflow(
        self,
        experiment_ids: List[str],
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Export experiments to MLflow.

        Args:
            experiment_ids: List of experiment IDs to export
            tracking_uri: MLflow tracking URI (optional)
            tags: Tags to add to the runs (optional)

        Returns:
            Dictionary mapping experiment IDs to status messages
        """
        results = {}

        for exp_id in experiment_ids:
            try:
                exp = self.manager.get_experiment(exp_id)
                if not exp:
                    results[exp_id] = "Experiment not found"
                    continue

                integration = MLflowIntegration(
                    experiment_name=exp.experiment_name,
                    tracking_uri=tracking_uri,
                    run_name=f"{exp.experiment_name}_{exp.experiment_id}",
                    tags=tags,
                )

                if not integration.run:
                    results[exp_id] = "Failed to initialize MLflow"
                    continue

                integration.log_hyperparameters(exp.hyperparameters)

                for entry in exp.metrics_history:
                    step = entry.get("step")
                    metrics = {
                        k: v
                        for k, v in entry.items()
                        if k not in ["timestamp", "step"] and isinstance(v, (int, float))
                    }
                    integration.log_metrics(metrics, step=step)

                for artifact_name, artifact_info in exp.artifacts.items():
                    integration.log_artifact(artifact_info["path"], artifact_name)

                integration.finish()
                results[exp_id] = f"Success: Run ID {integration.run_id}"
                logger.info(f"Exported experiment {exp_id} to MLflow")

            except Exception as e:
                results[exp_id] = f"Error: {str(e)}"
                logger.error(f"Failed to export experiment {exp_id} to MLflow: {str(e)}")

        return results

    def export_to_wandb(
        self,
        experiment_ids: List[str],
        project_name: str = "neural",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Export experiments to Weights & Biases.

        Args:
            experiment_ids: List of experiment IDs to export
            project_name: W&B project name
            tags: Tags to add to the runs (optional)

        Returns:
            Dictionary mapping experiment IDs to status messages
        """
        results = {}

        for exp_id in experiment_ids:
            try:
                exp = self.manager.get_experiment(exp_id)
                if not exp:
                    results[exp_id] = "Experiment not found"
                    continue

                integration = WandbIntegration(
                    experiment_name=exp.experiment_name,
                    project_name=project_name,
                    config=exp.hyperparameters,
                    tags=tags,
                )

                if not integration.run:
                    results[exp_id] = "Failed to initialize W&B"
                    continue

                for entry in exp.metrics_history:
                    step = entry.get("step")
                    metrics = {
                        k: v
                        for k, v in entry.items()
                        if k not in ["timestamp", "step"] and isinstance(v, (int, float))
                    }
                    integration.log_metrics(metrics, step=step)

                for artifact_name, artifact_info in exp.artifacts.items():
                    integration.log_artifact(artifact_info["path"], artifact_name)

                integration.finish()
                results[exp_id] = f"Success: Run ID {integration.run.id if integration.run else 'N/A'}"
                logger.info(f"Exported experiment {exp_id} to W&B")

            except Exception as e:
                results[exp_id] = f"Error: {str(e)}"
                logger.error(f"Failed to export experiment {exp_id} to W&B: {str(e)}")

        return results

    def export_to_tensorboard(
        self, experiment_ids: List[str], log_dir: str = "runs/neural"
    ) -> Dict[str, str]:
        """
        Export experiments to TensorBoard.

        Args:
            experiment_ids: List of experiment IDs to export
            log_dir: TensorBoard log directory

        Returns:
            Dictionary mapping experiment IDs to status messages
        """
        results = {}

        for exp_id in experiment_ids:
            try:
                exp = self.manager.get_experiment(exp_id)
                if not exp:
                    results[exp_id] = "Experiment not found"
                    continue

                integration = TensorBoardIntegration(
                    experiment_name=exp.experiment_name, log_dir=log_dir
                )

                if not integration.writer:
                    results[exp_id] = "Failed to initialize TensorBoard"
                    continue

                integration.log_hyperparameters(exp.hyperparameters)

                for entry in exp.metrics_history:
                    step = entry.get("step", 0)
                    metrics = {
                        k: v
                        for k, v in entry.items()
                        if k not in ["timestamp", "step"] and isinstance(v, (int, float))
                    }
                    integration.log_metrics(metrics, step=step)

                for artifact_name, artifact_info in exp.artifacts.items():
                    if artifact_info.get("type") in ["image", "figure", "text"]:
                        integration.log_artifact(artifact_info["path"], artifact_name)

                integration.finish()
                results[exp_id] = f"Success: {integration.log_dir}"
                logger.info(f"Exported experiment {exp_id} to TensorBoard")

            except Exception as e:
                results[exp_id] = f"Error: {str(e)}"
                logger.error(f"Failed to export experiment {exp_id} to TensorBoard: {str(e)}")

        return results

    def export_batch(
        self,
        experiment_ids: List[str],
        platforms: List[str],
        mlflow_uri: Optional[str] = None,
        wandb_project: str = "neural",
        tensorboard_logdir: str = "runs/neural",
    ) -> Dict[str, Dict[str, str]]:
        """
        Export experiments to multiple platforms at once.

        Args:
            experiment_ids: List of experiment IDs to export
            platforms: List of platforms ('mlflow', 'wandb', 'tensorboard')
            mlflow_uri: MLflow tracking URI (optional)
            wandb_project: W&B project name
            tensorboard_logdir: TensorBoard log directory

        Returns:
            Dictionary mapping platform names to export results
        """
        results = {}

        if "mlflow" in platforms:
            results["mlflow"] = self.export_to_mlflow(experiment_ids, tracking_uri=mlflow_uri)

        if "wandb" in platforms:
            results["wandb"] = self.export_to_wandb(experiment_ids, project_name=wandb_project)

        if "tensorboard" in platforms:
            results["tensorboard"] = self.export_to_tensorboard(
                experiment_ids, log_dir=tensorboard_logdir
            )

        return results
