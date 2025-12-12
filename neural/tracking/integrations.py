"""
Integrations with external experiment tracking tools.

This module provides integrations with popular experiment tracking tools like
MLflow, Weights & Biases, and TensorBoard.
"""

import os
import logging
import json
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt

# Configure logger
logger = logging.getLogger(__name__)

class BaseIntegration:
    """Base class for external tracking tool integrations."""

    def __init__(self, experiment_name: str = None):
        """
        Initialize the integration.

        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name or "neural_experiment"

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        raise NotImplementedError("Subclasses must implement log_hyperparameters")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Step or epoch number (optional)
        """
        raise NotImplementedError("Subclasses must implement log_metrics")

    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """
        Log an artifact.

        Args:
            artifact_path: Path to the artifact file
            artifact_name: Name to use for the artifact (defaults to filename)
        """
        raise NotImplementedError("Subclasses must implement log_artifact")

    def log_model(self, model_path: str, framework: str = "unknown"):
        """
        Log a model.

        Args:
            model_path: Path to the model file
            framework: Framework used for the model (tensorflow, pytorch, etc.)
        """
        raise NotImplementedError("Subclasses must implement log_model")

    def log_figure(self, figure: plt.Figure, figure_name: str):
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure
            figure_name: Name for the figure
        """
        # Default implementation: save figure to a temporary file and log as artifact
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            figure_path = tmp.name

        figure.savefig(figure_path)
        self.log_artifact(figure_path, figure_name)

        # Clean up the temporary file
        try:
            os.remove(figure_path)
        except:
            pass

    def finish(self):
        """Finish the experiment."""
        pass


class MLflowIntegration(BaseIntegration):
    """Integration with MLflow."""

    def __init__(self, experiment_name: str = None, tracking_uri: str = None, 
                 run_name: str = None, tags: Dict[str, str] = None):
        """
        Initialize the MLflow integration.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (optional)
            run_name: Name for the run (optional)
            tags: Tags for the run (optional)
        """
        super().__init__(experiment_name)
        self.tracking_uri = tracking_uri
        self.run_id = None
        self.run_name = run_name

        try:
            import mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(experiment_name)

            self.run = mlflow.start_run(run_name=run_name, tags=tags)
            self.run_id = self.run.info.run_id

            logger.info(f"Started MLflow run: {self.run_id}")

        except ImportError:
            logger.error("MLflow not installed. Please install it with 'pip install mlflow'.")
            self.run = None

        except Exception as e:
            logger.error(f"Error initializing MLflow: {str(e)}")
            self.run = None

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters to MLflow.

        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        if not self.run:
            logger.error("MLflow run not initialized")
            return

        try:
            import mlflow

            # Convert non-string values to strings
            params = {}
            for key, value in hyperparameters.items():
                if isinstance(value, (str, int, float, bool)):
                    params[key] = value
                else:
                    params[key] = str(value)

            mlflow.log_params(params)
            logger.debug(f"Logged hyperparameters to MLflow: {params}")

        except Exception as e:
            logger.error(f"Error logging hyperparameters to MLflow: {str(e)}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names and values
            step: Step or epoch number (optional)
        """
        if not self.run:
            logger.error("MLflow run not initialized")
            return

        try:
            import mlflow

            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

            if step is not None:
                mlflow.log_metrics(numeric_metrics, step=step)
            else:
                mlflow.log_metrics(numeric_metrics)

            logger.debug(f"Logged metrics to MLflow: {numeric_metrics}")

        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {str(e)}")

    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """
        Log an artifact to MLflow.

        Args:
            artifact_path: Path to the artifact file
            artifact_name: Name to use for the artifact (defaults to filename)
        """
        if not self.run:
            logger.error("MLflow run not initialized")
            return

        if not os.path.exists(artifact_path):
            logger.error(f"Artifact not found: {artifact_path}")
            return

        try:
            import mlflow

            # Log the artifact
            mlflow.log_artifact(artifact_path)
            logger.debug(f"Logged artifact to MLflow: {artifact_path}")

        except Exception as e:
            logger.error(f"Error logging artifact to MLflow: {str(e)}")

    def log_model(self, model_path: str, framework: str = "unknown"):
        """
        Log a model to MLflow.

        Args:
            model_path: Path to the model file
            framework: Framework used for the model (tensorflow, pytorch, etc.)
        """
        if not self.run:
            logger.error("MLflow run not initialized")
            return

        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return

        try:
            import mlflow

            # Log the model as an artifact
            mlflow.log_artifact(model_path, "models")

            # Add a tag for the framework
            mlflow.set_tag("framework", framework)

            logger.debug(f"Logged {framework} model to MLflow: {model_path}")

        except Exception as e:
            logger.error(f"Error logging model to MLflow: {str(e)}")

    def finish(self):
        """End the MLflow run."""
        if not self.run:
            return

        try:
            import mlflow
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")

        except Exception as e:
            logger.error(f"Error ending MLflow run: {str(e)}")


class WandbIntegration(BaseIntegration):
    """Integration with Weights & Biases."""

    def __init__(self, experiment_name: str = None, project_name: str = "neural",
                config: Dict[str, Any] = None, tags: List[str] = None):
        """
        Initialize the Weights & Biases integration.

        Args:
            experiment_name: Name of the experiment (used as run name)
            project_name: Name of the W&B project
            config: Initial configuration for the run
            tags: Tags for the run
        """
        super().__init__(experiment_name)
        self.project_name = project_name
        self.config = config or {}
        self.tags = tags or []
        self.run = None

        try:
            import wandb

            # Initialize W&B
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                tags=tags
            )

            logger.info(f"Started W&B run: {self.run.id}")

        except ImportError:
            logger.error("Weights & Biases not installed. Please install it with 'pip install wandb'.")

        except Exception as e:
            logger.error(f"Error initializing Weights & Biases: {str(e)}")

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters to W&B.

        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        if not self.run:
            logger.error("W&B run not initialized")
            return

        try:
            import wandb

            # Update config
            for key, value in hyperparameters.items():
                self.run.config[key] = value

            logger.debug(f"Logged hyperparameters to W&B: {hyperparameters}")

        except Exception as e:
            logger.error(f"Error logging hyperparameters to W&B: {str(e)}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Step or epoch number (optional)
        """
        if not self.run:
            logger.error("W&B run not initialized")
            return

        try:
            import wandb

            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

            if step is not None:
                self.run.log(numeric_metrics, step=step)
            else:
                self.run.log(numeric_metrics)

            logger.debug(f"Logged metrics to W&B: {numeric_metrics}")

        except Exception as e:
            logger.error(f"Error logging metrics to W&B: {str(e)}")

    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """
        Log an artifact to W&B.

        Args:
            artifact_path: Path to the artifact file
            artifact_name: Name to use for the artifact (defaults to filename)
        """
        if not self.run:
            logger.error("W&B run not initialized")
            return

        if not os.path.exists(artifact_path):
            logger.error(f"Artifact not found: {artifact_path}")
            return

        try:
            import wandb

            # Create an artifact
            artifact_name = artifact_name or os.path.basename(artifact_path)
            artifact_type = self._get_artifact_type(artifact_path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type
            )

            # Add the file to the artifact
            artifact.add_file(artifact_path)

            # Log the artifact
            self.run.log_artifact(artifact)

            logger.debug(f"Logged artifact to W&B: {artifact_path}")

        except Exception as e:
            logger.error(f"Error logging artifact to W&B: {str(e)}")

    def log_model(self, model_path: str, framework: str = "unknown"):
        """
        Log a model to W&B.

        Args:
            model_path: Path to the model file
            framework: Framework used for the model (tensorflow, pytorch, etc.)
        """
        if not self.run:
            logger.error("W&B run not initialized")
            return

        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return

        try:
            import wandb

            # Create a model artifact
            model_name = os.path.basename(model_path)

            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                metadata={"framework": framework}
            )

            # Add the model file to the artifact
            artifact.add_file(model_path)

            # Log the artifact
            self.run.log_artifact(artifact)

            logger.debug(f"Logged {framework} model to W&B: {model_path}")

        except Exception as e:
            logger.error(f"Error logging model to W&B: {str(e)}")

    def log_figure(self, figure: plt.Figure, figure_name: str):
        """
        Log a matplotlib figure to W&B.

        Args:
            figure: Matplotlib figure
            figure_name: Name for the figure
        """
        if not self.run:
            logger.error("W&B run not initialized")
            return

        try:
            import wandb

            # Log the figure directly
            self.run.log({figure_name: wandb.Image(figure)})

            logger.debug(f"Logged figure to W&B: {figure_name}")

        except Exception as e:
            logger.error(f"Error logging figure to W&B: {str(e)}")
            # Fall back to the default implementation
            super().log_figure(figure, figure_name)

    def finish(self):
        """End the W&B run."""
        if not self.run:
            return

        try:
            self.run.finish()
            logger.info(f"Ended W&B run: {self.run.id}")

        except Exception as e:
            logger.error(f"Error ending W&B run: {str(e)}")

    def _get_artifact_type(self, artifact_path: str) -> str:
        """
        Determine the type of an artifact based on its file extension.

        Args:
            artifact_path: Path to the artifact

        Returns:
            Type of the artifact
        """
        ext = os.path.splitext(artifact_path)[1].lower()

        if ext in ['.h5', '.pt', '.pth', '.onnx', '.pb', '.tflite']:
            return 'model'
        elif ext in ['.png', '.jpg', '.jpeg', '.svg', '.gif']:
            return 'image'
        elif ext in ['.csv', '.tsv']:
            return 'dataset'
        elif ext in ['.json', '.yaml', '.yml']:
            return 'config'
        elif ext in ['.txt', '.log', '.md']:
            return 'text'
        else:
            return 'other'


class TensorBoardIntegration(BaseIntegration):
    """Integration with TensorBoard."""

    def __init__(self, experiment_name: str = None, log_dir: str = "runs/neural"):
        """
        Initialize the TensorBoard integration.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for TensorBoard logs
        """
        super().__init__(experiment_name)
        self.log_dir = os.path.join(log_dir, experiment_name) if experiment_name else log_dir
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            # Create the log directory
            os.makedirs(self.log_dir, exist_ok=True)

            # Initialize the writer
            self.writer = SummaryWriter(log_dir=self.log_dir)

            logger.info(f"Initialized TensorBoard writer: {self.log_dir}")

        except ImportError:
            logger.error("TensorBoard not installed. Please install it with 'pip install tensorboard'.")

        except Exception as e:
            logger.error(f"Error initializing TensorBoard: {str(e)}")

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters to TensorBoard.

        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        if not self.writer:
            logger.error("TensorBoard writer not initialized")
            return

        try:
            # Convert hyperparameters to a string for text logging
            hparams_str = json.dumps(hyperparameters, indent=2)

            # Log as text
            self.writer.add_text("hyperparameters", hparams_str)

            # Try to log as hparams if available
            try:
                # Filter out non-primitive types
                filtered_hparams = {}
                for key, value in hyperparameters.items():
                    if isinstance(value, (str, int, float, bool)):
                        filtered_hparams[key] = value
                    else:
                        filtered_hparams[key] = str(value)

                # Log as hparams
                self.writer.add_hparams(filtered_hparams, {})

            except Exception as e:
                logger.debug(f"Could not log hyperparameters using add_hparams: {str(e)}")

            logger.debug(f"Logged hyperparameters to TensorBoard")

        except Exception as e:
            logger.error(f"Error logging hyperparameters to TensorBoard: {str(e)}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric names and values
            step: Step or epoch number (optional)
        """
        if not self.writer:
            logger.error("TensorBoard writer not initialized")
            return

        try:
            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

            # Use 0 as default step if not provided
            step = step if step is not None else 0

            # Log each metric
            for name, value in numeric_metrics.items():
                self.writer.add_scalar(name, value, step)

            logger.debug(f"Logged metrics to TensorBoard: {numeric_metrics}")

        except Exception as e:
            logger.error(f"Error logging metrics to TensorBoard: {str(e)}")

    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """
        Log an artifact to TensorBoard.

        Args:
            artifact_path: Path to the artifact file
            artifact_name: Name to use for the artifact (defaults to filename)
        """
        if not self.writer:
            logger.error("TensorBoard writer not initialized")
            return

        if not os.path.exists(artifact_path):
            logger.error(f"Artifact not found: {artifact_path}")
            return

        try:
            # TensorBoard doesn't have a direct way to log arbitrary artifacts
            # For images, we can use add_image
            ext = os.path.splitext(artifact_path)[1].lower()

            if ext in ['.png', '.jpg', '.jpeg']:
                try:
                    import numpy as np
                    from PIL import Image

                    # Load the image
                    img = Image.open(artifact_path)
                    img_array = np.array(img)

                    # Log the image
                    artifact_name = artifact_name or os.path.basename(artifact_path)
                    self.writer.add_image(artifact_name, img_array, dataformats='HWC')

                    logger.debug(f"Logged image to TensorBoard: {artifact_path}")

                except Exception as e:
                    logger.error(f"Error logging image to TensorBoard: {str(e)}")

            # For text files, we can use add_text
            elif ext in ['.txt', '.log', '.md', '.json', '.yaml', '.yml']:
                try:
                    with open(artifact_path, 'r') as f:
                        content = f.read()

                    # Log the text
                    artifact_name = artifact_name or os.path.basename(artifact_path)
                    self.writer.add_text(artifact_name, content)

                    logger.debug(f"Logged text to TensorBoard: {artifact_path}")

                except Exception as e:
                    logger.error(f"Error logging text to TensorBoard: {str(e)}")

            # For other files, we can't do much with TensorBoard
            else:
                logger.warning(f"TensorBoard doesn't support logging artifacts of type: {ext}")

        except Exception as e:
            logger.error(f"Error logging artifact to TensorBoard: {str(e)}")

    def log_model(self, model_path: str, framework: str = "unknown"):
        """
        Log a model to TensorBoard.

        Args:
            model_path: Path to the model file
            framework: Framework used for the model (tensorflow, pytorch, etc.)
        """
        if not self.writer:
            logger.error("TensorBoard writer not initialized")
            return

        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return

        try:
            # TensorBoard doesn't have a direct way to log models
            # We can log the model architecture if it's a PyTorch model
            if framework == "pytorch":
                try:
                    import torch

                    # Load the model
                    model = torch.load(model_path)

                    # Create a dummy input
                    if hasattr(model, 'input_shape'):
                        dummy_input = torch.randn(1, *model.input_shape)

                        # Log the model graph
                        self.writer.add_graph(model, dummy_input)

                        logger.debug(f"Logged PyTorch model graph to TensorBoard: {model_path}")

                except Exception as e:
                    logger.error(f"Error logging PyTorch model to TensorBoard: {str(e)}")

            # For TensorFlow models, we can use the TensorFlow profiler
            elif framework == "tensorflow":
                logger.warning("TensorBoard doesn't support logging TensorFlow models directly through this interface")

            else:
                logger.warning(f"TensorBoard doesn't support logging models of framework: {framework}")

        except Exception as e:
            logger.error(f"Error logging model to TensorBoard: {str(e)}")

    def log_figure(self, figure: plt.Figure, figure_name: str):
        """
        Log a matplotlib figure to TensorBoard.

        Args:
            figure: Matplotlib figure
            figure_name: Name for the figure
        """
        if not self.writer:
            logger.error("TensorBoard writer not initialized")
            return

        try:
            # Log the figure
            self.writer.add_figure(figure_name, figure)

            logger.debug(f"Logged figure to TensorBoard: {figure_name}")

        except Exception as e:
            logger.error(f"Error logging figure to TensorBoard: {str(e)}")
            # Fall back to the default implementation
            super().log_figure(figure, figure_name)

    def finish(self):
        """Close the TensorBoard writer."""
        if not self.writer:
            return

        try:
            self.writer.close()
            logger.info(f"Closed TensorBoard writer: {self.log_dir}")

        except Exception as e:
            logger.error(f"Error closing TensorBoard writer: {str(e)}")


def create_integration(integration_type: str, experiment_name: str = None, **kwargs) -> BaseIntegration:
    """
    Create an integration with an external tracking tool.

    Args:
        integration_type: Type of integration ('mlflow', 'wandb', 'tensorboard')
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for the integration

    Returns:
        Integration instance
    """
    if integration_type == 'mlflow':
        return MLflowIntegration(experiment_name=experiment_name, **kwargs)
    elif integration_type == 'wandb':
        return WandbIntegration(experiment_name=experiment_name, **kwargs)
    elif integration_type == 'tensorboard':
        return TensorBoardIntegration(experiment_name=experiment_name, **kwargs)
    else:
        logger.error(f"Unknown integration type: {integration_type}")
        return None
