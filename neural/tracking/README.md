# Neural Experiment Tracking

Comprehensive experiment tracking system for Neural with Aquarium dashboard, metrics visualization, and external platform integrations.

## Features

### Core Tracking
- **Experiment Management**: Track experiments with hyperparameters, metrics, and artifacts
- **Metric Logging**: Log metrics at each training step with automatic persistence
- **Artifact Storage**: Store models, figures, and other artifacts
- **Hyperparameter Tracking**: Record and compare hyperparameters across experiments

### Aquarium Dashboard
- **Interactive UI**: Web-based dashboard for viewing and comparing experiments
- **Training Curves**: Visualize metrics over time with interactive charts
- **Experiment Comparison**: Side-by-side comparison of multiple experiments
- **Export Integration**: Export to MLflow, Weights & Biases, or TensorBoard

### Visualization
- **Training Curves**: Plot metrics over time
- **Smoothed Curves**: Apply moving average for clearer trends
- **Distribution Plots**: View metric distributions
- **Correlation Matrix**: Analyze metric correlations
- **Heatmaps**: Visualize metrics across experiments

### Export Integrations
- **MLflow**: Export experiments to MLflow tracking server
- **Weights & Biases**: Sync experiments to W&B projects
- **TensorBoard**: Generate TensorBoard logs from experiments

## Quick Start

### Basic Tracking

```python
from neural.tracking import ExperimentTracker

# Create an experiment
tracker = ExperimentTracker(experiment_name="my_experiment")

# Log hyperparameters
tracker.log_hyperparameters({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
})

# Training loop
for epoch in range(100):
    # ... training code ...
    
    # Log metrics
    tracker.log_metrics({
        "loss": loss,
        "accuracy": accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }, step=epoch)

# Log artifacts
tracker.log_model("model.h5", framework="tensorflow")
tracker.log_figure(plt.gcf(), "training_curves.png")

# Set status
tracker.set_status("completed")

# Save summary
tracker.save_experiment_summary()
```

### Launch Aquarium Dashboard

```python
from neural.tracking import launch_aquarium

# Launch on default port (8053)
launch_aquarium()

# Or customize
launch_aquarium(
    base_dir="./my_experiments",
    port=8080,
    host="0.0.0.0",
    debug=True
)
```

Or from command line:
```bash
python -m neural.tracking.aquarium_app --port 8053 --base-dir neural_experiments
```

### Export to External Platforms

```python
from neural.tracking import ExperimentManager
from neural.tracking.export_manager import ExportManager

manager = ExperimentManager(base_dir="neural_experiments")
exporter = ExportManager(manager)

# Export to MLflow
results = exporter.export_to_mlflow(
    experiment_ids=["exp123", "exp456"],
    tracking_uri="http://localhost:5000"
)

# Export to Weights & Biases
results = exporter.export_to_wandb(
    experiment_ids=["exp123"],
    project_name="my-project",
    tags=["production", "v1.0"]
)

# Export to TensorBoard
results = exporter.export_to_tensorboard(
    experiment_ids=["exp123", "exp456"],
    log_dir="runs/my_experiments"
)
```

### Compare Experiments

```python
from neural.tracking import ExperimentManager

manager = ExperimentManager(base_dir="neural_experiments")

# List all experiments
experiments = manager.list_experiments()
for exp in experiments:
    print(f"{exp['experiment_name']}: {exp['status']}")

# Get specific experiment
exp = manager.get_experiment("exp123")
print(f"Best accuracy: {exp.get_best_metric('accuracy', mode='max')}")

# Compare experiments
plots = manager.compare_experiments(
    experiment_ids=["exp123", "exp456"],
    metric_names=["accuracy", "loss"]
)
```

### Advanced Visualization

```python
from neural.tracking import ExperimentManager
from neural.tracking.metrics_visualizer import MetricsVisualizerComponent

manager = ExperimentManager()
exp = manager.get_experiment("exp123")

visualizer = MetricsVisualizerComponent(exp)

# Create training curves
fig = visualizer.create_training_curves()
fig.show()

# Create smoothed curves
fig = visualizer.create_smoothed_curves(window_size=10)
fig.show()

# Create distribution plot
fig = visualizer.create_distribution_plot("accuracy")
fig.show()

# Create correlation matrix
fig = visualizer.create_correlation_matrix()
fig.show()
```

## Architecture

```
neural/tracking/
├── experiment_tracker.py      # Core tracking classes
├── integrations.py           # External platform integrations
├── aquarium_app.py           # Main Aquarium dashboard
├── comparison_component.py   # Experiment comparison UI
├── metrics_visualizer.py     # Advanced visualization
├── export_manager.py         # Export management
└── comparison_ui.py          # Legacy comparison UI
```

## Storage Format

Experiments are stored in the following structure:

```
neural_experiments/
└── experiment_name_expid123/
    ├── metadata.json           # Experiment metadata
    ├── hyperparameters.json    # Hyperparameters
    ├── metrics.json            # Metrics history
    ├── artifacts.json          # Artifact metadata
    ├── summary.json            # Experiment summary
    ├── artifacts/              # Artifact files
    │   ├── model.h5
    │   └── data.csv
    └── plots/                  # Generated plots
        └── training_curves.png
```

## Integration Details

### MLflow
- Logs hyperparameters as params
- Logs metrics with step information
- Uploads artifacts to MLflow artifact store
- Creates runs under specified experiment

### Weights & Biases
- Logs hyperparameters to run config
- Logs metrics with optional step
- Creates W&B artifacts for files
- Supports image logging

### TensorBoard
- Logs hyperparameters as text and hparams
- Logs metrics as scalars
- Supports image and text artifacts
- Creates structured log directories

## API Reference

### ExperimentTracker

Main class for tracking a single experiment.

**Methods:**
- `log_hyperparameters(params: Dict)`: Log hyperparameters
- `log_metrics(metrics: Dict, step: int)`: Log metrics at a step
- `log_artifact(path: str, name: str)`: Log an artifact file
- `log_model(path: str, framework: str)`: Log a model file
- `log_figure(fig: Figure, name: str)`: Log a matplotlib figure
- `set_status(status: str)`: Set experiment status
- `get_metrics(metric_name: str)`: Get metric history
- `get_best_metric(metric: str, mode: str)`: Get best metric value
- `plot_metrics(metrics: List[str])`: Plot training curves
- `save_experiment_summary()`: Save experiment summary

### ExperimentManager

Manager for multiple experiments.

**Methods:**
- `create_experiment(name: str)`: Create new experiment
- `get_experiment(id: str)`: Get experiment by ID
- `list_experiments()`: List all experiments
- `delete_experiment(id: str)`: Delete an experiment
- `compare_experiments(ids: List[str], metrics: List[str])`: Compare experiments

### ExportManager

Manager for exporting to external platforms.

**Methods:**
- `export_to_mlflow(ids: List[str], uri: str)`: Export to MLflow
- `export_to_wandb(ids: List[str], project: str)`: Export to W&B
- `export_to_tensorboard(ids: List[str], logdir: str)`: Export to TensorBoard
- `export_batch(ids: List[str], platforms: List[str])`: Export to multiple platforms

## Requirements

**Core:**
- numpy
- matplotlib
- pyyaml

**Dashboard:**
- dash
- dash-bootstrap-components
- plotly

**Integrations (optional):**
- mlflow (for MLflow integration)
- wandb (for Weights & Biases integration)
- tensorboard (for TensorBoard integration)

## Examples

See `examples/tracking_example.py` for complete usage examples.

## Command Line Usage

```bash
# Launch Aquarium dashboard
python -m neural.tracking.aquarium_app --port 8053

# Launch comparison UI
python -m neural.tracking.comparison_ui --port 8052

# With custom directory
python -m neural.tracking.aquarium_app --base-dir ./my_experiments --port 8080
```
