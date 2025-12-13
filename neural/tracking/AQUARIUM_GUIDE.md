# Aquarium Experiment Tracking Guide

Aquarium is Neural's comprehensive experiment tracking dashboard that provides a modern web interface for managing, visualizing, and exporting machine learning experiments.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Dashboard Interface](#dashboard-interface)
- [Tracking Experiments](#tracking-experiments)
- [Visualization](#visualization)
- [Export Integrations](#export-integrations)
- [CLI Commands](#cli-commands)
- [Python API](#python-api)
- [Best Practices](#best-practices)

## Features

### 🎯 Core Features

- **Experiment Management**: Track multiple experiments with automatic persistence
- **Interactive Dashboard**: Modern web UI with real-time updates
- **Metrics Visualization**: Interactive charts and training curves
- **Comparison Tools**: Side-by-side experiment comparison
- **Export Integration**: Export to MLflow, Weights & Biases, TensorBoard
- **Artifact Storage**: Store models, plots, and other artifacts
- **Hyperparameter Tracking**: Record and compare hyperparameters

### 🎨 Visualization Features

- Training curves with smoothing
- Distribution plots
- Correlation matrices
- Heatmaps
- Multi-metric comparison
- Custom plot generation

### 🔄 Export Targets

- **MLflow**: Full experiment export with metrics, params, and artifacts
- **Weights & Biases**: Project-based organization with artifact versioning
- **TensorBoard**: Scalar metrics and visualization logs

## Quick Start

### 1. Install Dependencies

```bash
# Core tracking (already included)
pip install neural-dsl

# Dashboard dependencies
pip install dash dash-bootstrap-components plotly

# Optional: External integrations
pip install mlflow wandb tensorboard
```

### 2. Launch Aquarium

From command line:

```bash
# Using Neural CLI
neural aquarium

# Direct Python
python -m neural.tracking.aquarium_app

# Custom configuration
neural aquarium --port 8080 --base-dir ./my_experiments
```

From Python:

```python
from neural.tracking import launch_aquarium

launch_aquarium(port=8053)
```

### 3. Track Your First Experiment

```python
from neural.tracking import ExperimentTracker

# Create experiment
tracker = ExperimentTracker(experiment_name="my_first_experiment")

# Log hyperparameters
tracker.log_hyperparameters({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50
})

# Training loop
for epoch in range(50):
    # ... your training code ...
    
    tracker.log_metrics({
        "loss": train_loss,
        "accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    }, step=epoch)

# Log artifacts
tracker.log_model("model.h5", framework="tensorflow")
tracker.set_status("completed")
```

### 4. View in Aquarium

Navigate to `http://127.0.0.1:8053` to see your experiments.

## Dashboard Interface

### Main Tabs

#### 1. Experiments Tab

Lists all tracked experiments with:
- Experiment name and ID
- Status (running, completed, failed)
- Start time and duration
- Latest metrics
- Interactive filtering and sorting
- Multi-select for comparison

**Features:**
- Click on experiment to view details
- Select multiple for comparison
- Filter by status, date, metrics
- Sort by any column
- Real-time updates (5-second refresh)

#### 2. Compare Tab

Side-by-side experiment comparison:
- **Summary Cards**: Overview of each experiment
- **Metrics Comparison**: Interactive line charts for each metric
- **Hyperparameter Table**: Tabular comparison of all hyperparameters
- **Performance Summary**: Best values for each metric

**Features:**
- Up to 6 metrics displayed simultaneously
- Hover for detailed values
- Unified x-axis for easy comparison
- Export comparison as image/HTML

#### 3. Export Tab

Export experiments to external platforms:

**MLflow Card:**
- Tracking URI configuration
- Experiment ID selection
- One-click export

**Weights & Biases Card:**
- Project name configuration
- Tag support
- Artifact syncing

**TensorBoard Card:**
- Log directory configuration
- Scalar metrics export
- Image/text artifact support

## Tracking Experiments

### Basic Tracking

```python
from neural.tracking import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="my_experiment",
    base_dir="./experiments"  # Custom directory
)

# Log hyperparameters
tracker.log_hyperparameters({
    "model": "resnet50",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 32
})

# Log metrics during training
tracker.log_metrics({
    "loss": 0.5,
    "accuracy": 0.92
}, step=10)

# Log artifacts
tracker.log_artifact("config.yaml")
tracker.log_model("model.pth", framework="pytorch")
tracker.log_figure(fig, "training_curve.png")

# Set status
tracker.set_status("completed")
```

### Advanced Features

```python
# Get best metric
best_acc, best_step = tracker.get_best_metric("accuracy", mode="max")
print(f"Best accuracy: {best_acc} at step {best_step}")

# Get metric history
acc_history = tracker.get_metrics("accuracy")

# Plot metrics
fig = tracker.plot_metrics(["loss", "accuracy"])
fig.savefig("metrics.png")

# Save summary
summary_path = tracker.save_experiment_summary()
```

### Using ExperimentManager

```python
from neural.tracking import ExperimentManager

manager = ExperimentManager(base_dir="./experiments")

# Create experiment
exp = manager.create_experiment("exp1")

# List all experiments
experiments = manager.list_experiments()
for exp_info in experiments:
    print(f"{exp_info['experiment_name']}: {exp_info['status']}")

# Get specific experiment
exp = manager.get_experiment("exp_id_123")

# Delete experiment
manager.delete_experiment("exp_id_123")

# Compare experiments
plots = manager.compare_experiments(
    experiment_ids=["exp1", "exp2"],
    metric_names=["accuracy", "loss"]
)
```

## Visualization

### Using MetricsVisualizerComponent

```python
from neural.tracking import ExperimentManager
from neural.tracking.metrics_visualizer import MetricsVisualizerComponent

manager = ExperimentManager()
exp = manager.get_experiment("exp_id")

visualizer = MetricsVisualizerComponent(exp)

# Training curves
fig = visualizer.create_training_curves()
fig.show()

# Smoothed curves
fig = visualizer.create_smoothed_curves(window_size=10)
fig.show()

# Distribution plot
fig = visualizer.create_distribution_plot("accuracy")
fig.show()

# Correlation matrix
fig = visualizer.create_correlation_matrix()
fig.show()

# Metrics heatmap
fig = visualizer.create_metrics_heatmap()
fig.show()
```

### Customizing Visualizations

```python
# Custom training curves
fig = visualizer.create_training_curves(
    metric_names=["loss", "val_loss"]
)
fig.update_layout(
    title="Custom Training Curves",
    template="plotly_dark"
)

# Save as HTML
fig.write_html("custom_curves.html")

# Save as PNG
fig.write_image("custom_curves.png")
```

## Export Integrations

### MLflow Export

```python
from neural.tracking import ExperimentManager
from neural.tracking.export_manager import ExportManager

manager = ExperimentManager()
exporter = ExportManager(manager)

# Export single experiment
results = exporter.export_to_mlflow(
    experiment_ids=["exp123"],
    tracking_uri="http://localhost:5000"
)

# With tags
results = exporter.export_to_mlflow(
    experiment_ids=["exp123"],
    tracking_uri="http://localhost:5000",
    tags={"team": "research", "version": "v1.0"}
)
```

### Weights & Biases Export

```python
# Export to W&B
results = exporter.export_to_wandb(
    experiment_ids=["exp123"],
    project_name="my-project",
    tags=["baseline", "v1.0"]
)

# Multiple experiments
results = exporter.export_to_wandb(
    experiment_ids=["exp123", "exp456"],
    project_name="my-project"
)
```

### TensorBoard Export

```python
# Export to TensorBoard
results = exporter.export_to_tensorboard(
    experiment_ids=["exp123"],
    log_dir="runs/my_experiments"
)

# View in TensorBoard
# tensorboard --logdir runs/my_experiments
```

### Batch Export

```python
# Export to multiple platforms
results = exporter.export_batch(
    experiment_ids=["exp123", "exp456"],
    platforms=["mlflow", "wandb", "tensorboard"],
    mlflow_uri="http://localhost:5000",
    wandb_project="my-project",
    tensorboard_logdir="runs/exports"
)

# Check results
for platform, platform_results in results.items():
    print(f"\n{platform}:")
    for exp_id, status in platform_results.items():
        print(f"  {exp_id}: {status}")
```

## CLI Commands

### Neural CLI Integration

```bash
# Launch Aquarium
neural aquarium
neural aquarium --port 8080
neural aquarium --base-dir ./experiments

# Track experiment initialization
neural track init my_experiment
neural track init --integration mlflow --tracking-uri http://localhost:5000

# Log data
neural track log --hyperparameters '{"lr": 0.001}'
neural track log --metrics '{"loss": 0.5}' --step 10
neural track log --model model.h5 --framework tensorflow
```

### Direct Module Execution

```bash
# Aquarium dashboard
python -m neural.tracking.aquarium_app --port 8053

# Legacy comparison UI
python -m neural.tracking.comparison_ui --port 8052
```

## Python API

### Complete API Example

```python
from neural.tracking import (
    ExperimentTracker,
    ExperimentManager,
    launch_aquarium
)
from neural.tracking.export_manager import ExportManager
from neural.tracking.metrics_visualizer import MetricsVisualizerComponent
from neural.tracking.comparison_component import ComparisonComponent

# 1. Create and track experiment
tracker = ExperimentTracker("my_experiment")
tracker.log_hyperparameters({"lr": 0.001, "batch_size": 32})
tracker.log_metrics({"loss": 0.5, "acc": 0.92}, step=10)
tracker.set_status("completed")

# 2. Manage experiments
manager = ExperimentManager()
experiments = manager.list_experiments()
exp = manager.get_experiment("exp_id")

# 3. Compare experiments
comparison = ComparisonComponent([exp1, exp2])
comparison_html = comparison.render()

# 4. Visualize metrics
visualizer = MetricsVisualizerComponent(exp)
fig = visualizer.create_training_curves()

# 5. Export
exporter = ExportManager(manager)
results = exporter.export_to_mlflow(["exp_id"])

# 6. Launch dashboard
launch_aquarium(port=8053)
```

## Best Practices

### 1. Naming Conventions

```python
# Use descriptive experiment names
tracker = ExperimentTracker(
    experiment_name="resnet50_imagenet_baseline"
)

# Include important parameters in name
tracker = ExperimentTracker(
    experiment_name=f"resnet50_lr{lr}_bs{batch_size}"
)
```

### 2. Comprehensive Logging

```python
# Log all relevant hyperparameters
tracker.log_hyperparameters({
    # Model architecture
    "model": "resnet50",
    "num_layers": 50,
    "hidden_dim": 512,
    
    # Training config
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam",
    
    # Data config
    "dataset": "imagenet",
    "augmentation": "standard",
    
    # Environment
    "gpu": "Tesla V100",
    "framework": "tensorflow",
    "version": "2.8.0"
})

# Log multiple metrics
tracker.log_metrics({
    "train_loss": train_loss,
    "train_accuracy": train_acc,
    "val_loss": val_loss,
    "val_accuracy": val_acc,
    "learning_rate": current_lr
}, step=epoch)
```

### 3. Status Management

```python
tracker.set_status("running")
try:
    # Training code
    for epoch in range(epochs):
        # ...
        tracker.log_metrics(metrics, step=epoch)
    tracker.set_status("completed")
except Exception as e:
    tracker.set_status("failed")
    tracker.log_artifact("error.log")
    raise
```

### 4. Artifact Organization

```python
# Log models at checkpoints
tracker.log_model(f"checkpoint_epoch_{epoch}.h5", framework="tensorflow")

# Log final model
tracker.log_model("final_model.h5", framework="tensorflow")

# Log configuration files
tracker.log_artifact("config.yaml")

# Log generated plots
fig = create_plot()
tracker.log_figure(fig, "training_curves.png")
```

### 5. Regular Summaries

```python
# Save summary at the end
tracker.save_experiment_summary()

# Or periodically
if epoch % 10 == 0:
    tracker.save_experiment_summary()
```

## Troubleshooting

### Common Issues

**Dashboard won't start:**
```bash
pip install dash dash-bootstrap-components plotly
```

**Export fails:**
```bash
# For MLflow
pip install mlflow

# For W&B
pip install wandb
wandb login

# For TensorBoard
pip install tensorboard
```

**Experiments not showing:**
- Check base_dir path is correct
- Verify experiments directory exists
- Check file permissions

**Slow dashboard:**
- Reduce refresh interval in code
- Archive old experiments
- Use filtering in experiment list

## Support

For issues, questions, or contributions:
- GitHub: https://github.com/neural-dsl/neural
- Documentation: See neural/tracking/README.md
- Examples: See examples/tracking_example.py
