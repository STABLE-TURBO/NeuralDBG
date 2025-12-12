# Neural DSL Experiment Tracking Guide

Complete guide for using the Neural DSL experiment tracking system with artifact versioning, automatic visualizations, and backend integrations.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Features](#core-features)
5. [Backend Integrations](#backend-integrations)
6. [Experiment Comparison UI](#experiment-comparison-ui)
7. [Best Practices](#best-practices)
8. [Advanced Usage](#advanced-usage)
9. [API Reference](#api-reference)

## Overview

The Neural DSL experiment tracking system provides:

- **Automatic metric logging and visualization** - Track training metrics with auto-generated plots
- **Artifact versioning** - Keep track of model checkpoints with SHA-256 checksums
- **Backend integrations** - Seamless integration with MLflow, Weights & Biases, and TensorBoard
- **Interactive comparison UI** - Web interface to compare multiple experiments
- **Persistent storage** - All data stored locally in organized directories

## Installation

### Basic Installation

```bash
pip install -e .
```

### Full Installation (includes UI and backends)

```bash
pip install -e ".[full]"
```

### Manual Backend Installation

```bash
# For MLflow
pip install mlflow

# For Weights & Biases
pip install wandb

# For TensorBoard
pip install tensorboard

# For Comparison UI
pip install dash dash-bootstrap-components plotly
```

## Quick Start

### 1. Basic Experiment Tracking

```python
from neural.tracking import ExperimentTracker

# Create tracker
tracker = ExperimentTracker(
    experiment_name="my_first_experiment",
    auto_visualize=True
)

# Log hyperparameters
tracker.log_hyperparameters({
    'learning_rate': 0.01,
    'batch_size': 32,
    'optimizer': 'adam',
    'epochs': 100
})

# Training loop
for epoch in range(100):
    # ... your training code ...
    
    # Log metrics
    tracker.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    }, step=epoch)
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        save_model(f"checkpoint_{epoch}.pt")
        tracker.log_model(
            f"checkpoint_{epoch}.pt",
            framework="pytorch",
            version=True
        )

# Finish experiment
tracker.finish()

print(f"Experiment ID: {tracker.experiment_id}")
print(f"Results saved to: {tracker.experiment_dir}")
```

### 2. View Results

After running your experiment:

```python
from neural.tracking import ExperimentManager

manager = ExperimentManager()
experiments = manager.list_experiments()

for exp in experiments:
    print(f"{exp['experiment_name']} - {exp['status']}")
```

### 3. Launch Comparison UI

```bash
python neural/tracking/comparison_ui.py
```

Then open http://127.0.0.1:8052 in your browser.

## Core Features

### Automatic Metric Visualization

The tracker automatically generates visualizations every N metric logs:

```python
tracker = ExperimentTracker(
    experiment_name="auto_viz_demo",
    auto_visualize=True  # Enable automatic visualization
)

# Visualizations are generated automatically every 10 steps
for step in range(100):
    tracker.log_metrics({'loss': loss, 'accuracy': acc}, step=step)
```

Generated visualizations include:
- Individual metric plots with min/max markers
- Combined metrics plot
- Metric correlation matrix
- Distribution histograms

### Artifact Versioning

Track multiple versions of the same artifact:

```python
tracker = ExperimentTracker(experiment_name="versioning_demo")

# Save multiple checkpoints
for epoch in [10, 20, 30, 40, 50]:
    save_checkpoint(f"model_{epoch}.pt")
    tracker.log_model(
        f"model_{epoch}.pt",
        framework="pytorch",
        version=True  # Enable versioning
    )

# List all versions
versions = tracker.list_artifact_versions("model.pt")
for v in versions:
    print(f"Version {v.version}: {v.checksum[:8]}... ({v.size} bytes)")

# Get specific version
latest = tracker.get_artifact_version("model.pt", version=-1)
v2 = tracker.get_artifact_version("model.pt", version=2)

print(f"Latest version: {latest.path}")
print(f"Version 2: {v2.path}")
```

Each version includes:
- Version number
- SHA-256 checksum
- File size
- Timestamp
- Path to versioned file

### Manual Visualization Control

```python
tracker = ExperimentTracker(
    experiment_name="manual_viz",
    auto_visualize=False  # Disable automatic visualization
)

# ... training code ...

# Generate visualizations manually when needed
saved_plots = tracker.generate_visualizations()

for plot_name, plot_path in saved_plots.items():
    print(f"{plot_name}: {plot_path}")
```

### Custom Figures

Log your own matplotlib figures:

```python
import matplotlib.pyplot as plt

# Create custom plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, losses)
ax.set_title("Custom Training Curve")

# Log it
tracker.log_figure(fig, "custom_training_curve.png")
plt.close(fig)
```

## Backend Integrations

### MLflow Integration

```python
tracker = ExperimentTracker(
    experiment_name="mlflow_experiment",
    backend="mlflow"
)

# All metrics and artifacts are automatically logged to MLflow
tracker.log_hyperparameters({'lr': 0.01})
tracker.log_metrics({'loss': 0.5}, step=1)
tracker.log_model("model.pt", framework="pytorch")

tracker.finish()

# View in MLflow UI
# mlflow ui --backend-store-uri ./mlruns
```

### Weights & Biases Integration

```python
# First time setup (one-time)
# wandb login

tracker = ExperimentTracker(
    experiment_name="wandb_experiment",
    backend="wandb"
)

# Automatically synced with W&B
tracker.log_hyperparameters({'lr': 0.01, 'batch_size': 32})

for step in range(100):
    tracker.log_metrics({'loss': loss, 'accuracy': acc}, step=step)

tracker.finish()
```

### TensorBoard Integration

```python
tracker = ExperimentTracker(
    experiment_name="tensorboard_experiment",
    backend="tensorboard"
)

# Logged to TensorBoard
for step in range(100):
    tracker.log_metrics({'loss': loss}, step=step)

tracker.finish()

# View in TensorBoard
# tensorboard --logdir=runs/tensorboard_experiment
```

### Using Multiple Backends

To use multiple backends simultaneously:

```python
from neural.tracking import ExperimentTracker, create_integration

tracker = ExperimentTracker(experiment_name="multi_backend")

# Add MLflow
mlflow_backend = create_integration('mlflow', 'multi_backend')

# Add W&B
wandb_backend = create_integration('wandb', 'multi_backend')

# Log to all backends
hyperparams = {'lr': 0.01}
tracker.log_hyperparameters(hyperparams)
mlflow_backend.log_hyperparameters(hyperparams)
wandb_backend.log_hyperparameters(hyperparams)

# ... training ...

tracker.finish()
mlflow_backend.finish()
wandb_backend.finish()
```

## Experiment Comparison UI

### Launching the UI

**Method 1: Command Line**
```bash
python neural/tracking/comparison_ui.py --port 8052 --base-dir ./neural_experiments
```

**Method 2: Python API**
```python
from neural.tracking import launch_comparison_ui

launch_comparison_ui(
    base_dir="neural_experiments",
    port=8052,
    debug=False
)
```

**Method 3: Direct Class Usage**
```python
from neural.tracking import ExperimentManager, ExperimentComparisonUI

manager = ExperimentManager(base_dir="neural_experiments")
ui = ExperimentComparisonUI(manager=manager, port=8052)
ui.run(debug=False)
```

### UI Features

The comparison UI provides four main tabs:

#### 1. Metrics Comparison
- Interactive line plots for each metric
- Compare multiple experiments on the same plot
- Hover to see exact values
- Auto-scaling axes

#### 2. Hyperparameters
- Side-by-side comparison table
- Highlights differences between experiments
- Easy to spot configuration changes

#### 3. Best Metrics
- Bar chart showing best achieved values
- Compare up to 10 metrics at once
- Grouped by experiment

#### 4. Summary
- Overview cards for each experiment
- Status, start time, metrics count
- Latest metric values
- Quick experiment comparison

### Programmatic Comparison

```python
from neural.tracking import ExperimentManager

manager = ExperimentManager()

# Compare specific experiments
experiment_ids = ['abc12345', 'def67890', 'ghi11121']
plots = manager.compare_experiments(experiment_ids)

# Save comparison plots
for plot_name, fig in plots.items():
    fig.savefig(f"comparison_{plot_name}.png")

# Export full comparison report
output_dir = manager.export_comparison(experiment_ids)
print(f"Comparison report saved to: {output_dir}")
```

## Best Practices

### 1. Naming Conventions

Use descriptive experiment names:

```python
# Good
tracker = ExperimentTracker("resnet50_imagenet_lr0.01_batch128")

# Less descriptive
tracker = ExperimentTracker("exp1")
```

### 2. Hyperparameter Organization

Log all important hyperparameters:

```python
hyperparams = {
    # Model
    'model_architecture': 'resnet50',
    'num_layers': 50,
    'dropout': 0.5,
    
    # Training
    'learning_rate': 0.01,
    'batch_size': 32,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
    
    # Data
    'dataset': 'imagenet',
    'augmentation': True,
    
    # Other
    'random_seed': 42
}

tracker.log_hyperparameters(hyperparams)
```

### 3. Consistent Metric Names

Use the same metric names across experiments:

```python
# Use consistent naming
tracker.log_metrics({
    'train_loss': ...,
    'val_loss': ...,
    'train_accuracy': ...,
    'val_accuracy': ...
}, step=epoch)

# Avoid inconsistent naming
# Bad: 'training_loss', 'train_loss', 'loss_train'
```

### 4. Regular Checkpointing

Save checkpoints regularly with versioning:

```python
checkpoint_frequency = 10

for epoch in range(num_epochs):
    # ... training ...
    
    if epoch % checkpoint_frequency == 0:
        checkpoint_path = save_checkpoint(model, epoch)
        tracker.log_model(
            checkpoint_path,
            framework="pytorch",
            version=True
        )
```

### 5. Error Handling

Always finish experiments properly:

```python
tracker = ExperimentTracker("my_experiment")

try:
    tracker.log_hyperparameters(hyperparams)
    
    for epoch in range(num_epochs):
        # Training code
        tracker.log_metrics(metrics, step=epoch)
    
    tracker.set_status("completed")
    
except Exception as e:
    tracker.set_status("failed")
    raise
    
finally:
    tracker.finish()
```

## Advanced Usage

### Custom Metric Visualizations

```python
from neural.tracking import MetricVisualizer

# Create custom visualizations
plots = MetricVisualizer.create_metric_plots(
    tracker.metrics_history,
    figsize=(15, 10)
)

# Distribution plots
dist_plots = MetricVisualizer.create_distribution_plots(
    tracker.metrics_history,
    figsize=(15, 8)
)
```

### Querying Experiments

```python
manager = ExperimentManager()

# Get all experiments
experiments = manager.list_experiments()

# Filter by status
completed = [e for e in experiments if e['status'] == 'completed']

# Sort by start time
sorted_exps = sorted(experiments, key=lambda x: x['start_time'])

# Get specific experiment
tracker = manager.get_experiment('abc12345')

# Query metrics
best_acc, step = tracker.get_best_metric('accuracy', mode='max')
print(f"Best accuracy: {best_acc:.4f} at step {step}")

# Get all values for a metric
losses = tracker.get_metrics('train_loss')
```

### Experiment Deletion

```python
manager = ExperimentManager()

# Delete specific experiment
success = manager.delete_experiment('abc12345')

if success:
    print("Experiment deleted successfully")
```

### Custom Storage Location

```python
tracker = ExperimentTracker(
    experiment_name="custom_location",
    base_dir="/path/to/my/experiments"
)
```

## API Reference

### ExperimentTracker

**Constructor:**
```python
ExperimentTracker(
    experiment_name: str = None,
    base_dir: str = "neural_experiments",
    auto_visualize: bool = True,
    backend: Optional[str] = None
)
```

**Methods:**

- `log_hyperparameters(hyperparameters: Dict[str, Any])` - Log hyperparameters
- `log_metrics(metrics: Dict[str, float], step: int)` - Log metrics
- `log_artifact(path: str, name: str, version: bool)` - Log artifact with versioning
- `log_model(path: str, framework: str, version: bool)` - Log model with versioning
- `log_figure(figure: plt.Figure, name: str)` - Log matplotlib figure
- `get_metrics(metric_name: str)` - Get metric history
- `get_best_metric(metric_name: str, mode: str)` - Get best metric value
- `get_artifact_version(name: str, version: int)` - Get specific artifact version
- `list_artifact_versions(name: str)` - List all artifact versions
- `generate_visualizations()` - Generate all visualizations
- `set_status(status: str)` - Set experiment status
- `finish()` - Complete experiment

### ExperimentManager

**Constructor:**
```python
ExperimentManager(base_dir: str = "neural_experiments")
```

**Methods:**

- `create_experiment(name: str, auto_visualize: bool, backend: str)` - Create new experiment
- `get_experiment(experiment_id: str)` - Load existing experiment
- `list_experiments()` - List all experiments
- `delete_experiment(experiment_id: str)` - Delete experiment
- `compare_experiments(ids: List[str], metrics: List[str])` - Compare experiments
- `export_comparison(ids: List[str], output_dir: str)` - Export comparison report

### ExperimentComparisonUI

**Constructor:**
```python
ExperimentComparisonUI(manager: ExperimentManager, port: int = 8052)
```

**Methods:**

- `run(debug: bool, host: str)` - Start the web server

## Troubleshooting

### Common Issues

**Issue: "Dash not available"**
```bash
pip install dash dash-bootstrap-components plotly
```

**Issue: "MLflow not installed"**
```bash
pip install mlflow
```

**Issue: "W&B authentication failed"**
```bash
wandb login
# Follow the prompts to enter your API key
```

**Issue: Visualizations not generating**
- Check that `auto_visualize=True`
- Ensure you've logged enough metrics (at least 10 steps)
- Check the `plots/` directory in your experiment folder

**Issue: Comparison UI shows no experiments**
- Verify experiments exist in the base directory
- Check that experiments have completed successfully
- Try refreshing the experiments list in the UI

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

tracker = ExperimentTracker("debug_experiment")
# ... will show detailed debug output
```

## Examples

See `examples/experiment_tracking_example.py` for comprehensive examples:

```bash
python examples/experiment_tracking_example.py
```

This demonstrates:
- Basic experiment tracking
- Artifact versioning
- Backend integration
- Experiment comparison
- Automatic visualizations

## Summary

The Neural DSL experiment tracking system provides a complete solution for managing machine learning experiments:

1. ✅ Track metrics, hyperparameters, and artifacts
2. ✅ Automatic versioning with checksums
3. ✅ Auto-generated visualizations
4. ✅ Integration with popular backends
5. ✅ Interactive comparison UI
6. ✅ Persistent local storage
7. ✅ Easy-to-use API

For more information, see:
- `neural/tracking/README.md` - Technical documentation
- `examples/experiment_tracking_example.py` - Code examples
- `tests/tracking/test_experiment_tracker.py` - Usage tests
