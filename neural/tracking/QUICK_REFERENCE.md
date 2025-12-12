# Neural Experiment Tracking - Quick Reference

## Installation

```bash
# Basic
pip install -e .

# With UI and backends
pip install -e ".[full]"
```

## Basic Usage

```python
from neural.tracking import ExperimentTracker

# Create tracker
tracker = ExperimentTracker("my_experiment", auto_visualize=True)

# Log hyperparameters
tracker.log_hyperparameters({'lr': 0.01, 'batch_size': 32})

# Log metrics (in training loop)
tracker.log_metrics({'loss': loss, 'accuracy': acc}, step=epoch)

# Log artifacts
tracker.log_model("model.pt", framework="pytorch", version=True)
tracker.log_figure(matplotlib_fig, "plot.png")

# Finish
tracker.finish()
```

## Backend Integration

```python
# MLflow
tracker = ExperimentTracker("exp", backend="mlflow")

# Weights & Biases
tracker = ExperimentTracker("exp", backend="wandb")

# TensorBoard
tracker = ExperimentTracker("exp", backend="tensorboard")
```

## Artifact Versioning

```python
# Log with versioning
tracker.log_model("model.pt", version=True)  # Creates v1, v2, v3...

# List versions
versions = tracker.list_artifact_versions("model.pt")

# Get specific version
latest = tracker.get_artifact_version("model.pt", version=-1)
v2 = tracker.get_artifact_version("model.pt", version=2)
```

## Experiment Management

```python
from neural.tracking import ExperimentManager

manager = ExperimentManager()

# List experiments
experiments = manager.list_experiments()

# Get experiment
tracker = manager.get_experiment("experiment_id")

# Compare experiments
plots = manager.compare_experiments(['id1', 'id2', 'id3'])

# Export comparison
manager.export_comparison(['id1', 'id2'])
```

## Comparison UI

```bash
# CLI
neural experiments --port 8052

# Python script
python neural/tracking/comparison_ui.py

# Python API
from neural.tracking import launch_comparison_ui
launch_comparison_ui()
```

## Querying

```python
# Get metrics
all_metrics = tracker.get_metrics()
losses = tracker.get_metrics('train_loss')

# Get best metric
best_acc, step = tracker.get_best_metric('accuracy', mode='max')
best_loss, step = tracker.get_best_metric('loss', mode='min')
```

## Visualizations

```python
# Manual generation
tracker = ExperimentTracker("exp", auto_visualize=False)
# ... training ...
plots = tracker.generate_visualizations()

# Custom visualizations
from neural.tracking import MetricVisualizer
plots = MetricVisualizer.create_metric_plots(metrics_history)
dist_plots = MetricVisualizer.create_distribution_plots(metrics_history)
```

## Common Patterns

### Simple Training Loop

```python
tracker = ExperimentTracker("training_run")
tracker.log_hyperparameters(config)

for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    tracker.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss
    }, step=epoch)

tracker.finish()
```

### With Checkpointing

```python
tracker = ExperimentTracker("checkpointed_training")

for epoch in range(num_epochs):
    # Training...
    
    if epoch % checkpoint_freq == 0:
        save_checkpoint(f"checkpoint_{epoch}.pt")
        tracker.log_model(
            f"checkpoint_{epoch}.pt",
            framework="pytorch",
            version=True
        )

tracker.finish()
```

### Hyperparameter Search

```python
from neural.tracking import ExperimentManager

manager = ExperimentManager()

for lr in [0.001, 0.01, 0.1]:
    tracker = manager.create_experiment(f"lr_{lr}")
    tracker.log_hyperparameters({'learning_rate': lr})
    
    # Train and log metrics...
    
    tracker.finish()

# Compare all runs
experiment_ids = [t.experiment_id for t in trackers]
plots = manager.compare_experiments(experiment_ids)
```

### With Error Handling

```python
tracker = ExperimentTracker("safe_experiment")

try:
    tracker.log_hyperparameters(config)
    # Training code...
    tracker.set_status("completed")
except Exception as e:
    tracker.set_status("failed")
    raise
finally:
    tracker.finish()
```

## CLI Commands

```bash
# Launch comparison UI
neural experiments

# With options
neural experiments --port 8052 --base-dir ./my_experiments

# Help
neural experiments --help
```

## File Locations

```
neural_experiments/
└── experiment_name_abc12345/
    ├── metadata.json              # Status, timestamps
    ├── hyperparameters.json       # Config
    ├── metrics.json               # All metrics
    ├── artifacts.json             # Artifact metadata
    ├── summary.json               # Summary
    ├── artifacts/                 # Current artifacts
    ├── plots/                     # Visualizations
    └── versions/                  # Versioned artifacts
        └── model.pt/
            ├── v1_model.pt
            └── v2_model.pt
```

## Tips

1. **Use descriptive experiment names**: `resnet50_lr0.01` not `exp1`
2. **Log hyperparameters first**: Before any metrics
3. **Consistent metric names**: Use same names across experiments
4. **Version checkpoints**: Always use `version=True` for models
5. **Call finish()**: Always finish experiments properly
6. **Use auto_visualize=True**: For automatic plots during training
7. **Backend for teams**: Use MLflow/W&B for collaboration

## Troubleshooting

```bash
# Install UI dependencies
pip install dash dash-bootstrap-components plotly

# Install MLflow
pip install mlflow

# Install W&B
pip install wandb
wandb login

# Install TensorBoard
pip install tensorboard
```

## More Info

- Full Guide: `docs/EXPERIMENT_TRACKING_GUIDE.md`
- Technical Docs: `neural/tracking/README.md`
- Examples: `examples/experiment_tracking_example.py`
- Tests: `tests/tracking/test_experiment_tracker.py`
