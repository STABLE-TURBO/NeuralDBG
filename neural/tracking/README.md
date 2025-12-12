# Neural Experiment Tracking

Comprehensive experiment tracking system for Neural DSL with artifact versioning, automatic visualizations, and backend integrations.

## Features

### 1. **Experiment Tracking**
- Track hyperparameters, metrics, and artifacts
- Automatic metric logging and persistence
- Support for multiple experiment runs
- Status tracking (created, running, completed, failed)

### 2. **Artifact Versioning**
- Automatic versioning of all logged artifacts
- SHA-256 checksums for artifact integrity
- Version history tracking
- Easy retrieval of specific artifact versions

### 3. **Automatic Metric Visualization**
- Auto-generated plots every N steps
- Individual and combined metric plots
- Metric correlation matrices
- Distribution histograms
- Configurable via `auto_visualize` parameter

### 4. **Backend Integrations**
- **MLflow**: Full integration with MLflow tracking
- **Weights & Biases**: Native W&B support
- **TensorBoard**: TensorBoard logging support
- Easy switching between backends

### 5. **Experiment Comparison UI**
- Interactive web interface for comparing experiments
- Side-by-side metric comparisons
- Hyperparameter comparison tables
- Best metrics visualization
- Export comparison reports

## Quick Start

### Basic Usage

```python
from neural.tracking import ExperimentTracker

# Create a tracker
tracker = ExperimentTracker(
    experiment_name="my_experiment",
    auto_visualize=True  # Enable automatic visualizations
)

# Log hyperparameters
tracker.log_hyperparameters({
    'learning_rate': 0.01,
    'batch_size': 32,
    'optimizer': 'adam'
})

# Log metrics during training
for epoch in range(100):
    # ... training code ...
    tracker.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'accuracy': accuracy
    }, step=epoch)

# Log artifacts
tracker.log_artifact("model.h5")
tracker.log_figure(matplotlib_figure, "training_curve.png")

# Finish experiment
tracker.finish()
```

### Artifact Versioning

```python
tracker = ExperimentTracker(experiment_name="versioned_experiment")

# Log multiple versions of the same artifact
for checkpoint in range(5):
    # ... training ...
    tracker.log_model("checkpoint.pt", framework="pytorch", version=True)

# List all versions
versions = tracker.list_artifact_versions("checkpoint.pt")
for v in versions:
    print(f"Version {v.version}: {v.checksum}")

# Get specific version
latest = tracker.get_artifact_version("checkpoint.pt", version=-1)
v2 = tracker.get_artifact_version("checkpoint.pt", version=2)
```

### Backend Integration

#### MLflow

```python
tracker = ExperimentTracker(
    experiment_name="mlflow_experiment",
    backend="mlflow"
)

# All metrics/artifacts are automatically logged to MLflow
tracker.log_hyperparameters({'lr': 0.01})
tracker.log_metrics({'loss': 0.5}, step=1)
```

#### Weights & Biases

```python
tracker = ExperimentTracker(
    experiment_name="wandb_experiment",
    backend="wandb"
)

# Automatically synced with W&B
tracker.log_hyperparameters({'lr': 0.01})
tracker.log_metrics({'loss': 0.5}, step=1)
```

#### TensorBoard

```python
tracker = ExperimentTracker(
    experiment_name="tensorboard_experiment",
    backend="tensorboard"
)

# Logged to TensorBoard
tracker.log_metrics({'loss': 0.5}, step=1)
```

### Experiment Comparison

#### Using ExperimentManager

```python
from neural.tracking import ExperimentManager

manager = ExperimentManager()

# Create multiple experiments
exp1 = manager.create_experiment("experiment_1")
exp2 = manager.create_experiment("experiment_2")

# ... run experiments ...

# Compare experiments
plots = manager.compare_experiments([exp1.experiment_id, exp2.experiment_id])

# Export comparison
manager.export_comparison([exp1.experiment_id, exp2.experiment_id])
```

#### Using Comparison UI

Launch the interactive comparison UI:

```bash
# Command line
python neural/tracking/comparison_ui.py --port 8052

# Or in Python
from neural.tracking import launch_comparison_ui
launch_comparison_ui(port=8052)
```

The UI provides:
- 📊 **Metrics Comparison**: Interactive plots comparing metrics across experiments
- ⚙️ **Hyperparameters**: Side-by-side hyperparameter comparison
- 📈 **Best Metrics**: Bar charts showing best achieved values
- 📋 **Summary**: Overview cards for each experiment

### Advanced Features

#### Custom Visualizations

```python
from neural.tracking import MetricVisualizer

# Create custom visualizations
plots = MetricVisualizer.create_metric_plots(
    metrics_history,
    figsize=(15, 10)
)

dist_plots = MetricVisualizer.create_distribution_plots(
    metrics_history
)
```

#### Manual Visualization Generation

```python
tracker = ExperimentTracker(
    experiment_name="manual_viz",
    auto_visualize=False  # Disable automatic visualization
)

# ... training ...

# Generate all visualizations manually
saved_plots = tracker.generate_visualizations()
print(f"Generated plots: {list(saved_plots.keys())}")
```

#### Experiment Queries

```python
manager = ExperimentManager()

# List all experiments
experiments = manager.list_experiments()
for exp in experiments:
    print(f"{exp['experiment_name']} - {exp['status']}")

# Get specific experiment
tracker = manager.get_experiment(experiment_id)

# Get best metric
best_acc, step = tracker.get_best_metric('accuracy', mode='max')
print(f"Best accuracy: {best_acc} at step {step}")

# Get all metrics for a specific metric name
losses = tracker.get_metrics('train_loss')
```

## Architecture

```
neural/tracking/
├── experiment_tracker.py       # Main tracking classes
│   ├── ExperimentTracker      # Individual experiment tracking
│   ├── ExperimentManager      # Manage multiple experiments
│   ├── ExperimentComparisonUI # Web UI for comparison
│   ├── ArtifactVersion        # Versioned artifact representation
│   └── MetricVisualizer       # Automatic visualization generation
├── integrations.py            # Backend integrations
│   ├── MLflowIntegration
│   ├── WandbIntegration
│   └── TensorBoardIntegration
├── comparison_ui.py           # Standalone UI launcher
└── README.md                  # This file
```

## Storage Structure

```
neural_experiments/
└── experiment_name_12345678/
    ├── metadata.json          # Experiment metadata
    ├── hyperparameters.json   # Logged hyperparameters
    ├── metrics.json           # All logged metrics
    ├── artifacts.json         # Artifact metadata
    ├── summary.json           # Experiment summary
    ├── artifacts/             # Current artifacts
    ├── plots/                 # Generated visualizations
    │   ├── auto_*.png        # Auto-generated plots
    │   ├── all_metrics_*.png
    │   └── metric_*.png
    └── versions/              # Versioned artifacts
        └── model.pt/
            ├── v1_model.pt
            ├── v2_model.pt
            └── v3_model.pt
```

## API Reference

### ExperimentTracker

```python
tracker = ExperimentTracker(
    experiment_name: str = None,
    base_dir: str = "neural_experiments",
    auto_visualize: bool = True,
    backend: Optional[str] = None  # 'mlflow', 'wandb', 'tensorboard'
)
```

**Methods:**
- `log_hyperparameters(hyperparameters: Dict)`: Log hyperparameters
- `log_metrics(metrics: Dict, step: int)`: Log metrics at a step
- `log_artifact(path: str, name: str, version: bool)`: Log an artifact
- `log_model(path: str, framework: str, version: bool)`: Log a model
- `log_figure(figure: plt.Figure, name: str)`: Log a matplotlib figure
- `get_metrics(metric_name: str)`: Get metric history
- `get_best_metric(metric_name: str, mode: str)`: Get best metric value
- `get_artifact_version(name: str, version: int)`: Get artifact version
- `list_artifact_versions(name: str)`: List all versions
- `generate_visualizations()`: Generate all visualizations
- `finish()`: Complete the experiment

### ExperimentManager

```python
manager = ExperimentManager(base_dir: str = "neural_experiments")
```

**Methods:**
- `create_experiment(name: str, auto_visualize: bool, backend: str)`: Create new experiment
- `get_experiment(experiment_id: str)`: Load existing experiment
- `list_experiments()`: List all experiments
- `delete_experiment(experiment_id: str)`: Delete an experiment
- `compare_experiments(ids: List[str], metrics: List[str])`: Compare experiments
- `export_comparison(ids: List[str], output_dir: str)`: Export comparison

### ExperimentComparisonUI

```python
ui = ExperimentComparisonUI(manager: ExperimentManager, port: int = 8052)
ui.run(debug: bool = False, host: str = "127.0.0.1")
```

## Examples

See `examples/experiment_tracking_example.py` for comprehensive examples:

```bash
python examples/experiment_tracking_example.py
```

This runs demonstrations of:
1. Basic experiment tracking
2. Artifact versioning
3. Backend integration
4. Experiment comparison
5. Automatic visualizations

## Requirements

**Core:**
- numpy
- matplotlib

**UI (optional):**
- dash
- dash-bootstrap-components
- plotly

**Backends (optional):**
- mlflow (for MLflow integration)
- wandb (for Weights & Biases)
- tensorboard (for TensorBoard)

Install with:
```bash
pip install -e ".[full]"  # All features
```

## Tips

1. **Auto-visualization frequency**: Set frequency by checking step count in `_auto_visualize_metrics`
2. **Storage space**: Versioning creates copies; clean old versions periodically
3. **Backend choice**: 
   - MLflow: Best for local/team tracking
   - W&B: Best for cloud/collaboration
   - TensorBoard: Best for TensorFlow workflows
4. **Comparison UI**: Works best with 2-5 experiments at a time
5. **Metrics naming**: Use consistent names across experiments for better comparison

## Troubleshooting

**Issue: Dash not installed**
```bash
pip install dash dash-bootstrap-components plotly
```

**Issue: Backend integration fails**
```bash
# For MLflow
pip install mlflow

# For Weights & Biases  
pip install wandb
wandb login

# For TensorBoard
pip install tensorboard
```

**Issue: Comparison UI shows no experiments**
- Check `base_dir` contains experiment folders
- Verify experiments have metadata.json files
- Try refreshing the experiments list in UI
