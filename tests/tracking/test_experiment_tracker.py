"""
Tests for experiment tracking functionality.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import matplotlib.pyplot as plt
from neural.tracking import (
    ExperimentTracker,
    ExperimentManager,
    ArtifactVersion,
    MetricVisualizer
)


@pytest.fixture
def temp_base_dir():
    """Create a temporary directory for experiments."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_experiment_tracker_creation(temp_base_dir):
    """Test basic experiment tracker creation."""
    tracker = ExperimentTracker(
        experiment_name="test_exp",
        base_dir=temp_base_dir,
        auto_visualize=False
    )
    
    assert tracker.experiment_name == "test_exp"
    assert os.path.exists(tracker.experiment_dir)
    assert os.path.exists(os.path.join(tracker.experiment_dir, "artifacts"))
    assert os.path.exists(os.path.join(tracker.experiment_dir, "plots"))
    assert os.path.exists(os.path.join(tracker.experiment_dir, "versions"))


def test_log_hyperparameters(temp_base_dir):
    """Test logging hyperparameters."""
    tracker = ExperimentTracker(base_dir=temp_base_dir, auto_visualize=False)
    
    hyperparams = {
        'learning_rate': 0.01,
        'batch_size': 32,
        'optimizer': 'adam'
    }
    
    tracker.log_hyperparameters(hyperparams)
    
    assert tracker.hyperparameters == hyperparams
    
    hparams_file = os.path.join(tracker.experiment_dir, "hyperparameters.json")
    assert os.path.exists(hparams_file)


def test_log_metrics(temp_base_dir):
    """Test logging metrics."""
    tracker = ExperimentTracker(base_dir=temp_base_dir, auto_visualize=False)
    
    for step in range(5):
        tracker.log_metrics({
            'train_loss': 1.0 - step * 0.1,
            'val_loss': 1.1 - step * 0.09,
            'accuracy': step * 0.2
        }, step=step)
    
    assert len(tracker.metrics_history) == 5
    assert tracker.metrics_history[0]['train_loss'] == 1.0
    assert tracker.metrics_history[4]['accuracy'] == 0.8


def test_get_best_metric(temp_base_dir):
    """Test getting best metric value."""
    tracker = ExperimentTracker(base_dir=temp_base_dir, auto_visualize=False)
    
    for step in range(10):
        tracker.log_metrics({
            'loss': 1.0 - step * 0.1,
            'accuracy': step * 0.1
        }, step=step)
    
    best_loss, step = tracker.get_best_metric('loss', mode='min')
    assert best_loss == 0.1
    assert step == 9
    
    best_acc, step = tracker.get_best_metric('accuracy', mode='max')
    assert best_acc == 0.9
    assert step == 9


def test_artifact_versioning(temp_base_dir):
    """Test artifact versioning functionality."""
    tracker = ExperimentTracker(base_dir=temp_base_dir, auto_visualize=False)
    
    temp_artifacts = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"Artifact version {i+1}")
            temp_artifacts.append(f.name)
        
        tracker.log_artifact(temp_artifacts[i], artifact_name="model.txt", version=True)
    
    versions = tracker.list_artifact_versions("model.txt")
    assert len(versions) == 3
    
    latest = tracker.get_artifact_version("model.txt", version=-1)
    assert latest.version == 3
    
    v1 = tracker.get_artifact_version("model.txt", version=1)
    assert v1.version == 1
    
    for path in temp_artifacts:
        os.remove(path)


def test_log_figure(temp_base_dir):
    """Test logging matplotlib figures."""
    tracker = ExperimentTracker(base_dir=temp_base_dir, auto_visualize=False)
    
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    
    tracker.log_figure(fig, "test_plot.png")
    plt.close(fig)
    
    plot_path = os.path.join(tracker.experiment_dir, "plots", "test_plot.png")
    assert os.path.exists(plot_path)
    assert "test_plot.png" in tracker.artifacts


def test_experiment_manager(temp_base_dir):
    """Test experiment manager functionality."""
    manager = ExperimentManager(base_dir=temp_base_dir)
    
    exp1 = manager.create_experiment("exp1", auto_visualize=False)
    exp2 = manager.create_experiment("exp2", auto_visualize=False)
    
    exp1.log_hyperparameters({'lr': 0.01})
    exp2.log_hyperparameters({'lr': 0.1})
    
    exp1.finish()
    exp2.finish()
    
    experiments = manager.list_experiments()
    assert len(experiments) == 2
    
    loaded_exp = manager.get_experiment(exp1.experiment_id)
    assert loaded_exp is not None
    assert loaded_exp.hyperparameters == {'lr': 0.01}


def test_experiment_comparison(temp_base_dir):
    """Test experiment comparison functionality."""
    manager = ExperimentManager(base_dir=temp_base_dir)
    
    exp_ids = []
    for i in range(2):
        exp = manager.create_experiment(f"exp_{i}", auto_visualize=False)
        exp.log_hyperparameters({'lr': 0.01 * (i + 1)})
        
        for step in range(10):
            exp.log_metrics({
                'loss': 1.0 - step * 0.1 * (i + 1),
                'accuracy': step * 0.1
            }, step=step)
        
        exp.finish()
        exp_ids.append(exp.experiment_id)
    
    plots = manager.compare_experiments(exp_ids)
    
    assert len(plots) > 0
    assert 'comparison_loss' in plots or 'comparison_accuracy' in plots


def test_metric_visualizer():
    """Test metric visualizer."""
    metrics_history = [
        {'step': i, 'loss': 1.0 - i * 0.1, 'accuracy': i * 0.1}
        for i in range(10)
    ]
    
    plots = MetricVisualizer.create_metric_plots(metrics_history)
    assert len(plots) > 0
    
    for fig in plots.values():
        plt.close(fig)
    
    dist_plots = MetricVisualizer.create_distribution_plots(metrics_history)
    assert len(dist_plots) > 0
    
    for fig in dist_plots.values():
        plt.close(fig)


def test_automatic_visualization(temp_base_dir):
    """Test automatic visualization generation."""
    tracker = ExperimentTracker(
        base_dir=temp_base_dir,
        auto_visualize=True
    )
    
    for step in range(20):
        tracker.log_metrics({
            'loss': 1.0 - step * 0.05,
            'accuracy': step * 0.05
        }, step=step)
    
    plots_dir = os.path.join(tracker.experiment_dir, "plots")
    auto_plots = [f for f in os.listdir(plots_dir) if f.startswith('auto_')]
    
    assert len(auto_plots) > 0


def test_experiment_finish(temp_base_dir):
    """Test experiment finish functionality."""
    tracker = ExperimentTracker(base_dir=temp_base_dir, auto_visualize=False)
    
    tracker.log_hyperparameters({'lr': 0.01})
    
    for step in range(10):
        tracker.log_metrics({'loss': 1.0 - step * 0.1}, step=step)
    
    tracker.finish()
    
    assert tracker.metadata['status'] == 'completed'
    assert 'end_time' in tracker.metadata
    
    summary_path = os.path.join(tracker.experiment_dir, "summary.json")
    assert os.path.exists(summary_path)


def test_artifact_checksum(temp_base_dir):
    """Test artifact checksum generation."""
    tracker = ExperimentTracker(base_dir=temp_base_dir, auto_visualize=False)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        temp_path = f.name
    
    tracker.log_artifact(temp_path, artifact_name="test.txt", version=True)
    
    version = tracker.get_artifact_version("test.txt", version=1)
    assert version.checksum is not None
    assert len(version.checksum) == 64
    
    os.remove(temp_path)


def test_export_comparison(temp_base_dir):
    """Test exporting experiment comparison."""
    manager = ExperimentManager(base_dir=temp_base_dir)
    
    exp_ids = []
    for i in range(2):
        exp = manager.create_experiment(f"exp_{i}", auto_visualize=False)
        exp.log_hyperparameters({'lr': 0.01})
        
        for step in range(5):
            exp.log_metrics({'loss': 1.0 - step * 0.1}, step=step)
        
        exp.finish()
        exp_ids.append(exp.experiment_id)
    
    output_dir = manager.export_comparison(exp_ids)
    
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "comparison_summary.json"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
