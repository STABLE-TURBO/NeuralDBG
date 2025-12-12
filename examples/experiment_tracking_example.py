"""
Example demonstrating the improved experiment tracking features.

This example shows:
1. Basic experiment tracking with automatic visualization
2. Artifact versioning
3. MLflow/Weights&Biases integration
4. Experiment comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from neural.tracking import ExperimentTracker, ExperimentManager


def simulate_training(tracker, epochs=50, lr=0.01):
    """Simulate a training loop with metric logging."""
    print(f"Starting training for {epochs} epochs with lr={lr}")
    
    for epoch in range(epochs):
        train_loss = 1.0 * np.exp(-epoch * lr) + np.random.randn() * 0.05
        val_loss = 1.2 * np.exp(-epoch * lr * 0.8) + np.random.randn() * 0.06
        train_acc = 1.0 - np.exp(-epoch * lr * 1.2) + np.random.randn() * 0.03
        val_acc = 1.0 - np.exp(-epoch * lr) + np.random.randn() * 0.04
        
        train_acc = np.clip(train_acc, 0, 1)
        val_acc = np.clip(val_acc, 0, 1)
        
        tracker.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'learning_rate': lr
        }, step=epoch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
    
    print("Training completed!")


def example_basic_tracking():
    """Example 1: Basic experiment tracking with auto-visualization."""
    print("\n" + "="*60)
    print("Example 1: Basic Experiment Tracking")
    print("="*60)
    
    tracker = ExperimentTracker(
        experiment_name="basic_training",
        auto_visualize=True
    )
    
    tracker.log_hyperparameters({
        'learning_rate': 0.01,
        'batch_size': 32,
        'optimizer': 'adam',
        'model_architecture': 'resnet50'
    })
    
    simulate_training(tracker, epochs=50, lr=0.01)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4, 5], [0.9, 0.85, 0.8, 0.75, 0.7])
    plt.title("Sample Confusion Matrix")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    tracker.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)
    
    print(f"\nExperiment ID: {tracker.experiment_id}")
    print(f"Experiment directory: {tracker.experiment_dir}")
    
    tracker.finish()
    
    return tracker.experiment_id


def example_artifact_versioning():
    """Example 2: Artifact versioning."""
    print("\n" + "="*60)
    print("Example 2: Artifact Versioning")
    print("="*60)
    
    tracker = ExperimentTracker(
        experiment_name="versioned_artifacts",
        auto_visualize=False
    )
    
    tracker.log_hyperparameters({
        'learning_rate': 0.001,
        'batch_size': 64
    })
    
    for checkpoint in range(1, 4):
        print(f"\nCheckpoint {checkpoint}")
        
        simulate_training(tracker, epochs=10, lr=0.001)
        
        model_data = f"Model checkpoint {checkpoint} data"
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(model_data)
            model_path = f.name
        
        tracker.log_model(model_path, framework="pytorch", version=True)
        
        import os
        os.remove(model_path)
        
        print(f"Logged model version {checkpoint}")
    
    print(f"\nModel versions:")
    for version in tracker.list_artifact_versions("model.txt"):
        print(f"  Version {version.version}: {version.path}")
    
    tracker.finish()
    
    return tracker.experiment_id


def example_backend_integration():
    """Example 3: Backend integration (MLflow/W&B)."""
    print("\n" + "="*60)
    print("Example 3: Backend Integration")
    print("="*60)
    
    print("Note: This example requires MLflow or W&B to be installed")
    print("Uncomment the backend parameter to enable integration")
    
    tracker = ExperimentTracker(
        experiment_name="backend_integration",
        auto_visualize=True,
    )
    
    tracker.log_hyperparameters({
        'learning_rate': 0.005,
        'batch_size': 128,
        'optimizer': 'sgd'
    })
    
    simulate_training(tracker, epochs=30, lr=0.005)
    
    tracker.finish()
    
    return tracker.experiment_id


def example_experiment_comparison():
    """Example 4: Comparing multiple experiments."""
    print("\n" + "="*60)
    print("Example 4: Experiment Comparison")
    print("="*60)
    
    manager = ExperimentManager()
    
    experiment_ids = []
    
    for i, lr in enumerate([0.001, 0.01, 0.1]):
        print(f"\nRunning experiment {i+1} with lr={lr}")
        
        tracker = manager.create_experiment(
            experiment_name=f"lr_comparison_{lr}",
            auto_visualize=False
        )
        
        tracker.log_hyperparameters({
            'learning_rate': lr,
            'batch_size': 32,
            'optimizer': 'adam'
        })
        
        simulate_training(tracker, epochs=30, lr=lr)
        
        tracker.finish()
        experiment_ids.append(tracker.experiment_id)
    
    print(f"\nComparing {len(experiment_ids)} experiments...")
    
    plots = manager.compare_experiments(experiment_ids)
    
    print(f"Generated {len(plots)} comparison plots")
    for plot_name in plots.keys():
        print(f"  - {plot_name}")
    
    output_dir = manager.export_comparison(experiment_ids)
    print(f"\nComparison exported to: {output_dir}")
    
    return experiment_ids


def example_visualizations():
    """Example 5: Automatic metric visualizations."""
    print("\n" + "="*60)
    print("Example 5: Automatic Visualizations")
    print("="*60)
    
    tracker = ExperimentTracker(
        experiment_name="visualization_demo",
        auto_visualize=True
    )
    
    tracker.log_hyperparameters({
        'learning_rate': 0.01,
        'batch_size': 32
    })
    
    simulate_training(tracker, epochs=100, lr=0.01)
    
    print("\nGenerating comprehensive visualizations...")
    saved_plots = tracker.generate_visualizations()
    
    print(f"\nGenerated {len(saved_plots)} visualizations:")
    for plot_name, plot_path in saved_plots.items():
        print(f"  - {plot_name}: {plot_path}")
    
    tracker.finish()
    
    return tracker.experiment_id


def main():
    """Run all examples."""
    print("=" * 60)
    print("Neural Experiment Tracking Examples")
    print("=" * 60)
    
    exp_id_1 = example_basic_tracking()
    
    exp_id_2 = example_artifact_versioning()
    
    exp_id_3 = example_backend_integration()
    
    comparison_ids = example_experiment_comparison()
    
    exp_id_5 = example_visualizations()
    
    print("\n" + "="*60)
    print("All Examples Complete!")
    print("="*60)
    print("\nExperiment IDs created:")
    print(f"  1. Basic tracking: {exp_id_1}")
    print(f"  2. Artifact versioning: {exp_id_2}")
    print(f"  3. Backend integration: {exp_id_3}")
    print(f"  4. Comparison experiments: {', '.join(comparison_ids)}")
    print(f"  5. Visualizations: {exp_id_5}")
    
    print("\nTo view experiment comparison UI, run:")
    print("  python neural/tracking/comparison_ui.py")
    print("  or")
    print("  from neural.tracking import launch_comparison_ui")
    print("  launch_comparison_ui()")


if __name__ == "__main__":
    main()
