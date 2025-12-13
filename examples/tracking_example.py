"""
Example usage of Neural's experiment tracking system with Aquarium dashboard.

This example demonstrates:
1. Creating and tracking experiments
2. Logging hyperparameters, metrics, and artifacts
3. Comparing experiments
4. Exporting to external platforms
5. Using the Aquarium dashboard
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural.tracking import ExperimentManager, ExperimentTracker, launch_aquarium
from neural.tracking.export_manager import ExportManager
from neural.tracking.metrics_visualizer import MetricsVisualizerComponent


def simulate_training(tracker: ExperimentTracker, epochs: int = 50):
    """
    Simulate a training run with metrics logging.

    Args:
        tracker: ExperimentTracker instance
        epochs: Number of epochs to simulate
    """
    print(f"Starting training simulation for {epochs} epochs...")

    for epoch in range(epochs):
        loss = 2.0 * np.exp(-epoch / 10) + np.random.normal(0, 0.1)
        val_loss = 2.2 * np.exp(-epoch / 10) + np.random.normal(0, 0.15)

        accuracy = 1.0 - np.exp(-epoch / 8) + np.random.normal(0, 0.02)
        accuracy = max(0, min(1, accuracy))

        val_accuracy = 1.0 - np.exp(-epoch / 9) + np.random.normal(0, 0.03)
        val_accuracy = max(0, min(1, val_accuracy))

        tracker.log_metrics(
            {
                "loss": loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "val_accuracy": val_accuracy,
            },
            step=epoch,
        )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")

    print("Training complete!")


def create_example_experiments():
    """Create several example experiments with different configurations."""
    manager = ExperimentManager(base_dir="neural_experiments")

    configs = [
        {
            "name": "baseline_model",
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
            "epochs": 50,
        },
        {
            "name": "high_lr_model",
            "learning_rate": 0.01,
            "batch_size": 32,
            "optimizer": "adam",
            "epochs": 50,
        },
        {
            "name": "large_batch_model",
            "learning_rate": 0.001,
            "batch_size": 128,
            "optimizer": "adam",
            "epochs": 50,
        },
        {
            "name": "sgd_model",
            "learning_rate": 0.01,
            "batch_size": 32,
            "optimizer": "sgd",
            "epochs": 50,
        },
    ]

    experiment_ids = []

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Creating experiment: {config['name']}")
        print(f"{'=' * 60}")

        tracker = manager.create_experiment(experiment_name=config["name"])
        experiment_ids.append(tracker.experiment_id)

        tracker.log_hyperparameters(
            {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "optimizer": config["optimizer"],
                "model_architecture": "resnet50",
                "dataset": "imagenet",
            }
        )

        tracker.set_status("running")

        simulate_training(tracker, epochs=config["epochs"])

        fig = tracker.plot_metrics()
        tracker.log_figure(fig, "training_curves.png")
        plt.close(fig)

        tracker.set_status("completed")

        summary_path = tracker.save_experiment_summary()
        print(f"Saved summary to: {summary_path}")

        best_acc, best_step = tracker.get_best_metric("accuracy", mode="max")
        print(f"Best accuracy: {best_acc:.4f} at step {best_step}")

    return experiment_ids


def demonstrate_comparison(experiment_ids):
    """Demonstrate experiment comparison."""
    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPARISON")
    print(f"{'=' * 60}")

    manager = ExperimentManager(base_dir="neural_experiments")

    print(f"\nComparing {len(experiment_ids)} experiments:")
    for exp_id in experiment_ids:
        exp = manager.get_experiment(exp_id)
        if exp:
            print(f"  - {exp.experiment_name} ({exp.experiment_id[:8]})")

    plots = manager.compare_experiments(
        experiment_ids=experiment_ids, metric_names=["accuracy", "loss"]
    )

    print(f"\nGenerated {len(plots)} comparison plots")

    for plot_name, fig in plots.items():
        print(f"  - {plot_name}")
        fig.savefig(f"comparison_{plot_name}.png")
        plt.close(fig)


def demonstrate_visualization(experiment_id):
    """Demonstrate advanced visualization."""
    print(f"\n{'=' * 60}")
    print("ADVANCED VISUALIZATION")
    print(f"{'=' * 60}")

    manager = ExperimentManager(base_dir="neural_experiments")
    exp = manager.get_experiment(experiment_id)

    if not exp:
        print("Experiment not found!")
        return

    visualizer = MetricsVisualizerComponent(exp)

    print(f"\nGenerating visualizations for: {exp.experiment_name}")

    fig = visualizer.create_training_curves()
    fig.write_html("visualization_training_curves.html")
    print("  - Training curves saved to visualization_training_curves.html")

    fig = visualizer.create_smoothed_curves(window_size=5)
    fig.write_html("visualization_smoothed_curves.html")
    print("  - Smoothed curves saved to visualization_smoothed_curves.html")

    fig = visualizer.create_distribution_plot("accuracy")
    fig.write_html("visualization_accuracy_distribution.html")
    print("  - Distribution plot saved to visualization_accuracy_distribution.html")

    fig = visualizer.create_correlation_matrix()
    fig.write_html("visualization_correlation_matrix.html")
    print("  - Correlation matrix saved to visualization_correlation_matrix.html")


def demonstrate_export(experiment_ids):
    """Demonstrate exporting to external platforms."""
    print(f"\n{'=' * 60}")
    print("EXPORT TO EXTERNAL PLATFORMS")
    print(f"{'=' * 60}")

    manager = ExperimentManager(base_dir="neural_experiments")
    exporter = ExportManager(manager)

    print("\nNote: Make sure the external platforms are installed and configured:")
    print("  - MLflow: pip install mlflow")
    print("  - W&B: pip install wandb")
    print("  - TensorBoard: pip install tensorboard")
    print("\nSkipping actual export in this example...")

    print("\nExample usage:")
    print("\n# Export to MLflow")
    print("results = exporter.export_to_mlflow(")
    print("    experiment_ids=['exp123'],")
    print("    tracking_uri='http://localhost:5000'")
    print(")")
    print("\n# Export to Weights & Biases")
    print("results = exporter.export_to_wandb(")
    print("    experiment_ids=['exp123'],")
    print("    project_name='neural-experiments'")
    print(")")
    print("\n# Export to TensorBoard")
    print("results = exporter.export_to_tensorboard(")
    print("    experiment_ids=['exp123'],")
    print("    log_dir='runs/neural'")
    print(")")


def list_experiments():
    """List all experiments."""
    print(f"\n{'=' * 60}")
    print("ALL EXPERIMENTS")
    print(f"{'=' * 60}")

    manager = ExperimentManager(base_dir="neural_experiments")
    experiments = manager.list_experiments()

    print(f"\nFound {len(experiments)} experiments:\n")

    for exp in experiments:
        print(f"Name: {exp['experiment_name']}")
        print(f"ID: {exp['experiment_id'][:8]}")
        print(f"Status: {exp['status']}")
        print(f"Start Time: {exp['start_time']}")

        if exp.get("summary"):
            summary = exp["summary"]
            if summary.get("metrics", {}).get("best"):
                print("Best Metrics:")
                for metric, info in summary["metrics"]["best"].items():
                    print(f"  - {metric}: {info['value']:.4f} (step {info['step']})")

        print("-" * 60)


def main():
    """Main function to run all examples."""
    print("=" * 60)
    print("Neural Experiment Tracking Example")
    print("=" * 60)

    experiment_ids = create_example_experiments()

    list_experiments()

    if experiment_ids:
        demonstrate_comparison(experiment_ids)

        demonstrate_visualization(experiment_ids[0])

        demonstrate_export(experiment_ids)

    print(f"\n{'=' * 60}")
    print("LAUNCH AQUARIUM DASHBOARD")
    print(f"{'=' * 60}")
    print("\nTo view your experiments in the Aquarium dashboard, run:")
    print("\n  python -m neural.tracking.aquarium_app --port 8053")
    print("\nOr in Python:")
    print("\n  from neural.tracking import launch_aquarium")
    print("  launch_aquarium()")
    print("\nThen navigate to: http://127.0.0.1:8053")

    print(f"\n{'=' * 60}")
    print("Example complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
