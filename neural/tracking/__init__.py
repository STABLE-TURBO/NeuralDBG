"""
Experiment tracking for Neural models.
"""

from .experiment_tracker import ExperimentManager, ExperimentTracker
from .integrations import (
    BaseIntegration,
    MLflowIntegration,
    TensorBoardIntegration,
    WandbIntegration,
    create_integration,
)


try:
    from .comparison_component import ComparisonComponent
    from .export_manager import ExportManager
    from .metrics_visualizer import MetricsVisualizerComponent

    _components_available = True
except ImportError:
    _components_available = False


__all__ = [
    "ExperimentTracker",
    "ExperimentManager",
    "BaseIntegration",
    "MLflowIntegration",
    "WandbIntegration",
    "TensorBoardIntegration",
    "create_integration",
    "launch_comparison_ui",
    "launch_aquarium",
]

if _components_available:
    __all__.extend(["ComparisonComponent", "ExportManager", "MetricsVisualizerComponent"])


def launch_comparison_ui(
    base_dir: str = "neural_experiments",
    port: int = 8052,
    host: str = "127.0.0.1",
    debug: bool = False,
):
    """
    Launch the experiment comparison UI.

    Args:
        base_dir: Base directory containing experiments
        port: Port to run the UI on
        host: Host to run the UI on
        debug: Run in debug mode

    Example:
        >>> from neural.tracking import launch_comparison_ui
        >>> launch_comparison_ui(port=8052)
    """
    try:
        from .comparison_ui import main as comparison_main
        import sys

        sys.argv = [
            "comparison_ui.py",
            "--base-dir",
            base_dir,
            "--port",
            str(port),
            "--host",
            host,
        ]
        if debug:
            sys.argv.append("--debug")
        comparison_main()
    except ImportError as e:
        print(f"Failed to launch comparison UI: {e}")
        print("Make sure dash and related dependencies are installed.")


def launch_aquarium(
    base_dir: str = "neural_experiments",
    port: int = 8053,
    host: str = "127.0.0.1",
    debug: bool = False,
):
    """
    Launch the Aquarium experiment tracking dashboard.

    Args:
        base_dir: Base directory containing experiments
        port: Port to run the dashboard on
        host: Host to run the dashboard on
        debug: Run in debug mode

    Example:
        >>> from neural.tracking import launch_aquarium
        >>> launch_aquarium(port=8053)
    """
    try:
        from .aquarium_app import AquariumDashboard

        dashboard = AquariumDashboard(base_dir=base_dir, port=port)
        dashboard.run(debug=debug, host=host)
    except ImportError as e:
        print(f"Failed to launch Aquarium: {e}")
        print("Make sure dash and related dependencies are installed.")
