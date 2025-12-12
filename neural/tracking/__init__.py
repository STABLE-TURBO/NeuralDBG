"""
Experiment tracking for Neural models.
"""

from .experiment_tracker import (
    ExperimentTracker,
    ExperimentManager,
    ExperimentComparisonUI,
    ArtifactVersion,
    MetricVisualizer
)
from .integrations import (
    BaseIntegration,
    MLflowIntegration,
    WandbIntegration,
    TensorBoardIntegration,
    create_integration
)

__all__ = [
    'ExperimentTracker',
    'ExperimentManager',
    'ExperimentComparisonUI',
    'ArtifactVersion',
    'MetricVisualizer',
    'BaseIntegration',
    'MLflowIntegration',
    'WandbIntegration',
    'TensorBoardIntegration',
    'create_integration',
    'launch_comparison_ui'
]


def launch_comparison_ui(base_dir: str = "neural_experiments", 
                        port: int = 8052, 
                        host: str = "127.0.0.1",
                        debug: bool = False):
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
    manager = ExperimentManager(base_dir=base_dir)
    ui = ExperimentComparisonUI(manager=manager, port=port)
    ui.run(debug=debug, host=host)
