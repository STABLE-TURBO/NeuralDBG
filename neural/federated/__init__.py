"""
Neural Federated Learning - Privacy-preserving distributed training.

.. deprecated:: 0.3.0
    The federated learning module will be extracted to a separate repository in v0.4.0.
    This feature is too specialized for the core DSL project.
    See docs/DEPRECATIONS.md for migration guide.
"""

import warnings

warnings.warn(
    "neural.federated is deprecated and will be extracted to a separate repository in v0.4.0. "
    "This specialized feature deserves its own focused project. "
    "See docs/DEPRECATIONS.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

from neural.federated.client import FederatedClient
from neural.federated.server import FederatedServer
from neural.federated.aggregation import (
    FedAvg,
    FedProx,
    FedAdam,
    FedYogi,
    SecureAggregator,
)
from neural.federated.privacy import (
    DifferentialPrivacy,
    GaussianDP,
    LaplacianDP,
    PrivacyAccountant,
)
from neural.federated.communication import (
    CompressionStrategy,
    QuantizationCompressor,
    SparsificationCompressor,
    AdaptiveCompressor,
)
from neural.federated.scenarios import (
    CrossDeviceScenario,
    CrossSiloScenario,
    HybridScenario,
)
from neural.federated.orchestrator import FederatedOrchestrator

__all__ = [
    'FederatedClient',
    'FederatedServer',
    'FedAvg',
    'FedProx',
    'FedAdam',
    'FedYogi',
    'SecureAggregator',
    'DifferentialPrivacy',
    'GaussianDP',
    'LaplacianDP',
    'PrivacyAccountant',
    'CompressionStrategy',
    'QuantizationCompressor',
    'SparsificationCompressor',
    'AdaptiveCompressor',
    'CrossDeviceScenario',
    'CrossSiloScenario',
    'HybridScenario',
    'FederatedOrchestrator',
]
