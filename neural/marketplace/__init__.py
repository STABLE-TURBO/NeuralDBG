"""
Neural Marketplace - Model upload/download, semantic search, and HuggingFace Hub integration.

.. deprecated:: 0.3.0
    The marketplace module will be removed in v0.4.0.
    Use HuggingFace Hub directly for model sharing and discovery.
    See docs/DEPRECATIONS.md for migration guide.
"""

import warnings

warnings.warn(
    "neural.marketplace is deprecated and will be removed in v0.4.0. "
    "Use HuggingFace Hub directly for model sharing and discovery. "
    "See docs/DEPRECATIONS.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

from .api import MarketplaceAPI
from .community_features import CommunityFeatures
from .discord_bot import DiscordWebhook, DiscordCommunityManager
from .education import EducationalResources, UniversityLicenseManager
from .huggingface_integration import HuggingFaceIntegration
from .registry import ModelRegistry
from .search import SemanticSearch


__all__ = [
    'ModelRegistry',
    'SemanticSearch',
    'MarketplaceAPI',
    'HuggingFaceIntegration',
    'CommunityFeatures',
    'DiscordWebhook',
    'DiscordCommunityManager',
    'EducationalResources',
    'UniversityLicenseManager',
]
