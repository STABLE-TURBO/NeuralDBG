"""
Version information for Neural CLI.
"""

import importlib.metadata
import logging
from .cli_aesthetics import print_warning

logger = logging.getLogger(__name__)

# Get version from package metadata
try:
    __version__ = importlib.metadata.version("neural")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
    # Avoid printing during module import to prevent encoding issues in some shells.
    # Log a concise warning instead; CLI can display version with a note when requested.
    logger.warning("neural package metadata not found; using fallback version 0.0.0")
