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
    print_warning("Could not determine Neural CLI version. Using fallback version 0.0.0. Try reinstalling with 'pip install neural --force-reinstall'.")
    logger.warning("Package metadata not found for 'neural'.")