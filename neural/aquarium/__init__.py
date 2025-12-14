"""
Neural Aquarium - AI-Powered Neural DSL Builder and IDE

Web-based interface for creating Neural DSL models through natural language conversation.
Includes integrated debugger with NeuralDbg dashboard embedding, settings panel, and model
compilation & execution runner.

.. deprecated:: 0.3.0
    The Aquarium IDE will be extracted to a separate repository in v0.4.0.
    Use VS Code, PyCharm, or other editors with Neural DSL language server support instead.
    See docs/DEPRECATIONS.md for migration guide.
"""

import warnings

warnings.warn(
    "neural.aquarium is deprecated and will be extracted to a separate repository in v0.4.0. "
    "Use VS Code, PyCharm, or other editors with Neural DSL language server support instead. "
    "See docs/DEPRECATIONS.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

__version__ = "0.3.0"
__author__ = "Neural DSL Team"

from pathlib import Path

AQUARIUM_ROOT = Path(__file__).parent
FRONTEND_ROOT = AQUARIUM_ROOT / "src"
BACKEND_ROOT = AQUARIUM_ROOT / "backend"

from .aquarium_ide import AquariumIDE

# Also expose standalone runner for backward compatibility
try:
    from .aquarium import main as run_aquarium
except ImportError:
    run_aquarium = None

__all__ = [
    "__version__",
    "__author__",
    "AQUARIUM_ROOT",
    "FRONTEND_ROOT",
    "BACKEND_ROOT",
    "AquariumIDE",
    "run_aquarium",
]
