"""
Neural Chat - Conversational interface for Neural DSL

.. deprecated:: 0.3.0
    Neural Chat will be removed in v0.4.0 as it doesn't align with the DSL-first approach.
    Use the clear DSL syntax and documentation instead.
    See docs/DEPRECATIONS.md for details.
"""

import warnings

warnings.warn(
    "neural.neural_chat is deprecated and will be removed in v0.4.0. "
    "Use the clear DSL syntax and documentation instead. "
    "See docs/DEPRECATIONS.md for details.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = []
