"""
Neural LLM - Language model integration for Neural DSL.

.. deprecated:: 0.3.0
    neural.neural_llm is deprecated and will be removed in v0.4.0.
    Use neural.ai module instead for AI-powered model generation.
    See docs/DEPRECATIONS.md for migration guide.
"""

import warnings

warnings.warn(
    "neural.neural_llm is deprecated and will be removed in v0.4.0. "
    "Use neural.ai.generate_model() instead. "
    "See docs/DEPRECATIONS.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = []
