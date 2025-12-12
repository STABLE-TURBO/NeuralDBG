"""
Neural CLI module.
This module provides the command-line interface for Neural.
Uses lazy imports for optimal startup performance.
"""

from .lazy_imports import lazy_import

_cli_module = lazy_import('neural.cli.cli')

def __getattr__(name):
    """Lazy load CLI components on demand."""
    if name == 'cli':
        return _cli_module.cli
    elif name == 'visualize':
        return _cli_module.visualize
    elif name == 'create_parser':
        parser_module = lazy_import('neural.parser.parser')
        return parser_module.create_parser
    elif name == 'ModelTransformer':
        parser_module = lazy_import('neural.parser.parser')
        return parser_module.ModelTransformer
    elif name == 'ShapePropagator':
        shape_module = lazy_import('neural.shape_propagation.shape_propagator')
        return shape_module.ShapePropagator
    raise AttributeError(f"module 'neural.cli' has no attribute '{name}'")

__all__ = ['cli', 'visualize', 'create_parser', 'ModelTransformer', 'ShapePropagator']
