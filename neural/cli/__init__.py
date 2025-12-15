"""
Neural CLI module.
This module provides the command-line interface for Neural.
Uses lazy imports for optimal startup performance.
"""

from .lazy_imports import lazy_import
from .cli import cli, visualize

def __getattr__(name):
    if name == 'create_parser':
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
