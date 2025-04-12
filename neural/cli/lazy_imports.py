"""
Lazy imports for Neural CLI.
This module provides lazy loading for heavy dependencies.
"""

import importlib
import sys
import os
import time
import logging
import warnings
from .cli_aesthetics import print_error

# Configure environment to suppress specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['MPLBACKEND'] = 'Agg'          # Non-interactive matplotlib backend

logger = logging.getLogger(__name__)

class LazyLoader:
    """
    Lazily import a module only when it's actually needed.
    """
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        self._cached_attrs = {}

    def __getattr__(self, name):
        if name in self._cached_attrs:
            return self._cached_attrs[name]
        if self.module is None:
            try:
                start_time = time.time()
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning)
                    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
                    self.module = importlib.import_module(self.module_name)
                end_time = time.time()
                logger.debug(f"Lazy-loaded {self.module_name} in {end_time - start_time:.2f} seconds")
            except ImportError as e:
                print_error(f"Failed to import {self.module_name}: {str(e)}")
                raise
            except Exception as e:
                print_error(f"Unexpected error loading {self.module_name}: {str(e)}")
                raise
        try:
            attr = getattr(self.module, name)
            self._cached_attrs[name] = attr
            return attr
        except AttributeError:
            print_error(f"Attribute {name} not found in {self.module_name}")
            raise

def lazy_import(module_name):
    """Create a lazy loader for a module."""
    return LazyLoader(module_name)

# Lazy loaders for heavy dependencies
tensorflow = lazy_import('tensorflow')
torch = lazy_import('torch')
jax = lazy_import('jax')
matplotlib = lazy_import('matplotlib')
plotly = lazy_import('plotly')
dash = lazy_import('dash')
optuna = lazy_import('optuna')

# Lazy loaders for Neural modules
shape_propagator = lazy_import('neural.shape_propagation.shape_propagator')
tensor_flow = lazy_import('neural.dashboard.tensor_flow')
hpo = lazy_import('neural.hpo.hpo')
code_generator = lazy_import('neural.code_generation.code_generator')

def get_module(lazy_loader):
    """Get the actual module from a lazy loader."""
    if isinstance(lazy_loader, LazyLoader):
        if lazy_loader.module is None:
            try:
                start_time = time.time()
                lazy_loader.module = importlib.import_module(lazy_loader.module_name)
                end_time = time.time()
                logger.debug(f"Lazy-loaded {lazy_loader.module_name} in {end_time - start_time:.2f} seconds")
            except ImportError as e:
                print_error(f"Failed to import {lazy_loader.module_name}: {str(e)}")
                raise
        return lazy_loader.module
    return lazy_loader