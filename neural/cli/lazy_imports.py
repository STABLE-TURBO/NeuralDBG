"""
Lazy imports for Neural CLI.
This module provides lazy loading for heavy dependencies.
"""

import importlib
import logging
import os
import sys
import time
import warnings
from functools import lru_cache

from .cli_aesthetics import print_error

# Configure environment to suppress specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['MPLBACKEND'] = 'Agg'          # Non-interactive matplotlib backend

logger = logging.getLogger(__name__)

_module_cache = {}

class LazyLoader:
    """
    Lazily import a module only when it's actually needed.
    Uses aggressive caching for optimal performance.
    """
    __slots__ = ('module_name', 'module', '_cached_attrs', '_import_lock')
    
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        self._cached_attrs = {}
        self._import_lock = False

    def _load_module(self):
        """Load module with caching and error handling."""
        if self.module is not None:
            return self.module
            
        if self._import_lock:
            return self.module
            
        self._import_lock = True
        
        if self.module_name in _module_cache:
            self.module = _module_cache[self.module_name]
            self._import_lock = False
            return self.module
            
        try:
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
                warnings.filterwarnings('ignore', category=FutureWarning)
                self.module = importlib.import_module(self.module_name)
                _module_cache[self.module_name] = self.module
            end_time = time.time()
            logger.debug(f"Lazy-loaded {self.module_name} in {end_time - start_time:.2f} seconds")
        except ImportError as e:
            logger.error(f"Failed to import {self.module_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {self.module_name}: {str(e)}")
            raise
        finally:
            self._import_lock = False
            
        return self.module

    def __getattr__(self, name):
        if name in ('module_name', 'module', '_cached_attrs', '_import_lock', '_load_module'):
            return object.__getattribute__(self, name)
            
        if name in self._cached_attrs:
            return self._cached_attrs[name]
            
        module = self._load_module()
        if module is None:
            raise ImportError(f"Module {self.module_name} not loaded")
            
        try:
            attr = getattr(module, name)
            self._cached_attrs[name] = attr
            return attr
        except AttributeError:
            logger.error(f"Attribute {name} not found in {self.module_name}")
            raise

@lru_cache(maxsize=128)
def lazy_import(module_name):
    """Create a lazy loader for a module with caching."""
    return LazyLoader(module_name)

# Note: Lazy imports have been simplified in the CLI refactor.
# The CLI now uses direct imports for better startup time and simpler code.
# This module is kept for backward compatibility with any code that might reference it.

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
