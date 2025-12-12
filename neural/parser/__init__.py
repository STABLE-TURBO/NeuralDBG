"""
Neural DSL Parser Module
"""

# Import and export the main parser functions
from .parser import (
    create_parser,
    ModelTransformer,
    NeuralParser,
    network_parser,
    layer_parser,
    research_parser
)

# Import DSLValidationError from parser_utils (refactored location)
from .parser_utils import DSLValidationError

# Export error handling modules
from . import error_handling
from . import validation
from . import learning_rate_schedules

# Export refactored utility modules
from . import layer_processors
from . import layer_handlers
from . import hpo_utils
from . import hpo_network_processor
from . import network_processors
from . import value_extractors
from . import parser_utils

__all__ = [
    'create_parser',
    'ModelTransformer',
    'NeuralParser',
    'DSLValidationError',
    'network_parser',
    'layer_parser',
    'research_parser',
    'error_handling',
    'validation',
    'learning_rate_schedules',
    'layer_processors',
    'layer_handlers',
    'hpo_utils',
    'hpo_network_processor',
    'network_processors',
    'value_extractors',
    'parser_utils'
]
