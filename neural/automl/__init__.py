"""
Neural AutoML - Automated Machine Learning and Neural Architecture Search

This module provides comprehensive AutoML capabilities including:
- Grid Search, Random Search, Bayesian Optimization, Evolutionary Algorithms
- Neural Architecture Search (NAS) with architecture space definition DSL
- Parallel trial execution with Ray/Dask support
- Early stopping strategies (median pruning, hyperband, ASHA)
- Automatic hyperparameter tuning
- Integration with existing HPO module

Features
--------
- Architecture space definition via DSL
- Multiple search strategies (Grid, Random, Bayesian, Evolutionary)
- Distributed execution support (Ray, Dask)
- Advanced early stopping mechanisms
- NAS-specific operations (layer composition, skip connections)
- Automatic architecture evaluation and ranking

Examples
--------
>>> from neural.automl import AutoMLEngine, ArchitectureSpace
>>> space = ArchitectureSpace.from_dsl(config_str)
>>> engine = AutoMLEngine(search_strategy='bayesian', executor='ray')
>>> best_arch = engine.search(space, max_trials=100)
"""

from .engine import AutoMLEngine
from .architecture_space import ArchitectureSpace, LayerChoice, ArchitectureBuilder
from .search_strategies import (
    GridSearchStrategy,
    RandomSearchStrategy,
    BayesianSearchStrategy,
    EvolutionarySearchStrategy,
    SearchStrategy
)
from .early_stopping import (
    EarlyStoppingStrategy,
    MedianPruner,
    HyperbandPruner,
    ASHAPruner,
    ThresholdPruner
)
from .executor import (
    BaseExecutor,
    SequentialExecutor,
    RayExecutor,
    DaskExecutor
)
from .nas_operations import (
    NASOperation,
    SkipConnection,
    FactorizedReduce,
    SepConv,
    DilatedConv,
    PoolBN
)
from .evaluation import (
    ArchitectureEvaluator,
    MetricTracker,
    PerformancePredictor
)

# Aliases for convenience
RandomSearch = RandomSearchStrategy
BayesianSearch = BayesianSearchStrategy
EvolutionarySearch = EvolutionarySearchStrategy

__all__ = [
    # Core
    'AutoMLEngine',
    'ArchitectureSpace',
    'LayerChoice',
    'ArchitectureBuilder',
    
    # Search strategies
    'SearchStrategy',
    'GridSearchStrategy',
    'RandomSearchStrategy',
    'BayesianSearchStrategy',
    'EvolutionarySearchStrategy',
    'RandomSearch',
    'BayesianSearch',
    'EvolutionarySearch',
    
    # Early stopping
    'EarlyStoppingStrategy',
    'MedianPruner',
    'HyperbandPruner',
    'ASHAPruner',
    'ThresholdPruner',
    
    # Executors
    'BaseExecutor',
    'SequentialExecutor',
    'RayExecutor',
    'DaskExecutor',
    
    # NAS operations
    'NASOperation',
    'SkipConnection',
    'FactorizedReduce',
    'SepConv',
    'DilatedConv',
    'PoolBN',
    
    # Evaluation
    'ArchitectureEvaluator',
    'MetricTracker',
    'PerformancePredictor',
]

__version__ = '0.1.0'
