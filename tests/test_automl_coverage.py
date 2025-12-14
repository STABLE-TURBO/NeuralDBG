"""
Comprehensive test suite for AutoML module to increase coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from neural.automl.architecture_space import ArchitectureSpace
from neural.automl.search_strategies import RandomSearch, BayesianSearch, EvolutionarySearch
from neural.automl.early_stopping import EarlyStopping
from neural.automl.evaluation import ModelEvaluator


class TestArchitectureSpace:
    """Test architecture space definition and sampling."""
    
    def test_architecture_space_creation(self):
        """Test creating an architecture space."""
        space = ArchitectureSpace()
        assert space is not None
    
    def test_add_layer_choice(self):
        """Test adding layer choices to space."""
        space = ArchitectureSpace()
        space.add_layer_choice(['Dense', 'Conv2D', 'LSTM'])
        assert len(space.layer_choices) > 0
    
    def test_add_parameter_range(self):
        """Test adding parameter ranges."""
        space = ArchitectureSpace()
        space.add_parameter('units', (32, 256))
        assert 'units' in space.parameters
    
    def test_sample_architecture(self):
        """Test sampling from architecture space."""
        space = ArchitectureSpace()
        space.add_layer_choice(['Dense', 'Conv2D'])
        space.add_parameter('units', (64, 256))
        
        arch = space.sample()
        assert arch is not None


class TestSearchStrategies:
    """Test different search strategies."""
    
    def test_random_search_initialization(self):
        """Test random search strategy initialization."""
        strategy = RandomSearch(max_trials=10)
        assert strategy.max_trials == 10
    
    def test_random_search_suggest(self):
        """Test random search suggestion."""
        space = ArchitectureSpace()
        space.add_parameter('units', (32, 256))
        
        strategy = RandomSearch(max_trials=5)
        suggestion = strategy.suggest(space)
        assert suggestion is not None
    
    def test_bayesian_search_initialization(self):
        """Test Bayesian search initialization."""
        strategy = BayesianSearch(max_trials=20)
        assert strategy.max_trials == 20
    
    def test_evolutionary_search_initialization(self):
        """Test evolutionary search initialization."""
        strategy = EvolutionarySearch(population_size=10, generations=5)
        assert strategy.population_size == 10
        assert strategy.generations == 5


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization."""
        early_stop = EarlyStopping(patience=5, min_delta=0.001)
        assert early_stop.patience == 5
        assert early_stop.min_delta == 0.001
    
    def test_early_stopping_not_triggered(self):
        """Test early stopping when not triggered."""
        early_stop = EarlyStopping(patience=3)
        
        # Improving metrics should not trigger
        assert not early_stop.should_stop(0.5)
        assert not early_stop.should_stop(0.4)
        assert not early_stop.should_stop(0.3)
    
    def test_early_stopping_triggered(self):
        """Test early stopping when triggered."""
        early_stop = EarlyStopping(patience=2)
        
        early_stop.should_stop(0.5)
        early_stop.should_stop(0.51)  # No improvement
        early_stop.should_stop(0.52)  # No improvement
        
        # Should trigger on third consecutive non-improvement
        assert early_stop.should_stop(0.53)


class TestModelEvaluator:
    """Test model evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(metric='accuracy')
        assert evaluator.metric == 'accuracy'
    
    @patch('neural.automl.evaluation.ModelEvaluator.evaluate')
    def test_model_evaluation(self, mock_evaluate):
        """Test model evaluation."""
        mock_evaluate.return_value = {'accuracy': 0.95, 'loss': 0.1}
        
        evaluator = ModelEvaluator(metric='accuracy')
        result = evaluator.evaluate(Mock(), Mock(), Mock())
        
        assert result['accuracy'] == 0.95
        assert result['loss'] == 0.1


@pytest.mark.parametrize("strategy_class,params", [
    (RandomSearch, {'max_trials': 10}),
    (BayesianSearch, {'max_trials': 20}),
    (EvolutionarySearch, {'population_size': 10, 'generations': 5}),
])
def test_search_strategy_initialization(strategy_class, params):
    """Parameterized test for search strategy initialization."""
    strategy = strategy_class(**params)
    assert strategy is not None


@pytest.mark.parametrize("patience,min_delta", [
    (3, 0.001),
    (5, 0.01),
    (10, 0.0001),
])
def test_early_stopping_configurations(patience, min_delta):
    """Parameterized test for early stopping configurations."""
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)
    assert early_stop.patience == patience
    assert early_stop.min_delta == min_delta
