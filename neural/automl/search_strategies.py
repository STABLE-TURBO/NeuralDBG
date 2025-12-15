"""
Search strategies for AutoML and NAS.

Implements various search strategies including Grid, Random, Bayesian, and Evolutionary approaches.
"""
from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SearchStrategy(ABC):
    """Base class for search strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def suggest(self, architecture_space, trial_number: int) -> Dict[str, Any]:
        """Suggest the next architecture to evaluate."""
        pass
    
    def update(self, architecture: Dict[str, Any], metrics: Dict[str, float]):
        """Update strategy with evaluation results."""
        self.history.append({
            'architecture': architecture,
            'metrics': metrics,
            'trial_number': len(self.history)
        })
    
    def get_best_architecture(self) -> Optional[Dict[str, Any]]:
        """Get the best architecture found so far."""
        if not self.history:
            return None
        
        best = max(self.history, key=lambda x: x['metrics'].get('accuracy', 0))
        return best['architecture']


class GridSearchStrategy(SearchStrategy):
    """Grid search over the architecture space."""
    
    def __init__(self):
        super().__init__('grid_search')
        self.grid_points: List[List[int]] = []
        self.current_idx = 0
    
    def suggest(self, architecture_space, trial_number: int) -> Dict[str, Any]:
        """Suggest next architecture using grid search."""
        if not self.grid_points:
            self.grid_points = self._generate_grid(architecture_space)
        
        if self.current_idx >= len(self.grid_points):
            self.current_idx = 0
        
        choices = self.grid_points[self.current_idx]
        self.current_idx += 1
        
        return architecture_space.sample_architecture(choices)
    
    def _generate_grid(self, architecture_space) -> List[List[int]]:
        """Generate grid points for the architecture space."""
        num_choices = [lc.num_choices for lc in architecture_space.layer_choices]
        
        if not num_choices:
            return [[]]
        
        grid = []
        self._recursive_grid(num_choices, [], grid)
        return grid
    
    def _recursive_grid(self, num_choices: List[int], current: List[int], grid: List[List[int]]):
        """Recursively generate grid combinations."""
        if len(current) == len(num_choices):
            grid.append(current.copy())
            return
        
        for i in range(num_choices[len(current)]):
            current.append(i)
            self._recursive_grid(num_choices, current, grid)
            current.pop()


class RandomSearchStrategy(SearchStrategy):
    """Random search over the architecture space."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__('random_search')
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def suggest(self, architecture_space, trial_number: int) -> Dict[str, Any]:
        """Suggest next architecture using random search."""
        return architecture_space.sample_architecture()


class BayesianSearchStrategy(SearchStrategy):
    """Bayesian optimization for architecture search."""
    
    def __init__(self, acquisition_function: str = 'ei', n_initial_random: int = 10):
        super().__init__('bayesian_search')
        self.acquisition_function = acquisition_function
        self.n_initial_random = n_initial_random
        self.surrogate_model = None
    
    def suggest(self, architecture_space, trial_number: int) -> Dict[str, Any]:
        """Suggest next architecture using Bayesian optimization."""
        if trial_number < self.n_initial_random or not self.history:
            return architecture_space.sample_architecture()
        
        try:
            return self._bayesian_suggest(architecture_space)
        except Exception as e:
            logger.warning(f"Bayesian suggestion failed: {e}, falling back to random")
            return architecture_space.sample_architecture()
    
    def _bayesian_suggest(self, architecture_space) -> Dict[str, Any]:
        """Generate suggestion using Bayesian optimization."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            logger.warning("scikit-learn not available, falling back to random search")
            return architecture_space.sample_architecture()
        
        X_train, y_train = self._prepare_training_data()
        
        if self.surrogate_model is None:
            kernel = Matern(nu=2.5)
            self.surrogate_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
        
        self.surrogate_model.fit(X_train, y_train)
        
        best_choice = self._optimize_acquisition(architecture_space)
        return architecture_space.sample_architecture(best_choice)
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for surrogate model."""
        X = []
        y = []
        
        for entry in self.history:
            arch = entry['architecture']
            metric = entry['metrics'].get('accuracy', 0)
            
            features = self._architecture_to_features(arch)
            X.append(features)
            y.append(metric)
        
        return np.array(X), np.array(y)
    
    def _architecture_to_features(self, architecture: Dict[str, Any]) -> List[float]:
        """Convert architecture to feature vector."""
        features = []
        
        for layer in architecture.get('layers', []):
            features.append(hash(layer['type']) % 1000 / 1000.0)
            
            if layer.get('params'):
                for param_val in layer['params'].values():
                    if isinstance(param_val, (int, float)):
                        features.append(float(param_val))
                    else:
                        features.append(hash(str(param_val)) % 1000 / 1000.0)
        
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _optimize_acquisition(self, architecture_space) -> List[int]:
        """Optimize acquisition function to find next point."""
        num_candidates = min(100, architecture_space.get_search_space_size())
        
        best_score = float('-inf')
        best_choice = None
        
        for _ in range(num_candidates):
            choices = [
                np.random.randint(lc.num_choices)
                for lc in architecture_space.layer_choices
            ]
            
            arch = architecture_space.sample_architecture(choices)
            features = np.array([self._architecture_to_features(arch)])
            
            mean, std = self.surrogate_model.predict(features, return_std=True)
            
            if self.acquisition_function == 'ei':
                score = self._expected_improvement(mean[0], std[0])
            elif self.acquisition_function == 'ucb':
                score = self._upper_confidence_bound(mean[0], std[0])
            else:
                score = mean[0]
            
            if score > best_score:
                best_score = score
                best_choice = choices
        
        return best_choice
    
    def _expected_improvement(self, mean: float, std: float) -> float:
        """Calculate expected improvement."""
        if not self.history:
            return 0.0
        
        best_y = max(entry['metrics'].get('accuracy', 0) for entry in self.history)
        
        if std < 1e-8:
            return 0.0
        
        z = (mean - best_y) / std
        
        from scipy.stats import norm
        ei = std * (z * norm.cdf(z) + norm.pdf(z))
        return ei
    
    def _upper_confidence_bound(self, mean: float, std: float, kappa: float = 2.0) -> float:
        """Calculate upper confidence bound."""
        return mean + kappa * std


class EvolutionarySearchStrategy(SearchStrategy):
    """Evolutionary algorithm for architecture search."""
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.5,
        tournament_size: int = 3
    ):
        super().__init__('evolutionary_search')
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.population: List[Dict[str, Any]] = []
        self.generation = 0
    
    def suggest(self, architecture_space, trial_number: int) -> Dict[str, Any]:
        """Suggest next architecture using evolutionary algorithm."""
        if len(self.population) < self.population_size:
            return architecture_space.sample_architecture()
        
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()
        
        if np.random.random() < self.crossover_rate:
            offspring = self._crossover(parent1, parent2, architecture_space)
        else:
            offspring = parent1['architecture']
        
        if np.random.random() < self.mutation_rate:
            offspring = self._mutate(offspring, architecture_space)
        
        return offspring
    
    def update(self, architecture: Dict[str, Any], metrics: Dict[str, float]):
        """Update population with new architecture."""
        super().update(architecture, metrics)
        
        entry = {
            'architecture': architecture,
            'metrics': metrics,
            'fitness': metrics.get('accuracy', 0)
        }
        
        if len(self.population) < self.population_size:
            self.population.append(entry)
        else:
            worst_idx = min(range(len(self.population)), key=lambda i: self.population[i]['fitness'])
            if entry['fitness'] > self.population[worst_idx]['fitness']:
                self.population[worst_idx] = entry
                self.generation += 1
    
    def _tournament_selection(self) -> Dict[str, Any]:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        architecture_space
    ) -> Dict[str, Any]:
        """Perform crossover between two parent architectures."""
        arch1 = parent1['architecture']
        arch2 = parent2['architecture']
        
        offspring = {
            'input': arch1['input'],
            'layers': [],
            'optimizer': arch1.get('optimizer', {}),
            'training_config': arch1.get('training_config', {})
        }
        
        min_layers = min(len(arch1.get('layers', [])), len(arch2.get('layers', [])))
        
        for i in range(min_layers):
            if np.random.random() < 0.5:
                offspring['layers'].append(arch1['layers'][i])
            else:
                offspring['layers'].append(arch2['layers'][i])
        
        return offspring
    
    def _mutate(self, architecture: Dict[str, Any], architecture_space) -> Dict[str, Any]:
        """Mutate an architecture."""
        mutated = {
            'input': architecture['input'],
            'layers': [],
            'optimizer': architecture.get('optimizer', {}),
            'training_config': architecture.get('training_config', {})
        }
        
        for layer in architecture.get('layers', []):
            if np.random.random() < 0.3:
                choices = [lc for lc in architecture_space.layer_choices if lc.name == layer['type']]
                if choices:
                    mutated_layer = choices[0].sample()
                else:
                    mutated_layer = layer
            else:
                mutated_layer = layer
            
            mutated['layers'].append(mutated_layer)
        
        return mutated


class RegularizedEvolutionStrategy(EvolutionarySearchStrategy):
    """Regularized evolution with aging mechanism."""
    
    def __init__(
        self,
        population_size: int = 20,
        sample_size: int = 10,
        mutation_rate: float = 0.2
    ):
        super().__init__(
            population_size=population_size,
            mutation_rate=mutation_rate
        )
        self.sample_size = sample_size
        self.age_property = 'age'
    
    def update(self, architecture: Dict[str, Any], metrics: Dict[str, float]):
        """Update with aging mechanism."""
        entry = {
            'architecture': architecture,
            'metrics': metrics,
            'fitness': metrics.get('accuracy', 0),
            'age': 0
        }
        
        self.population.append(entry)
        
        for individual in self.population:
            individual['age'] += 1
        
        if len(self.population) > self.population_size:
            oldest_idx = max(range(len(self.population)), key=lambda i: self.population[i]['age'])
            self.population.pop(oldest_idx)
        
        self.history.append({
            'architecture': architecture,
            'metrics': metrics,
            'trial_number': len(self.history)
        })
    
    def _tournament_selection(self) -> Dict[str, Any]:
        """Select from sample of population."""
        sample = random.sample(
            self.population,
            min(self.sample_size, len(self.population))
        )
        return max(sample, key=lambda x: x['fitness'])

class RandomSearch(SearchStrategy):
    def __init__(self, max_trials: int = 10, seed: Optional[int] = None):
        super().__init__('random_search')
        self.max_trials = max_trials
        self._delegate = RandomSearchStrategy(seed=seed)
    
    def suggest(self, architecture_space, trial_number: int = 0) -> Dict[str, Any]:
        return self._delegate.suggest(architecture_space, trial_number)

class BayesianSearch(SearchStrategy):
    def __init__(self, max_trials: int = 20, acquisition_function: str = 'ei'):
        super().__init__('bayesian_search')
        self.max_trials = max_trials
        self._delegate = BayesianSearchStrategy(acquisition_function=acquisition_function)
    
    def suggest(self, architecture_space, trial_number: int = 0) -> Dict[str, Any]:
        return self._delegate.suggest(architecture_space, trial_number)

class EvolutionarySearch(SearchStrategy):
    def __init__(self, population_size: int = 20, generations: int = 10, mutation_rate: float = 0.2, crossover_rate: float = 0.5):
        super().__init__('evolutionary_search')
        self.population_size = population_size
        self.generations = generations
        self._delegate = EvolutionarySearchStrategy(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
    
    def suggest(self, architecture_space, trial_number: int = 0) -> Dict[str, Any]:
        return self._delegate.suggest(architecture_space, trial_number)
