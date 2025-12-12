"""
Advanced search strategies for hyperparameter optimization.
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable

# Configure logger
logger = logging.getLogger(__name__)

class BaseStrategy:
    """Base class for HPO search strategies."""

    def __init__(self, name: str):
        self.name = name
        self.trials = []

    def suggest_parameters(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        raise NotImplementedError("Subclasses must implement suggest_parameters")

    def record_trial(self, trial_info: Dict[str, Any]):
        """Record trial results."""
        self.trials.append(trial_info)

    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters found so far."""
        if not self.trials:
            return {}
        # Default implementation: return parameters from best trial
        best_trial = max(self.trials, key=lambda t: t.get('score', float('-inf')))
        return best_trial.get('parameters', {})


class BayesianStrategy(BaseStrategy):
    """Bayesian optimization strategy using Gaussian Processes."""

    def __init__(self, acquisition_function: str = 'ei', n_initial_points: int = 10):
        """
        Initialize Bayesian optimization strategy.

        Args:
            acquisition_function: Acquisition function ('ei', 'pi', or 'ucb')
            n_initial_points: Number of initial random points before using GP
        """
        super().__init__(name="bayesian")
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.gp_model = None

    def suggest_parameters(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters using Bayesian optimization."""
        # For the first n_initial_points trials, use random search
        if len(self.trials) < self.n_initial_points:
            return self._random_suggestion(trial, search_space)

        # After initial points, use Gaussian Process
        try:
            import sklearn.gaussian_process as gp
            from scipy.stats import norm

            # Initialize GP if not already done
            if self.gp_model is None:
                self.gp_model = gp.GaussianProcessRegressor(
                    kernel=gp.kernels.Matern(nu=2.5),
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=5
                )

            # Extract parameters and scores from previous trials
            X = np.array([t['parameters_normalized'] for t in self.trials if 'parameters_normalized' in t])
            y = np.array([t['score'] for t in self.trials if 'score' in t])

            # Fit the GP model
            self.gp_model.fit(X, y)

            # Use acquisition function to suggest next point
            # Implementation depends on the specific acquisition function
            # This is a simplified version
            best_params = self._optimize_acquisition(search_space)

            # Convert normalized parameters back to actual values
            return self._denormalize_parameters(best_params, search_space)

        except (ImportError, Exception) as e:
            logger.warning(f"Error in Bayesian optimization: {str(e)}. Falling back to random search.")
            return self._random_suggestion(trial, search_space)

    def _random_suggestion(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random suggestions for initial points."""
        params = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] in ['float', 'range']:
                log_scale = param_config.get('log', False)
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    log=log_scale,
                    step=param_config.get('step', None)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    step=param_config.get('step', 1)
                )
        return params

    def _optimize_acquisition(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the acquisition function to find the next point."""
        # This is a placeholder for actual acquisition function optimization
        # In a real implementation, this would use methods like L-BFGS or CMA-ES
        # to find the point that maximizes the acquisition function
        return {}

    def _denormalize_parameters(self, normalized_params: Dict[str, float],
                               search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert normalized parameters (0-1) back to actual values."""
        # This is a placeholder for parameter denormalization
        return {}


class EvolutionaryStrategy(BaseStrategy):
    """Evolutionary algorithm strategy for HPO."""

    def __init__(self, population_size: int = 10, mutation_rate: float = 0.1,
                crossover_rate: float = 0.5, selection_size: int = 3):
        """
        Initialize evolutionary strategy.

        Args:
            population_size: Size of the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            selection_size: Number of individuals for tournament selection
        """
        super().__init__(name="evolutionary")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_size = selection_size
        self.population = []

    def suggest_parameters(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters using evolutionary algorithm."""
        # If population is not full yet, generate random individuals
        if len(self.population) < self.population_size:
            return self._generate_random_individual(trial, search_space)

        # Select parents using tournament selection
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()

        # Perform crossover with some probability
        if np.random.random() < self.crossover_rate:
            child = self._crossover(parent1, parent2)
        else:
            child = parent1.copy()

        # Perform mutation with some probability
        if np.random.random() < self.mutation_rate:
            child = self._mutate(child, search_space)

        # Convert the child's parameters to trial parameters
        return self._child_to_trial_params(trial, child, search_space)

    def _generate_random_individual(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a random individual."""
        params = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] in ['float', 'range']:
                log_scale = param_config.get('log', False)
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    log=log_scale,
                    step=param_config.get('step', None)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    step=param_config.get('step', 1)
                )
        return params

    def _tournament_selection(self) -> Dict[str, Any]:
        """Select an individual using tournament selection."""
        # Select random individuals for the tournament
        tournament = np.random.choice(self.population, size=self.selection_size, replace=False)
        # Return the best individual in the tournament
        return max(tournament, key=lambda ind: ind.get('score', float('-inf')))

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two parents."""
        child = {}
        for key in parent1:
            # For each parameter, randomly choose from either parent
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(self, individual: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        # Randomly select a parameter to mutate
        param_to_mutate = np.random.choice(list(individual.keys()))
        param_config = search_space[param_to_mutate]

        # Mutate the selected parameter based on its type
        if param_config['type'] == 'categorical':
            mutated[param_to_mutate] = np.random.choice(param_config['values'])
        elif param_config['type'] in ['float', 'range']:
            min_val = param_config['min']
            max_val = param_config['max']
            if param_config.get('log', False):
                # For log scale, use multiplicative perturbation
                factor = np.exp(np.random.normal(0, 0.2))
                new_val = individual[param_to_mutate] * factor
            else:
                # For linear scale, use additive perturbation
                sigma = (max_val - min_val) * 0.1
                new_val = individual[param_to_mutate] + np.random.normal(0, sigma)
            # Clip to valid range
            mutated[param_to_mutate] = np.clip(new_val, min_val, max_val)
        elif param_config['type'] == 'int':
            min_val = param_config['min']
            max_val = param_config['max']
            step = param_config.get('step', 1)
            # For integers, add/subtract a small number of steps
            steps_to_change = np.random.randint(-3, 4)  # -3 to +3 steps
            new_val = individual[param_to_mutate] + steps_to_change * step
            # Clip to valid range
            mutated[param_to_mutate] = int(np.clip(new_val, min_val, max_val))

        return mutated

    def _child_to_trial_params(self, trial, child: Dict[str, Any],
                              search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a child's parameters to trial parameters."""
        params = {}
        for param_name, param_value in child.items():
            param_config = search_space[param_name]
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, [param_value])
            elif param_config['type'] in ['float', 'range']:
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_value,
                    param_value
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_value,
                    param_value
                )
        return params


class PopulationBasedTraining(BaseStrategy):
    """Population Based Training (PBT) strategy."""

    def __init__(self, population_size: int = 10, exploit_fraction: float = 0.2,
                explore_factor: float = 0.2):
        """
        Initialize PBT strategy.

        Args:
            population_size: Size of the population
            exploit_fraction: Fraction of population to replace in each generation
            explore_factor: Factor for parameter perturbation during exploration
        """
        super().__init__(name="pbt")
        self.population_size = population_size
        self.exploit_fraction = exploit_fraction
        self.explore_factor = explore_factor
        self.population = []
        self.generation = 0

    def suggest_parameters(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters using PBT."""
        # If population is not full yet, generate random individuals
        if len(self.population) < self.population_size:
            return self._generate_random_individual(trial, search_space)

        # If we have a full population, perform PBT
        # 1. Evaluate current population (already done in record_trial)
        # 2. Exploit: Replace worst performers with copies of best performers
        # 3. Explore: Perturb the copied parameters

        # Sort population by score
        sorted_pop = sorted(self.population, key=lambda ind: ind.get('score', float('-inf')), reverse=True)

        # Determine which individuals to replace
        n_replace = int(self.population_size * self.exploit_fraction)
        best_individuals = sorted_pop[:n_replace]
        worst_individuals = sorted_pop[-n_replace:]

        # Replace worst with perturbed copies of best
        for i in range(n_replace):
            # Copy a best individual
            new_individual = best_individuals[i % len(best_individuals)].copy()
            # Explore: perturb the parameters
            new_individual = self._perturb_parameters(new_individual, search_space)
            # Replace a worst individual
            worst_idx = self.population.index(worst_individuals[i])
            self.population[worst_idx] = new_individual

        # Increment generation counter
        self.generation += 1

        # Return parameters for the next trial
        # For PBT, we typically evaluate all individuals in parallel,
        # but for sequential trials, we can cycle through the population
        individual_idx = trial.number % self.population_size
        return self._individual_to_trial_params(trial, self.population[individual_idx], search_space)

    def _generate_random_individual(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a random individual."""
        params = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_config['type'] in ['float', 'range']:
                log_scale = param_config.get('log', False)
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    log=log_scale,
                    step=param_config.get('step', None)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    step=param_config.get('step', 1)
                )
        return params

    def _perturb_parameters(self, individual: Dict[str, Any],
                           search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb parameters for exploration."""
        perturbed = individual.copy()

        for param_name, param_value in individual.items():
            # Randomly decide whether to perturb this parameter
            if np.random.random() < 0.3:  # 30% chance to perturb each parameter
                param_config = search_space[param_name]

                if param_config['type'] == 'categorical':
                    # For categorical, randomly select a different value
                    current_idx = param_config['values'].index(param_value)
                    other_values = [v for i, v in enumerate(param_config['values']) if i != current_idx]
                    if other_values:
                        perturbed[param_name] = np.random.choice(other_values)

                elif param_config['type'] in ['float', 'range']:
                    min_val = param_config['min']
                    max_val = param_config['max']

                    if param_config.get('log', False):
                        # For log scale, multiply by a random factor
                        factor = np.exp(np.random.normal(0, self.explore_factor))
                        new_val = param_value * factor
                    else:
                        # For linear scale, add/subtract a percentage of the range
                        range_size = max_val - min_val
                        perturbation = np.random.normal(0, self.explore_factor * range_size)
                        new_val = param_value + perturbation

                    # Clip to valid range
                    perturbed[param_name] = np.clip(new_val, min_val, max_val)

                elif param_config['type'] == 'int':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    step = param_config.get('step', 1)

                    # For integers, add/subtract a percentage of the range
                    range_size = max_val - min_val
                    perturbation = int(np.random.normal(0, self.explore_factor * range_size) / step) * step
                    new_val = param_value + perturbation

                    # Clip to valid range
                    perturbed[param_name] = int(np.clip(new_val, min_val, max_val))

        return perturbed

    def _individual_to_trial_params(self, trial, individual: Dict[str, Any],
                                  search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an individual's parameters to trial parameters."""
        params = {}
        for param_name, param_value in individual.items():
            param_config = search_space[param_name]
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, [param_value])
            elif param_config['type'] in ['float', 'range']:
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_value,
                    param_value
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_value,
                    param_value
                )
        return params


# Factory function to create strategy instances
def create_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """
    Create a strategy instance based on the strategy name.

    Args:
        strategy_name: Name of the strategy
        **kwargs: Additional arguments for the strategy

    Returns:
        A strategy instance
    """
    strategies = {
        'bayesian': BayesianStrategy,
        'evolutionary': EvolutionaryStrategy,
        'pbt': PopulationBasedTraining,
        # Add more strategies here
    }

    if strategy_name not in strategies:
        logger.warning(f"Unknown strategy: {strategy_name}. Falling back to Bayesian.")
        return BayesianStrategy(**kwargs)

    return strategies[strategy_name](**kwargs)
