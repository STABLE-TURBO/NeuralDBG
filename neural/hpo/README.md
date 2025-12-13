# Neural HPO - Hyperparameter Optimization

This module handles hyperparameter optimization for Neural DSL. We've integrated several state-of-the-art techniques because, let's be honest, manually tuning hyperparameters is tedious and inefficient.

## Features

### 1. Bayesian Optimization

Instead of randomly trying hyperparameters, Bayesian optimization builds a probabilistic model of your objective function and uses it to pick promising configurations.

- **TPE Sampler** - Tree-structured Parzen Estimator. This is our default because it works well in practice and scales reasonably.
- **CMA-ES Sampler** - Covariance Matrix Adaptation Evolution Strategy. Better for continuous parameters with complex interactions.
- **Gaussian Process** - For parameter importance analysis. Tells you which hyperparameters actually matter.

### 2. Multi-Objective Optimization

Sometimes you care about multiple things - accuracy, inference speed, model size. Multi-objective optimization finds the trade-offs.

**What you get:**
- Pareto front visualization (the set of non-dominated solutions)
- NSGA-II sampler for efficient multi-objective search
- 2D and 3D plotting of trade-offs

**Use case example:** You want high accuracy but also need your model to run in under 100ms for production. Multi-objective optimization helps you find the best accuracy you can get within that latency constraint.

### 3. Distributed HPO with Ray Tune

For large search spaces or expensive evaluations, distribute the work across multiple machines.

- Scale across multiple CPUs/GPUs
- ASHA scheduler for early stopping of bad trials (saves time)
- Population-Based Training (PBT) for online hyperparameter adjustment
- Asynchronous parallel trials

**Trade-off:** More setup complexity in exchange for much faster search times.

### 4. Parameter Importance Analysis

Which hyperparameters actually matter for your model? This tells you.

**Methods available:**
- Random Forest importance
- Gradient Boosting importance
- Permutation importance
- fANOVA (functional ANOVA) - the gold standard when you have enough data
- Gaussian Process-based importance
- Bootstrap uncertainty estimation

**Why this matters:** If a parameter isn't important, you can fix it to a reasonable value and reduce your search space. Makes optimization faster.

### 5. Visualizations

We've included comprehensive visualizations because staring at logs is terrible for understanding what's happening:

- Optimization history (how performance improved over time)
- Parameter importance plots with uncertainty bars
- Parallel coordinates (see relationships between parameters)
- Correlation heatmaps
- Contour plots for 2D parameter spaces
- Slice plots for individual parameters
- Marginal effects plots
- Parameter interaction heatmaps
- Convergence comparison across different runs
- HTML reports that bundle everything together

## Usage Examples

### Basic Bayesian Optimization

The simplest case - just find good hyperparameters:

```python
from neural.hpo import optimize_and_return

config = """
network MyModel {
    input: (28, 28, 1)
    
    Dense(units: hpo(categorical: [64, 128, 256])) -> relu
    Dropout(rate: hpo(range: [0.3, 0.7, step=0.1]))
    Dense(units: 10) -> softmax
    
    optimizer: adam(learning_rate: hpo(log_range: [1e-4, 1e-2]))
    training: {
        batch_size: hpo(categorical: [16, 32, 64])
    }
}
"""

# Run 50 trials with TPE sampler
results = optimize_and_return(
    config=config,
    n_trials=50,
    dataset_name='MNIST',
    backend='pytorch',
    device='auto',
    sampler='tpe',  # Bayesian optimization
    enable_pruning=True  # Stop bad trials early
)

print(f"Best parameters: {results}")
```

**Note:** 50 trials is reasonable for this search space. For larger spaces, you might need 100+ trials. Start small and increase if needed.

### Multi-Objective Optimization

When you care about multiple metrics:

```python
from neural.hpo import MultiObjectiveOptimizer, objective

# Define what you're optimizing
objectives = ['loss', 'accuracy', 'precision']
directions = ['minimize', 'maximize', 'maximize']

moo = MultiObjectiveOptimizer(objectives, directions)

results = moo.optimize(
    objective_fn=objective,
    n_trials=100,  # Multi-objective needs more trials
    sampler='nsgaii',
    config=config,
    dataset_name='MNIST',
    backend='pytorch',
    device='auto'
)

# Get Pareto-optimal solutions
pareto_front = moo.get_pareto_front()
print(f"Found {len(pareto_front)} Pareto-optimal solutions")

# Visualize the trade-offs
fig = moo.plot_pareto_front(obj_x=0, obj_y=1)
fig.savefig('pareto_front.png')
```

**Tip:** Multi-objective optimization typically needs 2-3x more trials than single-objective because the search space is effectively larger.

### Distributed HPO with Ray Tune

For when you have multiple GPUs or machines:

```python
from neural.hpo import DistributedHPO

dist_hpo = DistributedHPO(use_ray=True)

config_space = {
    'batch_size': dist_hpo.tune.choice([16, 32, 64]),
    'learning_rate': dist_hpo.tune.loguniform(1e-4, 1e-1),
    'dense_units': dist_hpo.tune.choice([64, 128, 256]),
    'dropout_rate': dist_hpo.tune.uniform(0.3, 0.7)
}

results = dist_hpo.optimize_with_ray(
    trainable_fn=my_training_function,
    config_space=config_space,
    n_trials=100,
    n_cpus=2,  # Per trial
    n_gpus=1,  # Per trial
    scheduler='asha',  # Asynchronous Successive Halving
    search_alg='optuna',
    metric='accuracy',
    mode='max'
)

print(f"Best config: {results['best_config']}")
print(f"Best result: {results['best_result']}")
```

**Setup note:** Ray needs to be installed separately: `pip install "ray[tune]"`

### Advanced Parameter Importance Analysis

Figure out which hyperparameters actually matter:

```python
from neural.hpo import ParameterImportanceAnalyzer, BayesianParameterImportance
from neural.hpo import plot_param_importance

# Standard importance analysis with Random Forest
analyzer = ParameterImportanceAnalyzer(method='random_forest')
importance = analyzer.analyze(trials, target_metric='accuracy')

# Plot with uncertainty (using bootstrap)
fig = analyzer.plot_importance_with_std(
    trials=trials,
    target_metric='accuracy',
    n_iterations=20  # Bootstrap iterations
)
fig.savefig('importance_with_uncertainty.png')

# Bayesian approach with Gaussian Process
bayesian_analyzer = BayesianParameterImportance()
gp_importance = bayesian_analyzer.analyze_with_gp(trials, 'accuracy')
fig = bayesian_analyzer.plot_importance_with_uncertainty(trials, 'accuracy')
fig.savefig('gp_importance.png')

# Check parameter interactions
fig = analyzer.plot_interaction_heatmap(trials, 'accuracy')
fig.savefig('interactions.png')

# Marginal effects (how each parameter affects the objective)
fig = analyzer.plot_marginal_effects(trials, 'accuracy')
fig.savefig('marginal_effects.png')

# fANOVA (functional ANOVA) - most reliable but needs more data
fanova_importance = analyzer.analyze_with_fanova(trials, 'accuracy')
print("fANOVA importances:", fanova_importance)
```

**Interpretation tip:** If a parameter has importance < 0.05, it probably doesn't matter much. Consider fixing it to save search time.

### Visualization Suite

Generate comprehensive visualizations of your optimization run:

```python
from neural.hpo.visualization import (
    plot_optimization_history,
    plot_parallel_coordinates,
    plot_correlation_heatmap,
    plot_contour,
    plot_multi_objective_pareto,
    plot_convergence_comparison,
    create_optimization_report
)

# Optimization history over time
fig = plot_optimization_history(trials, metric='accuracy')
fig.savefig('history.png')

# Parallel coordinates (shows parameter combinations)
fig = plot_parallel_coordinates(trials, metric='accuracy', top_n=10)
fig.savefig('parallel_coords.png')

# Correlation between parameters and objective
fig = plot_correlation_heatmap(trials, metric='accuracy')
fig.savefig('correlations.png')

# 2D contour plot for two parameters
fig = plot_contour(trials, 'learning_rate', 'batch_size', metric='accuracy')
fig.savefig('contour.png')

# Multi-objective Pareto front
fig = plot_multi_objective_pareto(
    trials, 
    obj_x='loss', 
    obj_y='accuracy',
    highlight_pareto=True
)
fig.savefig('pareto.png')

# Compare different optimization methods
trials_dict = {
    'TPE': tpe_trials,
    'Random': random_trials,
    'CMA-ES': cmaes_trials
}
fig = plot_convergence_comparison(trials_dict, metric='accuracy')
fig.savefig('convergence.png')

# Generate comprehensive HTML report
report_path = create_optimization_report(
    trials,
    metric='accuracy',
    output_path='hpo_report.html'
)
print(f"Report saved to: {report_path}")
```

**Note:** The HTML report is particularly useful for sharing results with team members who don't want to look at code.

### Using Different Samplers

Each sampler has different strengths:

```python
# TPE (default) - good all-around performance
results_tpe = optimize_and_return(
    config=config,
    n_trials=50,
    sampler='tpe'
)

# CMA-ES - better for continuous parameters
results_cmaes = optimize_and_return(
    config=config,
    n_trials=50,
    sampler='cmaes'
)

# Random - baseline for comparison
results_random = optimize_and_return(
    config=config,
    n_trials=50,
    sampler='random'
)

# NSGA-II - multi-objective only
results_nsgaii = optimize_and_return(
    config=config,
    n_trials=100,
    objectives=['loss', 'accuracy'],
    sampler='nsgaii'
)
```

**When to use what:**
- TPE: Default choice, works well in most cases
- CMA-ES: Your parameters are mostly continuous and you suspect interactions
- Random: Baseline for comparison or very small search spaces
- NSGA-II: You have multiple objectives

### Complete Example

Putting it all together:

```python
from neural.hpo import (
    optimize_and_return,
    ParameterImportanceAnalyzer,
    BayesianParameterImportance,
    create_optimization_report
)

# 1. Run optimization
results = optimize_and_return(
    config=config,
    n_trials=100,
    dataset_name='MNIST',
    backend='pytorch',
    device='cuda',
    sampler='tpe',
    enable_pruning=True,
    study_name='mnist_experiment'
)

# 2. Extract trial history
trials = results.get('_trials_history', [])
study = results.get('_study')

# 3. Analyze parameter importance
analyzer = ParameterImportanceAnalyzer(method='random_forest')
importance = analyzer.analyze(trials, target_metric='accuracy')
print("Parameter importances:", importance)

# 4. Create visualizations
fig1 = analyzer.plot_importance_with_std(trials, 'accuracy', n_iterations=20)
fig1.savefig('importance.png')

fig2 = analyzer.plot_interaction_heatmap(trials, 'accuracy')
fig2.savefig('interactions.png')

fig3 = analyzer.plot_marginal_effects(trials, 'accuracy')
fig3.savefig('marginal_effects.png')

# 5. Generate comprehensive report
report_path = create_optimization_report(
    trials,
    metric='accuracy',
    output_path='hpo_report.html'
)

print(f"Optimization complete!")
print(f"Best parameters: {results}")
print(f"Report saved to: {report_path}")
```

## API Reference

### Core Functions

- `optimize_and_return(config, n_trials, dataset_name, backend, device, sampler, objectives, use_ray, enable_pruning, study_name)` - Main optimization interface

- `objective(trial, config, dataset_name, backend, device)` - Objective function for a single trial (you can override this)

### Classes

- `MultiObjectiveOptimizer(objectives, directions)` - Multi-objective optimization with Pareto analysis

- `DistributedHPO(use_ray)` - Distributed optimization with Ray Tune

- `BayesianParameterImportance()` - Bayesian parameter importance analysis

- `ParameterImportanceAnalyzer(method)` - General parameter importance analysis

### Visualization Functions

Too many to list individually. See the visualization example above. They all follow the pattern: `plot_*(trials, metric, **kwargs)` and return matplotlib figures.

## Installation

The HPO module requires optional dependencies:

```bash
# Basic HPO with Optuna
pip install optuna

# For distributed HPO with Ray Tune
pip install "ray[tune]"

# For Bayesian optimization
pip install scikit-learn scipy

# For all features
pip install -e ".[full]"
```

**Note:** The full install is several hundred MB due to Ray and its dependencies. If you don't need distributed HPO, skip it.

## Performance Tips

Some things we've learned through experience:

1. **Start with fewer trials** - Use 10-20 trials initially to check that everything works. Scale up once you're confident.

2. **Enable pruning** - Set `enable_pruning=True` to stop unpromising trials early. Can save 30-50% of total time.

3. **Choose the right sampler:**
   - TPE for general use (good default)
   - CMA-ES for continuous parameters
   - Random as a baseline
   - NSGA-II for multi-objective

4. **Use distributed optimization** - If you have multiple GPUs, Ray Tune can parallelize trials. Worth the setup overhead for large searches.

5. **Focus on important parameters** - Run parameter importance analysis first, then narrow your search to the parameters that matter.

6. **Watch for overfitting** - If your validation performance is much worse than training, you're probably overfitting to your HPO search. Use a separate test set.

## Limitations and Trade-offs

Let's be realistic about what works and what doesn't:

**What works well:**
- Small to medium search spaces (< 100 dimensions)
- Expensive objective functions (minutes per evaluation)
- Continuous and mixed parameter spaces

**What's challenging:**
- Very large search spaces (100+ dimensions) - you'll need a lot of trials
- Very cheap objective functions (< 1 second) - overhead becomes significant
- Discrete-only spaces - random search might be competitive
- Multi-modal objectives - Bayesian optimization can get stuck in local optima

**Known issues:**
- Ray Tune adds significant setup complexity
- fANOVA needs at least 100+ trials to be reliable
- Multi-objective optimization with 3+ objectives is hard to visualize
- Parameter importance analysis can be misleading with few trials

## Citation

If you use Neural HPO in research:

```bibtex
@software{neural_dsl,
  title = {Neural DSL: A Domain-Specific Language for Neural Networks},
  author = {Neural DSL Team},
  year = {2024},
  url = {https://github.com/Lemniscate-SHA-256/Neural}
}
```

Though honestly, you're probably better off citing the underlying methods (Optuna, Ray Tune, etc.) depending on what you actually use.
