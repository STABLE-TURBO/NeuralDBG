"""
Early stopping strategies for AutoML trials.

Implements various pruning strategies to terminate unpromising trials early.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EarlyStoppingStrategy(ABC):
    """Base class for early stopping strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def should_stop(
        self,
        trial_id: int,
        step: int,
        metrics: Dict[str, float],
        trial_history: List[Dict[str, Any]]
    ) -> bool:
        """Determine if a trial should be stopped early."""
        pass


class MedianPruner(EarlyStoppingStrategy):
    """Prune trials below the median of all trials at the same step."""
    
    def __init__(self, n_startup_trials: int = 5, n_warmup_steps: int = 5):
        super().__init__('median_pruner')
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
    
    def should_stop(
        self,
        trial_id: int,
        step: int,
        metrics: Dict[str, float],
        trial_history: List[Dict[str, Any]]
    ) -> bool:
        """Stop if performance is below median."""
        if len(trial_history) < self.n_startup_trials:
            return False
        
        if step < self.n_warmup_steps:
            return False
        
        current_value = metrics.get('accuracy', metrics.get('loss', 0))
        
        values_at_step = []
        for trial in trial_history:
            if trial['trial_id'] != trial_id:
                step_metrics = trial.get('step_metrics', {}).get(step)
                if step_metrics:
                    value = step_metrics.get('accuracy', step_metrics.get('loss', 0))
                    values_at_step.append(value)
        
        if not values_at_step:
            return False
        
        median = np.median(values_at_step)
        
        if 'accuracy' in metrics:
            return current_value < median
        else:
            return current_value > median


class HyperbandPruner(EarlyStoppingStrategy):
    """Hyperband early stopping strategy."""
    
    def __init__(
        self,
        max_resource: int = 100,
        reduction_factor: int = 3,
        min_resource: Optional[int] = None
    ):
        super().__init__('hyperband_pruner')
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.min_resource = min_resource or max_resource // (reduction_factor ** 4)
        
        self.brackets = self._calculate_brackets()
        self.trial_brackets: Dict[int, int] = {}
        self.trial_rungs: Dict[int, int] = {}
    
    def _calculate_brackets(self) -> List[List[tuple]]:
        """Calculate hyperband brackets (rungs)."""
        s_max = int(np.log(self.max_resource / self.min_resource) / np.log(self.reduction_factor))
        
        brackets = []
        for s in range(s_max + 1):
            n = int(np.ceil(s_max / (s + 1) * self.reduction_factor ** s))
            r = self.max_resource * self.reduction_factor ** (-s)
            
            bracket = []
            for i in range(s + 1):
                n_i = n * self.reduction_factor ** (-i)
                r_i = r * self.reduction_factor ** i
                bracket.append((int(n_i), int(r_i)))
            
            brackets.append(bracket)
        
        return brackets
    
    def should_stop(
        self,
        trial_id: int,
        step: int,
        metrics: Dict[str, float],
        trial_history: List[Dict[str, Any]]
    ) -> bool:
        """Stop based on Hyperband schedule."""
        if trial_id not in self.trial_brackets:
            self.trial_brackets[trial_id] = trial_id % len(self.brackets)
            self.trial_rungs[trial_id] = 0
        
        bracket = self.brackets[self.trial_brackets[trial_id]]
        current_rung = self.trial_rungs[trial_id]
        
        if current_rung >= len(bracket):
            return False
        
        n_trials, resource = bracket[current_rung]
        
        if step < resource:
            return False
        
        current_value = metrics.get('accuracy', -metrics.get('loss', float('inf')))
        
        values = []
        for trial in trial_history:
            if trial['trial_id'] != trial_id:
                step_metrics = trial.get('step_metrics', {}).get(resource)
                if step_metrics:
                    value = step_metrics.get('accuracy', -step_metrics.get('loss', float('inf')))
                    values.append(value)
        
        if len(values) < n_trials:
            return False
        
        values.sort(reverse=True)
        threshold = values[min(n_trials - 1, len(values) - 1)]
        
        should_prune = current_value < threshold
        
        if not should_prune and step >= resource:
            self.trial_rungs[trial_id] += 1
        
        return should_prune


class ASHAPruner(EarlyStoppingStrategy):
    """Asynchronous Successive Halving Algorithm (ASHA) pruner."""
    
    def __init__(
        self,
        reduction_factor: int = 4,
        min_resource: int = 1,
        max_resource: Optional[int] = None
    ):
        super().__init__('asha_pruner')
        self.reduction_factor = reduction_factor
        self.min_resource = min_resource
        self.max_resource = max_resource
        
        self.rungs: List[int] = []
        self.trial_rungs: Dict[int, int] = {}
        self._initialize_rungs()
    
    def _initialize_rungs(self):
        """Initialize ASHA rungs."""
        if self.max_resource:
            resource = self.min_resource
            while resource <= self.max_resource:
                self.rungs.append(resource)
                resource *= self.reduction_factor
        else:
            for i in range(10):
                self.rungs.append(self.min_resource * (self.reduction_factor ** i))
    
    def should_stop(
        self,
        trial_id: int,
        step: int,
        metrics: Dict[str, float],
        trial_history: List[Dict[str, Any]]
    ) -> bool:
        """Stop based on ASHA algorithm."""
        if trial_id not in self.trial_rungs:
            self.trial_rungs[trial_id] = 0
        
        current_rung_idx = self.trial_rungs[trial_id]
        
        if current_rung_idx >= len(self.rungs):
            return False
        
        current_rung = self.rungs[current_rung_idx]
        
        if step < current_rung:
            return False
        
        current_value = metrics.get('accuracy', -metrics.get('loss', float('inf')))
        
        values = []
        for trial in trial_history:
            if trial['trial_id'] != trial_id:
                step_metrics = trial.get('step_metrics', {}).get(current_rung)
                if step_metrics:
                    value = step_metrics.get('accuracy', -step_metrics.get('loss', float('inf')))
                    values.append(value)
        
        if len(values) < self.reduction_factor:
            return False
        
        values.sort(reverse=True)
        top_k = len(values) // self.reduction_factor
        threshold = values[top_k - 1] if top_k > 0 else values[0]
        
        should_prune = current_value < threshold
        
        if not should_prune and step >= current_rung:
            self.trial_rungs[trial_id] += 1
        
        return should_prune


class ThresholdPruner(EarlyStoppingStrategy):
    """Prune trials that don't meet a threshold."""
    
    def __init__(
        self,
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        metric_name: str = 'accuracy',
        n_warmup_steps: int = 5
    ):
        super().__init__('threshold_pruner')
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.metric_name = metric_name
        self.n_warmup_steps = n_warmup_steps
    
    def should_stop(
        self,
        trial_id: int,
        step: int,
        metrics: Dict[str, float],
        trial_history: List[Dict[str, Any]]
    ) -> bool:
        """Stop if metric is outside threshold range."""
        if step < self.n_warmup_steps:
            return False
        
        value = metrics.get(self.metric_name)
        if value is None:
            return False
        
        if self.lower_threshold is not None and value < self.lower_threshold:
            return True
        
        if self.upper_threshold is not None and value > self.upper_threshold:
            return True
        
        return False


class PercentilePruner(EarlyStoppingStrategy):
    """Prune trials below a percentile threshold."""
    
    def __init__(
        self,
        percentile: float = 25.0,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 5
    ):
        super().__init__('percentile_pruner')
        self.percentile = percentile
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
    
    def should_stop(
        self,
        trial_id: int,
        step: int,
        metrics: Dict[str, float],
        trial_history: List[Dict[str, Any]]
    ) -> bool:
        """Stop if performance is below percentile."""
        if len(trial_history) < self.n_startup_trials:
            return False
        
        if step < self.n_warmup_steps:
            return False
        
        current_value = metrics.get('accuracy', -metrics.get('loss', float('inf')))
        
        values_at_step = []
        for trial in trial_history:
            if trial['trial_id'] != trial_id:
                step_metrics = trial.get('step_metrics', {}).get(step)
                if step_metrics:
                    value = step_metrics.get('accuracy', -step_metrics.get('loss', float('inf')))
                    values_at_step.append(value)
        
        if not values_at_step:
            return False
        
        threshold = np.percentile(values_at_step, self.percentile)
        
        return current_value < threshold


class PatientPruner(EarlyStoppingStrategy):
    """Prune trials that show no improvement for a patience period."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        super().__init__('patient_pruner')
        self.patience = patience
        self.min_delta = min_delta
        self.best_values: Dict[int, float] = {}
        self.wait_counts: Dict[int, int] = {}
    
    def should_stop(
        self,
        trial_id: int,
        step: int,
        metrics: Dict[str, float],
        trial_history: List[Dict[str, Any]]
    ) -> bool:
        """Stop if no improvement for patience steps."""
        current_value = metrics.get('accuracy', -metrics.get('loss', float('inf')))
        
        if trial_id not in self.best_values:
            self.best_values[trial_id] = current_value
            self.wait_counts[trial_id] = 0
            return False
        
        improvement = current_value - self.best_values[trial_id]
        
        if improvement > self.min_delta:
            self.best_values[trial_id] = current_value
            self.wait_counts[trial_id] = 0
            return False
        else:
            self.wait_counts[trial_id] += 1
            return self.wait_counts[trial_id] >= self.patience

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.wait_count: int = 0
    
    def should_stop(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            self.wait_count = 0
            return False
        improvement = current_value - self.best_value
        if improvement > self.min_delta:
            self.best_value = current_value
            self.wait_count = 0
            return False
        else:
            self.wait_count += 1
            return self.wait_count >= self.patience
