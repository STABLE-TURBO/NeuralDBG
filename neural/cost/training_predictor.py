"""
Training time and cost prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from neural.cost.estimator import CostEstimator, InstanceType


@dataclass
class TrainingEstimate:
    """Training time and cost estimate."""
    
    estimated_hours: float
    estimated_epochs: int
    estimated_iterations: int
    compute_cost: float
    storage_cost: float
    total_cost: float
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    factors: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'estimated_hours': self.estimated_hours,
            'estimated_epochs': self.estimated_epochs,
            'estimated_iterations': self.estimated_iterations,
            'compute_cost': self.compute_cost,
            'storage_cost': self.storage_cost,
            'total_cost': self.total_cost,
            'confidence_interval': {
                'lower': self.confidence_interval[0],
                'upper': self.confidence_interval[1]
            },
            'factors': self.factors
        }


@dataclass
class TrainingHistory:
    """Historical training data."""
    
    model_size: int
    dataset_size: int
    batch_size: int
    epochs: int
    actual_hours: float
    instance_type: str
    gpu_count: int
    convergence_epoch: Optional[int] = None


class TrainingPredictor:
    """Predict training time and costs."""
    
    def __init__(
        self,
        cost_estimator: Optional[CostEstimator] = None,
        history: Optional[List[TrainingHistory]] = None
    ):
        """
        Initialize training predictor.
        
        Parameters
        ----------
        cost_estimator : CostEstimator, optional
            Cost estimator instance
        history : list, optional
            Historical training data
        """
        self.cost_estimator = cost_estimator or CostEstimator()
        self.history = history or []
        self._fitted = False
        
    def predict_training_time(
        self,
        model_params: Dict[str, Any],
        dataset_size: int,
        batch_size: int,
        epochs: int,
        instance_type: InstanceType
    ) -> TrainingEstimate:
        """
        Predict training time and cost.
        
        Parameters
        ----------
        model_params : dict
            Model parameters (layers, parameters, etc.)
        dataset_size : int
            Number of training samples
        batch_size : int
            Training batch size
        epochs : int
            Number of training epochs
        instance_type : InstanceType
            Instance type to use
            
        Returns
        -------
        TrainingEstimate
            Training time and cost estimate
        """
        iterations_per_epoch = max(1, dataset_size // batch_size)
        total_iterations = iterations_per_epoch * epochs
        
        model_size = model_params.get('total_params', 1_000_000)
        
        if self._fitted and self.history:
            estimated_hours = self._predict_from_history(
                model_size,
                dataset_size,
                batch_size,
                epochs,
                instance_type.gpu_count
            )
        else:
            estimated_hours = self._predict_heuristic(
                model_size,
                total_iterations,
                instance_type
            )
        
        compute_cost = instance_type.spot_price_per_hour * estimated_hours
        storage_cost = self._estimate_storage_cost(model_size, estimated_hours)
        
        lower_bound = estimated_hours * 0.7
        upper_bound = estimated_hours * 1.5
        
        return TrainingEstimate(
            estimated_hours=estimated_hours,
            estimated_epochs=epochs,
            estimated_iterations=total_iterations,
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            total_cost=compute_cost + storage_cost,
            confidence_interval=(lower_bound, upper_bound),
            factors={
                'model_size': model_size,
                'dataset_size': dataset_size,
                'batch_size': batch_size,
                'iterations_per_epoch': iterations_per_epoch,
                'gpu_count': instance_type.gpu_count,
                'gpu_type': instance_type.gpu_type,
            }
        )
    
    def compare_configurations(
        self,
        model_params: Dict[str, Any],
        dataset_size: int,
        configs: List[Dict[str, Any]]
    ) -> List[TrainingEstimate]:
        """
        Compare different training configurations.
        
        Parameters
        ----------
        model_params : dict
            Model parameters
        dataset_size : int
            Dataset size
        configs : list
            List of configuration dicts (batch_size, epochs, instance_type)
            
        Returns
        -------
        list
            Estimates for each configuration
        """
        estimates = []
        
        for config in configs:
            batch_size = config['batch_size']
            epochs = config['epochs']
            instance_type = config['instance_type']
            
            estimate = self.predict_training_time(
                model_params,
                dataset_size,
                batch_size,
                epochs,
                instance_type
            )
            estimates.append(estimate)
        
        return sorted(estimates, key=lambda x: x.total_cost)
    
    def optimize_batch_size(
        self,
        model_params: Dict[str, Any],
        dataset_size: int,
        instance_type: InstanceType,
        target_epochs: int = 100
    ) -> Dict[str, Any]:
        """
        Find optimal batch size.
        
        Parameters
        ----------
        model_params : dict
            Model parameters
        dataset_size : int
            Dataset size
        instance_type : InstanceType
            Instance type
        target_epochs : int
            Target number of epochs
            
        Returns
        -------
        dict
            Optimal configuration
        """
        batch_sizes = [16, 32, 64, 128, 256, 512]
        
        configs = [
            {
                'batch_size': bs,
                'epochs': target_epochs,
                'instance_type': instance_type
            }
            for bs in batch_sizes
        ]
        
        estimates = self.compare_configurations(
            model_params,
            dataset_size,
            configs
        )
        
        best_estimate = estimates[0]
        
        return {
            'optimal_batch_size': best_estimate.factors['batch_size'],
            'estimated_hours': best_estimate.estimated_hours,
            'estimated_cost': best_estimate.total_cost,
            'all_estimates': [e.to_dict() for e in estimates]
        }
    
    def add_training_record(self, record: TrainingHistory):
        """Add historical training record."""
        self.history.append(record)
        self._fitted = False
    
    def fit_from_history(self):
        """Fit predictor from historical data."""
        if len(self.history) < 3:
            return
        
        self._fitted = True
    
    def _predict_from_history(
        self,
        model_size: int,
        dataset_size: int,
        batch_size: int,
        epochs: int,
        gpu_count: int
    ) -> float:
        """Predict using historical data."""
        similar_runs = self._find_similar_runs(
            model_size,
            dataset_size,
            batch_size,
            gpu_count
        )
        
        if not similar_runs:
            return self._predict_heuristic_time(
                model_size,
                dataset_size,
                batch_size,
                epochs,
                gpu_count
            )
        
        avg_time_per_epoch = np.mean([
            run.actual_hours / run.epochs for run in similar_runs
        ])
        
        return avg_time_per_epoch * epochs
    
    def _find_similar_runs(
        self,
        model_size: int,
        dataset_size: int,
        batch_size: int,
        gpu_count: int,
        tolerance: float = 0.3
    ) -> List[TrainingHistory]:
        """Find similar historical runs."""
        similar = []
        
        for run in self.history:
            size_ratio = abs(run.model_size - model_size) / model_size
            data_ratio = abs(run.dataset_size - dataset_size) / dataset_size
            batch_ratio = abs(run.batch_size - batch_size) / batch_size
            
            if (size_ratio < tolerance and 
                data_ratio < tolerance and 
                batch_ratio < tolerance and
                run.gpu_count == gpu_count):
                similar.append(run)
        
        return similar
    
    def _predict_heuristic(
        self,
        model_size: int,
        total_iterations: int,
        instance_type: InstanceType
    ) -> float:
        """Predict using heuristics."""
        base_time_per_iteration_ms = 10.0
        
        complexity_factor = (model_size / 1_000_000) ** 0.5
        
        gpu_speedup = {
            'T4': 1.0,
            'V100': 2.0,
            'A100': 4.0,
            'A10G': 1.5,
        }
        
        speedup = 1.0
        if instance_type.gpu_type:
            for gpu_key, gpu_speedup_val in gpu_speedup.items():
                if gpu_key in instance_type.gpu_type:
                    speedup = gpu_speedup_val * instance_type.gpu_count
                    break
        
        time_ms = (base_time_per_iteration_ms * complexity_factor * 
                   total_iterations / speedup)
        
        return time_ms / (1000 * 3600)
    
    def _predict_heuristic_time(
        self,
        model_size: int,
        dataset_size: int,
        batch_size: int,
        epochs: int,
        gpu_count: int
    ) -> float:
        """Predict time using heuristics."""
        iterations_per_epoch = max(1, dataset_size // batch_size)
        total_iterations = iterations_per_epoch * epochs
        
        base_time_per_iteration_ms = 10.0
        complexity_factor = (model_size / 1_000_000) ** 0.5
        gpu_speedup = max(1.0, gpu_count * 1.8)
        
        time_ms = (base_time_per_iteration_ms * complexity_factor * 
                   total_iterations / gpu_speedup)
        
        return time_ms / (1000 * 3600)
    
    def _estimate_storage_cost(
        self,
        model_size: int,
        training_hours: float
    ) -> float:
        """Estimate storage cost."""
        checkpoint_size_gb = (model_size * 4) / (1024 ** 3)
        
        num_checkpoints = max(1, int(training_hours / 2))
        
        total_storage_gb = checkpoint_size_gb * min(num_checkpoints, 5)
        
        storage_rate_per_gb_per_hour = 0.10 / (30 * 24)
        
        return total_storage_gb * storage_rate_per_gb_per_hour * training_hours
