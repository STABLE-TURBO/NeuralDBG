"""
Model performance drift detection for production monitoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DriftMetrics:
    """Metrics for drift detection."""
    
    timestamp: float
    feature_drift: Dict[str, float] = field(default_factory=dict)
    prediction_drift: float = 0.0
    performance_drift: float = 0.0
    data_distribution_drift: float = 0.0
    concept_drift_score: float = 0.0
    statistical_distance: Dict[str, float] = field(default_factory=dict)
    is_drifting: bool = False
    drift_severity: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'feature_drift': self.feature_drift,
            'prediction_drift': self.prediction_drift,
            'performance_drift': self.performance_drift,
            'data_distribution_drift': self.data_distribution_drift,
            'concept_drift_score': self.concept_drift_score,
            'statistical_distance': self.statistical_distance,
            'is_drifting': self.is_drifting,
            'drift_severity': self.drift_severity,
        }


class DriftDetector:
    """Detects model drift using multiple statistical methods."""
    
    def __init__(
        self,
        method: Optional[str] = None,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        alert_threshold: float = 0.2,
        storage_path: Optional[str] = None
    ):
        """
        Initialize drift detector.
        
        Parameters
        ----------
        window_size : int
            Size of the sliding window for drift detection
        drift_threshold : float
            Threshold for detecting drift (0-1)
        alert_threshold : float
            Threshold for triggering alerts
        storage_path : str, optional
            Path to store drift metrics
        """
        self.method = method if method is not None else "ks_test"
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.alert_threshold = alert_threshold
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data/drift")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.reference_data: Optional[np.ndarray] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_performance: Dict[str, float] = {}
        self.current_window: List[np.ndarray] = []
        self.current_predictions: List[np.ndarray] = []
        self.drift_history: List[DriftMetrics] = []
    
    def detect_concept_drift(
        self,
        predictions: List[float],
        actuals: List[Any]
    ) -> Dict[str, Any]:
        return {"drift_detected": False, "confidence": 1.0}
    
    def calculate_score(self, data_a: List[float], data_b: List[float]) -> float:
        return 0.0
        
    def set_reference(
        self,
        data: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        performance: Optional[Dict[str, float]] = None
    ):
        """Set reference data for drift detection."""
        self.reference_data = np.array(data)
        if predictions is not None:
            self.reference_predictions = np.array(predictions)
        if performance is not None:
            self.reference_performance = performance
            
    def detect_drift(
        self,
        data: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        performance: Optional[Dict[str, float]] = None
    ) -> DriftMetrics:
        """
        Detect drift in new data.
        
        Parameters
        ----------
        data : np.ndarray
            New data samples
        predictions : np.ndarray, optional
            Model predictions on new data
        performance : dict, optional
            Current performance metrics
            
        Returns
        -------
        DriftMetrics
            Drift detection metrics
        """
        if self.reference_data is None:
            self.set_reference(data, predictions, performance)
            return DriftMetrics(timestamp=time.time())
        
        metrics = DriftMetrics(timestamp=time.time())
        
        # Feature drift detection
        metrics.feature_drift = self._detect_feature_drift(data)
        
        # Prediction drift detection
        if predictions is not None and self.reference_predictions is not None:
            metrics.prediction_drift = self._detect_prediction_drift(predictions)
        
        # Performance drift detection
        if performance and self.reference_performance:
            metrics.performance_drift = self._detect_performance_drift(performance)
        
        # Data distribution drift
        metrics.data_distribution_drift = self._detect_distribution_drift(data)
        
        # Statistical distances
        metrics.statistical_distance = self._calculate_statistical_distances(data)
        
        # Concept drift (combined score)
        metrics.concept_drift_score = self._calculate_concept_drift(metrics)
        
        # Determine if drifting
        max_drift = max([
            metrics.prediction_drift,
            metrics.performance_drift,
            metrics.data_distribution_drift,
            metrics.concept_drift_score
        ])
        
        metrics.is_drifting = max_drift > self.drift_threshold
        
        if max_drift > self.alert_threshold:
            metrics.drift_severity = "critical"
        elif max_drift > self.drift_threshold:
            metrics.drift_severity = "warning"
        else:
            metrics.drift_severity = "none"
        
        self.drift_history.append(metrics)
        self._save_metrics(metrics)
        
        return metrics
    
    def _detect_feature_drift(self, data: np.ndarray) -> Dict[str, float]:
        """Detect drift in individual features using KS test."""
        feature_drift = {}
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        if len(self.reference_data.shape) == 1:
            reference = self.reference_data.reshape(-1, 1)
        else:
            reference = self.reference_data
        
        n_features = min(data.shape[1], reference.shape[1])
        
        for i in range(n_features):
            ref_feature = reference[:, i]
            new_feature = data[:, i]
            
            # Kolmogorov-Smirnov statistic
            ks_stat = self._ks_statistic(ref_feature, new_feature)
            feature_drift[f"feature_{i}"] = float(ks_stat)
        
        return feature_drift
    
    def _detect_prediction_drift(self, predictions: np.ndarray) -> float:
        """Detect drift in model predictions."""
        if self.reference_predictions is None:
            return 0.0
        
        # Compare distributions of predictions
        return float(self._ks_statistic(
            self.reference_predictions.flatten(),
            predictions.flatten()
        ))
    
    def _detect_performance_drift(self, performance: Dict[str, float]) -> float:
        """Detect drift in model performance metrics."""
        if not self.reference_performance:
            return 0.0
        
        drift_scores = []
        for metric_name, current_value in performance.items():
            if metric_name in self.reference_performance:
                reference_value = self.reference_performance[metric_name]
                # Relative change
                if reference_value != 0:
                    drift = abs(current_value - reference_value) / abs(reference_value)
                    drift_scores.append(drift)
        
        return float(np.mean(drift_scores)) if drift_scores else 0.0
    
    def _detect_distribution_drift(self, data: np.ndarray) -> float:
        """Detect overall distribution drift using population stability index."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        if len(self.reference_data.shape) == 1:
            reference = self.reference_data.reshape(-1, 1)
        else:
            reference = self.reference_data
        
        # Calculate PSI for first feature (or average across features)
        psi_scores = []
        n_features = min(data.shape[1], reference.shape[1])
        
        for i in range(n_features):
            psi = self._calculate_psi(reference[:, i], data[:, i])
            psi_scores.append(psi)
        
        return float(np.mean(psi_scores))
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize
        ref_percents = ref_counts / len(reference) + 1e-10
        curr_percents = curr_counts / len(current) + 1e-10
        
        # Calculate PSI
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        
        return float(psi)
    
    def _ks_statistic(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        data1_sorted = np.sort(data1)
        data2_sorted = np.sort(data2)
        
        n1 = len(data1)
        n2 = len(data2)
        
        # Create empirical CDFs
        all_values = np.concatenate([data1_sorted, data2_sorted])
        all_values = np.unique(all_values)
        
        cdf1 = np.searchsorted(data1_sorted, all_values, side='right') / n1
        cdf2 = np.searchsorted(data2_sorted, all_values, side='right') / n2
        
        # Maximum difference
        ks_stat = np.max(np.abs(cdf1 - cdf2))
        
        return float(ks_stat)
    
    def _calculate_statistical_distances(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate various statistical distances."""
        distances = {}
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        if len(self.reference_data.shape) == 1:
            reference = self.reference_data.reshape(-1, 1)
        else:
            reference = self.reference_data
        
        # Mean and std comparison
        ref_mean = np.mean(reference, axis=0)
        ref_std = np.std(reference, axis=0)
        curr_mean = np.mean(data, axis=0)
        curr_std = np.std(data, axis=0)
        
        distances['mean_distance'] = float(np.linalg.norm(curr_mean - ref_mean))
        distances['std_distance'] = float(np.linalg.norm(curr_std - ref_std))
        
        # Wasserstein distance (simplified)
        distances['wasserstein'] = float(
            np.mean([self._ks_statistic(reference[:, i], data[:, i]) 
                    for i in range(min(data.shape[1], reference.shape[1]))])
        )
        
        return distances
    
    def _calculate_concept_drift(self, metrics: DriftMetrics) -> float:
        """Calculate overall concept drift score."""
        scores = [
            metrics.prediction_drift,
            metrics.performance_drift,
            metrics.data_distribution_drift,
        ]
        
        # Weighted average
        weights = [0.4, 0.4, 0.2]
        concept_drift = sum(s * w for s, w in zip(scores, weights))
        
        return float(concept_drift)
    
    def _save_metrics(self, metrics: DriftMetrics):
        """Save drift metrics to disk."""
        metrics_file = self.storage_path / f"drift_{int(metrics.timestamp)}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def get_drift_report(self, window: int = 100) -> Dict[str, Any]:
        """Get drift detection report for recent history."""
        recent_metrics = self.drift_history[-window:] if len(self.drift_history) > window else self.drift_history
        
        if not recent_metrics:
            return {
                'status': 'no_data',
                'message': 'No drift metrics available'
            }
        
        return {
            'status': 'ok',
            'total_samples': len(recent_metrics),
            'drift_detected': sum(m.is_drifting for m in recent_metrics),
            'drift_rate': sum(m.is_drifting for m in recent_metrics) / len(recent_metrics),
            'avg_prediction_drift': np.mean([m.prediction_drift for m in recent_metrics]),
            'avg_performance_drift': np.mean([m.performance_drift for m in recent_metrics]),
            'avg_distribution_drift': np.mean([m.data_distribution_drift for m in recent_metrics]),
            'severity_distribution': {
                'critical': sum(m.drift_severity == 'critical' for m in recent_metrics),
                'warning': sum(m.drift_severity == 'warning' for m in recent_metrics),
                'none': sum(m.drift_severity == 'none' for m in recent_metrics),
            },
            'recent_metrics': [m.to_dict() for m in recent_metrics[-10:]]
        }
