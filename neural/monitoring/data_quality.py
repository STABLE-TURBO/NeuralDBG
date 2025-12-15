"""
Data quality monitoring for production ML systems.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class QualityReport:
    """Data quality report."""
    
    timestamp: float
    total_samples: int
    missing_values: Dict[str, int] = field(default_factory=dict)
    missing_rate: float = 0.0
    outliers: Dict[str, int] = field(default_factory=dict)
    outlier_rate: float = 0.0
    invalid_values: Dict[str, int] = field(default_factory=dict)
    invalid_rate: float = 0.0
    schema_violations: List[str] = field(default_factory=list)
    data_types: Dict[str, str] = field(default_factory=dict)
    value_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_score: float = 1.0
    is_healthy: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'total_samples': self.total_samples,
            'missing_values': self.missing_values,
            'missing_rate': self.missing_rate,
            'outliers': self.outliers,
            'outlier_rate': self.outlier_rate,
            'invalid_values': self.invalid_values,
            'invalid_rate': self.invalid_rate,
            'schema_violations': self.schema_violations,
            'data_types': self.data_types,
            'value_ranges': self.value_ranges,
            'quality_score': self.quality_score,
            'is_healthy': self.is_healthy,
            'warnings': self.warnings,
        }


class DataQualityMonitor:
    """Monitor data quality in production."""
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
        missing_threshold: float = 0.1,
        outlier_threshold: float = 3.0,
        storage_path: Optional[str] = None
    ):
        """
        Initialize data quality monitor.
        
        Parameters
        ----------
        feature_names : list, optional
            Names of features to monitor
        feature_types : dict, optional
            Expected types for each feature
        missing_threshold : float
            Threshold for missing value rate to trigger warning
        outlier_threshold : float
            Z-score threshold for outlier detection
        storage_path : str, optional
            Path to store quality reports
        """
        self.feature_names = feature_names or []
        self.feature_types = feature_types or {}
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data/quality")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.reference_statistics: Dict[str, Dict[str, float]] = {}
        self.quality_history: List[QualityReport] = []
        
    def set_reference_statistics(self, data: np.ndarray):
        """Set reference statistics for quality monitoring."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        for i in range(data.shape[1]):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_data = data[:, i]
            
            self.reference_statistics[feature_name] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'q25': float(np.percentile(feature_data, 25)),
                'q50': float(np.percentile(feature_data, 50)),
                'q75': float(np.percentile(feature_data, 75)),
            }
    
    def check_quality(self, data: np.ndarray) -> QualityReport:
        """
        Check data quality.
        
        Parameters
        ----------
        data : np.ndarray
            Data to check
            
        Returns
        -------
        QualityReport
            Quality report
        """
        report = QualityReport(
            timestamp=time.time(),
            total_samples=len(data)
        )
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Check missing values
        report.missing_values = self._check_missing_values(data)
        report.missing_rate = sum(report.missing_values.values()) / (data.shape[0] * data.shape[1])
        
        # Check outliers
        report.outliers = self._check_outliers(data)
        report.outlier_rate = sum(report.outliers.values()) / (data.shape[0] * data.shape[1])
        
        # Check invalid values
        report.invalid_values = self._check_invalid_values(data)
        report.invalid_rate = sum(report.invalid_values.values()) / (data.shape[0] * data.shape[1])
        
        # Check data types
        report.data_types = self._check_data_types(data)
        
        # Check value ranges
        report.value_ranges = self._check_value_ranges(data)
        
        # Check schema violations
        report.schema_violations = self._check_schema_violations(data)
        
        # Calculate quality score
        report.quality_score = self._calculate_quality_score(report)
        
        # Determine if healthy
        report.is_healthy = (
            report.missing_rate < self.missing_threshold and
            report.outlier_rate < 0.05 and
            report.invalid_rate < 0.01 and
            len(report.schema_violations) == 0
        )
        
        # Add warnings
        if report.missing_rate >= self.missing_threshold:
            report.warnings.append(f"High missing value rate: {report.missing_rate:.2%}")
        if report.outlier_rate >= 0.05:
            report.warnings.append(f"High outlier rate: {report.outlier_rate:.2%}")
        if report.invalid_rate >= 0.01:
            report.warnings.append(f"Invalid values detected: {report.invalid_rate:.2%}")
        if report.schema_violations:
            report.warnings.append(f"Schema violations: {len(report.schema_violations)}")
        
        self.quality_history.append(report)
        self._save_report(report)
        
        return report
    
    def _check_missing_values(self, data: np.ndarray) -> Dict[str, int]:
        """Check for missing values."""
        missing_values = {}
        
        for i in range(data.shape[1]):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_data = data[:, i]
            
            # Check for NaN, inf, or None
            n_missing = np.sum(np.isnan(feature_data) | np.isinf(feature_data))
            if n_missing > 0:
                missing_values[feature_name] = int(n_missing)
        
        return missing_values
    
    def _check_outliers(self, data: np.ndarray) -> Dict[str, int]:
        """Check for outliers using z-score."""
        outliers = {}
        
        for i in range(data.shape[1]):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_data = data[:, i]
            
            # Remove NaN/inf for statistics
            valid_data = feature_data[~(np.isnan(feature_data) | np.isinf(feature_data))]
            
            if len(valid_data) > 0:
                mean = np.mean(valid_data)
                std = np.std(valid_data)
                
                if std > 0:
                    z_scores = np.abs((valid_data - mean) / std)
                    n_outliers = np.sum(z_scores > self.outlier_threshold)
                    if n_outliers > 0:
                        outliers[feature_name] = int(n_outliers)
        
        return outliers
    
    def _check_invalid_values(self, data: np.ndarray) -> Dict[str, int]:
        """Check for invalid values based on feature types."""
        invalid_values = {}
        
        for i in range(data.shape[1]):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_data = data[:, i]
            
            if feature_name in self.feature_types:
                feature_type = self.feature_types[feature_name]
                
                if feature_type == 'positive':
                    n_invalid = np.sum(feature_data < 0)
                    if n_invalid > 0:
                        invalid_values[feature_name] = int(n_invalid)
                elif feature_type == 'probability':
                    n_invalid = np.sum((feature_data < 0) | (feature_data > 1))
                    if n_invalid > 0:
                        invalid_values[feature_name] = int(n_invalid)
                elif feature_type == 'binary':
                    n_invalid = np.sum(~np.isin(feature_data, [0, 1]))
                    if n_invalid > 0:
                        invalid_values[feature_name] = int(n_invalid)
        
        return invalid_values
    
    def _check_data_types(self, data: np.ndarray) -> Dict[str, str]:
        """Check data types."""
        data_types = {}
        
        for i in range(data.shape[1]):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_data = data[:, i]
            
            if np.all(feature_data == feature_data.astype(int)):
                data_types[feature_name] = 'integer'
            else:
                data_types[feature_name] = 'float'
        
        return data_types
    
    def _check_value_ranges(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Check value ranges."""
        value_ranges = {}
        
        for i in range(data.shape[1]):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_data = data[:, i]
            
            # Remove NaN/inf
            valid_data = feature_data[~(np.isnan(feature_data) | np.isinf(feature_data))]
            
            if len(valid_data) > 0:
                value_ranges[feature_name] = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                }
        
        return value_ranges
    
    def _check_schema_violations(self, data: np.ndarray) -> List[str]:
        """Check for schema violations."""
        violations = []
        
        # Check expected number of features
        expected_features = len(self.feature_names) if self.feature_names else None
        if expected_features and data.shape[1] != expected_features:
            violations.append(
                f"Feature count mismatch: expected {expected_features}, got {data.shape[1]}"
            )
        
        # Check value ranges against reference
        if self.reference_statistics:
            for i in range(data.shape[1]):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                
                if feature_name in self.reference_statistics:
                    feature_data = data[:, i]
                    valid_data = feature_data[~(np.isnan(feature_data) | np.isinf(feature_data))]
                    
                    if len(valid_data) > 0:
                        ref_stats = self.reference_statistics[feature_name]
                        curr_min = np.min(valid_data)
                        curr_max = np.max(valid_data)
                        
                        # Check if values are outside reference range with tolerance
                        tolerance = 3 * ref_stats['std']
                        if curr_min < ref_stats['min'] - tolerance:
                            violations.append(
                                f"{feature_name}: minimum value {curr_min:.4f} is outside reference range"
                            )
                        if curr_max > ref_stats['max'] + tolerance:
                            violations.append(
                                f"{feature_name}: maximum value {curr_max:.4f} is outside reference range"
                            )
        
        return violations
    
    def _calculate_quality_score(self, report: QualityReport) -> float:
        """Calculate overall quality score."""
        # Start with perfect score
        score = 1.0
        
        # Penalize missing values
        score -= report.missing_rate * 0.4
        
        # Penalize outliers
        score -= report.outlier_rate * 0.3
        
        # Penalize invalid values
        score -= report.invalid_rate * 0.2
        
        # Penalize schema violations
        score -= len(report.schema_violations) * 0.02
        
        return max(0.0, score)
    
    def _save_report(self, report: QualityReport):
        """Save quality report to disk."""
        report_file = self.storage_path / f"quality_{int(report.timestamp)}.json"
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    
    def get_quality_summary(self, window: int = 100) -> Dict[str, Any]:
        """Get quality summary for recent history."""
        recent_reports = self.quality_history[-window:] if len(self.quality_history) > window else self.quality_history
        
        if not recent_reports:
            return {
                'status': 'no_data',
                'message': 'No quality reports available'
            }
        
        return {
            'status': 'ok',
            'total_reports': len(recent_reports),
            'healthy_rate': sum(r.is_healthy for r in recent_reports) / len(recent_reports),
            'avg_quality_score': np.mean([r.quality_score for r in recent_reports]),
            'avg_missing_rate': np.mean([r.missing_rate for r in recent_reports]),
            'avg_outlier_rate': np.mean([r.outlier_rate for r in recent_reports]),
            'avg_invalid_rate': np.mean([r.invalid_rate for r in recent_reports]),
            'total_warnings': sum(len(r.warnings) for r in recent_reports),
            'recent_reports': [r.to_dict() for r in recent_reports[-10:]]
        }


class DataQualityChecker:
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
        missing_threshold: float = 0.1,
        outlier_threshold: float = 3.0,
        storage_path: Optional[str] = None
    ):
        self._monitor = DataQualityMonitor(
            feature_names=feature_names,
            feature_types=feature_types,
            missing_threshold=missing_threshold,
            outlier_threshold=outlier_threshold,
            storage_path=storage_path
        )
    
    def _to_array(self, data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        try:
            import pandas as pd  # type: ignore
            if isinstance(data, pd.DataFrame):
                return data.values
            if isinstance(data, pd.Series):
                return data.values.reshape(-1, 1)
        except Exception:
            pass
        if isinstance(data, list):
            return np.array(data)
        return np.asarray(data)
    
    def check_missing(self, data: Any) -> Dict[str, int]:
        arr = self._to_array(data)
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        return self._monitor._check_missing_values(arr)
    
    def check_outliers(self, data: Any) -> Dict[str, int]:
        arr = self._to_array(data)
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        return self._monitor._check_outliers(arr)
    
    def check_schema(self, data: Any, schema: Dict[str, str]) -> Dict[str, Any]:
        violations: List[str] = []
        arr = self._to_array(data)
        expected_features = len(schema) if schema else None
        if expected_features and len(arr.shape) > 1 and arr.shape[1] != expected_features:
            violations.append(f"Feature count mismatch: expected {expected_features}, got {arr.shape[1]}")
        return {"compliant": len(violations) == 0, "violations": violations}
    
    def generate_report(self, data: Any) -> Dict[str, Any]:
        arr = self._to_array(data)
        report = self._monitor.check_quality(arr)
        return report.to_dict()
