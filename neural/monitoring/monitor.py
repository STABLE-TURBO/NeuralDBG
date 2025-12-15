"""
Main monitoring integration for production ML systems.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from neural.monitoring.alerting import (
    AlertManager,
    AlertSeverity,
    Alert,
    create_drift_alert_rule,
)
from neural.monitoring.drift_detector import DriftDetector


class ModelMonitor:
    """Monitoring for ML models with drift detection and alerting."""
    
    def __init__(
        self,
        model_name: str = "default",
        model_version: str = "1.0",
        model_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        enable_alerting: bool = True,
        alert_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model monitor.
        
        Parameters
        ----------
        model_name : str
            Name of the model being monitored
        model_version : str
            Version of the model
        model_id : str, optional
            Unique model identifier
        storage_path : str, optional
            Base path for storing monitoring data
        enable_alerting : bool
            Enable alerting system
        alert_config : dict, optional
            Configuration for alerting (slack_webhook, email_config, etc.)
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model_id = model_id or model_name
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.drift_detector = DriftDetector(
            storage_path=str(self.storage_path / "drift")
        )
        
        if enable_alerting:
            alert_config = alert_config or {}
            self.alert_manager = AlertManager(
                slack_webhook_url=alert_config.get('slack_webhook'),
                email_config=alert_config.get('email_config'),
                webhook_url=alert_config.get('webhook_url'),
                storage_path=str(self.storage_path / "alerts")
            )
            self._setup_default_alert_rules()
        else:
            self.alert_manager = None
        
        self.start_time = time.time()
        self.total_predictions = 0
        self.total_errors = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = {
            "error_rate": self.total_errors / self.total_predictions if self.total_predictions > 0 else 0.0,
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "uptime_seconds": time.time() - self.start_time
        }
        
        if hasattr(self.drift_detector, 'get_current_metrics'):
            drift_metrics = self.drift_detector.get_current_metrics()
            if drift_metrics:
                metrics.update(drift_metrics)
        
        return metrics
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in monitoring data."""
        anomalies = []
        
        drift_report = self.drift_detector.get_drift_report(window=100)
        if drift_report.get('status') == 'ok' and drift_report.get('drift_rate', 0) > 0.2:
            anomalies.append({
                'type': 'drift',
                'severity': 'warning',
                'message': f"High drift rate: {drift_report['drift_rate']:.2%}",
                'timestamp': time.time()
            })
        
        if self.total_predictions > 0:
            error_rate = self.total_errors / self.total_predictions
            if error_rate > 0.05:
                anomalies.append({
                    'type': 'error_rate',
                    'severity': 'critical',
                    'message': f"High error rate: {error_rate:.2%}",
                    'timestamp': time.time()
                })
        
        return anomalies
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        if not self.alert_manager:
            return
        
        self.alert_manager.add_rule(create_drift_alert_rule(threshold=0.2))
    
    def set_reference_data(
        self,
        data: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        performance: Optional[Dict[str, float]] = None
    ):
        """
        Set reference data for monitoring.
        
        Parameters
        ----------
        data : np.ndarray
            Reference input data
        predictions : np.ndarray, optional
            Reference predictions
        performance : dict, optional
            Reference performance metrics
        """
        self.drift_detector.set_reference(data, predictions, performance)
    
    def check_drift(
        self,
        data: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        performance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Check for model drift.
        
        Parameters
        ----------
        data : np.ndarray
            New data samples
        predictions : np.ndarray, optional
            Model predictions
        performance : dict, optional
            Current performance metrics
            
        Returns
        -------
        dict
            Drift detection results
        """
        drift_metrics = self.drift_detector.detect_drift(data, predictions, performance)
        
        if self.alert_manager and drift_metrics.is_drifting:
            self.alert_manager.check_rules({
                'drift_score': drift_metrics.concept_drift_score,
                'prediction_drift': drift_metrics.prediction_drift,
                'performance_drift': drift_metrics.performance_drift,
            })
        
        return drift_metrics.to_dict()
    
    def record_prediction(self):
        """Record a prediction."""
        self.total_predictions += 1
    
    def record_error(self, error_type: str = "unknown"):
        """Record an error."""
        self.total_errors += 1
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.
        
        Returns
        -------
        dict
            Monitoring summary
        """
        summary = {
            'model': {
                'name': self.model_name,
                'version': self.model_version,
            },
            'uptime_seconds': time.time() - self.start_time,
            'total_predictions': self.total_predictions,
            'total_errors': self.total_errors,
            'error_rate': self.total_errors / self.total_predictions if self.total_predictions > 0 else 0.0,
        }
        
        summary['drift'] = self.drift_detector.get_drift_report(window=100)
        
        if self.alert_manager:
            summary['alerts'] = self.alert_manager.get_alert_summary(hours=24)
        
        return summary
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send a custom alert."""
        if not self.alert_manager:
            return
        
        alert = Alert(
            timestamp=time.time(),
            severity=severity,
            title=title,
            message=message,
            source="custom",
            metadata=metadata or {}
        )
        
        self.alert_manager.send_alert(alert)
    
    def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate health report.
        
        Returns
        -------
        dict
            Health report
        """
        report = {
            'timestamp': time.time(),
            'model': self.model_name,
            'version': self.model_version,
            'status': 'healthy',
            'issues': []
        }
        
        drift_report = self.drift_detector.get_drift_report(window=100)
        if drift_report.get('status') == 'ok' and drift_report.get('drift_rate', 0) > 0.2:
            report['status'] = 'warning'
            report['issues'].append(f"High drift rate: {drift_report['drift_rate']:.2%}")
        
        if self.total_predictions > 0:
            error_rate = self.total_errors / self.total_predictions
            if error_rate > 0.05:
                report['status'] = 'critical'
                report['issues'].append(f"High error rate: {error_rate:.2%}")
            elif error_rate > 0.01:
                report['status'] = 'warning' if report['status'] == 'healthy' else report['status']
                report['issues'].append(f"Elevated error rate: {error_rate:.2%}")
        
        if not report['issues']:
            report['issues'].append("All systems operational")
        
        return report
