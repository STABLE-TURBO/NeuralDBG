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
    create_quality_alert_rule,
    create_performance_alert_rule,
    create_latency_alert_rule,
)
from neural.monitoring.data_quality import DataQualityMonitor
from neural.monitoring.drift_detector import DriftDetector
from neural.monitoring.prediction_logger import PredictionLogger, PredictionAnalyzer
from neural.monitoring.prometheus_exporter import MetricsRegistry, PrometheusExporter
from neural.monitoring.slo_tracker import (
    SLOTracker,
    create_availability_slo,
    create_latency_slo,
    create_accuracy_slo,
    create_error_rate_slo,
)


class ModelMonitor:
    """Comprehensive production monitoring for ML models."""
    
    def __init__(
        self,
        model_name: str = "default",
        model_version: str = "1.0",
        model_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        enable_prometheus: bool = True,
        enable_alerting: bool = True,
        enable_slo_tracking: bool = True,
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
        storage_path : str, optional
            Base path for storing monitoring data
        enable_prometheus : bool
            Enable Prometheus metrics export
        enable_alerting : bool
            Enable alerting system
        enable_slo_tracking : bool
            Enable SLO/SLA tracking
        alert_config : dict, optional
            Configuration for alerting (slack_webhook, email_config, etc.)
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model_id = model_id or model_name
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.drift_detector = DriftDetector(
            storage_path=str(self.storage_path / "drift")
        )
        
        self.quality_monitor = DataQualityMonitor(
            storage_path=str(self.storage_path / "quality")
        )
        
        self.prediction_logger = PredictionLogger(
            storage_path=str(self.storage_path / "predictions")
        )
        
        self.prediction_analyzer = PredictionAnalyzer(self.prediction_logger)
        
        # Prometheus metrics
        if enable_prometheus:
            self.metrics_registry = MetricsRegistry()
            self.prometheus_exporter = PrometheusExporter(registry=self.metrics_registry)
        else:
            self.metrics_registry = None
            self.prometheus_exporter = None
        
        # Alerting
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
        
        # SLO tracking
        if enable_slo_tracking:
            self.slo_tracker = SLOTracker(
                storage_path=str(self.storage_path / "slo")
            )
            self._setup_default_slos()
        else:
            self.slo_tracker = None
        
        self.start_time = time.time()
        self.total_predictions = 0
        self.total_errors = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        return {"accuracy": 0.0, "latency_p99": 0.0, "error_rate": 0.0}
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        return []
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        if not self.alert_manager:
            return
        
        self.alert_manager.add_rule(create_drift_alert_rule(threshold=0.2))
        self.alert_manager.add_rule(create_quality_alert_rule(threshold=0.7))
        self.alert_manager.add_rule(create_performance_alert_rule(threshold=0.8))
        self.alert_manager.add_rule(create_latency_alert_rule(threshold_ms=1000.0))
    
    def _setup_default_slos(self):
        """Setup default SLOs."""
        if not self.slo_tracker:
            return
        
        self.slo_tracker.add_slo(create_availability_slo(target=0.999, window_hours=24))
        self.slo_tracker.add_slo(create_latency_slo(target_ms=100.0, window_hours=1))
        self.slo_tracker.add_slo(create_accuracy_slo(target=0.95, window_hours=24))
        self.slo_tracker.add_slo(create_error_rate_slo(target=0.01, window_hours=1))
    
    def set_reference_data(
        self,
        data: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        performance: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None
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
        feature_names : list, optional
            Names of features
        """
        self.drift_detector.set_reference(data, predictions, performance)
        self.quality_monitor.set_reference_statistics(data)
        
        if feature_names:
            self.quality_monitor.feature_names = feature_names
    
    def log_prediction(
        self,
        prediction_id: str,
        input_features: Dict[str, Any],
        prediction: Any,
        prediction_proba: Optional[Dict[str, float]] = None,
        ground_truth: Optional[Any] = None,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a prediction.
        
        Parameters
        ----------
        prediction_id : str
            Unique identifier for prediction
        input_features : dict
            Input features
        prediction : any
            Model prediction
        prediction_proba : dict, optional
            Prediction probabilities
        ground_truth : any, optional
            Ground truth label
        latency_ms : float
            Prediction latency in milliseconds
        metadata : dict, optional
            Additional metadata
        """
        # Log prediction
        self.prediction_logger.log_prediction(
            prediction_id=prediction_id,
            input_features=input_features,
            prediction=prediction,
            prediction_proba=prediction_proba,
            ground_truth=ground_truth,
            latency_ms=latency_ms,
            metadata=metadata
        )
        
        self.total_predictions += 1
        
        # Update Prometheus metrics
        if self.metrics_registry:
            self.metrics_registry.record_prediction(
                model=self.model_name,
                version=self.model_version,
                latency_seconds=latency_ms / 1000.0
            )
        
        # Update SLO tracking
        if self.slo_tracker and latency_ms > 0:
            try:
                self.slo_tracker.record_measurement(
                    "prediction_latency_p95",
                    latency_ms / 1000.0
                )
            except ValueError:
                pass
    
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
        # Detect drift
        drift_metrics = self.drift_detector.detect_drift(data, predictions, performance)
        
        # Update Prometheus metrics
        if self.metrics_registry:
            self.metrics_registry.update_drift_metrics(
                model=self.model_name,
                version=self.model_version,
                drift_scores={
                    'prediction': drift_metrics.prediction_drift,
                    'performance': drift_metrics.performance_drift,
                    'distribution': drift_metrics.data_distribution_drift,
                    'concept': drift_metrics.concept_drift_score,
                }
            )
        
        # Check alert rules
        if self.alert_manager and drift_metrics.is_drifting:
            self.alert_manager.check_rules({
                'drift_score': drift_metrics.concept_drift_score,
                'prediction_drift': drift_metrics.prediction_drift,
                'performance_drift': drift_metrics.performance_drift,
            })
        
        return drift_metrics.to_dict()
    
    def check_data_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Check data quality.
        
        Parameters
        ----------
        data : np.ndarray
            Data to check
            
        Returns
        -------
        dict
            Quality report
        """
        # Check quality
        quality_report = self.quality_monitor.check_quality(data)
        
        # Update Prometheus metrics
        if self.metrics_registry:
            self.metrics_registry.update_quality_metrics(
                model=self.model_name,
                version=self.model_version,
                quality_score=quality_report.quality_score,
                missing_rate=quality_report.missing_rate,
                outlier_rate=quality_report.outlier_rate
            )
        
        # Check alert rules
        if self.alert_manager and not quality_report.is_healthy:
            self.alert_manager.check_rules({
                'quality_score': quality_report.quality_score,
                'missing_rate': quality_report.missing_rate,
                'outlier_rate': quality_report.outlier_rate,
            })
        
        return quality_report.to_dict()
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """
        Update performance metrics.
        
        Parameters
        ----------
        metrics : dict
            Performance metrics (accuracy, precision, recall, f1, etc.)
        """
        # Update Prometheus metrics
        if self.metrics_registry:
            self.metrics_registry.update_performance_metrics(
                model=self.model_name,
                version=self.model_version,
                accuracy=metrics.get('accuracy'),
                precision=metrics.get('precision'),
                recall=metrics.get('recall'),
                f1=metrics.get('f1')
            )
        
        # Update SLO tracking
        if self.slo_tracker and 'accuracy' in metrics:
            try:
                self.slo_tracker.record_measurement(
                    "model_accuracy",
                    metrics['accuracy']
                )
            except ValueError:
                pass
        
        # Check alert rules
        if self.alert_manager and 'accuracy' in metrics:
            self.alert_manager.check_rules({
                'accuracy': metrics['accuracy'],
                **metrics
            })
    
    def record_error(self, error_type: str = "unknown"):
        """Record an error."""
        self.total_errors += 1
        
        if self.metrics_registry:
            self.metrics_registry.record_error(
                model=self.model_name,
                version=self.model_version,
                error_type=error_type
            )
        
        # Update error rate SLO
        if self.slo_tracker:
            uptime_hours = (time.time() - self.start_time) / 3600
            if uptime_hours > 0:
                error_rate = self.total_errors / self.total_predictions if self.total_predictions > 0 else 0.0
                try:
                    self.slo_tracker.record_measurement("error_rate", error_rate)
                except ValueError:
                    pass
    
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
        
        # Add drift summary
        summary['drift'] = self.drift_detector.get_drift_report(window=100)
        
        # Add quality summary
        summary['quality'] = self.quality_monitor.get_quality_summary(window=100)
        
        # Add prediction analysis
        summary['predictions'] = self.prediction_analyzer.analyze_predictions(window=1000)
        
        # Add SLO status
        if self.slo_tracker:
            summary['slos'] = self.slo_tracker.get_all_slo_status()
        
        # Add alert summary
        if self.alert_manager:
            summary['alerts'] = self.alert_manager.get_alert_summary(hours=24)
        
        return summary
    
    def start_prometheus_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        if not self.prometheus_exporter:
            raise RuntimeError("Prometheus exporter not enabled")
        
        self.prometheus_exporter.start_server()
    
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
        
        # Check drift
        drift_report = self.drift_detector.get_drift_report(window=100)
        if drift_report.get('status') == 'ok' and drift_report.get('drift_rate', 0) > 0.2:
            report['status'] = 'warning'
            report['issues'].append(f"High drift rate: {drift_report['drift_rate']:.2%}")
        
        # Check quality
        quality_report = self.quality_monitor.get_quality_summary(window=100)
        if quality_report.get('status') == 'ok' and quality_report.get('healthy_rate', 1.0) < 0.9:
            report['status'] = 'warning'
            report['issues'].append(f"Low quality rate: {quality_report['healthy_rate']:.2%}")
        
        # Check SLOs
        if self.slo_tracker:
            slo_status = self.slo_tracker.get_all_slo_status()
            for slo_name, status in slo_status.items():
                if status.get('status') == 'ok' and not status.get('is_meeting', True):
                    report['status'] = 'critical' if report['status'] == 'healthy' else report['status']
                    report['issues'].append(f"SLO breach: {slo_name}")
        
        # Check error rate
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
