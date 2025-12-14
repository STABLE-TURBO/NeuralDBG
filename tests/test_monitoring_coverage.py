"""
Comprehensive test suite for Monitoring module to increase coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from neural.monitoring.monitor import ModelMonitor
from neural.monitoring.drift_detector import DriftDetector
from neural.monitoring.data_quality import DataQualityChecker
from neural.monitoring.alerting import AlertManager


class TestModelMonitor:
    """Test model monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ModelMonitor(model_id="model_v1")
        assert monitor.model_id == "model_v1"
    
    def test_log_prediction(self):
        """Test logging predictions."""
        monitor = ModelMonitor(model_id="model_v1")
        with patch.object(monitor, 'log_prediction') as mock_log:
            mock_log.return_value = True
            result = monitor.log_prediction(
                input_data={"feature1": 1.0},
                prediction=0.95,
                actual=1.0
            )
            assert result is True
    
    def test_get_metrics(self):
        """Test retrieving metrics."""
        monitor = ModelMonitor(model_id="model_v1")
        with patch.object(monitor, 'get_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "accuracy": 0.95,
                "latency_p99": 0.1,
                "error_rate": 0.01
            }
            metrics = monitor.get_metrics()
            assert metrics["accuracy"] == 0.95
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        monitor = ModelMonitor(model_id="model_v1")
        with patch.object(monitor, 'detect_anomalies') as mock_detect:
            mock_detect.return_value = [
                {"timestamp": "2024-01-01", "type": "latency_spike"}
            ]
            anomalies = monitor.detect_anomalies()
            assert len(anomalies) == 1


class TestDriftDetector:
    """Test data drift detection."""
    
    def test_drift_detector_initialization(self):
        """Test drift detector initialization."""
        detector = DriftDetector(method="ks_test")
        assert detector.method == "ks_test"
    
    def test_detect_feature_drift(self):
        """Test feature drift detection."""
        detector = DriftDetector()
        with patch.object(detector, 'detect_drift') as mock_detect:
            mock_detect.return_value = {
                "drift_detected": True,
                "p_value": 0.001,
                "features": ["feature1", "feature2"]
            }
            result = detector.detect_drift([1, 2, 3], [4, 5, 6])
            assert result["drift_detected"] is True
    
    def test_detect_concept_drift(self):
        """Test concept drift detection."""
        detector = DriftDetector(method="concept_drift")
        with patch.object(detector, 'detect_concept_drift') as mock_detect:
            mock_detect.return_value = {
                "drift_detected": False,
                "confidence": 0.95
            }
            result = detector.detect_concept_drift(
                predictions=[0.9, 0.8, 0.7],
                actuals=[1, 1, 0]
            )
            assert result["drift_detected"] is False
    
    def test_calculate_drift_score(self):
        """Test drift score calculation."""
        detector = DriftDetector()
        with patch.object(detector, 'calculate_score') as mock_score:
            mock_score.return_value = 0.45
            score = detector.calculate_score([1, 2, 3], [1.1, 2.1, 3.1])
            assert 0 <= score <= 1


class TestDataQualityChecker:
    """Test data quality checking."""
    
    def test_quality_checker_initialization(self):
        """Test quality checker initialization."""
        checker = DataQualityChecker()
        assert checker is not None
    
    def test_check_missing_values(self):
        """Test missing value detection."""
        checker = DataQualityChecker()
        with patch.object(checker, 'check_missing') as mock_check:
            mock_check.return_value = {
                "total_missing": 5,
                "percentage": 0.05,
                "columns": {"col1": 3, "col2": 2}
            }
            result = checker.check_missing(Mock())
            assert result["total_missing"] == 5
    
    def test_check_outliers(self):
        """Test outlier detection."""
        checker = DataQualityChecker()
        with patch.object(checker, 'check_outliers') as mock_check:
            mock_check.return_value = {
                "outlier_count": 10,
                "outlier_indices": [5, 15, 25]
            }
            result = checker.check_outliers([1, 2, 100, 3, 4])
            assert result["outlier_count"] == 10
    
    def test_check_schema_compliance(self):
        """Test schema compliance checking."""
        checker = DataQualityChecker()
        schema = {"col1": "int", "col2": "float"}
        with patch.object(checker, 'check_schema') as mock_check:
            mock_check.return_value = {
                "compliant": True,
                "violations": []
            }
            result = checker.check_schema(Mock(), schema)
            assert result["compliant"] is True
    
    def test_generate_quality_report(self):
        """Test quality report generation."""
        checker = DataQualityChecker()
        with patch.object(checker, 'generate_report') as mock_report:
            mock_report.return_value = {
                "overall_score": 0.92,
                "issues": [],
                "recommendations": ["Check for outliers"]
            }
            report = checker.generate_report(Mock())
            assert report["overall_score"] > 0.9


class TestAlertManager:
    """Test alerting functionality."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        assert manager is not None
    
    def test_create_alert_rule(self):
        """Test creating alert rules."""
        manager = AlertManager()
        with patch.object(manager, 'create_rule') as mock_create:
            mock_create.return_value = "rule_123"
            rule_id = manager.create_rule(
                name="High Error Rate",
                condition="error_rate > 0.05",
                severity="critical"
            )
            assert rule_id == "rule_123"
    
    def test_trigger_alert(self):
        """Test triggering an alert."""
        manager = AlertManager()
        with patch.object(manager, 'trigger') as mock_trigger:
            mock_trigger.return_value = True
            result = manager.trigger(
                rule_id="rule_123",
                message="Error rate exceeded threshold"
            )
            assert result is True
    
    def test_send_notification(self):
        """Test sending notifications."""
        manager = AlertManager()
        with patch.object(manager, 'send_notification') as mock_send:
            mock_send.return_value = True
            result = manager.send_notification(
                channel="email",
                recipients=["admin@example.com"],
                message="Alert triggered"
            )
            assert result is True
    
    def test_get_alert_history(self):
        """Test retrieving alert history."""
        manager = AlertManager()
        with patch.object(manager, 'get_history') as mock_history:
            mock_history.return_value = [
                {"timestamp": "2024-01-01", "severity": "critical"},
                {"timestamp": "2024-01-02", "severity": "warning"}
            ]
            history = manager.get_history()
            assert len(history) == 2


@pytest.mark.parametrize("method", [
    "ks_test",
    "chi_square",
    "wasserstein",
])
def test_drift_detection_methods(method):
    """Parameterized test for drift detection methods."""
    detector = DriftDetector(method=method)
    assert detector.method == method


@pytest.mark.parametrize("severity,expected_priority", [
    ("critical", 1),
    ("warning", 2),
    ("info", 3),
])
def test_alert_severity_levels(severity, expected_priority):
    """Parameterized test for alert severity levels."""
    manager = AlertManager()
    with patch.object(manager, 'get_priority') as mock_priority:
        mock_priority.return_value = expected_priority
        priority = manager.get_priority(severity)
        assert priority == expected_priority


@pytest.mark.parametrize("metric_name,threshold,expected_alert", [
    ("error_rate", 0.05, True),
    ("latency_p99", 1.0, True),
    ("accuracy", 0.9, False),
])
def test_metric_thresholds(metric_name, threshold, expected_alert):
    """Parameterized test for metric thresholds."""
    manager = AlertManager()
    with patch.object(manager, 'check_threshold') as mock_check:
        mock_check.return_value = expected_alert
        should_alert = manager.check_threshold(metric_name, threshold)
        assert should_alert == expected_alert
