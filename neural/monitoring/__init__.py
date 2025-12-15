"""
Production monitoring and observability system for Neural DSL.

This module provides core monitoring capabilities for DSL debugging:
- Model performance drift detection
- Basic alerting system (Slack/email/webhook/log)
- Dashboard integration for visualization
"""

from neural.monitoring.drift_detector import DriftDetector, DriftMetrics
from neural.monitoring.alerting import AlertManager, AlertRule, AlertChannel
from neural.monitoring.monitor import ModelMonitor

__all__ = [
    'DriftDetector',
    'DriftMetrics',
    'AlertManager',
    'AlertRule',
    'AlertChannel',
    'ModelMonitor',
]
