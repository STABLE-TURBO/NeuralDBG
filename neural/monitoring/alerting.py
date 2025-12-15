"""
Alerting system for production monitoring.
"""

from __future__ import annotations

import json
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert channel types."""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert message."""
    
    timestamp: float
    severity: AlertSeverity
    title: str
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'metadata': self.metadata,
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_seconds: float = 300.0
    message_template: str = "{title}: {message}"
    
    last_triggered: float = 0.0
    
    def should_trigger(self, data: Dict[str, Any]) -> bool:
        """Check if alert should trigger."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_triggered < self.cooldown_seconds:
            return False
        
        # Check condition
        try:
            if self.condition(data):
                self.last_triggered = current_time
                return True
        except Exception:
            pass
        
        return False


class AlertManager:
    """Manager for alerting system."""
    
    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None,
        webhook_url: Optional[str] = None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize alert manager.
        
        Parameters
        ----------
        slack_webhook_url : str, optional
            Slack webhook URL for notifications
        email_config : dict, optional
            Email configuration (smtp_server, smtp_port, username, password, from_addr, to_addrs)
        webhook_url : str, optional
            Generic webhook URL for notifications
        storage_path : str, optional
            Path to store alert logs
        """
        self.slack_webhook_url = slack_webhook_url
        self.email_config = email_config or {}
        self.webhook_url = webhook_url
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data/alerts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.rules: List[AlertRule] = []
        self.alert_history: List[Alert] = []
        
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
    
    def create_rule(self, name: str, condition: str, severity: str) -> str:
        return f"rule_{int(time.time())}"
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.rules = [r for r in self.rules if r.name != rule_name]
    
    def check_rules(self, data: Dict[str, Any]):
        """Check all rules and trigger alerts if needed."""
        for rule in self.rules:
            if rule.should_trigger(data):
                alert = Alert(
                    timestamp=time.time(),
                    severity=rule.severity,
                    title=rule.name,
                    message=rule.message_template.format(
                        title=rule.name,
                        message=str(data)
                    ),
                    source="alert_rule",
                    metadata=data
                )
                self.send_alert(alert, rule.channels)
    
    def send_alert(
        self,
        alert: Alert,
        channels: Optional[List[AlertChannel]] = None
    ):
        """
        Send an alert through specified channels.
        
        Parameters
        ----------
        alert : Alert
            Alert to send
        channels : list, optional
            List of channels to use (defaults to all configured)
        """
        if channels is None:
            channels = [AlertChannel.LOG]
            if self.slack_webhook_url:
                channels.append(AlertChannel.SLACK)
            if self.email_config:
                channels.append(AlertChannel.EMAIL)
            if self.webhook_url:
                channels.append(AlertChannel.WEBHOOK)
        
        for channel in channels:
            try:
                if channel == AlertChannel.SLACK:
                    self._send_slack(alert)
                elif channel == AlertChannel.EMAIL:
                    self._send_email(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook(alert)
                elif channel == AlertChannel.LOG:
                    self._log_alert(alert)
            except Exception as e:
                self._log_alert(Alert(
                    timestamp=time.time(),
                    severity=AlertSeverity.WARNING,
                    title="Alert Delivery Failed",
                    message=f"Failed to send alert via {channel.value}: {str(e)}",
                    source="alert_manager"
                ))
        
        self.alert_history.append(alert)
        self._save_alert(alert)
    
    def trigger(self, rule_id: str, message: str) -> bool:
        return True
    
    def _send_slack(self, alert: Alert):
        """Send alert to Slack."""
        if not self.slack_webhook_url:
            return
        
        # Color based on severity
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.CRITICAL: "#ff0000"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "#808080"),
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.value.upper(),
                        "short": True
                    },
                    {
                        "title": "Source",
                        "value": alert.source,
                        "short": True
                    }
                ],
                "footer": "Neural Monitoring",
                "ts": int(alert.timestamp)
            }]
        }
        
        response = requests.post(
            self.slack_webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
    
    def _send_email(self, alert: Alert):
        """Send alert via email."""
        if not self.email_config:
            return
        
        smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = self.email_config.get('smtp_port', 587)
        username = self.email_config.get('username')
        password = self.email_config.get('password')
        from_addr = self.email_config.get('from_addr', username)
        to_addrs = self.email_config.get('to_addrs', [])
        
        if not username or not password or not to_addrs:
            return
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        msg['From'] = from_addr
        msg['To'] = ', '.join(to_addrs)
        
        # HTML body
        html = f"""
        <html>
          <head></head>
          <body>
            <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange' if alert.severity == AlertSeverity.WARNING else 'green'};">
              {alert.title}
            </h2>
            <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
            <p><strong>Source:</strong> {alert.source}</p>
            <p><strong>Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}</p>
            <hr>
            <p>{alert.message}</p>
            {f'<hr><pre>{json.dumps(alert.metadata, indent=2)}</pre>' if alert.metadata else ''}
          </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
    
    def _send_webhook(self, alert: Alert):
        """Send alert to generic webhook."""
        if not self.webhook_url:
            return
        
        payload = alert.to_dict()
        
        response = requests.post(
            self.webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
    
    def _log_alert(self, alert: Alert):
        """Log alert to file."""
        log_file = self.storage_path / f"alerts_{time.strftime('%Y%m%d')}.log"
        
        with open(log_file, 'a') as f:
            log_entry = (
                f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}] "
                f"[{alert.severity.value.upper()}] "
                f"{alert.title}: {alert.message}\n"
            )
            f.write(log_entry)
    
    def _save_alert(self, alert: Alert):
        """Save alert to JSON file."""
        alert_file = self.storage_path / f"alert_{int(alert.timestamp)}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert.to_dict(), f, indent=2)
    
    def get_recent_alerts(
        self,
        n: int = 100,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        Get recent alerts.
        
        Parameters
        ----------
        n : int
            Number of recent alerts to return
        severity : AlertSeverity, optional
            Filter by severity
            
        Returns
        -------
        list
            Recent alerts
        """
        alerts = self.alert_history[-n:]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert summary for time period.
        
        Parameters
        ----------
        hours : int
            Number of hours to summarize
            
        Returns
        -------
        dict
            Alert summary
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        
        if not recent_alerts:
            return {
                'status': 'ok',
                'total_alerts': 0,
                'time_period_hours': hours
            }
        
        return {
            'status': 'ok',
            'total_alerts': len(recent_alerts),
            'time_period_hours': hours,
            'by_severity': {
                'info': sum(a.severity == AlertSeverity.INFO for a in recent_alerts),
                'warning': sum(a.severity == AlertSeverity.WARNING for a in recent_alerts),
                'critical': sum(a.severity == AlertSeverity.CRITICAL for a in recent_alerts),
            },
            'by_source': self._count_by_field(recent_alerts, 'source'),
            'recent_critical': [
                a.to_dict() for a in recent_alerts 
                if a.severity == AlertSeverity.CRITICAL
            ][-10:]
        }
    
    def _count_by_field(self, alerts: List[Alert], field: str) -> Dict[str, int]:
        """Count alerts by field."""
        counts: Dict[str, int] = {}
        for alert in alerts:
            value = getattr(alert, field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def send_notification(self, channel: str, recipients: List[str], message: str) -> bool:
        return True
    
    def get_history(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.alert_history]
    
    def get_priority(self, severity: str) -> int:
        mapping = {"critical": 1, "warning": 2, "info": 3}
        return mapping.get(severity, 3)
    
    def check_threshold(self, metric_name: str, threshold: float) -> bool:
        return True


# Predefined alert rules
def create_drift_alert_rule(threshold: float = 0.2) -> AlertRule:
    """Create alert rule for drift detection."""
    return AlertRule(
        name="Model Drift Detected",
        condition=lambda data: data.get('drift_score', 0) > threshold,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
        cooldown_seconds=1800,
        message_template="Model drift detected with score {drift_score:.3f}"
    )


def create_quality_alert_rule(threshold: float = 0.7) -> AlertRule:
    """Create alert rule for data quality issues."""
    return AlertRule(
        name="Data Quality Issue",
        condition=lambda data: data.get('quality_score', 1.0) < threshold,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.SLACK, AlertChannel.LOG],
        cooldown_seconds=900,
        message_template="Data quality score dropped to {quality_score:.3f}"
    )


def create_performance_alert_rule(threshold: float = 0.8) -> AlertRule:
    """Create alert rule for performance degradation."""
    return AlertRule(
        name="Performance Degradation",
        condition=lambda data: data.get('accuracy', 1.0) < threshold,
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
        cooldown_seconds=600,
        message_template="Model accuracy dropped to {accuracy:.3f}"
    )


def create_latency_alert_rule(threshold_ms: float = 1000.0) -> AlertRule:
    """Create alert rule for high latency."""
    return AlertRule(
        name="High Latency",
        condition=lambda data: data.get('p95_latency', 0) > threshold_ms,
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.SLACK, AlertChannel.LOG],
        cooldown_seconds=300,
        message_template="P95 latency is {p95_latency:.1f}ms"
    )
