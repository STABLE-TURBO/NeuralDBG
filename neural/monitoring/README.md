# Neural Monitoring & Observability

Focused monitoring system for Neural DSL models with drift detection and basic alerting.

## Features

### 🔍 Model Performance Drift Detection
- Multiple statistical methods (KS test, PSI, Wasserstein distance)
- Feature-level drift tracking
- Prediction distribution drift
- Performance metrics drift
- Concept drift detection with combined scoring

### 🚨 Alerting System
- Multiple channels: Slack, Email, Webhook, Log
- Configurable alert rules
- Severity levels (Info, Warning, Critical)
- Cooldown periods to prevent alert storms
- Predefined rules for common issues

### 💻 Dashboard UI
- Real-time monitoring visualization
- Drift charts
- Recent alerts display
- Auto-refresh capability

## Installation

Install with monitoring dependencies:

```bash
pip install -e ".[full]"
# Or specific monitoring dependencies
pip install dash plotly dash-bootstrap-components
```

## Quick Start

### 1. Initialize Monitoring

```bash
neural monitor init \
  --model-name my-model \
  --model-version 1.0 \
  --enable-alerting \
  --slack-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 2. Set Reference Data (Python)

```python
import numpy as np
from neural.monitoring import ModelMonitor

# Initialize monitor
monitor = ModelMonitor(
    model_name="my-model",
    model_version="1.0",
    storage_path="monitoring_data",
    enable_alerting=True,
    alert_config={
        'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
    }
)

# Set reference data for drift detection
reference_data = np.random.randn(1000, 10)
reference_predictions = np.random.randint(0, 2, 1000)
reference_performance = {'accuracy': 0.95, 'f1': 0.94}

monitor.set_reference_data(
    data=reference_data,
    predictions=reference_predictions,
    performance=reference_performance
)
```

### 3. Check Drift

```python
# Check for drift with new data
new_data = np.random.randn(100, 10)
new_predictions = np.random.randint(0, 2, 100)
new_performance = {'accuracy': 0.92, 'f1': 0.91}

drift_report = monitor.check_drift(
    data=new_data,
    predictions=new_predictions,
    performance=new_performance
)

print(f"Drift detected: {drift_report['is_drifting']}")
print(f"Drift severity: {drift_report['drift_severity']}")
```

### 4. View Status

```bash
# CLI commands
neural monitor status --format text
neural monitor drift --window 100
neural monitor alerts --hours 24
neural monitor health
```

### 5. Start Dashboard

```bash
# Start monitoring dashboard
neural monitor dashboard --port 8052
```

## CLI Commands

### Initialize
```bash
neural monitor init [OPTIONS]
```
Options:
- `--model-name`: Model name (default: default)
- `--model-version`: Model version (default: 1.0)
- `--storage-path`: Storage path (default: monitoring_data)
- `--enable-alerting`: Enable alerting
- `--slack-webhook`: Slack webhook URL
- `--email-smtp`: SMTP server
- `--email-user`: Email username
- `--email-to`: Email recipients (multiple)

### Status
```bash
neural monitor status [OPTIONS]
```
Options:
- `--storage-path`: Storage path (default: monitoring_data)
- `--format`: Output format (text|json)

### Drift Report
```bash
neural monitor drift [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--window`: Number of samples to analyze (default: 100)
- `--format`: Output format (text|json)

### Alert Summary
```bash
neural monitor alerts [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--hours`: Time window in hours (default: 24)
- `--severity`: Filter by severity (info|warning|critical)
- `--format`: Output format (text|json)

### Health Check
```bash
neural monitor health [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--format`: Output format (text|json)

### Dashboard
```bash
neural monitor dashboard [OPTIONS]
```
Options:
- `--storage-path`: Storage path
- `--port`: Dashboard port (default: 8052)
- `--host`: Dashboard host (default: localhost)

## Architecture

```
neural/monitoring/
├── __init__.py                 # Module exports
├── monitor.py                  # Main ModelMonitor class
├── drift_detector.py           # Drift detection
├── alerting.py                 # Alert management
├── cli_commands.py             # CLI commands
├── dashboard.py                # Dashboard UI
└── README.md                   # This file
```

## Alerting Configuration

### Slack Integration

```python
alert_config = {
    'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
}
```

### Email Integration

```python
alert_config = {
    'email_config': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',
        'from_addr': 'your-email@gmail.com',
        'to_addrs': ['recipient@example.com']
    }
}
```

### Custom Webhook

```python
alert_config = {
    'webhook_url': 'https://your-webhook-endpoint.com/alerts'
}
```

### Custom Alert Rules

```python
from neural.monitoring import AlertManager, AlertRule, AlertSeverity, AlertChannel

alert_manager = AlertManager(alert_config=alert_config)

# Add custom rule
custom_rule = AlertRule(
    name="High Error Rate",
    condition=lambda data: data.get('error_rate', 0) > 0.05,
    severity=AlertSeverity.CRITICAL,
    channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
    cooldown_seconds=600,
    message_template="Error rate is {error_rate:.2%}"
)

alert_manager.add_rule(custom_rule)
```

## Best Practices

1. **Set Reference Data**: Always set reference data before monitoring production traffic
2. **Monitor Gradually**: Start with logging, then add drift detection, then alerting
3. **Tune Thresholds**: Adjust drift thresholds based on your model's behavior
4. **Alert Wisely**: Use cooldown periods to prevent alert fatigue
5. **Dashboard Monitoring**: Keep the dashboard visible for real-time monitoring

## Troubleshooting

### Missing Dependencies

If you get import errors:
```bash
pip install dash plotly dash-bootstrap-components
```

### Alert Delivery Issues

- Check webhook URLs are correct
- Verify email credentials
- Check network connectivity
- Review alert cooldown settings

## License

MIT License - see LICENSE.md for details.
