"""
CLI commands for monitoring functionality.
"""

import json
import os
import sys
import time
from pathlib import Path

import click
import numpy as np


@click.group()
def monitor():
    """Production monitoring and observability commands."""
    pass


@monitor.command('init')
@click.option('--model-name', default='default', help='Model name')
@click.option('--model-version', default='1.0', help='Model version')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--enable-alerting', is_flag=True, default=True, help='Enable alerting')
@click.option('--slack-webhook', help='Slack webhook URL for alerts')
@click.option('--email-smtp', help='SMTP server for email alerts')
@click.option('--email-user', help='Email username')
@click.option('--email-to', multiple=True, help='Email recipients')
def init_monitoring(
    model_name: str,
    model_version: str,
    storage_path: str,
    enable_alerting: bool,
    slack_webhook: str,
    email_smtp: str,
    email_user: str,
    email_to: tuple
):
    """Initialize monitoring for a model."""
    from neural.monitoring.monitor import ModelMonitor
    
    click.echo(f"Initializing monitoring for {model_name} v{model_version}")
    
    alert_config = {}
    if slack_webhook:
        alert_config['slack_webhook'] = slack_webhook
    
    if email_smtp and email_user:
        email_password = click.prompt('Email password', hide_input=True)
        alert_config['email_config'] = {
            'smtp_server': email_smtp,
            'smtp_port': 587,
            'username': email_user,
            'password': email_password,
            'from_addr': email_user,
            'to_addrs': list(email_to) if email_to else [email_user]
        }
    
    monitor = ModelMonitor(
        model_name=model_name,
        model_version=model_version,
        storage_path=storage_path,
        enable_alerting=enable_alerting,
        alert_config=alert_config
    )
    
    config = {
        'model_name': model_name,
        'model_version': model_version,
        'storage_path': storage_path,
        'enable_alerting': enable_alerting,
    }
    
    config_path = Path(storage_path) / 'monitor_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    click.echo(f"✓ Monitoring initialized at {storage_path}")
    click.echo(f"✓ Config saved to {config_path}")


@monitor.command('status')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def monitoring_status(storage_path: str, format: str):
    """Get monitoring status."""
    from neural.monitoring.monitor import ModelMonitor
    
    config_path = Path(storage_path) / 'monitor_config.json'
    if not config_path.exists():
        click.echo(f"Error: Monitoring not initialized. Run 'neural monitor init' first.", err=True)
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    monitor = ModelMonitor(
        model_name=config['model_name'],
        model_version=config['model_version'],
        storage_path=storage_path,
        enable_alerting=config.get('enable_alerting', True)
    )
    
    summary = monitor.get_monitoring_summary()
    
    if format == 'json':
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"\n=== Monitoring Status for {config['model_name']} v{config['model_version']} ===\n")
        
        click.echo(f"Total Predictions: {summary['total_predictions']}")
        click.echo(f"Total Errors: {summary['total_errors']}")
        click.echo(f"Error Rate: {summary['error_rate']:.2%}")
        click.echo(f"Uptime: {summary['uptime_seconds']:.0f} seconds")
        
        if summary['drift']['status'] == 'ok':
            click.echo(f"\nDrift Detection:")
            click.echo(f"  Drift Rate: {summary['drift'].get('drift_rate', 0):.2%}")
            click.echo(f"  Avg Prediction Drift: {summary['drift'].get('avg_prediction_drift', 0):.4f}")
            click.echo(f"  Avg Performance Drift: {summary['drift'].get('avg_performance_drift', 0):.4f}")
        
        if 'alerts' in summary and summary['alerts']['status'] == 'ok':
            click.echo(f"\nAlerts (24h):")
            click.echo(f"  Total: {summary['alerts']['total_alerts']}")
            by_severity = summary['alerts'].get('by_severity', {})
            click.echo(f"  Critical: {by_severity.get('critical', 0)}")
            click.echo(f"  Warning: {by_severity.get('warning', 0)}")
            click.echo(f"  Info: {by_severity.get('info', 0)}")


@monitor.command('drift')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--window', default=100, help='Number of recent samples to analyze')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def drift_report(storage_path: str, window: int, format: str):
    """Get drift detection report."""
    from neural.monitoring.drift_detector import DriftDetector
    
    detector = DriftDetector(storage_path=str(Path(storage_path) / 'drift'))
    report = detector.get_drift_report(window=window)
    
    if format == 'json':
        click.echo(json.dumps(report, indent=2))
    else:
        if report['status'] == 'no_data':
            click.echo("No drift data available")
            return
        
        click.echo(f"\n=== Drift Detection Report ===\n")
        click.echo(f"Total Samples: {report['total_samples']}")
        click.echo(f"Drift Detected: {report['drift_detected']} ({report['drift_rate']:.2%})")
        click.echo(f"Avg Prediction Drift: {report['avg_prediction_drift']:.4f}")
        click.echo(f"Avg Performance Drift: {report['avg_performance_drift']:.4f}")
        click.echo(f"Avg Distribution Drift: {report['avg_distribution_drift']:.4f}")
        
        severity_dist = report['severity_distribution']
        click.echo(f"\nSeverity Distribution:")
        click.echo(f"  Critical: {severity_dist['critical']}")
        click.echo(f"  Warning: {severity_dist['warning']}")
        click.echo(f"  None: {severity_dist['none']}")


@monitor.command('alerts')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--hours', default=24, help='Number of hours to show')
@click.option('--severity', type=click.Choice(['info', 'warning', 'critical']), help='Filter by severity')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def alert_summary(storage_path: str, hours: int, severity: str, format: str):
    """Get alert summary."""
    from neural.monitoring.alerting import AlertManager, AlertSeverity
    
    manager = AlertManager(storage_path=str(Path(storage_path) / 'alerts'))
    summary = manager.get_alert_summary(hours=hours)
    
    if format == 'json':
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"\n=== Alert Summary ({hours}h) ===\n")
        click.echo(f"Total Alerts: {summary['total_alerts']}")
        
        by_severity = summary.get('by_severity', {})
        click.echo(f"\nBy Severity:")
        click.echo(f"  Critical: {by_severity.get('critical', 0)}")
        click.echo(f"  Warning: {by_severity.get('warning', 0)}")
        click.echo(f"  Info: {by_severity.get('info', 0)}")
        
        if summary.get('recent_critical'):
            click.echo(f"\nRecent Critical Alerts:")
            for alert in summary['recent_critical'][-5:]:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert['timestamp']))
                click.echo(f"  [{timestamp}] {alert['title']}")


@monitor.command('dashboard')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--port', default=8052, help='Dashboard port')
@click.option('--host', default='localhost', help='Dashboard host')
def monitoring_dashboard(storage_path: str, port: int, host: str):
    """Start monitoring dashboard."""
    click.echo(f"Starting monitoring dashboard on {host}:{port}")
    click.echo(f"Press Ctrl+C to stop")
    
    try:
        from neural.monitoring.dashboard import create_app
        app = create_app(storage_path=storage_path)
        app.run_server(debug=False, host=host, port=port)
    except ImportError as e:
        click.echo(f"Error: Dashboard dependencies not available: {e}", err=True)
        click.echo("Install with: pip install dash plotly", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped")


@monitor.command('health')
@click.option('--storage-path', default='monitoring_data', help='Storage path for monitoring data')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def health_check(storage_path: str, format: str):
    """Get health report."""
    from neural.monitoring.monitor import ModelMonitor
    
    config_path = Path(storage_path) / 'monitor_config.json'
    if not config_path.exists():
        click.echo(f"Error: Monitoring not initialized. Run 'neural monitor init' first.", err=True)
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    monitor = ModelMonitor(
        model_name=config['model_name'],
        model_version=config['model_version'],
        storage_path=storage_path,
        enable_alerting=config.get('enable_alerting', True)
    )
    
    report = monitor.generate_health_report()
    
    if format == 'json':
        click.echo(json.dumps(report, indent=2))
    else:
        status_icon = {"healthy": "✓", "warning": "⚠", "critical": "✗"}.get(report['status'], "?")
        
        click.echo(f"\n=== Health Report ===\n")
        click.echo(f"Model: {report['model']} v{report['version']}")
        click.echo(f"Status: {status_icon} {report['status'].upper()}")
        click.echo(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['timestamp']))}")
        click.echo(f"\nIssues:")
        for issue in report['issues']:
            click.echo(f"  • {issue}")
        
        sys.exit(0 if report['status'] == 'healthy' else 1)


if __name__ == '__main__':
    monitor()
