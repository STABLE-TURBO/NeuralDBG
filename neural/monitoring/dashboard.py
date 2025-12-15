"""
Monitoring dashboard for production ML systems.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import dash
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

try:
    import dash_bootstrap_components as dbc
    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False
    dbc = None

try:
    from neural.security import (
        load_security_config,
        create_basic_auth,
        create_jwt_auth,
        apply_security_middleware,
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False


def create_app(storage_path: str = "monitoring_data") -> dash.Dash:
    """
    Create monitoring dashboard app.
    
    Parameters
    ----------
    storage_path : str
        Path to monitoring data
        
    Returns
    -------
    dash.Dash
        Dash application
    """
    if HAS_BOOTSTRAP:
        app = dash.Dash(
            __name__,
            title="Neural Monitoring Dashboard",
            external_stylesheets=[dbc.themes.DARKLY]
        )
    else:
        app = dash.Dash(
            __name__,
            title="Neural Monitoring Dashboard"
        )
    
    if SECURITY_AVAILABLE:
        security_config = load_security_config()
        server = app.server
        
        apply_security_middleware(
            server,
            cors_enabled=security_config.cors_enabled,
            cors_origins=security_config.cors_origins,
            cors_methods=security_config.cors_methods,
            cors_allow_headers=security_config.cors_allow_headers,
            cors_allow_credentials=security_config.cors_allow_credentials,
            rate_limit_enabled=security_config.rate_limit_enabled,
            rate_limit_requests=security_config.rate_limit_requests,
            rate_limit_window_seconds=security_config.rate_limit_window_seconds,
            security_headers_enabled=security_config.security_headers_enabled,
        )
        
        auth_middleware = None
        if security_config.auth_enabled:
            if security_config.auth_type == 'jwt' and security_config.jwt_secret_key:
                auth_middleware = create_jwt_auth(
                    security_config.jwt_secret_key,
                    security_config.jwt_algorithm,
                    security_config.jwt_expiration_hours
                )
            elif security_config.auth_type == 'basic':
                auth_middleware = create_basic_auth(
                    security_config.basic_auth_username,
                    security_config.basic_auth_password
                )
        
        if auth_middleware:
            @server.before_request
            def check_auth():
                from flask import request
                if request.path.startswith('/_dash'):
                    auth_data = auth_middleware.get_auth_data(request)
                    if not auth_data or not auth_middleware.check_auth(auth_data):
                        return auth_middleware.authenticate()
    
    storage_path = Path(storage_path)
    
    config_path = storage_path / 'monitor_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config.get('model_name', 'Unknown')
        model_version = config.get('model_version', '1.0')
    else:
        model_name = 'Unknown'
        model_version = '1.0'
    
    app.layout = html.Div([
        html.Div([
            html.H1("Neural Monitoring Dashboard", style={'color': '#00ff00'}),
            html.H3(f"Model: {model_name} v{model_version}", style={'color': '#888'}),
        ], style={'textAlign': 'center', 'padding': '20px'}),
        
        dcc.Interval(
            id='interval-component',
            interval=5*1000,
            n_intervals=0
        ),
        
        html.Div([
            html.Div([
                html.H4("Total Predictions", style={'color': '#fff'}),
                html.H2(id='total-predictions', style={'color': '#00ff00'}),
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': '#222', 'margin': '5px'}),
            
            html.Div([
                html.H4("Error Rate", style={'color': '#fff'}),
                html.H2(id='error-rate', style={'color': '#ff9900'}),
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': '#222', 'margin': '5px'}),
            
            html.Div([
                html.H4("Drift Score", style={'color': '#fff'}),
                html.H2(id='drift-score', style={'color': '#ff0000'}),
            ], style={'width': '32%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': '#222', 'margin': '5px'}),
        ], style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='drift-chart'),
            ], style={'width': '100%', 'padding': '10px'}),
        ]),
        
        html.Div([
            html.H3("Recent Alerts", style={'color': '#fff', 'padding': '10px'}),
            html.Div(id='alerts-list', style={'padding': '10px'}),
        ], style={'backgroundColor': '#222', 'margin': '10px'}),
        
    ], style={'backgroundColor': '#111', 'minHeight': '100vh', 'color': '#fff'})
    
    @app.callback(
        [
            Output('total-predictions', 'children'),
            Output('error-rate', 'children'),
            Output('drift-score', 'children'),
            Output('drift-chart', 'figure'),
            Output('alerts-list', 'children'),
        ],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        """Update dashboard data."""
        from neural.monitoring.monitor import ModelMonitor
        
        try:
            monitor = ModelMonitor(
                model_name=model_name,
                model_version=model_version,
                storage_path=str(storage_path),
                enable_alerting=True
            )
            
            summary = monitor.get_monitoring_summary()
            
            total_preds = summary.get('total_predictions', 0)
            error_rate = f"{summary.get('error_rate', 0):.2%}"
            
            drift_report = summary.get('drift', {})
            drift_score = "N/A"
            if drift_report.get('status') == 'ok':
                avg_drift = drift_report.get('avg_distribution_drift', 0)
                drift_score = f"{avg_drift:.3f}"
            
            drift_fig = create_drift_chart(drift_report)
            
            alert_summary = summary.get('alerts', {})
            alerts_html = create_alerts_html(alert_summary)
            
            return (
                total_preds,
                error_rate,
                drift_score,
                drift_fig,
                alerts_html
            )
        
        except Exception as e:
            return (
                "Error",
                "Error",
                "Error",
                go.Figure(),
                html.Div(f"Error loading data: {str(e)}", style={'color': '#ff0000'})
            )
    
    return app


def create_drift_chart(drift_report: dict) -> go.Figure:
    """Create drift chart."""
    fig = go.Figure()
    
    if drift_report.get('status') == 'ok':
        recent_metrics = drift_report.get('recent_metrics', [])
        
        if recent_metrics:
            timestamps = [m['timestamp'] for m in recent_metrics]
            pred_drift = [m['prediction_drift'] for m in recent_metrics]
            perf_drift = [m['performance_drift'] for m in recent_metrics]
            dist_drift = [m['data_distribution_drift'] for m in recent_metrics]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=pred_drift,
                mode='lines',
                name='Prediction Drift',
                line=dict(color='#ff9900')
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=perf_drift,
                mode='lines',
                name='Performance Drift',
                line=dict(color='#ff0000')
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=dist_drift,
                mode='lines',
                name='Distribution Drift',
                line=dict(color='#00ffff')
            ))
    
    fig.update_layout(
        title='Drift Detection',
        xaxis_title='Time',
        yaxis_title='Drift Score',
        plot_bgcolor='#222',
        paper_bgcolor='#222',
        font=dict(color='#fff'),
        hovermode='x unified'
    )
    
    return fig


def create_alerts_html(alert_summary: dict) -> html.Div:
    """Create alerts HTML."""
    if alert_summary.get('status') != 'ok':
        return html.Div("No alerts data available", style={'color': '#888'})
    
    recent_critical = alert_summary.get('recent_critical', [])
    
    if not recent_critical:
        return html.Div("No recent critical alerts", style={'color': '#00ff00'})
    
    alert_items = []
    for alert in recent_critical[-10:]:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert['timestamp']))
        
        alert_items.append(html.Div([
            html.Span(f"[{timestamp}] ", style={'color': '#888'}),
            html.Span(f"{alert['title']}: ", style={'color': '#ff0000', 'fontWeight': 'bold'}),
            html.Span(alert['message'], style={'color': '#fff'}),
        ], style={'padding': '5px', 'borderBottom': '1px solid #333'}))
    
    return html.Div(alert_items)


if __name__ == '__main__':
    import warnings
    warnings.warn(
        "\n" + "="*70 + "\n"
        "⚠️  DEPRECATION WARNING\n"
        "="*70 + "\n"
        "Running neural/monitoring/dashboard.py directly is deprecated.\n"
        "Use the unified server instead:\n\n"
        "  neural server start\n\n"
        "The unified server provides monitoring in the Monitor tab,\n"
        "along with Debug and Build features in a single interface.\n\n"
        "This entry point will be removed in v0.4.0.\n"
        "See neural/dashboard/DEPRECATED.md for migration guide.\n"
        + "="*70,
        DeprecationWarning,
        stacklevel=2
    )
    print("\n" + "⚠️  DEPRECATION WARNING: Use 'neural server start' instead\n")
    app = create_app()
    
    ssl_context = None
    if SECURITY_AVAILABLE:
        security_config = load_security_config()
        if security_config.ssl_enabled and security_config.ssl_cert_file and security_config.ssl_key_file:
            ssl_context = (security_config.ssl_cert_file, security_config.ssl_key_file)
    
    server = app.server
    server.run(
        debug=True,
        host='0.0.0.0',
        port=8053,
        ssl_context=ssl_context
    )
