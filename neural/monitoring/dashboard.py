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
    
    # Apply security if available
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
    
    # Load config
    config_path = storage_path / 'monitor_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config.get('model_name', 'Unknown')
        model_version = config.get('model_version', '1.0')
    else:
        model_name = 'Unknown'
        model_version = '1.0'
    
    # Layout
    app.layout = html.Div([
        html.Div([
            html.H1("Neural Monitoring Dashboard", style={'color': '#00ff00'}),
            html.H3(f"Model: {model_name} v{model_version}", style={'color': '#888'}),
        ], style={'textAlign': 'center', 'padding': '20px'}),
        
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # Update every 5 seconds
            n_intervals=0
        ),
        
        # Status Cards
        html.Div([
            html.Div([
                html.H4("Total Predictions", style={'color': '#fff'}),
                html.H2(id='total-predictions', style={'color': '#00ff00'}),
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': '#222', 'margin': '5px'}),
            
            html.Div([
                html.H4("Error Rate", style={'color': '#fff'}),
                html.H2(id='error-rate', style={'color': '#ff9900'}),
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': '#222', 'margin': '5px'}),
            
            html.Div([
                html.H4("Drift Score", style={'color': '#fff'}),
                html.H2(id='drift-score', style={'color': '#ff0000'}),
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': '#222', 'margin': '5px'}),
            
            html.Div([
                html.H4("Quality Score", style={'color': '#fff'}),
                html.H2(id='quality-score', style={'color': '#00ffff'}),
            ], style={'width': '24%', 'display': 'inline-block', 'padding': '10px', 'backgroundColor': '#222', 'margin': '5px'}),
        ], style={'textAlign': 'center'}),
        
        # Charts
        html.Div([
            html.Div([
                dcc.Graph(id='drift-chart'),
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                dcc.Graph(id='quality-chart'),
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(id='latency-chart'),
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                dcc.Graph(id='slo-chart'),
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
        
        # Alerts
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
            Output('quality-score', 'children'),
            Output('drift-chart', 'figure'),
            Output('quality-chart', 'figure'),
            Output('latency-chart', 'figure'),
            Output('slo-chart', 'figure'),
            Output('alerts-list', 'children'),
        ],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        """Update dashboard data."""
        from neural.monitoring.monitor import ModelMonitor
        
        try:
            # Create monitor
            monitor = ModelMonitor(
                model_name=model_name,
                model_version=model_version,
                storage_path=str(storage_path),
                enable_prometheus=False,
                enable_alerting=True,
                enable_slo_tracking=True
            )
            
            # Get summary
            summary = monitor.get_monitoring_summary()
            
            # Status values
            total_preds = summary.get('total_predictions', 0)
            error_rate = f"{summary.get('error_rate', 0):.2%}"
            
            drift_report = summary.get('drift', {})
            drift_score = "N/A"
            if drift_report.get('status') == 'ok':
                avg_drift = drift_report.get('avg_distribution_drift', 0)
                drift_score = f"{avg_drift:.3f}"
            
            quality_report = summary.get('quality', {})
            quality_score_val = "N/A"
            if quality_report.get('status') == 'ok':
                avg_quality = quality_report.get('avg_quality_score', 1.0)
                quality_score_val = f"{avg_quality:.3f}"
            
            # Drift chart
            drift_fig = create_drift_chart(drift_report)
            
            # Quality chart
            quality_fig = create_quality_chart(quality_report)
            
            # Latency chart
            pred_analysis = summary.get('predictions', {})
            latency_fig = create_latency_chart(pred_analysis)
            
            # SLO chart
            slo_status = summary.get('slos', {})
            slo_fig = create_slo_chart(slo_status)
            
            # Alerts
            alert_summary = summary.get('alerts', {})
            alerts_html = create_alerts_html(alert_summary)
            
            return (
                total_preds,
                error_rate,
                drift_score,
                quality_score_val,
                drift_fig,
                quality_fig,
                latency_fig,
                slo_fig,
                alerts_html
            )
        
        except Exception as e:
            # Return empty/error states
            return (
                "Error",
                "Error",
                "Error",
                "Error",
                go.Figure(),
                go.Figure(),
                go.Figure(),
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


def create_quality_chart(quality_report: dict) -> go.Figure:
    """Create quality chart."""
    fig = go.Figure()
    
    if quality_report.get('status') == 'ok':
        recent_reports = quality_report.get('recent_reports', [])
        
        if recent_reports:
            timestamps = [r['timestamp'] for r in recent_reports]
            quality_scores = [r['quality_score'] for r in recent_reports]
            missing_rates = [r['missing_rate'] for r in recent_reports]
            outlier_rates = [r['outlier_rate'] for r in recent_reports]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=quality_scores,
                mode='lines',
                name='Quality Score',
                line=dict(color='#00ff00')
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=missing_rates,
                mode='lines',
                name='Missing Rate',
                line=dict(color='#ff9900'),
                yaxis='y2'
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=outlier_rates,
                mode='lines',
                name='Outlier Rate',
                line=dict(color='#ff0000'),
                yaxis='y2'
            ))
    
    fig.update_layout(
        title='Data Quality',
        xaxis_title='Time',
        yaxis_title='Quality Score',
        yaxis2=dict(
            title='Rate',
            overlaying='y',
            side='right'
        ),
        plot_bgcolor='#222',
        paper_bgcolor='#222',
        font=dict(color='#fff'),
        hovermode='x unified'
    )
    
    return fig


def create_latency_chart(pred_analysis: dict) -> go.Figure:
    """Create latency chart."""
    fig = go.Figure()
    
    if pred_analysis.get('status') == 'ok' and 'latency' in pred_analysis:
        latency = pred_analysis['latency']
        
        fig.add_trace(go.Bar(
            x=['Mean', 'Median', 'P95', 'P99'],
            y=[
                latency.get('mean', 0),
                latency.get('median', 0),
                latency.get('p95', 0),
                latency.get('p99', 0)
            ],
            marker_color='#00ffff'
        ))
    
    fig.update_layout(
        title='Prediction Latency (ms)',
        xaxis_title='Metric',
        yaxis_title='Latency (ms)',
        plot_bgcolor='#222',
        paper_bgcolor='#222',
        font=dict(color='#fff')
    )
    
    return fig


def create_slo_chart(slo_status: dict) -> go.Figure:
    """Create SLO chart."""
    fig = go.Figure()
    
    if slo_status:
        slo_names = []
        compliance_rates = []
        targets = []
        colors = []
        
        for slo_name, status in slo_status.items():
            if status.get('status') == 'ok':
                slo_names.append(slo_name)
                compliance_rates.append(status.get('compliance_rate', 0) * 100)
                targets.append(status.get('slo', {}).get('target', 0) * 100)
                
                # Color based on meeting SLO
                if status.get('is_meeting', True):
                    colors.append('#00ff00')
                else:
                    colors.append('#ff0000')
        
        if slo_names:
            fig.add_trace(go.Bar(
                x=slo_names,
                y=compliance_rates,
                name='Actual',
                marker_color=colors
            ))
            
            fig.add_trace(go.Scatter(
                x=slo_names,
                y=targets,
                mode='markers',
                name='Target',
                marker=dict(color='#fff', size=12, symbol='line-ew-open')
            ))
    
    fig.update_layout(
        title='SLO Compliance',
        xaxis_title='SLO',
        yaxis_title='Compliance (%)',
        plot_bgcolor='#222',
        paper_bgcolor='#222',
        font=dict(color='#fff'),
        yaxis=dict(range=[0, 105])
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
    app = create_app()
    
    ssl_context = None
    if SECURITY_AVAILABLE:
        security_config = load_security_config()
        if security_config.ssl_enabled and security_config.ssl_cert_file and security_config.ssl_key_file:
            ssl_context = (security_config.ssl_cert_file, security_config.ssl_key_file)
    
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8053,
        ssl_context=ssl_context
    )
