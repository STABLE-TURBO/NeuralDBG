import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Optional
import numpy as np


def create_layer_profiling_view(layer_stats: Dict[str, Any]) -> go.Figure:
    if not layer_stats:
        return go.Figure()
    
    layers = list(layer_stats.keys())
    mean_times = [stats.get('mean_time', 0) * 1000 for stats in layer_stats.values()]
    std_times = [stats.get('std_time', 0) * 1000 for stats in layer_stats.values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=layers,
        y=mean_times,
        error_y=dict(type='data', array=std_times),
        name='Execution Time',
        marker_color='rgb(55, 83, 109)',
    ))
    
    fig.update_layout(
        title='Layer-by-Layer Execution Time',
        xaxis_title='Layer',
        yaxis_title='Time (ms)',
        template='plotly_dark',
        height=500,
        showlegend=True,
    )
    
    return fig


def create_memory_profiling_view(memory_snapshots: List[Dict[str, Any]]) -> go.Figure:
    if not memory_snapshots:
        return go.Figure()
    
    timestamps = [s['timestamp'] for s in memory_snapshots]
    cpu_memory = [s['cpu_memory_mb'] for s in memory_snapshots]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cpu_memory,
        mode='lines+markers',
        name='CPU Memory',
        line=dict(color='rgb(0, 176, 246)', width=2),
    ))
    
    if 'gpu_memory_mb' in memory_snapshots[0]:
        gpu_memory = [s.get('gpu_memory_mb', 0) for s in memory_snapshots]
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=gpu_memory,
            mode='lines+markers',
            name='GPU Memory',
            line=dict(color='rgb(231, 107, 243)', width=2),
        ))
    
    fig.update_layout(
        title='Memory Usage Timeline',
        xaxis_title='Timestamp',
        yaxis_title='Memory (MB)',
        template='plotly_dark',
        height=500,
        showlegend=True,
    )
    
    return fig


def create_bottleneck_view(bottlenecks: List[Dict[str, Any]]) -> go.Figure:
    if not bottlenecks:
        fig = go.Figure()
        fig.add_annotation(
            text="No bottlenecks detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(template='plotly_dark', height=400)
        return fig
    
    layers = [b['layer'] for b in bottlenecks[:10]]
    overhead = [b['overhead_percentage'] for b in bottlenecks[:10]]
    severity = [b['severity_score'] for b in bottlenecks[:10]]
    
    colors = ['red' if s > 100 else 'orange' if s > 50 else 'yellow' for s in severity]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=layers,
        x=overhead,
        orientation='h',
        marker_color=colors,
        text=[f"{o:.1f}%" for o in overhead],
        textposition='outside',
    ))
    
    fig.update_layout(
        title='Top Bottlenecks by Overhead',
        xaxis_title='Overhead (%)',
        yaxis_title='Layer',
        template='plotly_dark',
        height=500,
    )
    
    return fig


def create_comparative_profiling_view(comparison: Dict[str, Any]) -> go.Figure:
    if not comparison or 'backends' not in comparison:
        return go.Figure()
    
    backends = comparison['backends']
    layer_comparisons = comparison.get('layer_comparisons', {})
    
    if not layer_comparisons:
        return go.Figure()
    
    layers = list(layer_comparisons.keys())[:15]
    
    fig = go.Figure()
    
    for backend in backends:
        times = []
        for layer in layers:
            layer_comp = layer_comparisons[layer]
            backend_data = layer_comp.get('backend_data', {})
            if backend in backend_data:
                times.append(backend_data[backend]['mean_time'] * 1000)
            else:
                times.append(0)
        
        fig.add_trace(go.Bar(
            name=backend,
            x=layers,
            y=times,
        ))
    
    fig.update_layout(
        title='Backend Comparison: Layer Execution Times',
        xaxis_title='Layer',
        yaxis_title='Time (ms)',
        barmode='group',
        template='plotly_dark',
        height=500,
    )
    
    return fig


def create_gpu_utilization_view(gpu_metrics: List[Dict[str, Any]]) -> go.Figure:
    if not gpu_metrics:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GPU Utilization', 'Memory Usage', 'Temperature', 'Power Usage'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    timestamps = list(range(len(gpu_metrics)))
    
    gpu_util = [m.get('gpu_utilization_percent', 0) for m in gpu_metrics]
    if any(u is not None for u in gpu_util):
        fig.add_trace(
            go.Scatter(x=timestamps, y=gpu_util, name='GPU Util %', 
                      line=dict(color='cyan')),
            row=1, col=1
        )
    
    memory = [m.get('memory_allocated_mb', 0) for m in gpu_metrics]
    fig.add_trace(
        go.Scatter(x=timestamps, y=memory, name='Memory MB',
                  line=dict(color='magenta')),
        row=1, col=2
    )
    
    temps = [m.get('temperature_celsius') for m in gpu_metrics]
    if any(t is not None for t in temps):
        fig.add_trace(
            go.Scatter(x=timestamps, y=temps, name='Temp °C',
                      line=dict(color='orange')),
            row=2, col=1
        )
    
    power = [m.get('power_watts') for m in gpu_metrics]
    if any(p is not None for p in power):
        fig.add_trace(
            go.Scatter(x=timestamps, y=power, name='Power W',
                      line=dict(color='yellow')),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text='GPU Utilization Metrics',
        template='plotly_dark',
        height=700,
        showlegend=False,
    )
    
    return fig


def create_memory_leak_view(leak_summary: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    
    if not leak_summary.get('has_leaks', False):
        fig.add_annotation(
            text="No memory leaks detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color='green')
        )
    else:
        leaks = leak_summary.get('leaks', [])
        if leaks:
            indices = list(range(len(leaks)))
            increases = [leak['increase_mb'] for leak in leaks]
            
            fig.add_trace(go.Bar(
                x=indices,
                y=increases,
                marker_color='red',
                text=[f"+{inc:.1f}MB" for inc in increases],
                textposition='outside',
            ))
            
            fig.update_layout(
                title=f'Memory Leaks Detected: {len(leaks)} instances',
                xaxis_title='Leak Instance',
                yaxis_title='Memory Increase (MB)',
            )
    
    fig.update_layout(template='plotly_dark', height=400)
    return fig


def create_distributed_profiling_view(distributed_metrics: Dict[str, Any]) -> go.Figure:
    if not distributed_metrics:
        return go.Figure()
    
    load_balance = distributed_metrics.get('load_balance', {})
    
    if not load_balance or 'node_times' not in load_balance:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Node Execution Times', 'Load Balance Analysis'),
        row_heights=[0.6, 0.4]
    )
    
    node_times = load_balance['node_times']
    nodes = list(node_times.keys())
    times = list(node_times.values())
    
    colors = ['red' if node == load_balance.get('slowest_node') else 
              'green' if node == load_balance.get('fastest_node') else 
              'blue' for node in nodes]
    
    fig.add_trace(
        go.Bar(x=nodes, y=times, marker_color=colors, name='Execution Time'),
        row=1, col=1
    )
    
    mean_time = load_balance.get('mean_time', 0)
    fig.add_hline(y=mean_time, line_dash="dash", line_color="yellow", 
                  annotation_text="Mean", row=1, col=1)
    
    metrics = ['Imbalance Factor', 'Std Dev']
    values = [load_balance.get('imbalance_factor', 0), 
              load_balance.get('std_time', 0)]
    
    fig.add_trace(
        go.Bar(x=metrics, y=values, marker_color='orange'),
        row=2, col=1
    )
    
    fig.update_layout(
        title_text='Distributed Training Analysis',
        template='plotly_dark',
        height=700,
        showlegend=False,
    )
    
    return fig


def create_recommendations_view(recommendations: List[Dict[str, Any]]) -> go.Figure:
    if not recommendations:
        fig = go.Figure()
        fig.add_annotation(
            text="No recommendations at this time",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(template='plotly_dark', height=400)
        return fig
    
    categories = {}
    for rec in recommendations:
        cat = rec.get('category', 'other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(rec)
    
    category_counts = {cat: len(recs) for cat, recs in categories.items()}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(category_counts.keys()),
            y=list(category_counts.values()),
            marker_color='lightblue',
            text=list(category_counts.values()),
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title='Optimization Recommendations by Category',
        xaxis_title='Category',
        yaxis_title='Count',
        template='plotly_dark',
        height=400,
    )
    
    return fig


def create_execution_timeline(execution_history: List[Dict[str, Any]]) -> go.Figure:
    if not execution_history:
        return go.Figure()
    
    layers = [e['layer'] for e in execution_history]
    times = [e['execution_time'] * 1000 for e in execution_history]
    cumulative = [e.get('cumulative_time', 0) * 1000 for e in execution_history]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=list(range(len(layers))), y=times, name='Execution Time',
              marker_color='rgb(55, 83, 109)'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(len(layers))), y=cumulative, 
                  name='Cumulative Time', line=dict(color='red', width=2)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Execution Step")
    fig.update_yaxes(title_text="Time (ms)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Time (ms)", secondary_y=True)
    
    fig.update_layout(
        title='Execution Timeline',
        template='plotly_dark',
        height=500,
    )
    
    return fig
