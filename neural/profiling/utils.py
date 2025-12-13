from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional
import functools
import time


def profile_function(profiler_manager, layer_name: str, metadata: Optional[Dict[str, Any]] = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler_manager.profile_layer(layer_name, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def timed_execution(label: str = ""):
    start_time = time.perf_counter()
    result = {'execution_time': 0}
    
    try:
        yield result
    finally:
        result['execution_time'] = time.perf_counter() - start_time
        if label:
            print(f"{label}: {result['execution_time']*1000:.2f}ms")


def format_profiling_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("PROFILING REPORT")
    lines.append("=" * 80)
    
    if 'profiling_session' in report:
        session = report['profiling_session']
        lines.append("\nSession Information:")
        if session.get('duration_seconds'):
            lines.append(f"  Duration: {session['duration_seconds']:.2f}s")
    
    if 'layer_profiling' in report:
        layer_prof = report['layer_profiling']
        lines.append("\nLayer Profiling Summary:")
        lines.append(f"  Total Layers: {layer_prof.get('total_layers', 0)}")
        lines.append(f"  Total Time: {layer_prof.get('total_time', 0)*1000:.2f}ms")
        lines.append(f"  Total Calls: {layer_prof.get('total_calls', 0)}")
        
        if 'slowest_layers' in layer_prof:
            lines.append("\n  Slowest Layers:")
            for layer, time_val in layer_prof['slowest_layers'][:5]:
                lines.append(f"    {layer}: {time_val*1000:.2f}ms")
    
    if 'memory_profiling' in report:
        mem_prof = report['memory_profiling']
        lines.append("\nMemory Profiling:")
        if 'peak_memory_mb' in mem_prof:
            lines.append(f"  Peak Memory: {mem_prof['peak_memory_mb']:.2f}MB")
        if 'baseline_memory_mb' in mem_prof:
            lines.append(f"  Baseline Memory: {mem_prof['baseline_memory_mb']:.2f}MB")
    
    if 'memory_leaks' in report:
        leak_info = report['memory_leaks']
        if leak_info.get('has_leaks'):
            lines.append("\nMemory Leak Detection:")
            lines.append(f"  WARNING: {leak_info.get('leak_count', 0)} potential leak(s) detected")
            lines.append(f"  Total Leaked: {leak_info.get('total_memory_leaked_mb', 0):.2f}MB")
    
    if 'bottlenecks' in report:
        bottleneck_info = report['bottlenecks']
        if 'summary' in bottleneck_info:
            summary = bottleneck_info['summary']
            lines.append("\nBottleneck Analysis:")
            lines.append(f"  Total Bottlenecks: {summary.get('total_bottlenecks', 0)}")
            lines.append(f"  Critical Bottlenecks: {summary.get('critical_bottlenecks', 0)}")
            
            if 'top_bottlenecks' in summary:
                lines.append("\n  Top Bottlenecks:")
                for b in summary['top_bottlenecks']:
                    lines.append(f"    {b['layer']}: {b['overhead_percentage']:.1f}% overhead")
        
        if 'recommendations' in bottleneck_info and bottleneck_info['recommendations']:
            lines.append("\n  Recommendations:")
            for rec in bottleneck_info['recommendations'][:5]:
                lines.append(f"    [{rec.get('priority', 'medium').upper()}] {rec.get('recommendation', 'N/A')}")
    
    if 'gpu_utilization' in report:
        gpu_info = report['gpu_utilization']
        lines.append("\nGPU Utilization:")
        if 'mean_gpu_utilization_percent' in gpu_info:
            lines.append(f"  Mean Utilization: {gpu_info['mean_gpu_utilization_percent']:.1f}%")
        if 'peak_memory_allocated_mb' in gpu_info:
            lines.append(f"  Peak Memory: {gpu_info['peak_memory_allocated_mb']:.2f}MB")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def export_profiling_data_to_json(profiler_manager, filepath: str):
    profiler_manager.export_report(filepath)


def compare_profiling_sessions(session1: Dict[str, Any], session2: Dict[str, Any]) -> Dict[str, Any]:
    comparison = {
        'session1': session1.get('profiling_session', {}),
        'session2': session2.get('profiling_session', {}),
        'improvements': {},
    }
    
    if 'layer_profiling' in session1 and 'layer_profiling' in session2:
        time1 = session1['layer_profiling'].get('total_time', 0)
        time2 = session2['layer_profiling'].get('total_time', 0)
        
        if time1 > 0:
            speedup = (time1 - time2) / time1 * 100
            comparison['improvements']['total_time'] = {
                'session1_ms': time1 * 1000,
                'session2_ms': time2 * 1000,
                'speedup_percent': speedup,
            }
    
    if 'memory_profiling' in session1 and 'memory_profiling' in session2:
        mem1 = session1['memory_profiling'].get('peak_memory_mb', 0)
        mem2 = session2['memory_profiling'].get('peak_memory_mb', 0)
        
        if mem1 > 0:
            reduction = (mem1 - mem2) / mem1 * 100
            comparison['improvements']['memory'] = {
                'session1_mb': mem1,
                'session2_mb': mem2,
                'reduction_percent': reduction,
            }
    
    return comparison
