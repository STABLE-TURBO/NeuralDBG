import click
import json
from pathlib import Path
from typing import Optional

from .profiler_manager import ProfilerManager
from .utils import format_profiling_report


@click.group()
def profiling_cli():
    """Profiling and performance analysis commands."""
    pass


@profiling_cli.command()
@click.argument('report_file', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
def show_report(report_file, format):
    """Display a profiling report."""
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    if format == 'json':
        click.echo(json.dumps(report, indent=2))
    else:
        click.echo(format_profiling_report(report))


@profiling_cli.command()
@click.argument('report_file', type=click.Path(exists=True))
@click.option('--top-n', default=10, help='Number of top items to show')
def bottlenecks(report_file, top_n):
    """Show bottleneck analysis from a report."""
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    if 'bottlenecks' not in report:
        click.echo("No bottleneck data in report")
        return
    
    bottleneck_data = report['bottlenecks']
    
    if 'bottlenecks' in bottleneck_data:
        click.echo("\nTop Bottlenecks:")
        click.echo("=" * 80)
        
        for i, b in enumerate(bottleneck_data['bottlenecks'][:top_n], 1):
            click.echo(f"\n{i}. {b['layer']}")
            click.echo(f"   Type: {', '.join(b['bottleneck_type'])}")
            click.echo(f"   Mean Time: {b['mean_execution_time']*1000:.2f}ms")
            click.echo(f"   Overhead: {b['overhead_percentage']:.1f}%")
            click.echo(f"   Severity Score: {b['severity_score']:.2f}")
    
    if 'recommendations' in bottleneck_data:
        click.echo("\n\nRecommendations:")
        click.echo("=" * 80)
        
        for i, rec in enumerate(bottleneck_data['recommendations'][:top_n], 1):
            priority_color = 'red' if rec['priority'] == 'high' else 'yellow'
            click.secho(f"\n{i}. [{rec['priority'].upper()}] {rec['category'].upper()}", 
                       fg=priority_color, bold=True)
            click.echo(f"   {rec['recommendation']}")
            click.echo(f"   {rec['details']}")


@profiling_cli.command()
@click.argument('report_file', type=click.Path(exists=True))
def memory_analysis(report_file):
    """Show memory profiling analysis."""
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    click.echo("\nMemory Analysis:")
    click.echo("=" * 80)
    
    if 'memory_profiling' in report:
        mem = report['memory_profiling']
        click.echo(f"\nBaseline Memory: {mem.get('baseline_memory_mb', 0):.2f}MB")
        click.echo(f"Peak Memory: {mem.get('peak_memory_mb', 0):.2f}MB")
        
        if 'peak_gpu_memory_mb' in mem:
            click.echo(f"Peak GPU Memory: {mem['peak_gpu_memory_mb']:.2f}MB")
    
    if 'memory_leaks' in report:
        leaks = report['memory_leaks']
        
        if leaks.get('has_leaks'):
            click.secho(f"\n⚠ WARNING: {leaks['leak_count']} memory leak(s) detected!", 
                       fg='red', bold=True)
            click.echo(f"Total Leaked: {leaks.get('total_memory_leaked_mb', 0):.2f}MB")
            click.echo(f"Average Growth Rate: {leaks.get('average_growth_rate', 0):.4f}MB/step")
            
            if 'leaks' in leaks:
                click.echo("\nLeak Details:")
                for i, leak in enumerate(leaks['leaks'][:5], 1):
                    click.echo(f"\n  {i}. Increase: {leak['increase_mb']:.2f}MB")
                    click.echo(f"     Growth Rate: {leak['growth_rate_mb_per_step']:.4f}MB/step")
                    click.echo(f"     Layers: {', '.join(leak['layers_involved'][:3])}...")
        else:
            click.secho("\n✓ No memory leaks detected", fg='green')


@profiling_cli.command()
@click.argument('report1', type=click.Path(exists=True))
@click.argument('report2', type=click.Path(exists=True))
def compare(report1, report2):
    """Compare two profiling reports."""
    with open(report1, 'r') as f:
        r1 = json.load(f)
    with open(report2, 'r') as f:
        r2 = json.load(f)
    
    from .utils import compare_profiling_sessions
    comparison = compare_profiling_sessions(r1, r2)
    
    click.echo("\nProfiling Report Comparison:")
    click.echo("=" * 80)
    
    if 'total_time' in comparison.get('improvements', {}):
        time_imp = comparison['improvements']['total_time']
        speedup = time_imp['speedup_percent']
        
        click.echo(f"\nExecution Time:")
        click.echo(f"  Report 1: {time_imp['session1_ms']:.2f}ms")
        click.echo(f"  Report 2: {time_imp['session2_ms']:.2f}ms")
        
        if speedup > 0:
            click.secho(f"  Improvement: {speedup:.1f}% faster", fg='green', bold=True)
        else:
            click.secho(f"  Change: {abs(speedup):.1f}% slower", fg='red', bold=True)
    
    if 'memory' in comparison.get('improvements', {}):
        mem_imp = comparison['improvements']['memory']
        reduction = mem_imp['reduction_percent']
        
        click.echo(f"\nMemory Usage:")
        click.echo(f"  Report 1: {mem_imp['session1_mb']:.2f}MB")
        click.echo(f"  Report 2: {mem_imp['session2_mb']:.2f}MB")
        
        if reduction > 0:
            click.secho(f"  Improvement: {reduction:.1f}% reduction", fg='green', bold=True)
        else:
            click.secho(f"  Change: {abs(reduction):.1f}% increase", fg='red', bold=True)


@profiling_cli.command()
@click.argument('report_file', type=click.Path(exists=True))
def gpu_stats(report_file):
    """Show GPU utilization statistics."""
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    if 'gpu_utilization' not in report:
        click.echo("No GPU utilization data in report")
        return
    
    gpu = report['gpu_utilization']
    
    click.echo("\nGPU Utilization Statistics:")
    click.echo("=" * 80)
    
    if gpu.get('torch_available'):
        click.echo(f"\nDevice Count: {gpu.get('device_count', 0)}")
        
        if 'device_names' in gpu:
            click.echo(f"Devices: {', '.join(gpu['device_names'])}")
        
        if 'mean_gpu_utilization_percent' in gpu:
            util = gpu['mean_gpu_utilization_percent']
            color = 'green' if util > 70 else 'yellow' if util > 40 else 'red'
            click.secho(f"\nMean GPU Utilization: {util:.1f}%", fg=color, bold=True)
            click.echo(f"Min: {gpu.get('min_gpu_utilization_percent', 0):.1f}%")
            click.echo(f"Max: {gpu.get('max_gpu_utilization_percent', 0):.1f}%")
        
        click.echo(f"\nMemory:")
        click.echo(f"  Mean Allocated: {gpu.get('mean_memory_allocated_mb', 0):.2f}MB")
        click.echo(f"  Peak Allocated: {gpu.get('peak_memory_allocated_mb', 0):.2f}MB")
        
        if 'mean_temperature_celsius' in gpu:
            click.echo(f"\nTemperature:")
            click.echo(f"  Mean: {gpu['mean_temperature_celsius']:.1f}°C")
            click.echo(f"  Max: {gpu.get('max_temperature_celsius', 0):.1f}°C")
        
        if 'mean_power_watts' in gpu:
            click.echo(f"\nPower:")
            click.echo(f"  Mean: {gpu['mean_power_watts']:.1f}W")
            click.echo(f"  Max: {gpu.get('max_power_watts', 0):.1f}W")
        
        if 'utilization_warning' in gpu:
            click.secho(f"\n⚠ {gpu['utilization_warning']}", fg='yellow', bold=True)
    else:
        click.echo("GPU not available or PyTorch not installed")
    
    if 'gpu_recommendations' in report and report['gpu_recommendations']:
        click.echo("\nRecommendations:")
        for rec in report['gpu_recommendations']:
            click.secho(f"  [{rec['priority'].upper()}] {rec['recommendation']}", 
                       fg='yellow' if rec['priority'] == 'high' else 'white')


@profiling_cli.command()
@click.argument('report_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for summary')
def summary(report_file, output):
    """Generate a summary of the profiling report."""
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    summary_text = format_profiling_report(report)
    
    if output:
        with open(output, 'w') as f:
            f.write(summary_text)
        click.echo(f"Summary written to {output}")
    else:
        click.echo(summary_text)


if __name__ == '__main__':
    profiling_cli()
