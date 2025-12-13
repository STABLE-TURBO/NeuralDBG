"""
Example of integrating profiling with Neural DSL model execution.
This shows how to profile a model defined in Neural DSL.
"""

import time
from neural.profiling import ProfilerManager
from neural.profiling.utils import format_profiling_report


def simulate_neural_dsl_model_execution():
    """Simulate execution of a Neural DSL model with profiling."""
    
    # Neural DSL model structure (simulated)
    model_structure = {
        'input': {'shape': [32, 224, 224, 3]},
        'layers': [
            {'name': 'Conv2D_1', 'type': 'Conv2D', 'filters': 64, 'duration': 0.08},
            {'name': 'BatchNorm_1', 'type': 'BatchNormalization', 'duration': 0.02},
            {'name': 'ReLU_1', 'type': 'ReLU', 'duration': 0.01},
            {'name': 'MaxPool_1', 'type': 'MaxPooling2D', 'duration': 0.04},
            {'name': 'Conv2D_2', 'type': 'Conv2D', 'filters': 128, 'duration': 0.12},
            {'name': 'BatchNorm_2', 'type': 'BatchNormalization', 'duration': 0.02},
            {'name': 'ReLU_2', 'type': 'ReLU', 'duration': 0.01},
            {'name': 'MaxPool_2', 'type': 'MaxPooling2D', 'duration': 0.04},
            {'name': 'Flatten', 'type': 'Flatten', 'duration': 0.005},
            {'name': 'Dense_1', 'type': 'Dense', 'units': 512, 'duration': 0.10},
            {'name': 'Dropout_1', 'type': 'Dropout', 'rate': 0.5, 'duration': 0.005},
            {'name': 'Dense_2', 'type': 'Dense', 'units': 256, 'duration': 0.08},
            {'name': 'Dropout_2', 'type': 'Dropout', 'rate': 0.3, 'duration': 0.005},
            {'name': 'Dense_Output', 'type': 'Dense', 'units': 10, 'duration': 0.03},
            {'name': 'Softmax', 'type': 'Softmax', 'duration': 0.01},
        ]
    }
    
    return model_structure


def execute_layer_with_profiling(profiler, layer_config):
    """Execute a single layer with profiling."""
    layer_name = layer_config['name']
    layer_type = layer_config['type']
    duration = layer_config['duration']
    
    # Prepare metadata
    metadata = {
        'layer_type': layer_type,
    }
    
    # Add type-specific metadata
    if 'filters' in layer_config:
        metadata['filters'] = layer_config['filters']
    if 'units' in layer_config:
        metadata['units'] = layer_config['units']
    if 'rate' in layer_config:
        metadata['dropout_rate'] = layer_config['rate']
    
    # Estimate memory usage based on layer type
    if layer_type == 'Conv2D':
        metadata['memory_usage'] = 150 * layer_config.get('filters', 64) / 64
    elif layer_type == 'Dense':
        metadata['memory_usage'] = 100 * layer_config.get('units', 512) / 512
    else:
        metadata['memory_usage'] = 20
    
    # Execute with profiling
    with profiler.profile_layer(layer_name, metadata):
        # Simulate layer computation
        time.sleep(duration)
    
    return True


def main():
    print("=" * 80)
    print("Neural DSL Model Profiling Integration Example")
    print("=" * 80)
    
    # Get model structure
    model = simulate_neural_dsl_model_execution()
    
    print(f"\nModel Input Shape: {model['input']['shape']}")
    print(f"Total Layers: {len(model['layers'])}")
    
    # Initialize profiler with all features
    profiler = ProfilerManager(enable_all=True)
    
    print("\nStarting profiling session...")
    profiler.start_profiling()
    
    # Execute model layers
    print("\nExecuting model layers:")
    for layer_config in model['layers']:
        execute_layer_with_profiling(profiler, layer_config)
        print(f"  ✓ {layer_config['name']} ({layer_config['type']})")
    
    print("\nEnding profiling session...")
    profiler.end_profiling()
    
    # Analyze bottlenecks
    print("\nAnalyzing bottlenecks...")
    bottleneck_analysis = profiler.analyze_bottlenecks()
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = profiler.get_comprehensive_report()
    
    # Display formatted report
    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)
    print(format_profiling_report(report))
    
    # Display bottleneck details
    if bottleneck_analysis.get('bottlenecks'):
        print("\n" + "=" * 80)
        print("BOTTLENECK ANALYSIS")
        print("=" * 80)
        
        summary = bottleneck_analysis.get('summary', {})
        print(f"\nTotal Bottlenecks: {summary.get('total_bottlenecks', 0)}")
        print(f"Critical Bottlenecks: {summary.get('critical_bottlenecks', 0)}")
        print(f"Total Overhead: {summary.get('total_overhead_ms', 0):.2f}ms")
        
        if summary.get('top_bottlenecks'):
            print("\nTop 5 Bottlenecks:")
            for i, b in enumerate(summary['top_bottlenecks'], 1):
                print(f"  {i}. {b['layer']}: {b['overhead_percentage']:.1f}% overhead")
    
    # Display optimization recommendations
    if bottleneck_analysis.get('recommendations'):
        print("\n" + "=" * 80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 80)
        
        high_priority = [r for r in bottleneck_analysis['recommendations'] if r['priority'] == 'high']
        medium_priority = [r for r in bottleneck_analysis['recommendations'] if r['priority'] == 'medium']
        
        if high_priority:
            print("\nHIGH PRIORITY:")
            for i, rec in enumerate(high_priority, 1):
                print(f"\n  {i}. {rec['layer']} ({rec['category']})")
                print(f"     {rec['recommendation']}")
                print(f"     Details: {rec['details']}")
        
        if medium_priority:
            print("\nMEDIUM PRIORITY:")
            for i, rec in enumerate(medium_priority[:3], 1):
                print(f"\n  {i}. {rec['layer']} ({rec['category']})")
                print(f"     {rec['recommendation']}")
    
    # Check for memory leaks
    if report.get('memory_leaks', {}).get('has_leaks'):
        print("\n" + "=" * 80)
        print("MEMORY LEAK WARNING")
        print("=" * 80)
        
        leaks = report['memory_leaks']
        print(f"\n⚠ {leaks['leak_count']} potential memory leak(s) detected!")
        print(f"Total Memory Leaked: {leaks.get('total_memory_leaked_mb', 0):.2f}MB")
        print(f"Average Growth Rate: {leaks.get('average_growth_rate', 0):.4f}MB/step")
    
    # Export reports
    print("\n" + "=" * 80)
    print("EXPORTING REPORTS")
    print("=" * 80)
    
    json_report_path = "neural_dsl_profiling_report.json"
    profiler.export_report(json_report_path)
    print(f"\n✓ Full profiling report: {json_report_path}")
    
    # Get dashboard data
    dashboard_data = profiler.get_dashboard_data()
    print(f"✓ Dashboard data ready: {len(dashboard_data['execution_history'])} execution records")
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    
    print("""
To view profiling results:

1. Start the dashboard:
   python neural/dashboard/dashboard.py

2. Navigate to http://localhost:8050

3. Select profiling tabs to view:
   - Layer Profiling: Execution time analysis
   - Memory Profiling: Memory usage timeline
   - Bottleneck Analysis: Performance bottlenecks
   - GPU Utilization: GPU metrics (if available)

4. View the report with CLI:
   python -m neural.profiling.cli_integration show-report neural_dsl_profiling_report.json
""")


if __name__ == "__main__":
    main()
