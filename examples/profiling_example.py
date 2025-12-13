import time
import numpy as np
from neural.profiling import ProfilerManager
from neural.profiling.utils import format_profiling_report


def simulate_layer_computation(layer_name, duration=0.1, memory_mb=50):
    time.sleep(duration)
    data = np.random.randn(int(memory_mb * 1000))
    return data


def main():
    print("=" * 80)
    print("Neural DSL Profiling Example")
    print("=" * 80)
    
    profiler = ProfilerManager(enable_all=True)
    profiler.start_profiling()
    
    print("\nSimulating neural network execution with profiling...")
    
    layers = [
        ("Input", 0.05, 10),
        ("Conv2D_1", 0.15, 100),
        ("BatchNorm_1", 0.03, 20),
        ("ReLU_1", 0.02, 5),
        ("MaxPool_1", 0.08, 30),
        ("Conv2D_2", 0.20, 150),
        ("BatchNorm_2", 0.03, 20),
        ("ReLU_2", 0.02, 5),
        ("MaxPool_2", 0.08, 30),
        ("Flatten", 0.01, 5),
        ("Dense_1", 0.12, 80),
        ("ReLU_3", 0.02, 5),
        ("Dropout", 0.01, 2),
        ("Dense_2", 0.10, 60),
        ("Softmax", 0.03, 10),
    ]
    
    for layer_name, duration, memory_mb in layers:
        metadata = {'memory_usage': memory_mb}
        with profiler.profile_layer(layer_name, metadata):
            _ = simulate_layer_computation(layer_name, duration, memory_mb)
        print(f"  Profiled: {layer_name}")
    
    print("\nAnalyzing bottlenecks...")
    bottleneck_analysis = profiler.analyze_bottlenecks()
    
    profiler.end_profiling()
    
    print("\nGenerating comprehensive report...")
    report = profiler.get_comprehensive_report()
    
    print("\n" + format_profiling_report(report))
    
    if bottleneck_analysis.get('recommendations'):
        print("\nOptimization Recommendations:")
        for i, rec in enumerate(bottleneck_analysis['recommendations'][:5], 1):
            print(f"\n{i}. [{rec['priority'].upper()}] {rec['category'].upper()}")
            print(f"   {rec['recommendation']}")
            print(f"   Details: {rec['details']}")
    
    export_path = "profiling_report.json"
    profiler.export_report(export_path)
    print(f"\nFull report exported to: {export_path}")
    
    dashboard_data = profiler.get_dashboard_data()
    print(f"\nDashboard data available with {len(dashboard_data.get('execution_history', []))} execution records")


if __name__ == "__main__":
    main()
