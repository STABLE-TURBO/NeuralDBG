import time
import numpy as np
from neural.profiling import ProfilerManager


def simulate_gpu_operations():
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU simulation")
            return False
        
        return True
    except ImportError:
        print("PyTorch not available, skipping GPU profiling")
        return False


def main():
    print("=" * 80)
    print("GPU Utilization Profiling Example")
    print("=" * 80)
    
    gpu_available = simulate_gpu_operations()
    
    profiler = ProfilerManager(enable_all=False)
    profiler.enable_profiler('layer')
    profiler.enable_profiler('gpu')
    profiler.start_profiling()
    
    print("\nSimulating GPU operations...")
    
    layers = [
        ("Conv2D_1", 0.05),
        ("BatchNorm_1", 0.01),
        ("ReLU_1", 0.005),
        ("MaxPool_1", 0.02),
        ("Conv2D_2", 0.08),
        ("Dense_1", 0.04),
        ("Softmax", 0.01),
    ]
    
    for layer_name, duration in layers:
        with profiler.profile_layer(layer_name):
            time.sleep(duration)
            if gpu_available:
                import torch
                temp_tensor = torch.randn(1000, 1000).cuda()
                _ = temp_tensor @ temp_tensor
        
        print(f"  Profiled: {layer_name}")
    
    profiler.end_profiling()
    
    print("\nGenerating GPU profiling report...")
    report = profiler.get_comprehensive_report()
    
    if 'gpu_utilization' in report:
        gpu_stats = report['gpu_utilization']
        
        print("\n" + "=" * 80)
        print("GPU Utilization Summary")
        print("=" * 80)
        
        print(f"\nGPU Available: {gpu_stats.get('torch_available', False)}")
        print(f"Device Count: {gpu_stats.get('device_count', 0)}")
        
        if gpu_stats.get('device_names'):
            print(f"Devices: {', '.join(gpu_stats['device_names'])}")
        
        if 'mean_gpu_utilization_percent' in gpu_stats:
            print(f"\nMean GPU Utilization: {gpu_stats['mean_gpu_utilization_percent']:.1f}%")
            print(f"Min Utilization: {gpu_stats.get('min_gpu_utilization_percent', 0):.1f}%")
            print(f"Max Utilization: {gpu_stats.get('max_gpu_utilization_percent', 0):.1f}%")
        
        print(f"\nMemory Statistics:")
        print(f"  Mean Allocated: {gpu_stats.get('mean_memory_allocated_mb', 0):.2f}MB")
        print(f"  Peak Allocated: {gpu_stats.get('peak_memory_allocated_mb', 0):.2f}MB")
        
        if 'mean_temperature_celsius' in gpu_stats:
            print(f"\nTemperature:")
            print(f"  Mean: {gpu_stats['mean_temperature_celsius']:.1f}°C")
            print(f"  Max: {gpu_stats.get('max_temperature_celsius', 0):.1f}°C")
        
        if 'mean_power_watts' in gpu_stats:
            print(f"\nPower Consumption:")
            print(f"  Mean: {gpu_stats['mean_power_watts']:.1f}W")
            print(f"  Max: {gpu_stats.get('max_power_watts', 0):.1f}W")
        
        if 'utilization_warning' in gpu_stats:
            print(f"\n⚠ WARNING: {gpu_stats['utilization_warning']}")
    
    if 'gpu_recommendations' in report and report['gpu_recommendations']:
        print("\n" + "=" * 80)
        print("GPU Optimization Recommendations")
        print("=" * 80)
        
        for i, rec in enumerate(report['gpu_recommendations'], 1):
            print(f"\n{i}. [{rec['priority'].upper()}] {rec['category'].upper()}")
            print(f"   {rec['recommendation']}")
    
    print("\n" + "=" * 80)
    
    export_path = "gpu_profiling_report.json"
    profiler.export_report(export_path)
    print(f"\nFull report exported to: {export_path}")


if __name__ == "__main__":
    main()
