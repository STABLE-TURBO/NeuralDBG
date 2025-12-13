import time
import numpy as np
from neural.profiling import ProfilerManager, ComparativeProfiler


def simulate_backend_execution(backend_name, layers, speed_multiplier=1.0):
    profiler = ProfilerManager(enable_all=False)
    profiler.enable_profiler('layer')
    profiler.start_profiling()
    
    print(f"\nRunning on {backend_name}...")
    
    for layer_name, base_duration in layers:
        duration = base_duration * speed_multiplier
        with profiler.profile_layer(layer_name, {}):
            time.sleep(duration)
        print(f"  {layer_name}: {duration*1000:.2f}ms")
    
    profiler.end_profiling()
    return profiler.layer_profiler.export_to_dict()


def main():
    print("=" * 80)
    print("Comparative Backend Profiling Example")
    print("=" * 80)
    
    layers = [
        ("Conv2D_1", 0.10),
        ("BatchNorm_1", 0.02),
        ("ReLU_1", 0.01),
        ("MaxPool_1", 0.05),
        ("Conv2D_2", 0.15),
        ("Dense_1", 0.08),
        ("Softmax", 0.02),
    ]
    
    comparative = ComparativeProfiler()
    
    backends = {
        'TensorFlow': 1.0,
        'PyTorch': 0.9,
        'ONNX': 0.85,
    }
    
    for backend_name, speed_mult in backends.items():
        profile_data = simulate_backend_execution(backend_name, layers, speed_mult)
        comparative.add_backend_profile(backend_name, profile_data)
    
    print("\n" + "=" * 80)
    print("Comparing backends...")
    comparison = comparative.compare_backends()
    
    print("\nBackend Ranking:")
    rankings = comparative.get_backend_ranking()
    for i, (backend, total_time) in enumerate(rankings, 1):
        print(f"  {i}. {backend}: {total_time*1000:.2f}ms total")
    
    if 'summary' in comparison:
        summary = comparison['summary']
        print(f"\nFastest Backend: {summary['fastest_backend']}")
        print(f"Slowest Backend: {summary['slowest_backend']}")
        print(f"Overall Speedup: {summary['overall_speedup']:.2f}x")
        print(f"\nRecommendation: {summary['recommendation']}")
    
    print("\nLayer-by-Layer Comparison:")
    if 'layer_comparisons' in comparison:
        for layer, comp in list(comparison['layer_comparisons'].items())[:5]:
            print(f"\n  {layer}:")
            print(f"    Fastest: {comp['fastest_backend']} ({comp['backend_data'][comp['fastest_backend']]['mean_time']*1000:.2f}ms)")
            print(f"    Slowest: {comp['slowest_backend']} ({comp['backend_data'][comp['slowest_backend']]['mean_time']*1000:.2f}ms)")
            print(f"    Speedup: {comp['speedup']:.2f}x")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
