"""
Example: Using Enhanced NeuralDbg Dashboard Features

This example demonstrates:
1. Real-time layer-by-layer inspection
2. Breakpoint debugging with conditions
3. Enhanced anomaly detection
4. Performance profiling with flame graphs
"""

import numpy as np
import time
from neural.dashboard.dashboard import (
    update_dashboard_data,
    breakpoint_manager,
    anomaly_detector,
    profiler
)
from neural.dashboard.profiler_utils import (
    layer_profiler,
    breakpoint_helper,
    anomaly_monitor,
    LayerExecutionWrapper
)


def example_1_basic_profiling():
    """Example 1: Basic layer profiling"""
    print("\n=== Example 1: Basic Layer Profiling ===\n")
    
    @layer_profiler.profile("Conv2D_1")
    def conv_layer():
        time.sleep(0.05)
        return np.random.rand(64, 64, 32)
    
    @layer_profiler.profile("Dense_1")
    def dense_layer():
        time.sleep(0.03)
        return np.random.rand(128)
    
    for i in range(5):
        print(f"Iteration {i+1}")
        conv_layer()
        dense_layer()
    
    summary = profiler.get_summary()
    print("\nPerformance Summary:")
    for layer_name, stats in summary.items():
        print(f"  {layer_name}:")
        print(f"    Total time: {stats['total_time']:.4f}s")
        print(f"    Avg time: {stats['avg_time']:.4f}s")
        print(f"    Calls: {stats['call_count']}")


def example_2_breakpoints():
    """Example 2: Using breakpoints with conditions"""
    print("\n=== Example 2: Breakpoint Debugging ===\n")
    
    breakpoint_manager.add_breakpoint(
        "SlowLayer",
        condition="layer_data['execution_time'] > 0.08"
    )
    
    def simulate_layer(layer_name, duration):
        profiler.start_layer(layer_name)
        time.sleep(duration)
        
        layer_data = {
            "layer": layer_name,
            "execution_time": duration,
            "memory": np.random.rand() * 500,
            "mean_activation": np.random.rand()
        }
        
        breakpoint_helper.check(layer_name, layer_data)
        
        profiler.end_layer(layer_name)
        return layer_data
    
    print("Simulating layers (breakpoint on execution_time > 0.08s)...")
    simulate_layer("FastLayer", 0.02)
    print("FastLayer executed")
    
    simulate_layer("SlowLayer", 0.1)
    print("SlowLayer executed (would have hit breakpoint if paused)")
    
    breakpoint_info = breakpoint_manager.get_breakpoint_info()
    print(f"\nBreakpoint info: {breakpoint_info}")


def example_3_anomaly_detection():
    """Example 3: Anomaly detection"""
    print("\n=== Example 3: Enhanced Anomaly Detection ===\n")
    
    layer_name = "Conv2D_Test"
    
    print("Training anomaly detector with normal data...")
    for i in range(20):
        normal_metrics = {
            "execution_time": 0.05 + np.random.normal(0, 0.005),
            "memory": 250 + np.random.normal(0, 10),
            "mean_activation": 0.35 + np.random.normal(0, 0.02),
            "grad_norm": 0.02 + np.random.normal(0, 0.002)
        }
        anomaly_detector.add_sample(layer_name, normal_metrics)
    
    print("Testing with normal sample...")
    normal_test = {
        "execution_time": 0.051,
        "memory": 248,
        "mean_activation": 0.34,
        "grad_norm": 0.021
    }
    result = anomaly_detector.detect_anomalies(layer_name, normal_test)
    print(f"  Is anomaly: {result['is_anomaly']}")
    print(f"  Scores: {result['scores']}")
    
    print("\nTesting with anomalous sample...")
    anomalous_test = {
        "execution_time": 0.15,
        "memory": 400,
        "mean_activation": 0.8,
        "grad_norm": 0.001
    }
    result = anomaly_detector.detect_anomalies(layer_name, anomalous_test)
    print(f"  Is anomaly: {result['is_anomaly']}")
    print(f"  Scores: {result['scores']}")
    
    stats = anomaly_detector.get_layer_statistics(layer_name)
    print(f"\nLayer statistics:")
    for metric, values in stats.items():
        print(f"  {metric}: mean={values['mean']:.4f}, std={values['std']:.4f}")


def example_4_comprehensive_wrapper():
    """Example 4: Using the comprehensive wrapper"""
    print("\n=== Example 4: Comprehensive Layer Execution Wrapper ===\n")
    
    wrapper = LayerExecutionWrapper()
    
    breakpoint_manager.add_breakpoint("WrapperLayer", condition="layer_data['memory'] > 300")
    
    @wrapper.wrap("WrapperLayer")
    def my_layer(x):
        time.sleep(0.04)
        return x * 2
    
    print("Training with normal memory usage...")
    for i in range(5):
        result = my_layer(np.random.rand(10))
    
    print("\nExecuting layer with context manager...")
    with wrapper.wrap_context("WrapperLayer") as ctx:
        output = np.random.rand(100, 100)
        memory_used = output.nbytes / (1024 * 1024)
        ctx.set_metrics({
            "memory": memory_used,
            "mean_activation": np.mean(output)
        })
    
    print("Layer executed successfully")


def example_5_flame_graph_data():
    """Example 5: Generate data for flame graph"""
    print("\n=== Example 5: Flame Graph Data Generation ===\n")
    
    def simulate_nested_execution():
        profiler.start_layer("MainNetwork")
        time.sleep(0.01)
        
        profiler.start_layer("Encoder")
        time.sleep(0.02)
        
        profiler.start_layer("Conv2D_1")
        time.sleep(0.015)
        profiler.end_layer("Conv2D_1")
        
        profiler.start_layer("Conv2D_2")
        time.sleep(0.01)
        profiler.end_layer("Conv2D_2")
        
        profiler.end_layer("Encoder")
        
        profiler.start_layer("Decoder")
        time.sleep(0.025)
        
        profiler.start_layer("Dense_1")
        time.sleep(0.012)
        profiler.end_layer("Dense_1")
        
        profiler.end_layer("Decoder")
        
        profiler.end_layer("MainNetwork")
    
    print("Simulating nested layer execution...")
    simulate_nested_execution()
    
    flame_data = profiler.get_flame_graph_data()
    print(f"\nGenerated {len(flame_data)} flame graph entries")
    print("Sample entries:")
    for entry in flame_data[:5]:
        print(f"  {entry['name']}: {entry['duration']:.4f}s at depth {entry['depth']}")


def example_6_dashboard_integration():
    """Example 6: Full dashboard integration"""
    print("\n=== Example 6: Full Dashboard Integration ===\n")
    
    layers = ["Conv2D_1", "MaxPool_1", "Conv2D_2", "Flatten", "Dense_1", "Output"]
    
    trace_data = []
    
    for i, layer_name in enumerate(layers):
        exec_time = 0.05 + np.random.rand() * 0.03
        memory = 100 + i * 50 + np.random.rand() * 20
        
        entry = {
            "layer": layer_name,
            "execution_time": exec_time,
            "memory": memory,
            "flops": int((i + 1) * 1e6 * np.random.rand()),
            "mean_activation": 0.3 + np.random.rand() * 0.2,
            "grad_norm": 0.01 + np.random.rand() * 0.02,
            "dead_ratio": np.random.rand() * 0.1
        }
        
        trace_data.append(entry)
        
        profiler.start_layer(layer_name)
        time.sleep(exec_time)
        profiler.end_layer(layer_name)
        
        anomaly_result = anomaly_detector.detect_anomalies(layer_name, entry)
        if anomaly_result["is_anomaly"]:
            print(f"  ANOMALY in {layer_name}: {anomaly_result['scores']}")
    
    update_dashboard_data(new_trace_data=trace_data)
    
    print(f"\nUpdated dashboard with {len(trace_data)} layer entries")
    print("Performance summary:")
    summary = profiler.get_summary()
    for layer_name, stats in sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)[:3]:
        print(f"  {layer_name}: {stats['total_time']:.4f}s")


def example_7_anomaly_callback():
    """Example 7: Custom anomaly callback"""
    print("\n=== Example 7: Custom Anomaly Alert Callback ===\n")
    
    def anomaly_alert(layer_name, metrics, result):
        print(f"🚨 ALERT: Anomaly detected in {layer_name}")
        print(f"   Metrics: execution_time={metrics.get('execution_time', 0):.4f}s")
        print(f"   Scores: {result['scores']}")
    
    anomaly_monitor.set_alert_callback(anomaly_alert)
    
    test_layer = "AlertTest"
    
    for i in range(10):
        metrics = {
            "execution_time": 0.05 + np.random.normal(0, 0.005),
            "memory": 250
        }
        anomaly_detector.add_sample(test_layer, metrics)
    
    print("Testing with anomalous execution time...")
    anomalous_metrics = {
        "execution_time": 0.2,
        "memory": 250
    }
    anomaly_monitor.check(test_layer, anomalous_metrics)


def main():
    """Run all examples"""
    print("="*70)
    print("NeuralDbg Enhanced Dashboard Examples")
    print("="*70)
    
    try:
        example_1_basic_profiling()
        example_2_breakpoints()
        example_3_anomaly_detection()
        example_4_comprehensive_wrapper()
        example_5_flame_graph_data()
        example_6_dashboard_integration()
        example_7_anomaly_callback()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        print("\nTo view the dashboard, run:")
        print("  python neural/dashboard/dashboard.py")
        print("\nThen open http://localhost:8050 in your browser")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
