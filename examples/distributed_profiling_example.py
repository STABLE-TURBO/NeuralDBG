import time
import numpy as np
from neural.profiling import DistributedTrainingProfiler, ProfilerManager


def simulate_node_training(node_id, layers, speed_factor=1.0):
    profiler = ProfilerManager(enable_all=False)
    profiler.enable_profiler('layer')
    profiler.start_profiling()
    
    for layer_name, duration in layers:
        with profiler.profile_layer(layer_name, {}):
            time.sleep(duration * speed_factor)
    
    profiler.end_profiling()
    return profiler.layer_profiler.export_to_dict()


def main():
    print("=" * 80)
    print("Distributed Training Profiling Example")
    print("=" * 80)
    
    dist_profiler = DistributedTrainingProfiler()
    
    layers = [
        ("Forward_Pass", 0.10),
        ("Backward_Pass", 0.15),
        ("Gradient_Computation", 0.12),
        ("Weight_Update", 0.08),
    ]
    
    nodes = {
        'node_0': 1.0,
        'node_1': 1.1,
        'node_2': 0.95,
        'node_3': 1.3,
    }
    
    print("\nSimulating distributed training across nodes...")
    
    for node_id, speed_factor in nodes.items():
        print(f"\n  Training on {node_id} (speed factor: {speed_factor:.2f})...")
        profile_data = simulate_node_training(node_id, layers, speed_factor)
        dist_profiler.add_node_profile(node_id, profile_data)
    
    print("\nSimulating communication overhead...")
    dist_profiler.record_communication('node_0', 'node_1', data_size_mb=50, duration_ms=25, operation_type='gradient_transfer')
    dist_profiler.record_communication('node_1', 'node_2', data_size_mb=50, duration_ms=30, operation_type='gradient_transfer')
    dist_profiler.record_communication('node_2', 'node_3', data_size_mb=50, duration_ms=28, operation_type='gradient_transfer')
    dist_profiler.record_communication('node_3', 'node_0', data_size_mb=20, duration_ms=15, operation_type='parameter_sync')
    
    print("Simulating synchronization barriers...")
    dist_profiler.record_synchronization(
        list(nodes.keys()),
        'all_reduce',
        duration_ms=45,
        barrier_wait_ms=12
    )
    
    print("\n" + "=" * 80)
    print("Analyzing distributed training...")
    
    load_balance = dist_profiler.analyze_load_balance()
    print("\nLoad Balance Analysis:")
    print(f"  Mean Time: {load_balance['mean_time']*1000:.2f}ms")
    print(f"  Imbalance Factor: {load_balance['imbalance_factor']:.3f}")
    print(f"  Slowest Node: {load_balance['slowest_node']}")
    print(f"  Fastest Node: {load_balance['fastest_node']}")
    print(f"  Is Balanced: {'Yes' if load_balance['is_balanced'] else 'No'}")
    
    comm_overhead = dist_profiler.analyze_communication_overhead()
    print("\nCommunication Overhead:")
    print(f"  Total Time: {comm_overhead['total_communication_time_ms']:.2f}ms")
    print(f"  Total Data: {comm_overhead['total_data_transferred_mb']:.2f}MB")
    print(f"  Mean Bandwidth: {comm_overhead['mean_bandwidth_mbps']:.2f}Mbps")
    
    sync_overhead = dist_profiler.analyze_synchronization_overhead()
    print("\nSynchronization Overhead:")
    print(f"  Total Sync Time: {sync_overhead['total_sync_time_ms']:.2f}ms")
    print(f"  Total Barrier Wait: {sync_overhead['total_barrier_wait_ms']:.2f}ms")
    
    bottlenecks = dist_profiler.get_bottleneck_analysis()
    print("\nBottleneck Detection:")
    if bottlenecks['bottlenecks']:
        for b in bottlenecks['bottlenecks']:
            print(f"\n  [{b['severity'].upper()}] {b['type']}")
            print(f"    {b['details']}")
            print(f"    Recommendation: {b['recommendation']}")
    else:
        print("  No significant bottlenecks detected")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
