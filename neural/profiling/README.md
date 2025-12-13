# Neural DSL Profiling & Debugging Tools

Advanced profiling and debugging tools for neural networks, extending NeuralDbg with comprehensive performance analysis capabilities.

## Features

### 1. Layer-by-Layer Profiling (`LayerProfiler`)
- Track execution time for each layer
- Statistical analysis (mean, std, min, max)
- Identify slowest and most variable layers
- Execution timeline tracking

**Usage:**
```python
from neural.profiling import ProfilerManager

profiler = ProfilerManager()
profiler.start_profiling()

with profiler.profile_layer("Conv2D_1"):
    # Your layer computation
    output = conv2d_operation(input)

report = profiler.get_comprehensive_report()
```

### 2. Memory Profiling (`MemoryProfiler`)
- CPU and GPU memory tracking
- Memory growth analysis
- Layer-specific memory usage
- Peak memory detection

### 3. Memory Leak Detection (`MemoryLeakDetector`)
- Automatic leak detection using sliding window analysis
- Memory growth rate calculation
- Threshold-based alerting
- Detailed leak reports with affected layers

### 4. Bottleneck Analysis (`BottleneckAnalyzer`)
- Statistical bottleneck identification (90th percentile)
- Multi-dimensional analysis (compute, memory, I/O, GPU)
- Severity scoring
- Actionable optimization recommendations

**Bottleneck Types:**
- `compute`: High computation time
- `memory`: Excessive memory usage
- `io`: I/O-bound operations
- `underutilization`: Low GPU utilization

### 5. Comparative Profiling (`ComparativeProfiler`)
- Compare performance across backends (TensorFlow, PyTorch, ONNX)
- Layer-by-layer comparison
- Backend ranking
- Speedup calculations
- Automated recommendations

### 6. Distributed Training Diagnostics (`DistributedTrainingProfiler`)
- Load balance analysis across nodes
- Communication overhead tracking
- Synchronization barrier profiling
- Network bandwidth analysis
- Bottleneck detection (imbalance, communication, synchronization)

### 7. GPU Utilization Analysis (`GPUUtilizationProfiler`)
- GPU utilization percentage
- Memory allocation tracking
- Temperature monitoring
- Power consumption analysis
- Efficiency recommendations

## Quick Start

### Basic Profiling
```python
from neural.profiling import ProfilerManager

# Initialize with all profilers enabled
profiler = ProfilerManager(enable_all=True)
profiler.start_profiling()

# Profile your layers
layers = ["Input", "Conv2D", "ReLU", "MaxPool", "Dense", "Softmax"]
for layer_name in layers:
    with profiler.profile_layer(layer_name):
        # Your layer computation here
        pass

profiler.end_profiling()

# Get comprehensive report
report = profiler.get_comprehensive_report()
print(report)
```

### Backend Comparison
```python
from neural.profiling import ComparativeProfiler

comparative = ComparativeProfiler()

# Profile with different backends
for backend in ['tensorflow', 'pytorch', 'onnx']:
    profile_data = run_model_with_backend(backend)
    comparative.add_backend_profile(backend, profile_data)

# Compare results
comparison = comparative.compare_backends()
print(f"Fastest: {comparison['summary']['fastest_backend']}")
print(f"Recommendation: {comparison['summary']['recommendation']}")
```

### Distributed Training Analysis
```python
from neural.profiling import DistributedTrainingProfiler

dist_profiler = DistributedTrainingProfiler()

# Add node profiles
for node_id, profile_data in node_profiles.items():
    dist_profiler.add_node_profile(node_id, profile_data)

# Record communication
dist_profiler.record_communication(
    source_node='node_0',
    dest_node='node_1',
    data_size_mb=100,
    duration_ms=50,
    operation_type='gradient_transfer'
)

# Analyze
analysis = dist_profiler.get_bottleneck_analysis()
```

## Dashboard Integration

The profiling tools are integrated with the NeuralDbg dashboard, providing real-time visualization:

### Profiling Views
1. **Layer Profiling**: Execution time charts with standard deviations
2. **Memory Profiling**: Timeline of CPU/GPU memory usage
3. **Bottleneck Analysis**: Top bottlenecks with severity indicators
4. **GPU Utilization**: Multi-panel GPU metrics (utilization, memory, temperature, power)
5. **Backend Comparison**: Side-by-side performance comparison
6. **Distributed Training**: Load balance and communication analysis

### Accessing the Dashboard
```bash
python neural/dashboard/dashboard.py
```

Then navigate to `http://localhost:8050` and select the profiling tabs.

### Updating Dashboard Data
```python
from neural.dashboard.dashboard import update_profiling_data

# Get data from profiler
dashboard_data = profiler.get_dashboard_data()

# Update dashboard
update_profiling_data(dashboard_data)
```

## Examples

See the `examples/` directory for complete examples:
- `profiling_example.py`: Basic profiling workflow
- `comparative_profiling_example.py`: Backend comparison
- `distributed_profiling_example.py`: Distributed training analysis

## API Reference

### ProfilerManager

Main interface for all profiling operations.

**Methods:**
- `start_profiling()`: Begin profiling session
- `end_profiling()`: End profiling session
- `profile_layer(name, metadata)`: Context manager for layer profiling
- `analyze_bottlenecks()`: Run bottleneck analysis
- `compare_backends(backend_name)`: Add backend for comparison
- `get_comprehensive_report()`: Generate full profiling report
- `export_report(filepath)`: Save report to JSON
- `get_dashboard_data()`: Get data formatted for dashboard
- `reset_all()`: Clear all profiling data

### LayerProfiler

**Methods:**
- `start_layer(name)`: Start timing a layer
- `end_layer(name, metadata)`: End timing and record
- `get_layer_stats(name)`: Get statistics for a specific layer
- `get_slowest_layers(n)`: Get top N slowest layers
- `get_execution_timeline()`: Get chronological execution history

### BottleneckAnalyzer

**Methods:**
- `record_layer_metrics(name, metrics)`: Record metrics for analysis
- `analyze()`: Perform bottleneck analysis
- `get_bottlenecks()`: Get detected bottlenecks
- `get_recommendations()`: Get optimization recommendations

### ComparativeProfiler

**Methods:**
- `add_backend_profile(name, data)`: Add backend profile
- `compare_backends(backends)`: Compare specified backends
- `get_backend_ranking()`: Get backends sorted by performance

### DistributedTrainingProfiler

**Methods:**
- `add_node_profile(node_id, data)`: Add node profiling data
- `record_communication(source, dest, size, duration, type)`: Log communication
- `record_synchronization(nodes, type, duration)`: Log sync events
- `analyze_load_balance()`: Analyze load distribution
- `get_bottleneck_analysis()`: Identify distributed bottlenecks

### GPUUtilizationProfiler

**Methods:**
- `record_gpu_metrics(layer_name, device_id)`: Record GPU state
- `get_utilization_summary()`: Get overall GPU statistics
- `detect_memory_inefficiency()`: Identify memory issues
- `get_recommendations()`: Get GPU optimization tips

## Configuration

Profiling behavior can be configured when initializing the ProfilerManager:

```python
# Enable only specific profilers
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('layer')
profiler.enable_profiler('memory')
profiler.enable_profiler('bottleneck')

# Configure bottleneck threshold
from neural.profiling import BottleneckAnalyzer
analyzer = BottleneckAnalyzer(threshold_percentile=95.0)

# Configure memory leak detection sensitivity
from neural.profiling import MemoryLeakDetector
leak_detector = MemoryLeakDetector(threshold_mb=20.0, window_size=15)
```

## Performance Overhead

The profiling tools are designed to minimize performance impact:
- Layer profiling: ~0.1-0.5% overhead
- Memory profiling: ~1-2% overhead
- GPU profiling: ~2-3% overhead (requires NVML)

For production use, consider enabling only the profilers you need.

## Dependencies

### Core (always available):
- numpy
- psutil

### Optional (for GPU profiling):
- torch (CUDA support)
- pynvml (NVIDIA GPU metrics)

### Dashboard Integration:
- dash
- plotly

## Troubleshooting

### GPU Profiling Not Working
Ensure CUDA is available and pynvml is installed:
```bash
pip install pynvml
```

### High Memory Usage During Profiling
Reduce the number of snapshots by profiling fewer layers or increasing snapshot interval.

### Dashboard Not Updating
Ensure profiling data is being pushed to the dashboard:
```python
from neural.dashboard.dashboard import update_profiling_data
update_profiling_data(profiler.get_dashboard_data())
```

## Contributing

To add a new profiler:
1. Create a new file in `neural/profiling/`
2. Implement the profiler class
3. Add to `__init__.py`
4. Create visualization in `dashboard_integration.py`
5. Add callback in `dashboard.py`

## License

Part of Neural DSL - see main LICENSE.md
