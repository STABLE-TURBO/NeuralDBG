# Neural DSL Profiling & Debugging Guide

Comprehensive guide for using the advanced profiling and debugging tools in Neural DSL.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Profiling Tools](#profiling-tools)
5. [Dashboard Integration](#dashboard-integration)
6. [Advanced Usage](#advanced-usage)
7. [CLI Commands](#cli-commands)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

Neural DSL provides a comprehensive profiling suite that extends the existing NeuralDbg dashboard with:

- **Layer-by-layer execution profiling** - Detailed timing analysis for each layer
- **Memory profiling** - Track CPU and GPU memory usage
- **Memory leak detection** - Automatic detection of memory growth issues
- **Bottleneck identification** - Find performance bottlenecks with AI-powered recommendations
- **Comparative profiling** - Compare performance across different backends
- **Distributed training diagnostics** - Analyze multi-node training performance
- **GPU utilization analysis** - Monitor GPU efficiency and resource usage

## Installation

The profiling tools are included with Neural DSL. To use all features:

```bash
# Install with full dependencies
pip install -e ".[full]"

# Or install specific dependencies
pip install -e ".[backends,dashboard]"

# For GPU profiling, install pynvml
pip install pynvml
```

## Quick Start

### Basic Profiling

```python
from neural.profiling import ProfilerManager

# Initialize profiler
profiler = ProfilerManager(enable_all=True)
profiler.start_profiling()

# Profile your model layers
for layer_name, layer_fn in model_layers:
    with profiler.profile_layer(layer_name):
        output = layer_fn(input)

# End profiling and get report
profiler.end_profiling()
report = profiler.get_comprehensive_report()

# Print formatted report
from neural.profiling.utils import format_profiling_report
print(format_profiling_report(report))

# Export to JSON
profiler.export_report('profiling_report.json')
```

### Running Examples

```bash
# Basic profiling
python examples/profiling_example.py

# Backend comparison
python examples/comparative_profiling_example.py

# Distributed training analysis
python examples/distributed_profiling_example.py

# GPU profiling
python examples/gpu_profiling_example.py
```

## Profiling Tools

### 1. Layer Profiler

Tracks execution time and statistics for each layer.

```python
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('layer')

profiler.start_profiling()

with profiler.profile_layer('Conv2D_1'):
    output = conv2d(input)

# Get layer statistics
stats = profiler.layer_profiler.get_layer_stats('Conv2D_1')
print(f"Mean time: {stats['mean_time']*1000:.2f}ms")
print(f"Std dev: {stats['std_time']*1000:.2f}ms")

# Get slowest layers
slowest = profiler.layer_profiler.get_slowest_layers(top_n=5)
for layer, time in slowest:
    print(f"{layer}: {time*1000:.2f}ms")
```

**Features:**
- Mean, std dev, min, max execution times
- Call count tracking
- Execution timeline
- Identifies slowest and most variable layers

### 2. Memory Profiler

Monitors memory usage during execution.

```python
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('memory')

profiler.start_profiling()

with profiler.profile_layer('Dense_1', {'memory_usage': 100}):
    output = dense_layer(input)

# Get memory summary
summary = profiler.memory_profiler.get_summary()
print(f"Peak memory: {summary['peak_memory_mb']:.2f}MB")

# Get memory growth
growth = profiler.memory_profiler.get_memory_growth()
for entry in growth:
    print(f"{entry['layer']}: +{entry['memory_growth_mb']:.2f}MB")
```

**Features:**
- CPU and GPU memory tracking
- Memory growth analysis
- Layer-specific memory usage
- Peak memory detection

### 3. Memory Leak Detector

Automatically detects memory leaks using sliding window analysis.

```python
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('leak')

# Configure sensitivity
from neural.profiling import MemoryLeakDetector
profiler.memory_leak_detector = MemoryLeakDetector(
    threshold_mb=10.0,  # Alert if memory increases by 10MB
    window_size=10       # Check over 10 measurements
)

# After profiling
leak_summary = profiler.memory_leak_detector.get_summary()
if leak_summary['has_leaks']:
    print(f"⚠ {leak_summary['leak_count']} leaks detected!")
    print(f"Total leaked: {leak_summary['total_memory_leaked_mb']:.2f}MB")
```

**Features:**
- Automatic leak detection
- Growth rate calculation
- Identifies affected layers
- Configurable thresholds

### 4. Bottleneck Analyzer

Identifies performance bottlenecks and provides recommendations.

```python
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('bottleneck')

# After profiling multiple layers
bottlenecks = profiler.analyze_bottlenecks()

# Show bottlenecks
for b in bottlenecks['bottlenecks'][:5]:
    print(f"{b['layer']}: {b['overhead_percentage']:.1f}% overhead")
    print(f"  Type: {', '.join(b['bottleneck_type'])}")
    print(f"  Severity: {b['severity_score']:.2f}")

# Show recommendations
for rec in bottlenecks['recommendations']:
    print(f"[{rec['priority']}] {rec['recommendation']}")
    print(f"  {rec['details']}")
```

**Bottleneck Types:**
- `compute`: High computation time
- `memory`: Excessive memory usage
- `io`: I/O-bound operations
- `underutilization`: Low GPU utilization

**Features:**
- Statistical analysis (90th percentile threshold)
- Multi-dimensional bottleneck detection
- Severity scoring
- AI-powered optimization recommendations

### 5. Comparative Profiler

Compare performance across different backends.

```python
from neural.profiling import ComparativeProfiler

comparative = ComparativeProfiler()

# Profile each backend
for backend in ['tensorflow', 'pytorch', 'onnx']:
    profiler = ProfilerManager(enable_all=False)
    profiler.enable_profiler('layer')
    
    # Run model with backend
    profiler.start_profiling()
    run_model(backend)
    profiler.end_profiling()
    
    # Add to comparison
    profile_data = profiler.layer_profiler.export_to_dict()
    comparative.add_backend_profile(backend, profile_data)

# Compare
comparison = comparative.compare_backends()

# Show results
print(f"Fastest: {comparison['summary']['fastest_backend']}")
print(f"Speedup: {comparison['summary']['overall_speedup']:.2f}x")
print(f"Recommendation: {comparison['summary']['recommendation']}")

# Layer-by-layer comparison
for layer, comp in comparison['layer_comparisons'].items():
    print(f"\n{layer}:")
    print(f"  Fastest: {comp['fastest_backend']}")
    print(f"  Speedup: {comp['speedup']:.2f}x")
```

**Features:**
- Multi-backend comparison
- Layer-by-layer analysis
- Overall performance ranking
- Automated recommendations

### 6. Distributed Training Profiler

Analyze distributed training performance.

```python
from neural.profiling import DistributedTrainingProfiler

dist_profiler = DistributedTrainingProfiler()

# Add node profiles
for node_id in ['node_0', 'node_1', 'node_2']:
    profile_data = get_node_profile(node_id)
    dist_profiler.add_node_profile(node_id, profile_data)

# Record communication
dist_profiler.record_communication(
    source_node='node_0',
    dest_node='node_1',
    data_size_mb=100,
    duration_ms=50,
    operation_type='gradient_transfer'
)

# Record synchronization
dist_profiler.record_synchronization(
    node_ids=['node_0', 'node_1', 'node_2'],
    sync_type='all_reduce',
    duration_ms=45,
    barrier_wait_ms=12
)

# Analyze
load_balance = dist_profiler.analyze_load_balance()
print(f"Imbalance factor: {load_balance['imbalance_factor']:.3f}")
print(f"Slowest node: {load_balance['slowest_node']}")

bottlenecks = dist_profiler.get_bottleneck_analysis()
for b in bottlenecks['bottlenecks']:
    print(f"[{b['severity']}] {b['type']}: {b['recommendation']}")
```

**Features:**
- Load balance analysis
- Communication overhead tracking
- Synchronization profiling
- Network bandwidth analysis
- Distributed bottleneck detection

### 7. GPU Utilization Profiler

Monitor GPU efficiency and resource usage.

```python
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('gpu')

profiler.start_profiling()

for layer_name in layers:
    with profiler.profile_layer(layer_name):
        output = layer_fn(input)

# Get GPU summary
gpu_summary = profiler.gpu_profiler.get_utilization_summary()
print(f"Mean GPU utilization: {gpu_summary['mean_gpu_utilization_percent']:.1f}%")
print(f"Peak memory: {gpu_summary['peak_memory_allocated_mb']:.2f}MB")

# Get recommendations
recommendations = profiler.gpu_profiler.get_recommendations()
for rec in recommendations:
    print(f"[{rec['priority']}] {rec['recommendation']}")

# Detect memory inefficiencies
inefficiency = profiler.gpu_profiler.detect_memory_inefficiency()
if inefficiency['has_issues']:
    print(f"⚠ {inefficiency['issue_count']} memory issues detected")
```

**Features:**
- GPU utilization percentage
- Memory allocation tracking
- Temperature monitoring (requires NVML)
- Power consumption analysis (requires NVML)
- Memory efficiency analysis
- Optimization recommendations

## Dashboard Integration

The profiling tools are integrated into the NeuralDbg dashboard.

### Starting the Dashboard

```bash
python neural/dashboard/dashboard.py
```

Navigate to `http://localhost:8050`

### Dashboard Tabs

1. **Real-Time Monitoring** - Original NeuralDbg features
2. **Layer Profiling** - Execution time charts with error bars
3. **Memory Profiling** - CPU/GPU memory timeline and leak detection
4. **Bottleneck Analysis** - Visual bottleneck identification and recommendations
5. **GPU Utilization** - Multi-panel GPU metrics
6. **Backend Comparison** - Side-by-side backend performance
7. **Distributed Training** - Load balance and communication analysis

### Updating Dashboard Data

```python
from neural.dashboard.dashboard import update_profiling_data

profiler = ProfilerManager(enable_all=True)
# ... run profiling ...

# Get dashboard-formatted data
dashboard_data = profiler.get_dashboard_data()

# Update dashboard
update_profiling_data(dashboard_data)
```

## Advanced Usage

### Custom Profiler Configuration

```python
# Create profiler with specific features
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('layer')
profiler.enable_profiler('memory')
profiler.enable_profiler('bottleneck')

# Configure bottleneck analyzer
from neural.profiling import BottleneckAnalyzer
profiler.bottleneck_analyzer = BottleneckAnalyzer(
    threshold_percentile=95.0  # More aggressive bottleneck detection
)
```

### Decorator-Based Profiling

```python
from neural.profiling.utils import profile_function

profiler = ProfilerManager()
profiler.start_profiling()

@profile_function(profiler, 'my_layer')
def my_layer_function(x):
    return process(x)

result = my_layer_function(input_data)
```

### Comparing Profiling Sessions

```python
from neural.profiling.utils import compare_profiling_sessions

# Run two profiling sessions
session1_report = run_and_profile(version1)
session2_report = run_and_profile(version2)

# Compare
comparison = compare_profiling_sessions(session1_report, session2_report)

if comparison['improvements']['total_time']['speedup_percent'] > 0:
    print("Version 2 is faster!")
```

## CLI Commands

Use the profiling CLI for report analysis:

```bash
# Show report summary
python -m neural.profiling.cli_integration show-report profiling_report.json

# Show bottlenecks
python -m neural.profiling.cli_integration bottlenecks profiling_report.json --top-n 10

# Show memory analysis
python -m neural.profiling.cli_integration memory-analysis profiling_report.json

# Compare two reports
python -m neural.profiling.cli_integration compare report1.json report2.json

# Show GPU statistics
python -m neural.profiling.cli_integration gpu-stats profiling_report.json

# Generate summary
python -m neural.profiling.cli_integration summary profiling_report.json -o summary.txt
```

## Best Practices

### 1. Minimize Profiling Overhead

```python
# For production, enable only needed profilers
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('layer')  # Only ~0.5% overhead
```

### 2. Profile Representative Workloads

```python
# Profile with realistic data
for epoch in range(warmup_epochs):
    run_epoch(train_data)  # Warmup

profiler.start_profiling()
for epoch in range(profile_epochs):
    run_epoch(train_data)  # Profile
profiler.end_profiling()
```

### 3. Use Metadata for Context

```python
with profiler.profile_layer('Conv2D_1', {
    'batch_size': 32,
    'input_shape': (32, 224, 224, 3),
    'filters': 64,
}):
    output = conv2d(input)
```

### 4. Regular Exports

```python
# Export after each major change
profiler.export_report(f'profile_{version}_{timestamp}.json')
```

### 5. Analyze Trends

```python
# Compare across versions
reports = load_reports(['v1.json', 'v2.json', 'v3.json'])
for i in range(1, len(reports)):
    comparison = compare_profiling_sessions(reports[i-1], reports[i])
    print(f"v{i} to v{i+1}: {comparison['improvements']}")
```

## Troubleshooting

### GPU Profiling Not Working

**Problem:** GPU metrics showing as unavailable

**Solution:**
```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Install pynvml for advanced metrics
pip install pynvml
```

### High Memory Usage During Profiling

**Problem:** Profiler consuming too much memory

**Solution:**
```python
# Reduce snapshot frequency
profiler = ProfilerManager(enable_all=False)
profiler.enable_profiler('layer')  # Only timing, no memory snapshots
```

### Dashboard Not Showing Profiling Data

**Problem:** Profiling tabs empty in dashboard

**Solution:**
```python
# Ensure data is pushed to dashboard
from neural.dashboard.dashboard import update_profiling_data
update_profiling_data(profiler.get_dashboard_data())

# Or check if profiling module is importable
python -c "from neural.profiling import ProfilerManager; print('OK')"
```

### Bottleneck Analysis Not Detecting Issues

**Problem:** No bottlenecks found despite slow layers

**Solution:**
```python
# Lower the threshold
from neural.profiling import BottleneckAnalyzer
profiler.bottleneck_analyzer = BottleneckAnalyzer(
    threshold_percentile=75.0  # Lower from default 90.0
)
```

## Performance Impact

Typical overhead per profiler:

| Profiler | Overhead | Notes |
|----------|----------|-------|
| Layer Profiler | ~0.1-0.5% | Minimal impact |
| Memory Profiler | ~1-2% | Depends on snapshot frequency |
| Bottleneck Analyzer | ~0.5% | Lightweight analysis |
| GPU Profiler | ~2-3% | Requires NVML queries |
| Memory Leak Detector | ~0.5% | Window-based, efficient |

**Recommendation:** For production, use Layer Profiler only. Enable others for development and debugging.

## Additional Resources

- [Profiling Module README](../neural/profiling/README.md)
- [Example Scripts](../examples/)
- [Dashboard Documentation](../neural/dashboard/README.md)
- [API Reference](../docs/API.md)

## Contributing

To add new profiling features:

1. Create profiler in `neural/profiling/`
2. Add visualization in `dashboard_integration.py`
3. Update `ProfilerManager` integration
4. Add dashboard callback
5. Write examples and tests
6. Update documentation

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
