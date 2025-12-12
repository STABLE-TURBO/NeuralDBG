# NeuralDbg Dashboard - Enhanced Features

## Overview

The NeuralDbg dashboard provides comprehensive real-time monitoring and debugging capabilities for neural network execution. This document describes the enhanced features added to the dashboard.

## New Features

### 1. Real-Time Layer-by-Layer Inspection

The layer inspector allows you to examine detailed metrics for individual layers during execution.

**Features:**
- Select any layer from a dropdown to view its detailed statistics
- View current metrics: execution time, memory usage, FLOPs, activations, gradients
- Statistical summaries: mean, std, min, max, median for historical data
- Real-time updates as the network executes

**Usage:**
```python
# The layer inspector is automatically populated with layers from trace_data
# Select a layer from the dropdown to view its detailed inspection
```

**REST API:**
```bash
# Inspect a specific layer
curl http://localhost:8050/api/layer_inspect/Conv2D_1
```

### 2. Breakpoint Debugging Support

Set breakpoints on specific layers with optional conditions to pause execution and inspect state.

**Features:**
- Add/remove breakpoints on any layer
- Conditional breakpoints using Python expressions
- Hit count tracking for each breakpoint
- Enable/disable breakpoints without removing them
- REST API for programmatic control

**Usage in Dashboard:**
1. Enter layer name in the "Layer name" field
2. (Optional) Add a condition like: `layer_data['execution_time'] > 0.5`
3. Click "Add Breakpoint"
4. Toggle or remove breakpoints using the buttons

**REST API:**
```bash
# Add a breakpoint
curl -X POST http://localhost:8050/api/breakpoint \
  -H "Content-Type: application/json" \
  -d '{"layer_name": "Conv2D_1", "condition": "layer_data[\"execution_time\"] > 0.5"}'

# List all breakpoints
curl http://localhost:8050/api/breakpoint

# Remove a breakpoint
curl -X DELETE http://localhost:8050/api/breakpoint \
  -H "Content-Type: application/json" \
  -d '{"layer_name": "Conv2D_1"}'

# Continue execution after pause
curl -X POST http://localhost:8050/api/continue
```

**Condition Examples:**
- `layer_data['execution_time'] > 0.5` - Break if layer takes more than 0.5s
- `layer_data['memory'] > 1000` - Break if layer uses more than 1000 MB
- `layer_data['grad_norm'] < 0.001` - Break if gradient is too small
- `layer_data['mean_activation'] == 0` - Break if all activations are zero

### 3. Enhanced Anomaly Detection

Advanced anomaly detection using multiple statistical methods to identify unusual layer behavior.

**Algorithms:**

1. **Z-Score Detection**: Identifies values that deviate significantly from the mean
   - Flags values more than 3 standard deviations from the mean
   - Tracks multiple metrics: execution_time, memory, activations, gradients

2. **IQR (Interquartile Range) Detection**: Identifies outliers using quartile-based bounds
   - More robust to extreme values than Z-score
   - Uses 1.5 * IQR rule for outlier detection

3. **Isolation Forest** (framework for future ML-based detection)
   - Unsupervised learning approach
   - Can detect complex anomaly patterns

**Features:**
- Real-time anomaly scoring for each layer
- Multi-metric anomaly detection
- Visual indicators: red bars for detected anomalies
- Anomaly score overlay showing severity (Z-score values)
- Historical tracking of anomalies

**Interpretation:**
- Red bars: Anomaly detected in this layer
- Orange line: Anomaly severity score (higher = more unusual)
- Z-scores > 3: Significant deviation from normal behavior

### 4. Performance Profiling with Flame Graphs

Comprehensive performance profiling to identify bottlenecks and understand execution patterns.

**Components:**

#### Flame Graph
- Visual representation of execution time across all layers
- Horizontal bars show duration of each layer execution
- Colors indicate call depth (useful for nested operations)
- Hover to see detailed timing information

**Features:**
- Call stack visualization
- Time-based layout showing temporal execution order
- Color-coded by execution depth
- Interactive tooltips with precise timing

#### Performance Summary Table
- Top 10 slowest layers ranked by total time
- Statistics for each layer:
  - Total execution time across all calls
  - Number of calls
  - Average, minimum, and maximum execution time
- Helps identify optimization targets

**Usage:**
The profiler automatically tracks layer execution when layers report timing data. The flame graph and summary update in real-time.

**Profiler API (for integration):**
```python
from neural.dashboard.dashboard import profiler

# Start timing a layer
profiler.start_layer("Conv2D_1")

# ... layer execution ...

# End timing
profiler.end_layer("Conv2D_1")

# Get summary data
summary = profiler.get_summary()
flame_data = profiler.get_flame_graph_data()
```

## Dashboard Layout

The enhanced dashboard is organized into sections:

### Top Row
1. **Model Structure** (left): Network architecture visualization
2. **Gradient Flow / Dead Neurons / Anomaly Detection** (right): Training health metrics

### Middle Row
1. **Layer-by-Layer Inspector** (left): Detailed per-layer metrics
2. **Breakpoint Manager** (right): Breakpoint control interface

### Bottom Sections
1. **Performance Flame Graph**: Visual execution timeline
2. **Performance Summary**: Statistical breakdown of layer performance
3. **Resource Monitoring**: CPU/GPU/Memory usage

## Configuration

Update `config.yaml` to customize behavior:

```yaml
websocket_interval: 1000  # Update interval in milliseconds

# Anomaly detection sensitivity (0-1, higher = fewer anomalies)
anomaly_sensitivity: 0.95

# Breakpoint settings
breakpoints:
  enabled: true
  pause_on_condition: true
```

## Integration Example

```python
from neural.dashboard.dashboard import (
    update_dashboard_data,
    breakpoint_manager,
    anomaly_detector,
    profiler
)

# Update dashboard with trace data
trace_data = [
    {
        "layer": "Conv2D_1",
        "execution_time": 0.045,
        "memory": 256.5,
        "flops": 1000000,
        "mean_activation": 0.35,
        "grad_norm": 0.02
    },
    # ... more layers
]

update_dashboard_data(new_trace_data=trace_data)

# Add a breakpoint programmatically
breakpoint_manager.add_breakpoint(
    "Conv2D_2",
    condition="layer_data['execution_time'] > 0.1"
)

# Check for anomalies
result = anomaly_detector.detect_anomalies("Conv2D_1", trace_data[0])
if result["is_anomaly"]:
    print(f"Anomaly detected! Scores: {result['scores']}")

# Profile execution
profiler.start_layer("Conv2D_1")
# ... execute layer ...
profiler.end_layer("Conv2D_1")
```

## Development Notes

### Adding New Metrics

To add a new metric to layer inspection:

1. Ensure the metric is included in trace_data entries
2. The anomaly detector will automatically track it if named appropriately
3. Add visualization in the layer inspector callback if needed

### Custom Anomaly Detection Algorithms

Extend the `AnomalyDetector` class:

```python
class AnomalyDetector:
    def detect_custom_anomaly(self, layer_name, metrics):
        # Your custom logic here
        pass
```

### Performance Considerations

- Layer inspection history: Limited to 100 most recent entries
- Anomaly history: Limited to 1000 most recent samples
- Flame graph: Tracks all layer executions (consider clearing periodically for long runs)

## Troubleshooting

**Issue**: Layer inspector shows "No data available"
- **Solution**: Ensure trace_data includes the selected layer

**Issue**: Breakpoints not triggering
- **Solution**: Check that the layer name matches exactly (case-sensitive)

**Issue**: Anomaly detection flags too many/too few layers
- **Solution**: Adjust sensitivity parameter (0.95 default, lower = more sensitive)

**Issue**: Flame graph is empty
- **Solution**: Ensure profiler.start_layer() and end_layer() are called properly

## Future Enhancements

Planned features:
- Multi-GPU profiling support
- Export flame graph data to Chrome tracing format
- Machine learning-based anomaly detection training
- Breakpoint groups and batch operations
- Timeline scrubbing for historical analysis
- Comparative analysis between runs
