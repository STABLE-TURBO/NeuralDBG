# NeuralDbg Dashboard - Enhanced Features

## Overview

The NeuralDbg dashboard started as a simple monitoring tool but has grown into something more comprehensive. It helps you understand what's happening inside your neural network during training - which layers are slow, where gradients vanish, which neurons are dead. Basically, the stuff that's hard to debug with print statements.

## New Features

### 1. Real-Time Layer-by-Layer Inspection

This feature lets you drill into individual layers while your network is running. It's particularly useful when you're trying to figure out why one specific layer is causing problems.

**What you can see:**
- Current execution time for each layer
- Memory usage (useful for finding memory hogs)
- FLOPs count (for performance optimization)
- Activation statistics (mean, std, min, max)
- Gradient information (helps spot vanishing/exploding gradients)

**How to use it:**
```python
# The layer inspector populates automatically from your trace data
# Just select a layer from the dropdown to see its stats
```

**REST API:**
```bash
# If you want to inspect programmatically
curl http://localhost:8050/api/layer_inspect/Conv2D_1
```

**Note:** The inspector keeps history for the last 100 executions. Older data gets dropped to save memory.

### 2. Breakpoint Debugging Support

Yes, breakpoints for neural networks. Set them on specific layers, add conditions, and pause execution when something interesting happens. It's more useful than it sounds.

**What you can do:**
- Add breakpoints to any layer
- Set conditions like "pause when execution time > 0.5s"
- Track how many times each breakpoint triggers
- Enable/disable breakpoints without removing them
- Control everything via REST API

**In the dashboard:**
1. Type the layer name (case-sensitive, unfortunately)
2. Add a condition if you want (it's Python syntax)
3. Click "Add Breakpoint"
4. Use the toggle buttons to enable/disable

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

# Continue after hitting a breakpoint
curl -X POST http://localhost:8050/api/continue
```

**Useful conditions:**
- `layer_data['execution_time'] > 0.5` - Slow layers
- `layer_data['memory'] > 1000` - Memory-heavy operations
- `layer_data['grad_norm'] < 0.001` - Vanishing gradients
- `layer_data['mean_activation'] == 0` - Dead layer

**Caveat:** Complex conditions can slow things down. Keep them simple.

### 3. Enhanced Anomaly Detection

We've implemented multiple statistical methods to catch unusual behavior in your layers. It's not perfect, but it catches most obvious issues.

**Detection methods:**

1. **Z-Score Detection** - Flags values more than 3 standard deviations from the mean. This is the basic approach and works well for normally-distributed metrics.

2. **IQR (Interquartile Range) Detection** - More robust when you have outliers. Uses the 1.5 * IQR rule that statisticians like.

3. **Isolation Forest** - Framework for ML-based detection. We haven't fully implemented this yet, but the structure is there for future expansion.

**What it tracks:**
- Execution time anomalies (sudden slowdowns)
- Memory spikes
- Unusual activation patterns
- Gradient anomalies

**Visual indicators:**
- Red bars: Something is definitely weird here
- Orange line: Severity score (how weird is it?)
- Z-scores > 3: This is statistically significant

**Tuning sensitivity:**
You can adjust how sensitive the detector is in your config. Lower values = more sensitive, more false positives. We default to a Z-score threshold of 3, which seems to work well in practice.

**Limitations:**
- Needs some data history to establish baselines
- First few epochs will trigger false positives as patterns stabilize
- Very dynamic networks might trigger more false positives

### 4. Performance Profiling with Flame Graphs

Flame graphs are a standard tool in performance profiling. We've adapted them for neural networks.

**Components:**

#### Flame Graph
- Shows execution time for each layer
- Horizontal bars = duration
- Colors = call depth (useful if you have nested operations)
- Interactive tooltips with precise timing

This helps you quickly spot which layers are consuming the most time. The visual representation makes it easier than staring at numbers.

#### Performance Summary Table
Shows the top 10 slowest layers with:
- Total time across all calls
- Call count
- Average, min, and max execution time

This is useful for identifying optimization targets. Usually, optimizing the top 3 slowest layers gives you 80% of the benefit.

**Usage:**
The profiler runs automatically when layers report timing data. No setup required.

**Profiler API:**
```python
from neural.dashboard.dashboard import profiler

# Manual timing if needed
profiler.start_layer("Conv2D_1")
# ... layer execution ...
profiler.end_layer("Conv2D_1")

# Get the data
summary = profiler.get_summary()
flame_data = profiler.get_flame_graph_data()
```

**Performance tip:** The profiler itself has minimal overhead, but it does keep all timing data in memory. For very long training runs, you might want to periodically clear it.

## Dashboard Layout

We've organized the dashboard to minimize scrolling:

### Top Row
1. **Model Structure** (left) - Your network architecture as a graph
2. **Gradient Flow / Dead Neurons / Anomaly Detection** (right) - Training health at a glance

### Middle Row
1. **Layer-by-Layer Inspector** (left) - Detailed metrics for selected layer
2. **Breakpoint Manager** (right) - Control your breakpoints

### Bottom Sections
1. **Performance Flame Graph** - Where is time being spent?
2. **Performance Summary** - Top slowest layers
3. **Resource Monitoring** - CPU/GPU/Memory usage

## Configuration

Edit `config.yaml` to customize:

```yaml
websocket_interval: 1000  # Update frequency in ms (lower = more responsive, more CPU)

# Anomaly detection sensitivity (0-1)
# Higher = fewer false positives, might miss subtle issues
# Lower = catch more issues, more false positives
anomaly_sensitivity: 0.95

# Breakpoint settings
breakpoints:
  enabled: true
  pause_on_condition: true  # Set to false for logging-only mode
```

## Integration Example

Here's how you'd integrate this into your training code:

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
    print(f"Something unusual in Conv2D_1: {result['scores']}")

# Profile execution
profiler.start_layer("Conv2D_1")
# ... execute layer ...
profiler.end_layer("Conv2D_1")
```

## Development Notes

### Adding New Metrics

To track a new metric:

1. Add it to your trace_data entries
2. The anomaly detector will pick it up automatically (if the name is sensible)
3. Add visualization in the layer inspector callback if you want it displayed

### Custom Anomaly Detection

Extend the `AnomalyDetector` class if you want custom logic:

```python
class AnomalyDetector:
    def detect_custom_anomaly(self, layer_name, metrics):
        # Your detection logic here
        # Return True/False and a score
        pass
```

We've set this up to be extensible, so add whatever makes sense for your use case.

### Performance Considerations

Things to watch out for:

- **Layer inspection history**: Limited to 100 most recent entries per layer. This prevents memory issues during long training runs.
- **Anomaly history**: Capped at 1000 samples. After that, old data gets dropped.
- **Flame graph data**: Keeps everything. For very long runs, you might want to periodically call `profiler.clear()`.

These limits are configurable if you need to change them.

## Troubleshooting

**Layer inspector shows "No data available"**

Make sure the layer name in trace_data matches exactly what you selected. Layer names are case-sensitive.

**Breakpoints not triggering**

- Check the layer name matches exactly (case-sensitive)
- Verify your condition syntax (it's evaluated as Python)
- Check that the layer is actually executing (sounds obvious, but we've all been there)

**Too many/too few anomalies**

Adjust the `anomaly_sensitivity` parameter in config.yaml:
- Default is 0.95
- Lower it (e.g., 0.90) for more sensitivity
- Raise it (e.g., 0.98) for fewer false positives

This usually requires some experimentation to get right for your specific network.

**Flame graph is empty**

Ensure you're calling `profiler.start_layer()` and `profiler.end_layer()` correctly. The calls need to be paired and in the right order.

## Limitations

Let's be honest about what doesn't work well:

- **Overhead**: Running the full dashboard does add overhead. It's minimal, but for production training you might want to disable it.
- **Memory usage**: Long training runs accumulate data. We've added limits, but very long runs can still use significant memory.
- **False positives**: Anomaly detection is statistical. You'll get false positives, especially early in training.
- **Breakpoint performance**: Complex conditions are evaluated on every layer execution. Keep them simple.

## Future Enhancements

Things we're considering:

- Multi-GPU profiling (currently focused on single-GPU scenarios)
- Export flame graph to Chrome tracing format (for use with Chrome DevTools)
- Train the ML-based anomaly detector (Isolation Forest is there but not trained)
- Breakpoint groups (enable/disable multiple breakpoints at once)
- Timeline scrubbing (replay historical executions)
- Comparative analysis (compare multiple training runs side-by-side)

We add features based on what people actually need, so if you have ideas, open an issue.
