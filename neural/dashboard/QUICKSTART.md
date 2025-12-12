# NeuralDbg Dashboard - Quick Start Guide

## Installation

```bash
# Install with full dashboard dependencies
pip install -e ".[full]"
```

## Starting the Dashboard

```bash
# Start the dashboard server
python neural/dashboard/dashboard.py
```

Then open your browser to `http://localhost:8050`

## Basic Usage

### 1. Sending Data to the Dashboard

```python
from neural.dashboard import update_dashboard_data

# Prepare trace data with layer metrics
trace_data = [
    {
        "layer": "Conv2D_1",
        "execution_time": 0.045,
        "memory": 256.5,
        "flops": 1000000,
        "mean_activation": 0.35,
        "grad_norm": 0.02,
        "dead_ratio": 0.01
    },
    # Add more layers...
]

# Update the dashboard
update_dashboard_data(new_trace_data=trace_data)
```

### 2. Using the Profiler

```python
from neural.dashboard import profiler

# Option 1: Manual profiling
profiler.start_layer("Conv2D_1")
# ... your layer code ...
profiler.end_layer("Conv2D_1")

# Option 2: Using decorator
from neural.dashboard.profiler_utils import layer_profiler

@layer_profiler.profile("Conv2D_1")
def my_layer():
    # ... your layer code ...
    pass

# Option 3: Using context manager
from neural.dashboard.profiler_utils import layer_profiler

with layer_profiler.profile_context("Conv2D_1"):
    # ... your layer code ...
    pass
```

### 3. Setting Breakpoints

```python
from neural.dashboard import breakpoint_manager

# Add a simple breakpoint
breakpoint_manager.add_breakpoint("Conv2D_1")

# Add a conditional breakpoint
breakpoint_manager.add_breakpoint(
    "Conv2D_2",
    condition="layer_data['execution_time'] > 0.1"
)

# Via REST API
import requests
requests.post('http://localhost:8050/api/breakpoint', json={
    'layer_name': 'Conv2D_1',
    'condition': 'layer_data["memory"] > 500'
})
```

### 4. Monitoring Anomalies

```python
from neural.dashboard import anomaly_detector

# The detector automatically learns from trace data
# Check for anomalies manually
result = anomaly_detector.detect_anomalies("Conv2D_1", {
    "execution_time": 0.2,  # Unusually high
    "memory": 1000,
    "mean_activation": 0.9
})

if result["is_anomaly"]:
    print(f"Anomaly detected! Scores: {result['scores']}")

# Get statistics
stats = anomaly_detector.get_layer_statistics("Conv2D_1")
print(f"Average execution time: {stats['execution_times']['mean']:.4f}s")
```

### 5. Comprehensive Wrapper (All Features)

```python
from neural.dashboard.profiler_utils import LayerExecutionWrapper

wrapper = LayerExecutionWrapper()

# As a decorator
@wrapper.wrap("MyLayer")
def my_layer(x):
    # ... process x ...
    return result

# As a context manager with custom metrics
with wrapper.wrap_context("MyLayer") as ctx:
    result = my_computation()
    ctx.set_metrics({
        "memory": compute_memory_usage(),
        "mean_activation": result.mean(),
        "custom_metric": some_value
    })
```

## Dashboard Features Overview

### Real-Time Monitoring
- **Model Structure**: Visual representation of your network architecture
- **Layer Performance**: FLOPs and memory usage per layer
- **Gradient Flow**: Gradient magnitude visualization
- **Dead Neurons**: Detection of inactive neurons
- **Resource Monitoring**: CPU/GPU/Memory usage

### New Enhanced Features

#### Layer-by-Layer Inspector
- Select any layer from the dropdown
- View detailed metrics and statistics
- See historical trends

#### Breakpoint Manager
- Add/remove breakpoints on layers
- Set conditional breakpoints
- Track breakpoint hit counts
- Continue/pause execution

#### Enhanced Anomaly Detection
- Multiple detection algorithms (Z-score, IQR)
- Visual anomaly indicators
- Anomaly severity scores
- Historical tracking

#### Performance Profiling
- **Flame Graph**: Visual call stack and timing
- **Performance Summary**: Top slow layers
- Call counts and statistics
- Nested layer support

## REST API Endpoints

### Breakpoints
- `POST /api/breakpoint` - Add breakpoint
- `GET /api/breakpoint` - List breakpoints
- `DELETE /api/breakpoint` - Remove breakpoint
- `POST /api/continue` - Continue execution

### Layer Inspection
- `GET /api/layer_inspect/<layer_name>` - Get layer details

## Configuration

Create `config.yaml` in your project root:

```yaml
# Update interval for dashboard (milliseconds)
websocket_interval: 1000

# Anomaly detection sensitivity (0-1, higher = less sensitive)
anomaly_sensitivity: 0.95

# Breakpoint settings
breakpoints:
  enabled: true
  pause_on_condition: true
```

## Example: Complete Integration

```python
from neural.dashboard import (
    update_dashboard_data,
    breakpoint_manager,
    profiler
)
from neural.dashboard.profiler_utils import LayerExecutionWrapper

# Initialize
wrapper = LayerExecutionWrapper()
breakpoint_manager.add_breakpoint("SlowLayer", "layer_data['execution_time'] > 0.1")

# Define your layers with monitoring
@wrapper.wrap("Conv2D_1")
def conv_layer_1(x):
    return conv2d(x, filters=32)

@wrapper.wrap("Conv2D_2")
def conv_layer_2(x):
    return conv2d(x, filters=64)

# Execute your model
input_data = load_data()
x = conv_layer_1(input_data)
x = conv_layer_2(x)

# View results in dashboard at http://localhost:8050
```

## Troubleshooting

**Dashboard won't start**
- Ensure all dependencies are installed: `pip install -e ".[full]"`
- Check port 8050 is available

**No data showing**
- Verify trace_data format includes required fields
- Call `update_dashboard_data()` after generating data

**Breakpoints not working**
- Check layer names match exactly (case-sensitive)
- Verify condition syntax is valid Python

**Anomaly detection too sensitive**
- Increase `anomaly_sensitivity` in config
- Ensure enough training data (>10 samples per layer)

## Next Steps

- See `neural/dashboard/README.md` for detailed feature documentation
- Run `examples/dashboard_enhanced_usage.py` for complete examples
- Explore the dashboard UI to discover all features

## Support

For issues or questions:
- Check the README.md for detailed documentation
- Review example code in `examples/`
- Examine the source code in `neural/dashboard/`
