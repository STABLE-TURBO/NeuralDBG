---
sidebar_position: 1
---

# NeuralDbg - Built-in Debugger

NeuralDbg is Neural DSL's built-in neural network debugger providing real-time execution tracing, gradient analysis, and anomaly detection.

## Features

### 🔍 Real-Time Execution Tracing
Monitor layer-by-layer execution, activations, and tensor shapes as your model trains.

### 📈 Gradient Flow Analysis
Detect vanishing and exploding gradients with interactive visualizations.

### 💀 Dead Neuron Detection
Identify inactive neurons that never fire during training.

### ⚠️ Anomaly Detection
Automatically flag NaNs, infinite values, and extreme activations.

### 📊 Memory & FLOP Profiling
Track memory usage and floating-point operations per layer.

### 🎯 Step Debugging
Pause execution at any layer and inspect tensors manually.

## Quick Start

Start the debugger:

```bash
neural debug model.neural
```

Open http://localhost:8050 in your browser.

## Dashboard Overview

The NeuralDbg dashboard provides:

### Execution Trace Tab
- Layer-by-layer execution timeline
- Input/output shapes
- Execution time per layer
- Memory consumption

### Gradient Analysis Tab
- Gradient magnitude per layer
- Gradient flow visualization
- Vanishing/exploding gradient detection
- Histogram of gradients

### Activation Statistics Tab
- Mean, std, min, max per layer
- Activation distribution
- Dead neuron identification
- Saturation detection

### Memory Profiling Tab
- Memory usage over time
- Peak memory per layer
- Memory leaks detection
- Optimization suggestions

### Anomaly Detection Tab
- NaN/Inf detection
- Extreme activation alerts
- Weight explosion warnings
- Training instability indicators

## Usage Examples

### Basic Debugging

```bash
# Start debugger
neural debug mnist.neural

# With specific backend
neural debug mnist.neural --backend pytorch

# Custom port
neural debug mnist.neural --port 8888
```

### Gradient Analysis

```bash
# Focus on gradient issues
neural debug model.neural --gradients

# With custom thresholds
neural debug model.neural --gradients --vanishing-threshold 1e-5
```

### Dead Neuron Detection

```bash
# Identify inactive neurons
neural debug model.neural --dead-neurons

# With activation threshold
neural debug model.neural --dead-neurons --activation-threshold 0.01
```

### Anomaly Detection

```bash
# Monitor for training issues
neural debug model.neural --anomalies

# With alerts
neural debug model.neural --anomalies --alert-on-nan
```

### Step Debugging

```bash
# Pause at each layer
neural debug model.neural --step

# Pause at specific layer
neural debug model.neural --breakpoint Dense_1
```

## Integration with Cloud

Debug models running in cloud environments:

```python
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()

# Start debugger with tunnel for remote access
dashboard_info = executor.start_debug_dashboard(
    dsl_code,
    setup_tunnel=True
)

print(f"Access debugger at: {dashboard_info['tunnel_url']}")
```

## Advanced Features

### Custom Hooks

Add custom inspection hooks:

```python
from neural.dashboard.hooks import register_hook

@register_hook('after_layer')
def inspect_activations(layer_name, activations):
    print(f"{layer_name}: mean={activations.mean()}")
```

### Export Debug Data

Save debugging session:

```bash
neural debug model.neural --export debug_session.json
```

### Compare Runs

Compare multiple debugging sessions:

```bash
neural debug model.neural --compare debug_session_1.json debug_session_2.json
```

## Troubleshooting Common Issues

### Vanishing Gradients

**Symptoms:**
- Gradients approaching zero
- Training stalls early
- Weights not updating

**Solutions:**
1. Use batch normalization
2. Switch to ReLU activation
3. Reduce network depth
4. Use skip connections

**NeuralDbg helps by:**
- Showing exactly which layers have vanishing gradients
- Visualizing gradient flow
- Suggesting architecture changes

### Exploding Gradients

**Symptoms:**
- Gradients become very large
- Loss becomes NaN
- Training diverges

**Solutions:**
1. Reduce learning rate
2. Use gradient clipping
3. Add batch normalization
4. Check data preprocessing

**NeuralDbg helps by:**
- Detecting gradient explosions early
- Identifying problematic layers
- Recommending gradient clipping values

### Dead Neurons

**Symptoms:**
- Neurons never activate
- Reduced model capacity
- Poor performance

**Solutions:**
1. Use Leaky ReLU instead of ReLU
2. Check weight initialization
3. Reduce dropout rate
4. Adjust learning rate

**NeuralDbg helps by:**
- Identifying which neurons are dead
- Showing activation patterns
- Suggesting alternative activations

## Performance Considerations

NeuralDbg adds minimal overhead:
- ~5-10% slower training
- Memory overhead: ~100MB
- Negligible impact on accuracy

Disable for production:

```bash
# Training without debugger
neural run model.neural --backend tensorflow
```

## Dashboard Customization

Configure dashboard appearance in `neural_config.yaml`:

```yaml
dashboard:
  theme: dark
  refresh_rate: 1.0  # seconds
  max_history: 1000  # data points
  charts:
    - execution_trace
    - gradient_flow
    - memory_profile
```

## API Reference

See [API: NeuralDbg](/docs/api/neuraldbg) for programmatic access.

## Examples

Check out example debugging sessions:
- [Debugging CNN](/docs/guides/debugging-cnn)
- [Debugging RNN](/docs/guides/debugging-rnn)
- [Debugging GAN](/docs/guides/debugging-gan)

## Learn More

- [Tutorial: Debugging](/docs/tutorial/debugging)
- [Guide: Performance Optimization](/docs/guides/optimization)
- [Video: Debugging Workshop](https://youtube.com/watch?v=example)
