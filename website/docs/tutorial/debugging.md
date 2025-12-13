---
sidebar_position: 4
---

# Debugging with NeuralDbg

Learn to debug neural networks effectively with NeuralDbg.

## Starting the Debugger

```bash
neural debug model.neural
```

Open http://localhost:8050 in your browser.

## Common Debugging Scenarios

### 1. Training Not Converging

**Symptoms:**
- Loss not decreasing
- Accuracy stuck
- Slow learning

**Debug Steps:**

```bash
# Check gradient flow
neural debug model.neural --gradients

# Look for:
# - Vanishing gradients (very small values)
# - Exploding gradients (very large values)
# - Dead neurons (zero activations)
```

**Solutions:**
- Adjust learning rate
- Add/remove batch normalization
- Change activation functions
- Check data preprocessing

### 2. Loss Becomes NaN

**Symptoms:**
- Training starts, then loss → NaN
- Weights become infinite

**Debug Steps:**

```bash
# Monitor anomalies
neural debug model.neural --anomalies

# Check:
# - Gradient explosions
# - Data issues (NaN/Inf in inputs)
# - Too high learning rate
```

**Solutions:**
- Lower learning rate
- Use gradient clipping
- Check data for NaN/Inf
- Add batch normalization

### 3. Model Overfitting

**Symptoms:**
- Training accuracy high
- Validation accuracy low
- Large gap between train/val

**Debug Steps:**

```bash
# Monitor training progress
neural debug model.neural

# Watch for:
# - Diverging train/val metrics
# - Increasing validation loss
```

**Solutions:**
- Add dropout
- Reduce model capacity
- Use data augmentation
- Early stopping

## Dashboard Features

### Execution Trace
- See layer-by-layer execution
- Monitor shapes and timing
- Identify bottlenecks

### Gradient Analysis
- Visualize gradient flow
- Detect vanishing/exploding
- Per-layer statistics

### Activation Statistics
- Mean, std, min, max
- Distribution plots
- Dead neuron detection

### Memory Profiling
- Memory usage per layer
- Peak memory tracking
- Optimization suggestions

## Best Practices

1. **Debug early**: Don't wait for full training
2. **Check gradients**: Monitor gradient health
3. **Watch activations**: Ensure neurons are firing
4. **Profile memory**: Optimize for large models
5. **Use step mode**: Inspect specific layers

## Example Workflow

```bash
# 1. Start with validation
neural validate model.neural

# 2. Check shapes visually
neural visualize model.neural --format html

# 3. Debug first epoch
neural debug model.neural --step

# 4. Monitor full training
neural debug model.neural

# 5. Export when satisfied
neural export model.neural --format onnx
```

## Next Steps

- [NeuralDbg Features](/docs/features/neuraldbg)
- [Deployment](deployment)
