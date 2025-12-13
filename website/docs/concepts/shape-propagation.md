---
sidebar_position: 2
---

# Shape Propagation

Understanding how Neural DSL validates tensor shapes throughout your network.

## What is Shape Propagation?

Shape propagation is the process of tracking tensor dimensions through each layer of your network. Neural DSL automatically validates that shapes are compatible before any training occurs.

## Why It Matters

Traditional frameworks only check shapes at runtime, meaning you might:
- Wait hours for training to start
- Encounter shape errors deep into training
- Waste GPU resources on invalid models

Neural DSL catches these errors **before runtime**.

## How It Works

### 1. Define Input Shape

```yaml
network MyModel {
  input: (28, 28, 1)
  # Shape: (None, 28, 28, 1) - batch dimension added automatically
}
```

### 2. Track Through Layers

```yaml
layers:
  Conv2D(32, (3,3), "relu")
  # Input:  (None, 28, 28, 1)
  # Output: (None, 26, 26, 32)
  
  MaxPooling2D((2,2))
  # Input:  (None, 26, 26, 32)
  # Output: (None, 13, 13, 32)
  
  Flatten()
  # Input:  (None, 13, 13, 32)
  # Output: (None, 5408)
  
  Dense(128, "relu")
  # Input:  (None, 5408)
  # Output: (None, 128)
  
  Output(10, "softmax")
  # Input:  (None, 128)
  # Output: (None, 10)
```

### 3. Validate Compatibility

Neural DSL ensures:
- Conv kernels don't exceed input dimensions
- Dense layers receive flattened input
- Output dimensions match expected classes

## Visualizing Shapes

Generate interactive visualizations:

```bash
neural visualize model.neural --format html
```

This creates:
- **shape_propagation.html**: Interactive shape flow diagram
- **tensor_flow.html**: Detailed tensor transformations

## Common Shape Errors

### Error 1: Dense Without Flatten

```yaml
# ❌ Wrong
layers:
  Conv2D(32, (3,3), "relu")
  Dense(128, "relu")  # Error! Expects 1D input

# ✅ Correct
layers:
  Conv2D(32, (3,3), "relu")
  Flatten()           # Convert to 1D
  Dense(128, "relu")
```

### Error 2: Kernel Too Large

```yaml
# ❌ Wrong
input: (28, 28, 1)
layers:
  Conv2D(32, (30,30), "relu")  # Error! Kernel > input

# ✅ Correct
input: (28, 28, 1)
layers:
  Conv2D(32, (3,3), "relu", padding="same")
```

### Error 3: Output Mismatch

```yaml
# ❌ Wrong
layers:
  ...
  Output(10, "softmax")  # 10 classes

# But training data has 5 classes - Runtime error!

# ✅ Correct
layers:
  ...
  Output(5, "softmax")   # Matches data
```

## Dynamic Shapes

Handle variable-length sequences:

```yaml
input: (None, 100)  # Variable sequence length

layers:
  LSTM(64, return_sequences=True)
  # Input:  (None, None, 100)
  # Output: (None, None, 64)
  
  LSTM(32)
  # Input:  (None, None, 64)
  # Output: (None, 32)
```

## Debugging Shape Issues

Use the debugger to inspect shapes:

```bash
neural debug model.neural --shapes
```

This shows:
- Input/output shapes for each layer
- Where shape mismatches occur
- Suggested fixes

## Shape Constraints

### Convolutional Layers

```python
output_height = (input_height - kernel_height + 2*padding) / stride + 1
output_width = (input_width - kernel_width + 2*padding) / stride + 1
output_channels = filters
```

### Pooling Layers

```python
output_height = input_height / pool_height
output_width = input_width / pool_width
output_channels = input_channels  # Unchanged
```

### Dense Layers

```python
output_shape = (batch_size, units)
# Requires 2D input: (batch_size, features)
```

## Best Practices

1. **Start with input**: Always define input shape first
2. **Use Flatten before Dense**: Conv/Pool → Flatten → Dense
3. **Check output dimensions**: Match your problem (classification, regression)
4. **Visualize early**: Use `neural visualize` to see shape flow
5. **Use padding="same"**: Maintain spatial dimensions in conv layers

## Advanced: Custom Shapes

For complex architectures:

```yaml
network MultiInput {
  # Multiple inputs
  input_image: (224, 224, 3)
  input_metadata: (10,)
  
  layers:
    # Process image
    [input_image] Conv2D(32, (3,3), "relu")
    Flatten()
    
    # Concatenate with metadata
    Concatenate([flattened_image, input_metadata])
    
    Dense(128, "relu")
    Output(1, "sigmoid")
}
```

## Learn More

- [Tutorial: Working with Layers](/docs/tutorial/layers)
- [Guide: Custom Architectures](/docs/guides/custom-architectures)
- [API: Layer Reference](/docs/api/layers)
