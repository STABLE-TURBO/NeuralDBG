---
sidebar_position: 1
---

# DSL Syntax

Complete reference for Neural DSL syntax.

## Basic Structure

```yaml
network NetworkName {
  input: shape
  layers: [layer_definitions]
  loss: loss_function
  optimizer: optimizer_config
  train: training_config
}
```

## Data Types

### Shapes

Tuples representing tensor dimensions:

```yaml
input: (28, 28, 1)          # 3D tensor
input: (None, 100)           # Variable first dimension
input: (784,)                # 1D tensor
```

### Numbers

```yaml
filters: 32                  # Integer
learning_rate: 0.001         # Float
rate: 0.5                    # Decimal
```

### Strings

```yaml
activation: "relu"
loss: "categorical_crossentropy"
```

### Lists

```yaml
metrics: ["accuracy", "precision"]
callbacks: ["EarlyStopping(patience=3)"]
```

## Layer Syntax

### Basic Form

```yaml
LayerType(param1=value1, param2=value2)
```

### Shorthand

```yaml
Conv2D(32, (3,3), "relu")    # Positional parameters
Dense(128, "relu")            # Common case
Output(10, "softmax")        # Output layer
```

### Full Form

```yaml
Conv2D(
  filters=32,
  kernel_size=(3,3),
  activation="relu",
  padding="same"
)
```

## Activation Functions

```yaml
"relu"
"sigmoid"
"tanh"
"softmax"
"linear"
"elu"
"selu"
"leaky_relu"
```

## Loss Functions

```yaml
"categorical_crossentropy"
"sparse_categorical_crossentropy"
"binary_crossentropy"
"mse"              # Mean Squared Error
"mae"              # Mean Absolute Error
"huber"
```

## Optimizers

```yaml
Adam(learning_rate=0.001)
SGD(learning_rate=0.01, momentum=0.9)
RMSprop(learning_rate=0.001)
Adagrad(learning_rate=0.01)
```

## Training Configuration

```yaml
train {
  epochs: 100
  batch_size: 32
  validation_split: 0.2
  shuffle: true
  
  callbacks: [
    "EarlyStopping(patience=5)",
    "ReduceLROnPlateau(factor=0.5)",
    "ModelCheckpoint(filepath='model.h5')"
  ]
}
```

## Comments

```yaml
# This is a comment
network MyModel {
  input: (28, 28, 1)  # Input shape
  
  layers:
    # Feature extraction
    Conv2D(32, (3,3), "relu")
    
    # Classification
    Dense(10, "softmax")
}
```

## Best Practices

1. **Use descriptive names**: `network ImageClassifier` not `network M1`
2. **Comment complex sections**: Explain non-obvious choices
3. **Consistent formatting**: Use same indentation throughout
4. **Group related layers**: Separate feature extraction from classification
5. **Document hyperparameters**: Note why specific values were chosen

## Examples

See [Tutorial: Basics](/docs/tutorial/basics) for complete examples.
