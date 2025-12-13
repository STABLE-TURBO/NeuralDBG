---
sidebar_position: 1
---

# Tutorial Basics

Learn the fundamentals of Neural DSL through hands-on examples.

## Network Structure

Every Neural DSL model follows this structure:

```yaml
network NetworkName {
  input: (dimensions)
  
  layers:
    LayerType(parameters)
    ...
  
  loss: "loss_function"
  optimizer: OptimizerType(parameters)
  
  train {
    epochs: number
    batch_size: number
  }
}
```

## Input Definition

Define input shape as a tuple:

```yaml
# Grayscale image
input: (28, 28, 1)

# RGB image
input: (224, 224, 3)

# Sequence
input: (None, 100)  # Variable length sequences

# Tabular data
input: (784,)  # Flat vector
```

## Layers

### Convolutional Layers

```yaml
# 2D convolution
Conv2D(filters=32, kernel_size=(3,3), activation="relu")

# With padding
Conv2D(32, (3,3), "relu", padding="same")

# With stride
Conv2D(32, (3,3), "relu", strides=(2,2))
```

### Pooling Layers

```yaml
# Max pooling
MaxPooling2D(pool_size=(2,2))

# Average pooling
AveragePooling2D((2,2))

# Global pooling
GlobalMaxPooling2D()
GlobalAveragePooling2D()
```

### Dense Layers

```yaml
# Fully connected
Dense(units=128, activation="relu")

# Output layer
Output(units=10, activation="softmax")
```

### Normalization

```yaml
BatchNormalization()
LayerNormalization()
```

### Regularization

```yaml
Dropout(rate=0.5)
```

## Training Configuration

```yaml
train {
  epochs: 15
  batch_size: 64
  validation_split: 0.2
  
  callbacks: [
    "EarlyStopping(patience=3)",
    "ModelCheckpoint(filepath='best.h5')"
  ]
}
```

## Complete Example

```yaml
network ImageClassifier {
  input: (32, 32, 3)
  
  layers:
    Conv2D(32, (3,3), "relu")
    BatchNormalization()
    MaxPooling2D((2,2))
    
    Conv2D(64, (3,3), "relu")
    BatchNormalization()
    MaxPooling2D((2,2))
    
    Flatten()
    Dense(128, "relu")
    Dropout(0.5)
    Output(10, "softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 50
    batch_size: 32
    validation_split: 0.2
  }
}
```

## Next Steps

- [Working with Layers](layers)
- [Training Configuration](training)
- [Debugging Models](debugging)
