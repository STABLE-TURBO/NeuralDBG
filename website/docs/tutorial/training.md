---
sidebar_position: 3
---

# Training Configuration

Configure training parameters in Neural DSL.

## Basic Training Block

```yaml
train {
  epochs: 50
  batch_size: 32
  validation_split: 0.2
}
```

## Loss Functions

### Classification

```yaml
# Multi-class classification
loss: "categorical_crossentropy"

# Multi-class with integer labels
loss: "sparse_categorical_crossentropy"

# Binary classification
loss: "binary_crossentropy"
```

### Regression

```yaml
# Mean Squared Error
loss: "mse"

# Mean Absolute Error
loss: "mae"

# Huber loss
loss: "huber"
```

## Optimizers

```yaml
# Adam (recommended)
optimizer: Adam(learning_rate=0.001)

# SGD with momentum
optimizer: SGD(learning_rate=0.01, momentum=0.9)

# RMSprop
optimizer: RMSprop(learning_rate=0.001)
```

## Learning Rate Schedules

```yaml
scheduler: ExponentialDecay(
  initial_learning_rate=0.001,
  decay_steps=1000,
  decay_rate=0.9
)
```

## Metrics

```yaml
metrics: ["accuracy", "precision", "recall"]
```

## Callbacks

```yaml
train {
  epochs: 100
  batch_size: 32
  
  callbacks: [
    "EarlyStopping(patience=10, restore_best_weights=true)",
    "ReduceLROnPlateau(factor=0.5, patience=5)",
    "ModelCheckpoint(filepath='best_model.h5', save_best_only=true)"
  ]
}
```

## Complete Example

```yaml
network TrainedModel {
  input: (28, 28, 1)
  
  layers:
    Conv2D(32, (3,3), "relu")
    MaxPooling2D((2,2))
    Flatten()
    Dense(128, "relu")
    Dropout(0.5)
    Output(10, "softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  
  train {
    epochs: 50
    batch_size: 64
    validation_split: 0.2
    shuffle: true
    
    callbacks: [
      "EarlyStopping(patience=5)",
      "ModelCheckpoint(filepath='best.h5')"
    ]
  }
}
```

## Next Steps

- [Debugging Models](debugging)
- [Deployment](deployment)
