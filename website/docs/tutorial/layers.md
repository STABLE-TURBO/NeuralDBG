---
sidebar_position: 2
---

# Working with Layers

Comprehensive guide to using layers in Neural DSL.

## Core Layer Types

### Convolutional Layers

```yaml
# 2D Convolution
Conv2D(filters=32, kernel_size=(3,3), activation="relu")

# With padding and stride
Conv2D(32, (3,3), "relu", padding="same", strides=(2,2))

# 1D Convolution for sequences
Conv1D(filters=64, kernel_size=3, activation="relu")
```

### Dense (Fully Connected) Layers

```yaml
# Basic dense layer
Dense(units=128, activation="relu")

# Output layer for classification
Output(units=10, activation="softmax")

# Binary classification
Output(units=1, activation="sigmoid")
```

### Recurrent Layers

```yaml
# LSTM
LSTM(units=64, return_sequences=True)
LSTM(64, return_sequences=False)

# GRU
GRU(units=32, return_sequences=True)

# Simple RNN
SimpleRNN(units=16)
```

### Transformer Layers

```yaml
# Transformer Encoder
TransformerEncoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1)

# Transformer Decoder with cross-attention
# Supports causal masking for autoregressive decoding
TransformerDecoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1, use_causal_mask=true)

# Stacked transformer blocks
TransformerEncoder(num_heads=8, ff_dim=512) * 6
TransformerDecoder(num_heads=8, ff_dim=512) * 6
```

### Pooling Layers

```yaml
# Max pooling
MaxPooling2D(pool_size=(2,2))
MaxPooling1D(pool_size=2)

# Average pooling
AveragePooling2D((2,2))

# Global pooling
GlobalMaxPooling2D()
GlobalAveragePooling2D()
```

## Normalization Layers

```yaml
# Batch normalization
BatchNormalization()

# Layer normalization
LayerNormalization()
```

## Regularization Layers

```yaml
# Dropout
Dropout(rate=0.5)

# Spatial dropout for CNNs
SpatialDropout2D(rate=0.2)
```

## Activation Layers

```yaml
# ReLU
Activation("relu")

# Leaky ReLU
LeakyReLU(alpha=0.2)

# PReLU
PReLU()
```

## Reshape Layers

```yaml
# Flatten
Flatten()

# Reshape
Reshape(target_shape=(7, 7, 512))
```

## Example Architectures

### CNN for Image Classification

```yaml
network ImageCNN {
  input: (224, 224, 3)
  
  layers:
    Conv2D(64, (3,3), "relu", padding="same")
    BatchNormalization()
    MaxPooling2D((2,2))
    
    Conv2D(128, (3,3), "relu", padding="same")
    BatchNormalization()
    MaxPooling2D((2,2))
    
    GlobalAveragePooling2D()
    Dense(256, "relu")
    Dropout(0.5)
    Output(10, "softmax")
}
```

### RNN for Sequence Processing

```yaml
network SequenceRNN {
  input: (None, 100)
  
  layers:
    LSTM(64, return_sequences=True)
    Dropout(0.3)
    LSTM(32)
    Dense(64, "relu")
    Output(1, "sigmoid")
}
```

### Transformer Encoder-Decoder for Sequence-to-Sequence

```yaml
network Seq2SeqTransformer {
  input: (None, 100, 512)
  
  layers:
    # Encoder stack
    TransformerEncoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1)
    TransformerEncoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1)
    
    # Decoder stack with cross-attention
    TransformerDecoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1, use_causal_mask=true)
    TransformerDecoder(num_heads=8, d_model=512, ff_dim=2048, dropout=0.1, use_causal_mask=true)
    
    # Output projection
    Dense(10000, "softmax")
    
  optimizer: Adam(learning_rate=0.0001)
  loss: sparse_categorical_crossentropy
}
```

## Best Practices

1. Use BatchNormalization after Conv layers
2. Add Dropout for regularization
3. Use GlobalPooling before Dense layers in CNNs
4. Start with standard activations (ReLU, softmax)
5. Match output layer activation to task

## Next Steps

- [Training Configuration](training)
- [Debugging Models](debugging)
