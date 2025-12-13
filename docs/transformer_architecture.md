# Transformer Architecture Guide

## Overview

Transformers are a powerful neural network architecture introduced in "Attention is All You Need" (Vaswani et al., 2017). They have revolutionized natural language processing and are increasingly used in computer vision, audio processing, and multimodal tasks.

This guide explains how to build transformer architectures using Neural DSL, covering both the theoretical foundations and practical implementation.

## Core Components

### 1. TransformerEncoder

The encoder processes input sequences and generates contextualized representations.

```yaml
network BasicEncoder {
  input: (512, 768)  # (sequence_length, embedding_dim)
  layers:
    TransformerEncoder(
      num_heads=8,
      d_model=768,
      dff=2048,
      dropout_rate=0.1
    )
    GlobalAveragePooling1D()
    Dense(256, activation="relu")
    Output(10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

**Parameters:**
- `num_heads`: Number of attention heads (typically 8, 12, or 16)
- `d_model`: Model dimension (embedding size, e.g., 512, 768, 1024)
- `dff`: Feed-forward network dimension (typically 4x d_model)
- `dropout_rate`: Dropout probability for regularization (0.0-0.3)

### 2. TransformerDecoder

The decoder generates output sequences conditioned on encoder outputs (for seq2seq tasks).

```yaml
network EncoderDecoder {
  input: (100, 512)  # Source sequence
  layers:
    Embedding(input_dim=10000, output_dim=512)
    TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6
    
    # Decoder takes encoder output as context
    TransformerDecoder(num_heads=8, d_model=512, dff=2048) * 6
    Dense(10000, activation="softmax")  # Vocabulary size
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

### 3. Multi-Head Attention

While typically encapsulated within TransformerEncoder/Decoder layers, you can use attention mechanisms independently:

```yaml
network CustomAttention {
  input: (100, 512)
  layers:
    MultiHeadAttention(num_heads=8, key_dim=64)
    LayerNormalization()
    Dense(512, activation="relu")
    Output(10, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
}
```

## Stacking Transformer Layers

Neural DSL supports layer repetition using the `*` operator, which is especially useful for transformers:

```yaml
# Stack 12 encoder layers (BERT-base style)
TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12

# Stack 6 encoder and 6 decoder layers (original Transformer)
TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6
TransformerDecoder(num_heads=8, d_model=512, dff=2048) * 6
```

## Common Architectures

### Vision Transformer (ViT)

Apply transformers to image patches:

```yaml
network VisionTransformer {
  input: (224, 224, 3)
  layers:
    # Patch embedding: convert image to sequence of patches
    Conv2D(768, kernel_size=(16, 16), strides=16, padding="valid")
    Reshape(target_shape=(196, 768))  # (14*14 patches, embedding_dim)
    
    # Add positional embeddings (handled automatically in some backends)
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    
    # Classification head
    GlobalAveragePooling1D()
    Dense(1000, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

### BERT-style Encoder

Bidirectional encoder for text understanding:

```yaml
network BERTEncoder {
  input: (512, 768)  # Max sequence length, embedding dimension
  layers:
    # Pre-embedded input (typically done in preprocessing)
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    
    # Task-specific head
    GlobalAveragePooling1D()
    Dense(768, activation="tanh")
    Dropout(0.1)
    Output(2, activation="softmax")  # Binary classification
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=2e-5)
  train {
    epochs: 3
    batch_size: 32
  }
}
```

### GPT-style Decoder

Autoregressive language model:

```yaml
network GPTDecoder {
  input: (1024, 768)  # Context length, embedding dimension
  layers:
    TransformerDecoder(num_heads=12, d_model=768, dff=3072) * 12
    
    # Language modeling head
    Dense(50257, activation="softmax")  # Vocabulary size
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(
    learning_rate=ExponentialDecay(
      6e-4,      # Initial learning rate
      10000,     # Decay steps
      0.95       # Decay rate
    )
  )
  train {
    epochs: 10
    batch_size: 8
  }
}
```

### Transformer for Sequence Classification

Text or sequence classification with transformers:

```yaml
network TransformerClassifier {
  input: (128, 512)  # Sequence length, embedding dimension
  layers:
    Embedding(input_dim=30000, output_dim=512)
    TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6
    
    # Pooling strategy
    GlobalAveragePooling1D()  # Or use first token (CLS token)
    
    # Classification layers
    Dense(256, activation="relu")
    Dropout(0.3)
    Dense(128, activation="relu")
    Dropout(0.2)
    Output(5, activation="softmax")  # 5 classes
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
  train {
    epochs: 20
    batch_size: 64
    validation_split: 0.2
  }
}
```

## Input Shape Considerations

### Sequence Length and Embedding Dimension

Transformer inputs must specify both sequence length and embedding dimension:

```yaml
# Text: (max_sequence_length, embedding_dim)
input: (512, 768)

# Vision: Convert images to sequences
# (batch, height, width, channels) → (batch, num_patches, patch_embedding_dim)
input: (224, 224, 3)
Conv2D(768, kernel_size=(16, 16), strides=16)  # Creates 14x14=196 patches
Reshape(target_shape=(196, 768))
```

### Variable Length Sequences

While the model defines a maximum sequence length, shorter sequences are typically padded:

```yaml
# Maximum sequence length of 512 tokens
input: (512, 768)

# In practice, use padding and masking in your data pipeline
# Neural DSL backends handle attention masking automatically
```

## Positional Encoding

Transformers need positional information since they lack inherent sequence order:

```yaml
network TransformerWithPositionalEncoding {
  input: (100, 512)
  layers:
    # Positional encoding is typically handled automatically by backend implementations
    # or can be added as a preprocessing step
    
    TransformerEncoder(num_heads=8, d_model=512, dff=2048)
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

**Note:** Most backend implementations (TensorFlow, PyTorch) include positional encoding within their transformer layers. If you need custom positional encoding, implement it in your data preprocessing pipeline.

## Attention Masking

### Padding Mask

Prevents attention to padding tokens:

```yaml
# Handled automatically by backend when using standard implementations
# Provide a padding mask in your training data
```

### Causal Mask (Look-Ahead Mask)

For autoregressive models (GPT-style), prevents attention to future tokens:

```yaml
network CausalTransformer {
  input: (512, 768)
  layers:
    # Decoder layers automatically apply causal masking
    TransformerDecoder(num_heads=8, d_model=768, dff=2048) * 6
    Dense(10000, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

## Normalization Strategies

### Pre-Norm vs Post-Norm

Modern transformers typically use Pre-LayerNorm for better training stability:

```yaml
# Pre-Norm (recommended for deep models)
network PreNormTransformer {
  input: (100, 512)
  layers:
    # Most modern TransformerEncoder implementations use Pre-Norm by default
    TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 12
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

## Architecture Scaling

### Model Size Guidelines

| Model Size | num_heads | d_model | dff   | num_layers |
|------------|-----------|---------|-------|------------|
| Small      | 4         | 256     | 1024  | 4          |
| Base       | 8         | 512     | 2048  | 6          |
| Medium     | 8         | 768     | 3072  | 12         |
| Large      | 16        | 1024    | 4096  | 24         |
| XL         | 24        | 1536    | 6144  | 48         |

```yaml
# Example: Base model
TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6

# Example: Large model
TransformerEncoder(num_heads=16, d_model=1024, dff=4096) * 24
```

## Device Placement

For large transformer models, GPU acceleration is essential:

```yaml
network LargeTransformer {
  input: (512, 1024)
  layers:
    # Place transformer layers on GPU
    TransformerEncoder(num_heads=16, d_model=1024, dff=4096) @ "cuda:0" * 24
    GlobalAveragePooling1D() @ "cuda:0"
    Dense(1000) @ "cuda:0"
  
  optimizer: Adam(learning_rate=0.0001)
  execution {
    device: "cuda:0"
  }
}
```

## Next Steps

- [Attention Mechanism Explained](transformer_attention.md) - Deep dive into attention
- [Training Best Practices](transformer_training.md) - Optimize transformer training
- [Migration Guide](transformer_migration.md) - Convert existing transformer code
