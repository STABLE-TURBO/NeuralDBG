# Attention Mechanism Explained

## Overview

Attention is the core mechanism that enables transformers to weigh the importance of different parts of the input when processing sequences. This guide explains how attention works in Neural DSL transformers.

## What is Attention?

Attention allows models to focus on relevant parts of the input sequence when producing each output element. Unlike RNNs that process sequences sequentially, attention mechanisms enable parallel processing and long-range dependencies.

### Key Concepts

**Query (Q)**: What we're looking for  
**Key (K)**: What we're comparing against  
**Value (V)**: The actual information to retrieve  

The attention mechanism computes how much each value should contribute to the output based on the similarity between queries and keys.

## Scaled Dot-Product Attention

The fundamental attention operation:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q, K, V are query, key, and value matrices
- d_k is the dimension of the keys (used for scaling)
- softmax normalizes attention scores to sum to 1

### Why Scaling?

Dividing by √d_k prevents the dot products from becoming too large, which would push softmax into regions with small gradients.

## Multi-Head Attention

Instead of a single attention operation, multi-head attention runs multiple attention operations in parallel:

```yaml
network MultiHeadExample {
  input: (100, 512)
  layers:
    # 8 parallel attention heads
    MultiHeadAttention(num_heads=8, key_dim=64)
    LayerNormalization()
    Dense(512, activation="relu")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
}
```

### Benefits of Multiple Heads

1. **Different Representation Subspaces**: Each head can learn different aspects of the relationships
2. **Ensemble Effect**: Multiple heads provide redundancy and robustness
3. **Parallel Processing**: All heads compute simultaneously

### Head Configuration

```yaml
# Number of heads must divide d_model evenly
TransformerEncoder(num_heads=8, d_model=512, dff=2048)
# Each head dimension: 512 / 8 = 64

# More heads for larger models
TransformerEncoder(num_heads=16, d_model=1024, dff=4096)
# Each head dimension: 1024 / 16 = 64
```

**Typical head dimension**: 64 or 128  
**Typical number of heads**: 8, 12, or 16

## Self-Attention

In self-attention, the queries, keys, and values all come from the same input sequence:

```yaml
network SelfAttentionExample {
  input: (128, 512)
  layers:
    Embedding(input_dim=30000, output_dim=512)
    
    # Self-attention: each token attends to all other tokens
    TransformerEncoder(num_heads=8, d_model=512, dff=2048)
    
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

### What Self-Attention Learns

- **Local patterns**: Words that commonly appear together
- **Syntactic relationships**: Subject-verb agreement, noun-modifier relationships
- **Long-range dependencies**: Anaphora resolution (pronouns to their referents)
- **Semantic similarity**: Related concepts even far apart in the sequence

## Cross-Attention

In cross-attention (encoder-decoder attention), queries come from one sequence while keys and values come from another:

```yaml
network EncoderDecoderWithCrossAttention {
  input: (100, 512)
  layers:
    Embedding(input_dim=10000, output_dim=512)
    
    # Encoder: self-attention
    TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6
    
    # Decoder: self-attention + cross-attention to encoder
    TransformerDecoder(num_heads=8, d_model=512, dff=2048) * 6
    
    Dense(10000, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

Cross-attention enables the decoder to focus on relevant parts of the encoded input when generating each output token.

## Attention Patterns

### Bidirectional Attention (BERT-style)

Each position attends to all positions (past and future):

```yaml
network BidirectionalAttention {
  input: (512, 768)
  layers:
    # Bidirectional: sees full context
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(2, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=2e-5)
}
```

**Use cases**: Text classification, named entity recognition, question answering

### Causal Attention (GPT-style)

Each position only attends to past positions (autoregressive):

```yaml
network CausalAttention {
  input: (512, 768)
  layers:
    # Causal: only sees past context
    TransformerDecoder(num_heads=12, d_model=768, dff=3072) * 12
    Dense(50257, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

**Use cases**: Language generation, autoregressive prediction

## Attention Masking

### Padding Mask

Prevents attention to padding tokens in variable-length sequences:

```python
# In your data pipeline (not in Neural DSL)
attention_mask = (input_ids != pad_token_id).astype(float)
```

Most Neural DSL backends handle padding masks automatically when you provide them in the training data.

### Causal Mask

For autoregressive models, prevents attending to future tokens:

```
Position:  1  2  3  4
Token 1:  [✓]
Token 2:  [✓][✓]
Token 3:  [✓][✓][✓]
Token 4:  [✓][✓][✓][✓]
```

TransformerDecoder layers apply causal masking automatically.

### Combined Masks

You can combine padding and causal masks:

```python
# Combined mask (handled by backend)
final_mask = padding_mask * causal_mask
```

## Attention Visualization

Understanding what the model attends to:

```yaml
network VisualizableAttention {
  input: (100, 512)
  layers:
    TransformerEncoder(num_heads=8, d_model=512, dff=2048)
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

Use Neural's debugging tools to visualize attention:

```bash
# Debug mode to inspect attention weights
neural debug visualizable_attention.neural --attention-weights
```

## Attention Computation Cost

Attention has quadratic complexity in sequence length:

**Time Complexity**: O(n²d)
- n: sequence length
- d: model dimension

**Memory Complexity**: O(n²)
- Must store attention matrix for backpropagation

### Efficient Attention Variants

For very long sequences, consider:

1. **Local Attention**: Attend to nearby tokens only
2. **Sparse Attention**: Attend to subset of tokens
3. **Linear Attention**: Approximate attention with linear complexity

```yaml
# Standard attention (quadratic)
TransformerEncoder(num_heads=8, d_model=512, dff=2048)

# For sequences > 1024, consider chunking or sparse variants
# (implementation depends on backend)
```

## Attention and Positional Information

Attention is permutation-invariant without positional encoding:

```yaml
network AttentionWithPositionalEncoding {
  input: (512, 768)
  layers:
    # Positional encoding added automatically by backend
    TransformerEncoder(num_heads=8, d_model=768, dff=3072)
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

Positional encodings can be:
- **Sinusoidal**: Fixed mathematical function
- **Learned**: Trainable embeddings
- **Relative**: Encode relative positions between tokens

## Practical Guidelines

### Choosing Number of Heads

```yaml
# Small models: 4-8 heads
TransformerEncoder(num_heads=4, d_model=256, dff=1024)

# Medium models: 8-12 heads
TransformerEncoder(num_heads=8, d_model=512, dff=2048)

# Large models: 12-16 heads
TransformerEncoder(num_heads=16, d_model=1024, dff=4096)
```

**Rule of thumb**: head_dim = d_model / num_heads ≈ 64

### Attention Dropout

Regularize attention by randomly dropping attention weights:

```yaml
TransformerEncoder(
  num_heads=8,
  d_model=512,
  dff=2048,
  dropout_rate=0.1  # Applied to attention and FFN
)
```

**Typical values**: 0.1 for base models, 0.2-0.3 for large models

### Key Dimension

When using MultiHeadAttention directly:

```yaml
# key_dim = d_model / num_heads
MultiHeadAttention(num_heads=8, key_dim=64)  # For d_model=512

# Or specify explicitly
MultiHeadAttention(num_heads=8, key_dim=96)  # Non-standard but valid
```

## Common Issues and Solutions

### Attention Collapse

**Problem**: All attention weights converge to uniform distribution

**Solutions**:
1. Use Pre-LayerNorm (done by default in modern implementations)
2. Reduce learning rate
3. Add attention dropout
4. Use gradient clipping

```yaml
optimizer: Adam(
  learning_rate=ExponentialDecay(0.0001, 10000, 0.96)
)
```

### Memory Issues with Long Sequences

**Problem**: Out of memory for sequences > 512-1024 tokens

**Solutions**:
1. Reduce batch size
2. Use gradient checkpointing
3. Split into chunks
4. Consider efficient attention variants

```yaml
train {
  batch_size: 16  # Reduce for longer sequences
  gradient_accumulation_steps: 4  # Effective batch size: 64
}
```

### Attention Not Learning

**Problem**: Model trains but attention patterns seem random

**Solutions**:
1. Check positional encodings are added
2. Verify input preprocessing (tokenization, padding)
3. Ensure sufficient training data
4. Use pre-trained weights when possible

## Attention in Different Domains

### Natural Language Processing

```yaml
network NLPAttention {
  input: (512, 768)
  layers:
    Embedding(input_dim=30000, output_dim=768)
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=2e-5)
}
```

### Computer Vision

```yaml
network VisionAttention {
  input: (224, 224, 3)
  layers:
    # Convert image to patch sequence
    Conv2D(768, kernel_size=(16, 16), strides=16)
    Reshape(target_shape=(196, 768))
    
    # Apply attention to patches
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    
    GlobalAveragePooling1D()
    Dense(1000, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}
```

### Time Series

```yaml
network TimeSeriesAttention {
  input: (100, 64)  # 100 timesteps, 64 features
  layers:
    TransformerEncoder(num_heads=8, d_model=64, dff=256)
    GlobalAveragePooling1D()
    Dense(1)  # Regression
  
  loss: "mse"
  optimizer: Adam(learning_rate=0.001)
}
```

## Next Steps

- [Architecture Guide](transformer_architecture.md) - Full transformer architectures
- [Training Best Practices](transformer_training.md) - Optimize training
- [Migration Guide](transformer_migration.md) - Convert existing code
