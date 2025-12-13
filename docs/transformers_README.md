# Transformer Documentation

Comprehensive guide to building and training transformer models with Neural DSL.

## Documentation Overview

This collection provides everything you need to work with transformers in Neural DSL:

### 📚 Core Guides

1. **[Architecture Guide](transformer_architecture.md)**
   - Understanding transformer components (Encoder, Decoder, Multi-Head Attention)
   - Common architectures (BERT, GPT, ViT, Seq2Seq)
   - Model scaling and device placement
   - Input shape considerations and positional encoding

2. **[Attention Mechanism Explained](transformer_attention.md)**
   - How attention works (Query, Key, Value)
   - Multi-head attention benefits
   - Self-attention vs cross-attention
   - Attention patterns (bidirectional, causal)
   - Masking strategies and visualization

3. **[Training Best Practices](transformer_training.md)**
   - Learning rate strategies and warmup
   - Batch size and gradient accumulation
   - Optimizer selection (Adam, AdamW)
   - Regularization techniques (dropout, weight decay)
   - Mixed precision training
   - Multi-GPU strategies
   - Common training issues and solutions

4. **[Migration Guide](transformer_migration.md)**
   - Converting from raw TensorFlow code
   - Converting from raw PyTorch code
   - Layer mapping reference
   - Optimizer and training loop migration
   - Code reduction examples (50-75% less code)

## Quick Start

### Basic Transformer Encoder

```yaml
network SimpleTransformer {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=8, d_model=768, dff=2048)
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
  
  train {
    epochs: 20
    batch_size: 32
    validation_split: 0.2
  }
}
```

### Build and Run

```bash
# Compile with TensorFlow
neural compile simple_transformer.neural --backend tensorflow

# Or with PyTorch
neural compile simple_transformer.neural --backend pytorch

# Run training
neural run simple_transformer.neural --device cuda

# Debug and visualize
neural debug simple_transformer.neural --gradients
```

## Common Use Cases

### Text Classification

```yaml
network TextClassifier {
  input: (128, 512)
  layers:
    Embedding(input_dim=30000, output_dim=512)
    TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6
    GlobalAveragePooling1D()
    Dense(256, activation="relu")
    Dropout(0.3)
    Output(5, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
  train {
    epochs: 20
    batch_size: 64
  }
}
```

### Language Generation (GPT-style)

```yaml
network LanguageModel {
  input: (1024, 768)
  layers:
    TransformerDecoder(num_heads=12, d_model=768, dff=3072) * 12
    Dense(50257, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(
    learning_rate=ExponentialDecay(6e-4, 10000, 0.95)
  )
  train {
    epochs: 10
    batch_size: 8
  }
}
```

### Vision Transformer

```yaml
network VisionTransformer {
  input: (224, 224, 3)
  layers:
    Conv2D(768, kernel_size=(16, 16), strides=16, padding="valid")
    Reshape(target_shape=(196, 768))
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Dense(1000, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
  train {
    epochs: 100
    batch_size: 64
  }
}
```

### Sequence-to-Sequence

```yaml
network Seq2Seq {
  input: (100, 512)
  layers:
    Embedding(input_dim=10000, output_dim=512)
    TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6
    TransformerDecoder(num_heads=8, d_model=512, dff=2048) * 6
    Dense(10000, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
  train {
    epochs: 20
    batch_size: 64
  }
}
```

## Key Features

### Layer Repetition

Stack multiple transformer layers easily:

```yaml
# 12 encoder layers
TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12

# 6 encoder + 6 decoder layers
TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6
TransformerDecoder(num_heads=8, d_model=512, dff=2048) * 6
```

### Hyperparameter Optimization

Built-in HPO for transformers:

```yaml
network OptimizedTransformer {
  input: (512, 768)
  layers:
    TransformerEncoder(
      num_heads=HPO(choice(8, 12, 16)),
      d_model=768,
      dff=HPO(choice(2048, 3072, 4096)),
      dropout_rate=HPO(range(0.1, 0.3, step=0.05))
    ) * HPO(choice(6, 12))
    
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(
    learning_rate=HPO(log_range(1e-5, 1e-3))
  )
  
  train {
    batch_size: HPO(choice(16, 32, 64))
    search_method: "bayesian"
    num_trials: 50
  }
}
```

Run HPO:
```bash
neural compile model.neural --backend tensorflow --hpo
```

### Device Placement

Easy GPU utilization:

```yaml
# Place all layers on GPU
execution {
  device: "cuda:0"
}

# Per-layer placement for model parallelism
layers:
  TransformerEncoder(...) @ "cuda:0" * 12
  TransformerEncoder(...) @ "cuda:1" * 12
```

### Mixed Precision Training

Enable FP16 for faster training:

```yaml
execution {
  device: "cuda:0"
  mixed_precision: true
}
```

## Model Size Reference

Choose the right size for your use case:

| Size   | Heads | d_model | dff  | Layers | Params | Use Case                    |
|--------|-------|---------|------|--------|--------|-----------------------------|
| Tiny   | 4     | 256     | 1024 | 4      | ~10M   | Prototyping, edge devices   |
| Small  | 4     | 512     | 2048 | 6      | ~40M   | Resource-constrained apps   |
| Base   | 8     | 512     | 2048 | 6      | ~65M   | Standard applications       |
| Medium | 8     | 768     | 3072 | 12     | ~110M  | BERT-base equivalent        |
| Large  | 16    | 1024    | 4096 | 24     | ~340M  | High-quality applications   |
| XL     | 24    | 1536    | 6144 | 48     | ~1.5B  | State-of-the-art performance|

### Tiny Model (for testing)

```yaml
network TinyTransformer {
  input: (128, 256)
  layers:
    TransformerEncoder(num_heads=4, d_model=256, dff=1024) * 4
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(learning_rate=0.001)
}
```

### Base Model (recommended starting point)

```yaml
network BaseTransformer {
  input: (512, 512)
  layers:
    TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(learning_rate=0.0001)
}
```

### Large Model (production)

```yaml
network LargeTransformer {
  input: (512, 1024)
  layers:
    TransformerEncoder(num_heads=16, d_model=1024, dff=4096) * 24
    GlobalAveragePooling1D()
    Output(1000, activation="softmax")
  
  optimizer: Adam(learning_rate=0.00005)
  execution {
    device: "cuda"
    mixed_precision: true
  }
}
```

## Training Tips

### Start Small, Scale Up

1. **Prototype** with tiny model (4 layers, d_model=256)
2. **Validate** with base model (6 layers, d_model=512)
3. **Deploy** with larger model if needed

### Learning Rate Guidelines

```yaml
# Small models: higher learning rate
optimizer: Adam(learning_rate=0.001)

# Base models: moderate learning rate
optimizer: Adam(learning_rate=0.0001)

# Large models: lower learning rate with warmup
optimizer: Adam(
  learning_rate=ExponentialDecay(0.00005, 10000, 0.96)
)

# Fine-tuning: very low learning rate
optimizer: Adam(learning_rate=2e-5)
```

### Batch Size Strategy

```yaml
# Small model: larger batches
train {
  batch_size: 128
}

# Large model: smaller batches with gradient accumulation
train {
  batch_size: 8
  gradient_accumulation_steps: 16  # Effective: 128
}
```

## Debugging and Visualization

### Real-time Debugging

```bash
# Gradient flow analysis
neural debug model.neural --gradients

# Dead neuron detection
neural debug model.neural --dead-neurons

# Interactive debugging
neural debug model.neural --step

# With visualization
neural debug model.neural --attention-weights --theme dark
```

### Model Profiling

```bash
# Memory and latency profiling
neural profile model.neural --memory --latency

# Detailed profiling
neural profile model.neural --detailed --output profile.json
```

### Architecture Visualization

```bash
# Generate architecture diagram
neural visualize model.neural --format svg

# HTML interactive visualization
neural visualize model.neural --format html
```

## Performance Optimization

### Checklist for Production

- [ ] Use mixed precision training (`mixed_precision: true`)
- [ ] Enable gradient accumulation for large effective batch sizes
- [ ] Use Pre-LayerNorm architecture (default)
- [ ] Apply dropout (0.1-0.2) to prevent overfitting
- [ ] Use weight decay (AdamW with `weight_decay=0.01`)
- [ ] Enable gradient checkpointing for very deep models
- [ ] Profile and optimize data loading pipeline
- [ ] Use appropriate learning rate schedule with warmup
- [ ] Monitor gradient norms and attention patterns
- [ ] Validate on held-out data regularly

### Example Production Configuration

```yaml
network ProductionTransformer {
  input: (512, 1024)
  layers:
    TransformerEncoder(
      num_heads=16,
      d_model=1024,
      dff=4096,
      dropout_rate=0.1
    ) * 24
    GlobalAveragePooling1D()
    Output(1000, activation="softmax")
  
  optimizer: AdamW(
    learning_rate=ExponentialDecay(0.0001, 50000, 0.98),
    weight_decay=0.01
  )
  
  loss: "categorical_crossentropy"
  
  train {
    epochs: 100
    batch_size: 16
    validation_split: 0.1
    gradient_accumulation_steps: 8
    early_stopping_patience: 5
  }
  
  execution {
    device: "cuda"
    mixed_precision: true
    gradient_checkpointing: true
  }
}
```

## Common Issues

### Out of Memory

**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training
- Reduce sequence length
- Use model parallelism

### Training Instability

**Solutions:**
- Add gradient clipping
- Lower learning rate
- Use Pre-LayerNorm (default)
- Increase warmup steps
- Check for data issues

### Slow Convergence

**Solutions:**
- Increase learning rate
- Increase batch size
- Use better initialization (pre-trained weights)
- Simplify model if too large
- Check data preprocessing

### Attention Collapse

**Solutions:**
- Use Pre-LayerNorm
- Add attention dropout
- Reduce learning rate
- Check positional encodings

## Additional Resources

### Neural DSL Documentation
- [Main Documentation](../dsl.md)
- [CLI Reference](../cli.md)
- [HPO Guide](../dsl.md#hyperparameter-optimization)

### Examples
- [Basic Transformer](../../examples/transformer.neural)
- [Advanced HPO](../../examples/advanced_hpo_v0.2.7.neural)
- [Vision Examples](../../examples/use_cases/)

### Getting Help
- [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- [Contributing Guide](../../CONTRIBUTING.md)
- [Security Policy](../../SECURITY.md)

## Quick Reference

### TransformerEncoder Parameters

```yaml
TransformerEncoder(
  num_heads=8,        # Number of attention heads (4, 8, 12, 16)
  d_model=512,        # Model dimension (256, 512, 768, 1024)
  dff=2048,          # Feed-forward dimension (typically 4x d_model)
  dropout_rate=0.1   # Dropout rate (0.0-0.3)
)
```

### TransformerDecoder Parameters

```yaml
TransformerDecoder(
  num_heads=8,        # Number of attention heads
  d_model=512,        # Model dimension
  dff=2048,          # Feed-forward dimension
  dropout_rate=0.1   # Dropout rate
)
```

### MultiHeadAttention Parameters

```yaml
MultiHeadAttention(
  num_heads=8,   # Number of attention heads
  key_dim=64     # Dimension per head (typically d_model/num_heads)
)
```

## License

Neural DSL is open source. See [LICENSE](../../LICENSE.md) for details.
