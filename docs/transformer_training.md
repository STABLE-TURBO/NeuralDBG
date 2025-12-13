# Transformer Training Best Practices

## Overview

Training transformers effectively requires careful attention to hyperparameters, optimization strategies, and computational resources. This guide provides best practices for training transformer models in Neural DSL.

## Learning Rate Strategies

### Learning Rate Warmup

Gradually increase learning rate at the start of training to stabilize optimization:

```yaml
network WarmupExample {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(
    learning_rate=ExponentialDecay(
      2e-5,      # Peak learning rate after warmup
      10000,     # Decay after this many steps
      0.96       # Decay rate
    )
  )
  loss: "categorical_crossentropy"
  train {
    epochs: 10
    batch_size: 32
  }
}
```

### Recommended Learning Rates by Model Size

| Model Size | Base LR  | With Warmup | Warmup Steps |
|------------|----------|-------------|--------------|
| Small      | 1e-3     | 1e-3        | 4,000        |
| Base       | 5e-4     | 6e-4        | 8,000        |
| Large      | 1e-4     | 2e-4        | 10,000       |
| XL         | 5e-5     | 1e-4        | 15,000       |

### Learning Rate Schedules

```yaml
# Exponential decay (recommended)
optimizer: Adam(
  learning_rate=ExponentialDecay(
    HPO(log_range(1e-5, 1e-3)),  # Use HPO to find optimal rate
    10000,
    0.96
  )
)

# Cosine annealing (alternative)
optimizer: Adam(
  learning_rate=CosineDecay(
    1e-4,    # Initial learning rate
    50000    # Total training steps
  )
)

# Step decay
optimizer: Adam(
  learning_rate=PiecewiseConstantDecay(
    boundaries=[10000, 20000, 30000],
    values=[1e-4, 5e-5, 1e-5, 5e-6]
  )
)
```

## Batch Size and Gradient Accumulation

### Effective Batch Size

Transformers benefit from large batch sizes, but memory constraints often require compromises:

```yaml
network LargeBatchTraining {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(learning_rate=0.0001)
  loss: "categorical_crossentropy"
  train {
    epochs: 10
    batch_size: 64               # Per-device batch size
    gradient_accumulation_steps: 4  # Effective batch size: 256
  }
}
```

### Batch Size Guidelines

| Model Size | Min Batch | Recommended | Large Scale |
|------------|-----------|-------------|-------------|
| Small      | 32        | 64-128      | 256         |
| Base       | 16        | 32-64       | 128         |
| Large      | 8         | 16-32       | 64          |
| XL         | 4         | 8-16        | 32          |

**Tip**: If memory-constrained, reduce batch size and use gradient accumulation to maintain effective batch size.

## Optimizer Selection

### Adam and Variants

Adam is the standard optimizer for transformers:

```yaml
# Standard Adam
optimizer: Adam(
  learning_rate=0.0001,
  beta_1=0.9,
  beta_2=0.999,
  epsilon=1e-8
)

# AdamW (Adam with weight decay - recommended)
optimizer: AdamW(
  learning_rate=0.0001,
  beta_1=0.9,
  beta_2=0.999,
  weight_decay=0.01
)
```

### Optimizer HPO

Use hyperparameter optimization to find optimal settings:

```yaml
network HPOOptimizer {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(
    learning_rate=HPO(log_range(1e-5, 1e-3)),
    beta_1=HPO(range(0.85, 0.95, step=0.01)),
    beta_2=HPO(range(0.995, 0.9999, step=0.0001))
  )
  loss: "categorical_crossentropy"
  train {
    epochs: 5
    batch_size: 32
    search_method: "bayesian"
  }
}
```

## Regularization Techniques

### Dropout

Apply dropout to attention and feed-forward layers:

```yaml
network RegularizedTransformer {
  input: (512, 768)
  layers:
    TransformerEncoder(
      num_heads=12,
      d_model=768,
      dff=3072,
      dropout_rate=0.1  # Applied to attention and FFN
    ) * 12
    GlobalAveragePooling1D()
    Dropout(0.1)  # Additional dropout before output
    Output(10, activation="softmax")
  
  optimizer: Adam(learning_rate=0.0001)
  loss: "categorical_crossentropy"
}
```

**Dropout rates by model size:**
- Small models: 0.1
- Base models: 0.1
- Large models: 0.1-0.2
- XL models: 0.2-0.3

### Weight Decay

Apply L2 regularization to prevent overfitting:

```yaml
optimizer: AdamW(
  learning_rate=0.0001,
  weight_decay=0.01  # Standard for BERT-style models
)
```

### Label Smoothing

Soften target distributions to prevent overconfidence:

```yaml
loss: LabelSmoothingCrossentropy(
  smoothing=0.1
)
```

### Gradient Clipping

Prevent exploding gradients in deep transformers:

```yaml
optimizer: Adam(
  learning_rate=0.0001,
  clipnorm=1.0  # Clip by global norm
)
```

## Mixed Precision Training

Use FP16 training to reduce memory and increase speed:

```yaml
network MixedPrecisionTransformer {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(learning_rate=0.0001)
  loss: "categorical_crossentropy"
  execution {
    device: "cuda:0"
    mixed_precision: true  # Enable FP16 training
  }
}
```

**Benefits:**
- 2-3x faster training
- 50% less memory usage
- Minimal accuracy loss with proper loss scaling

## Data Strategies

### Data Augmentation

For text:
```python
# In your data pipeline (not in Neural DSL)
- Random token masking
- Synonym replacement
- Back-translation
- Mixup at embedding level
```

For vision transformers:
```python
- Random cropping
- Color jittering
- Cutout/cutmix
- Mixup
```

### Sequence Length Strategy

Start with shorter sequences and gradually increase:

```yaml
# Phase 1: Short sequences (first 50% of training)
network Phase1 {
  input: (128, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
  train {
    epochs: 5
    batch_size: 64
  }
}

# Phase 2: Full sequences (remaining 50%)
network Phase2 {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
  train {
    epochs: 5
    batch_size: 16
  }
}
```

## Model Initialization

### Pre-trained Weights

When possible, start from pre-trained models:

```bash
# Load pre-trained BERT weights
neural compile my_model.neural --load-weights bert-base-uncased.h5

# Fine-tune with lower learning rate
```

```yaml
network FineTuning {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(2, activation="softmax")  # Task-specific head
  
  optimizer: Adam(
    learning_rate=2e-5  # Much lower for fine-tuning
  )
  loss: "categorical_crossentropy"
  train {
    epochs: 3
    batch_size: 32
  }
}
```

### From Scratch Initialization

For training from scratch, use proper initialization:
- Xavier/Glorot for most layers (default in most frameworks)
- Scaled initialization for deep models

## Training Stability

### Gradient Checkpointing

Trade compute for memory:

```yaml
execution {
  device: "cuda:0"
  gradient_checkpointing: true  # Saves memory for deep models
}
```

### Pre-LayerNorm

Use Pre-LayerNorm architecture for better stability (default in modern implementations):

```yaml
# Modern implementations use Pre-LayerNorm by default
TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
```

### Monitoring Training

Key metrics to track:
1. **Training loss**: Should decrease steadily
2. **Validation loss**: Watch for overfitting
3. **Learning rate**: Verify schedule is working
4. **Gradient norm**: Check for gradient issues
5. **Attention entropy**: Ensure attention isn't collapsing

```bash
# Use Neural's debugging tools
neural debug model.neural --gradients --dead-neurons

# Real-time dashboard
neural dashboard model.neural
```

## Hyperparameter Optimization

### Comprehensive HPO Example

```yaml
network ComprehensiveHPO {
  input: (512, 768)
  layers:
    TransformerEncoder(
      num_heads=HPO(choice(8, 12, 16)),
      d_model=768,
      dff=HPO(choice(2048, 3072, 4096)),
      dropout_rate=HPO(range(0.1, 0.3, step=0.05))
    ) * HPO(choice(6, 12))
    
    GlobalAveragePooling1D()
    Dropout(HPO(range(0.1, 0.3, step=0.05)))
    Output(10, activation="softmax")
  
  optimizer: Adam(
    learning_rate=HPO(log_range(1e-5, 1e-3)),
    beta_1=HPO(range(0.85, 0.95, step=0.01)),
    beta_2=HPO(range(0.995, 0.9999, step=0.0001))
  )
  
  loss: "categorical_crossentropy"
  
  train {
    epochs: 10
    batch_size: HPO(choice(16, 32, 64))
    validation_split: 0.2
    search_method: "bayesian"
    num_trials: 50
  }
}
```

### HPO Search Space Design

**Critical parameters** (highest impact):
1. Learning rate
2. Batch size
3. Number of layers
4. Model dimension (d_model)

**Secondary parameters**:
5. Number of heads
6. FFN dimension (dff)
7. Dropout rates
8. Optimizer parameters (beta_1, beta_2)

**Low priority**:
9. Epsilon in optimizer
10. Gradient clipping threshold

## Multi-GPU Training

### Data Parallel

Distribute batch across GPUs:

```yaml
network DataParallel {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(learning_rate=0.0001)
  loss: "categorical_crossentropy"
  
  execution {
    device: "cuda"  # Use all available GPUs
    strategy: "data_parallel"
  }
  
  train {
    epochs: 10
    batch_size: 32  # Per-GPU batch size
  }
}
```

### Model Parallel

Split large models across GPUs:

```yaml
network ModelParallel {
  input: (512, 1024)
  layers:
    # First 12 layers on GPU 0
    TransformerEncoder(num_heads=16, d_model=1024, dff=4096) @ "cuda:0" * 12
    
    # Next 12 layers on GPU 1
    TransformerEncoder(num_heads=16, d_model=1024, dff=4096) @ "cuda:1" * 12
    
    GlobalAveragePooling1D() @ "cuda:1"
    Output(1000) @ "cuda:1"
  
  optimizer: Adam(learning_rate=0.0001)
  execution {
    strategy: "model_parallel"
  }
}
```

## Common Training Issues

### Vanishing Gradients

**Symptoms**: Loss plateaus, deep layers don't learn

**Solutions**:
1. Use Pre-LayerNorm (default)
2. Reduce learning rate
3. Add gradient clipping
4. Check initialization

```yaml
optimizer: Adam(
  learning_rate=0.00005,  # Lower rate
  clipnorm=1.0
)
```

### Exploding Gradients

**Symptoms**: Loss becomes NaN, training unstable

**Solutions**:
1. Add gradient clipping
2. Reduce learning rate
3. Check data for outliers
4. Use mixed precision carefully

```yaml
optimizer: Adam(
  learning_rate=0.00001,
  clipnorm=0.5  # Aggressive clipping
)
```

### Overfitting

**Symptoms**: Training loss decreases, validation loss increases

**Solutions**:
1. Increase dropout
2. Add weight decay
3. Use more data
4. Reduce model size

```yaml
TransformerEncoder(
  num_heads=12,
  d_model=768,
  dff=3072,
  dropout_rate=0.2  # Increase from 0.1
)

optimizer: AdamW(
  learning_rate=0.0001,
  weight_decay=0.01  # Add weight decay
)
```

### Slow Convergence

**Symptoms**: Training takes too long

**Solutions**:
1. Increase learning rate
2. Increase batch size
3. Use learning rate warmup
4. Check data pipeline efficiency

```yaml
optimizer: Adam(
  learning_rate=0.001  # Increase from 0.0001
)

train {
  batch_size: 128  # Increase from 32
  gradient_accumulation_steps: 2
}
```

### Out of Memory

**Symptoms**: CUDA OOM errors

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision
4. Reduce sequence length
5. Use model parallelism

```yaml
train {
  batch_size: 8  # Reduce from 32
}

execution {
  mixed_precision: true
  gradient_checkpointing: true
}
```

## Training Schedules

### Standard Training

```yaml
network StandardTraining {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(
    learning_rate=ExponentialDecay(0.0001, 10000, 0.96)
  )
  
  train {
    epochs: 20
    batch_size: 32
    validation_split: 0.2
  }
}
```

### Quick Experimentation

```yaml
network QuickExperiment {
  input: (128, 512)  # Shorter sequences
  layers:
    TransformerEncoder(num_heads=8, d_model=512, dff=2048) * 6  # Fewer layers
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 5
    batch_size: 64
  }
}
```

### Production Training

```yaml
network ProductionTraining {
  input: (512, 1024)
  layers:
    TransformerEncoder(num_heads=16, d_model=1024, dff=4096) * 24
    GlobalAveragePooling1D()
    Output(1000, activation="softmax")
  
  optimizer: AdamW(
    learning_rate=ExponentialDecay(0.0001, 50000, 0.98),
    weight_decay=0.01
  )
  
  train {
    epochs: 100
    batch_size: 16
    validation_split: 0.1
    gradient_accumulation_steps: 8
  }
  
  execution {
    device: "cuda"
    mixed_precision: true
    gradient_checkpointing: true
  }
}
```

## Performance Optimization

### Profiling

Identify bottlenecks:

```bash
# Profile training
neural profile model.neural --memory --latency

# Detailed profiling
neural profile model.neural --detailed --output profile.json
```

### Optimization Checklist

- [ ] Use mixed precision training
- [ ] Enable cudnn autotuner (automatic in most backends)
- [ ] Optimize data loading pipeline
- [ ] Use gradient accumulation for effective large batches
- [ ] Enable XLA compilation (TensorFlow)
- [ ] Use torch.compile (PyTorch 2.0+)
- [ ] Profile and eliminate bottlenecks
- [ ] Use efficient attention implementations

## Next Steps

- [Architecture Guide](transformer_architecture.md) - Build transformer models
- [Attention Mechanism](transformer_attention.md) - Understand attention
- [Migration Guide](transformer_migration.md) - Convert existing code
