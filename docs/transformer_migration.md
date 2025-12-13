# Migration Guide: From Raw TensorFlow/PyTorch to Neural DSL

## Overview

This guide helps you migrate existing transformer implementations in TensorFlow or PyTorch to Neural DSL. Neural DSL provides a simpler, more declarative syntax while maintaining compatibility with both frameworks.

## Why Migrate?

**Benefits of Neural DSL:**
- **Concise syntax**: Define models in 10-20 lines instead of 100-200
- **Backend agnostic**: Switch between TensorFlow and PyTorch without code changes
- **Built-in HPO**: Easy hyperparameter optimization
- **Type safety**: Automatic shape validation and error detection
- **Debugging tools**: Real-time visualization with NeuralDbg

## Migration Strategy

1. **Identify model architecture**: Understand your current model structure
2. **Map layers**: Convert framework-specific layers to Neural DSL equivalents
3. **Convert training loop**: Map optimizer and training configuration
4. **Validate output**: Ensure equivalent behavior
5. **Optimize**: Use Neural DSL features like HPO and device placement

## Basic Transformer Migration

### From TensorFlow

**Before (TensorFlow):**
```python
import tensorflow as tf
from tensorflow.keras import layers

# Define model
inputs = layers.Input(shape=(512, 768))
x = layers.MultiHeadAttention(num_heads=8, key_dim=96)(x, x)
x = layers.LayerNormalization()(x)
x = layers.Dense(2048, activation='relu')(x)
x = layers.Dense(768)(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_dataset,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)
```

**After (Neural DSL):**
```yaml
network TransformerModel {
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

**Reduction**: ~50 lines → ~15 lines (70% less code)

### From PyTorch

**Before (PyTorch):**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(768, 10)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return self.softmax(x)

model = TransformerModel()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(20):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
```

**After (Neural DSL):**
```yaml
network TransformerModel {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=8, d_model=768, dff=2048) * 6
    GlobalAveragePooling1D()
    Output(10, activation="softmax")
  
  loss: "cross_entropy"
  optimizer: Adam(learning_rate=0.0001)
  
  train {
    epochs: 20
    batch_size: 32
  }
}
```

**Reduction**: ~60 lines → ~15 lines (75% less code)

## BERT-Style Encoder

### From TensorFlow

**Before:**
```python
import tensorflow as tf
from tensorflow.keras import layers

class BERTEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = layers.Embedding(30000, 768)
        self.encoder_layers = [
            layers.MultiHeadAttention(num_heads=12, key_dim=64)
            for _ in range(12)
        ]
        self.norm_layers = [
            layers.LayerNormalization() for _ in range(12)
        ]
        self.ffn_layers = [
            tf.keras.Sequential([
                layers.Dense(3072, activation='relu'),
                layers.Dense(768)
            ]) for _ in range(12)
        ]
        self.pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(2, activation='softmax')
    
    def call(self, inputs):
        x = self.embedding(inputs)
        for i in range(12):
            attn_output = self.encoder_layers[i](x, x)
            x = self.norm_layers[i](x + attn_output)
            ffn_output = self.ffn_layers[i](x)
            x = self.norm_layers[i](x + ffn_output)
        x = self.pool(x)
        return self.classifier(x)

model = BERTEncoder()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='categorical_crossentropy'
)
```

**After:**
```yaml
network BERTEncoder {
  input: (512, 768)
  layers:
    TransformerEncoder(num_heads=12, d_model=768, dff=3072) * 12
    GlobalAveragePooling1D()
    Output(2, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=2e-5)
  
  train {
    epochs: 3
    batch_size: 32
  }
}
```

## GPT-Style Decoder

### From PyTorch

**Before:**
```python
import torch
import torch.nn as nn

class GPTDecoder(nn.Module):
    def __init__(self, vocab_size=50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 768)
        self.pos_encoding = nn.Embedding(1024, 768)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
        self.output_layer = nn.Linear(768, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        x = self.embedding(x) + self.pos_encoding(positions)
        
        # Create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        
        x = self.decoder(x, x, tgt_mask=mask)
        return self.output_layer(x)

model = GPTDecoder()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=6e-4,
    betas=(0.9, 0.95)
)
```

**After:**
```yaml
network GPTDecoder {
  input: (1024, 768)
  layers:
    TransformerDecoder(num_heads=12, d_model=768, dff=3072) * 12
    Dense(50257, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(
    learning_rate=ExponentialDecay(6e-4, 10000, 0.95),
    beta_1=0.9,
    beta_2=0.95
  )
  
  train {
    epochs: 10
    batch_size: 8
  }
}
```

## Vision Transformer (ViT)

### From TensorFlow

**Before:**
```python
import tensorflow as tf
from tensorflow.keras import layers

class VisionTransformer(tf.keras.Model):
    def __init__(self, patch_size=16, num_classes=1000):
        super().__init__()
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_proj = layers.Conv2D(
            768,
            kernel_size=patch_size,
            strides=patch_size
        )
        
        # Transformer encoder
        self.encoder_layers = [
            layers.MultiHeadAttention(num_heads=12, key_dim=64)
            for _ in range(12)
        ]
        self.norm_layers = [
            layers.LayerNormalization() for _ in range(12)
        ]
        
        # Classification head
        self.global_pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        # Create patches
        x = self.patch_proj(x)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, 768])
        
        # Apply transformer
        for i in range(12):
            attn = self.encoder_layers[i](x, x)
            x = self.norm_layers[i](x + attn)
        
        # Classify
        x = self.global_pool(x)
        return self.classifier(x)

model = VisionTransformer()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy'
)
```

**After:**
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

## Sequence-to-Sequence Model

### From PyTorch

**Before:**
```python
import torch
import torch.nn as nn

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size=10000, tgt_vocab_size=10000):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, 512)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, 512)
        
        self.transformer = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048
        )
        
        self.output_layer = nn.Linear(512, tgt_vocab_size)
    
    def forward(self, src, tgt):
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        
        # Create masks
        src_mask = self.create_pad_mask(src)
        tgt_mask = self.create_causal_mask(tgt)
        
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        return self.output_layer(output)
    
    def create_pad_mask(self, seq):
        return (seq == 0).transpose(0, 1)
    
    def create_causal_mask(self, seq):
        seq_len = seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()

model = Seq2SeqTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)
```

**After:**
```yaml
network Seq2SeqTransformer {
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

## Custom Attention Patterns

### From TensorFlow

**Before:**
```python
import tensorflow as tf
from tensorflow.keras import layers

class CustomAttentionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=8, key_dim=64)
        self.layernorm1 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(128)
        ])
        self.layernorm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)
        self.output_layer = layers.Dense(10, activation='softmax')
    
    def call(self, x, training=False):
        attn_output = self.mha(x, x)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return self.output_layer(x)
```

**After:**
```yaml
network CustomAttention {
  input: (100, 128)
  layers:
    MultiHeadAttention(num_heads=8, key_dim=64)
    LayerNormalization()
    Dense(512, activation="relu")
    Dense(128)
    LayerNormalization()
    Dropout(0.1)
    Output(10, activation="softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 20
    batch_size: 32
  }
}
```

## Layer Mapping Reference

### Encoder Layers

| TensorFlow/PyTorch | Neural DSL |
|-------------------|------------|
| `tf.keras.layers.MultiHeadAttention` | `MultiHeadAttention` |
| `nn.MultiheadAttention` | `MultiHeadAttention` |
| `tf.keras.layers.TransformerEncoderLayer` | `TransformerEncoder` |
| `nn.TransformerEncoderLayer` | `TransformerEncoder` |
| `tf.keras.layers.LayerNormalization` | `LayerNormalization` |
| `nn.LayerNorm` | `LayerNormalization` |

### Decoder Layers

| TensorFlow/PyTorch | Neural DSL |
|-------------------|------------|
| `tf.keras.layers.TransformerDecoderLayer` | `TransformerDecoder` |
| `nn.TransformerDecoderLayer` | `TransformerDecoder` |

### Pooling Layers

| TensorFlow/PyTorch | Neural DSL |
|-------------------|------------|
| `tf.keras.layers.GlobalAveragePooling1D` | `GlobalAveragePooling1D` |
| `nn.AdaptiveAvgPool1d` | `GlobalAveragePooling1D` |
| `tf.keras.layers.GlobalMaxPooling1D` | `GlobalMaxPooling1D` |
| `nn.AdaptiveMaxPool1d` | `GlobalMaxPooling1D` |

### Regularization

| TensorFlow/PyTorch | Neural DSL |
|-------------------|------------|
| `tf.keras.layers.Dropout` | `Dropout` |
| `nn.Dropout` | `Dropout` |

## Optimizer Migration

### TensorFlow to Neural DSL

```python
# Before (TensorFlow)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
```

```yaml
# After (Neural DSL)
optimizer: Adam(
  learning_rate=0.0001,
  beta_1=0.9,
  beta_2=0.999,
  epsilon=1e-8
)
```

### PyTorch to Neural DSL

```python
# Before (PyTorch)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

```yaml
# After (Neural DSL)
optimizer: Adam(
  learning_rate=0.0001,
  beta_1=0.9,
  beta_2=0.999,
  epsilon=1e-8
)
```

## Learning Rate Schedules

### TensorFlow

```python
# Before
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.96
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

```yaml
# After
optimizer: Adam(
  learning_rate=ExponentialDecay(0.0001, 10000, 0.96)
)
```

### PyTorch

```python
# Before
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.96
)
```

```yaml
# After
optimizer: Adam(
  learning_rate=ExponentialDecay(0.0001, 10000, 0.96)
)
```

## Training Loop Migration

### TensorFlow

**Before:**
```python
model.fit(
    train_dataset,
    epochs=20,
    batch_size=32,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5')
    ]
)
```

**After:**
```yaml
train {
  epochs: 20
  batch_size: 32
  validation_split: 0.2
  early_stopping_patience: 3
  checkpoint_path: "best_model.h5"
}
```

### PyTorch

**Before:**
```python
for epoch in range(20):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            outputs = model(batch['input'])
            val_loss += criterion(outputs, batch['labels'])
```

**After:**
```yaml
train {
  epochs: 20
  batch_size: 32
  validation_split: 0.2
}
```

## Device Placement

### TensorFlow

```python
# Before
with tf.device('/GPU:0'):
    model = create_model()
```

```yaml
# After
execution {
  device: "cuda:0"
}
```

### PyTorch

```python
# Before
device = torch.device('cuda:0')
model = model.to(device)
```

```yaml
# After
execution {
  device: "cuda:0"
}

# Or per-layer
layers:
  TransformerEncoder(...) @ "cuda:0"
```

## Mixed Precision Training

### TensorFlow

```python
# Before
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

```yaml
# After
execution {
  mixed_precision: true
}
```

### PyTorch

```python
# Before
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

```yaml
# After
execution {
  mixed_precision: true
}
```

## Adding HPO to Existing Models

One major advantage of Neural DSL is easy hyperparameter optimization:

```yaml
# Simply wrap parameters with HPO
network HPOModel {
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
    epochs: 10
    batch_size: HPO(choice(16, 32, 64))
    search_method: "bayesian"
    num_trials: 50
  }
}
```

Compile with HPO:
```bash
neural compile model.neural --backend tensorflow --hpo
```

## Running Your Migrated Model

### Compile

```bash
# TensorFlow backend
neural compile model.neural --backend tensorflow

# PyTorch backend
neural compile model.neural --backend pytorch
```

### Run

```bash
# Execute the model
neural run model.neural --device cuda

# With debugging
neural debug model.neural --gradients --dead-neurons
```

### Visualize

```bash
# View model architecture
neural visualize model.neural --format svg

# Real-time dashboard
neural dashboard model.neural
```

## Validation Checklist

After migration, verify:

- [ ] Model architecture matches original
- [ ] Input/output shapes are correct
- [ ] Loss function is equivalent
- [ ] Optimizer configuration matches
- [ ] Learning rate schedule is similar
- [ ] Training produces similar results
- [ ] Inference outputs match (within numerical precision)

## Getting Help

If you encounter issues during migration:

1. Check the [documentation](../dsl.md)
2. Look at [examples](../../examples/)
3. Use the debug mode: `neural debug model.neural --verbose`
4. File an issue on [GitHub](https://github.com/Lemniscate-world/Neural/issues)

## Next Steps

- [Architecture Guide](transformer_architecture.md) - Learn more about transformers
- [Training Best Practices](transformer_training.md) - Optimize your models
- [Attention Mechanism](transformer_attention.md) - Understand attention
