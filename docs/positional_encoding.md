# PositionalEncoding Layer

The `PositionalEncoding` layer adds positional information to input embeddings, which is essential for transformer-based models that process sequential data.

## Overview

Transformer models lack inherent positional awareness since they process all positions in parallel. The PositionalEncoding layer injects position-dependent signals into the input embeddings, enabling the model to understand token order.

## Syntax

```
PositionalEncoding([max_len], [encoding_type])
PositionalEncoding(max_len: value, encoding_type: "sinusoidal" | "learnable")
```

## Parameters

### max_len (optional)
- **Type**: Integer
- **Default**: 5000
- **Description**: Maximum sequence length that the model will encounter. This determines the size of the positional encoding matrix.

### encoding_type (optional)
- **Type**: String
- **Default**: "sinusoidal"
- **Valid values**: "sinusoidal", "learnable"
- **Description**: Type of positional encoding to use
  - **"sinusoidal"**: Fixed positional encodings using sine and cosine functions (as described in "Attention Is All You Need")
  - **"learnable"**: Trainable positional embeddings that are learned during training

## Shape Requirements

- **Input Shape**: `(batch_size, sequence_length, embedding_dim)`
- **Output Shape**: `(batch_size, sequence_length, embedding_dim)`

The layer preserves the input shape while adding positional information.

## Usage Examples

### Basic Usage (Sinusoidal)

```
network TransformerModel {
    input: (None, 512)
    layers:
        PositionalEncoding()
        TransformerEncoder(num_heads: 8, ff_dim: 2048)
        Output(10, "softmax")
    optimizer: "Adam" { learning_rate: 0.001 }
    loss: "categorical_crossentropy"
}
```

### With Custom Parameters

```
network CustomTransformer {
    input: (None, 256)
    layers:
        PositionalEncoding(max_len: 1000, encoding_type: "sinusoidal")
        TransformerEncoder(num_heads: 4, ff_dim: 1024)
        Output(5, "softmax")
    optimizer: "Adam" { learning_rate: 0.0001 }
    loss: "categorical_crossentropy"
}
```

### Learnable Positional Encoding

```
network LearnableTransformer {
    input: (None, 256)
    layers:
        PositionalEncoding(max_len: 512, encoding_type: "learnable")
        TransformerEncoder(num_heads: 8, ff_dim: 2048)
        Output(10, "softmax")
    optimizer: "Adam" { learning_rate: 0.001 }
    loss: "categorical_crossentropy"
}
```

### Named Parameters

```
network NamedParamsExample {
    input: (None, 384)
    layers:
        PositionalEncoding(encoding_type: "learnable", max_len: 2048)
        TransformerEncoder(num_heads: 6, ff_dim: 1536)
        Output(20, "softmax")
    optimizer: "Adam" { learning_rate: 0.0005 }
    loss: "categorical_crossentropy"
}
```

## Implementation Details

### TensorFlow

**Sinusoidal Encoding:**
```python
def get_positional_encoding(seq_len, d_model, max_len=5000):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pos_encoding, dtype=tf.float32)
```

**Learnable Encoding:**
```python
pos_embedding = layers.Embedding(input_dim=max_len, output_dim=d_model)
positions = tf.range(start=0, limit=seq_len, delta=1)
x = x + pos_embedding(positions)
```

### PyTorch

**Sinusoidal Encoding:**
```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.max_len = max_len

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) * 
                            -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_len, d_model, device=x.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return x + pos_encoding.unsqueeze(0)
```

**Learnable Encoding:**
```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len=5000, d_model=512):
        super(LearnablePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.pos_embedding = None

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        if self.pos_embedding is None or self.pos_embedding.size(0) != self.max_len or 
           self.pos_embedding.size(1) != d_model:
            self.pos_embedding = nn.Parameter(torch.randn(self.max_len, d_model, device=x.device))
        positions = self.pos_embedding[:seq_len, :].unsqueeze(0)
        return x + positions
```

## Mathematical Background

### Sinusoidal Encoding Formula

The sinusoidal positional encoding uses the following formulas:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` is the position in the sequence
- `i` is the dimension index
- `d_model` is the embedding dimension

This formulation has several advantages:
1. It allows the model to easily learn to attend by relative positions
2. It can extrapolate to sequence lengths longer than those seen during training
3. It provides a unique encoding for each position

### Learnable Encoding

Learnable positional encodings are simply trainable parameters initialized randomly. The model learns the optimal positional representations during training. This approach:
1. Can potentially learn more task-specific position information
2. Requires the max_len to be at least as large as the longest training sequence
3. Cannot extrapolate to longer sequences

## Best Practices

1. **Choose encoding type based on your task:**
   - Use **sinusoidal** for tasks requiring extrapolation to longer sequences
   - Use **learnable** when you have fixed-length sequences and want task-specific position learning

2. **Set max_len appropriately:**
   - For sinusoidal: Can be set higher than training data for potential extrapolation
   - For learnable: Must be at least as large as the longest sequence you'll encounter

3. **Position in architecture:**
   - Place immediately after input embeddings and before transformer layers
   - Always use with models that lack inherent position awareness (e.g., Transformers)

4. **Memory considerations:**
   - Learnable encoding adds parameters: `max_len * embedding_dim`
   - Sinusoidal encoding has no trainable parameters

## Common Patterns

### With Multiple Transformer Layers

```
network MultiLayerTransformer {
    input: (None, 512)
    layers:
        PositionalEncoding(max_len: 1024, encoding_type: "sinusoidal")
        TransformerEncoder(num_heads: 8, ff_dim: 2048) * 6
        Output(100, "softmax")
    optimizer: "Adam" { learning_rate: 0.0001 }
    loss: "categorical_crossentropy"
}
```

### With Dropout for Regularization

```
network RegularizedTransformer {
    input: (None, 512)
    layers:
        PositionalEncoding(max_len: 2048)
        Dropout(0.1)
        TransformerEncoder(num_heads: 8, ff_dim: 2048, dropout: 0.1)
        Output(50, "softmax")
    optimizer: "Adam" { learning_rate: 0.0001 }
    loss: "categorical_crossentropy"
}
```

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - Original transformer paper introducing sinusoidal positional encoding
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) - Uses learnable positional embeddings

## See Also

- [TransformerEncoder](transformer_encoder.md)
- [TransformerDecoder](transformer_decoder.md)
- [LayerNormalization](layer_normalization.md)
