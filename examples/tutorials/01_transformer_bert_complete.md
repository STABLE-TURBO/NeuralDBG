# Tutorial: Building BERT with Neural DSL

## Overview

This tutorial provides a complete guide to implementing BERT (Bidirectional Encoder Representations from Transformers) using Neural DSL. You'll learn how to build encoder-only transformers for masked language modeling and fine-tuning on downstream tasks.

## Table of Contents

1. [Introduction to BERT](#introduction-to-bert)
2. [Architecture Overview](#architecture-overview)
3. [Implementation](#implementation)
4. [Pre-training](#pre-training)
5. [Fine-tuning](#fine-tuning)
6. [Multi-Backend Support](#multi-backend-support)
7. [Production Deployment](#production-deployment)

## Introduction to BERT

BERT revolutionized NLP by introducing bidirectional pre-training. Unlike GPT's unidirectional approach, BERT can see context from both directions, making it ideal for:

- **Masked Language Modeling (MLM)**: Predict masked tokens using full context
- **Sentence Classification**: Sentiment analysis, intent detection
- **Token Classification**: Named Entity Recognition, POS tagging
- **Question Answering**: SQuAD, extractive QA
- **Sentence Pair Tasks**: Natural Language Inference, paraphrase detection

## Architecture Overview

### Key Components

1. **Embeddings Layer**: Token + Position + Segment embeddings
2. **Encoder Stack**: 12-24 transformer encoder layers
3. **Attention Mechanism**: Multi-head self-attention (bidirectional)
4. **Feed-Forward Networks**: Position-wise transformations
5. **Output Heads**: MLM head and NSP head for pre-training

### BERT Variants

| Model | Layers | Hidden Size | Attention Heads | Parameters |
|-------|--------|-------------|-----------------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

## Implementation

### Basic BERT-Base Model

See `examples/bert_encoder.neural` for the complete implementation. Here's the core structure:

```neural
define BertLayer(num_heads, d_model, d_ff, dropout) {
  # Multi-head self-attention (bidirectional)
  MultiHeadAttention(num_heads=$num_heads, d_model=$d_model, dropout=$dropout)
  Add()
  LayerNormalization(epsilon=1e-12)
  
  # Position-wise feed-forward network
  Dense(units=$d_ff, activation="gelu")
  Dropout(rate=$dropout)
  Dense(units=$d_model, activation="linear")
  Dropout(rate=$dropout)
  Add()
  LayerNormalization(epsilon=1e-12)
}

network BertEncoder {
  input: (512,)  # Max sequence length
  
  layers:
    # Token embeddings
    Embedding(input_dim=30522, output_dim=768, mask_zero=True)
    LayerNormalization(epsilon=1e-12)
    Dropout(rate=0.1)
    
    # 12 encoder layers for BERT-Base
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    
    # MLM head
    Dense(units=768, activation="gelu")
    LayerNormalization(epsilon=1e-12)
    Dense(units=30522, activation="linear")
    Activation("softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-6)
  metrics: ["accuracy"]
  
  train {
    epochs: 40
    batch_size: 32
    validation_split: 0.05
  }
}
```

### Compile for TensorFlow

```bash
neural compile examples/bert_encoder.neural --backend tensorflow --output bert_tf.py
```

### Compile for PyTorch

```bash
neural compile examples/bert_encoder.neural --backend pytorch --output bert_pt.py
```

## Pre-training

### Data Preparation

**1. Tokenization with WordPiece**

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text
text = "Neural DSL makes transformers easy!"
tokens = tokenizer.tokenize(text)
# ['neural', 'ds', '##l', 'makes', 'transform', '##ers', 'easy', '!']

token_ids = tokenizer.encode(text)
# [101, 15756, 2852, 2140, 3084, 10938, 2869, 2066, 999, 102]
# [CLS] neural ds ##l makes transform ##ers easy ! [SEP]
```

**2. Creating MLM Examples**

```python
import random

def create_mlm_examples(text, tokenizer, mask_prob=0.15):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    masked_tokens = []
    labels = []
    
    for token_id in token_ids:
        if random.random() < mask_prob:
            # 80% mask, 10% random, 10% unchanged
            rand = random.random()
            if rand < 0.8:
                masked_tokens.append(tokenizer.mask_token_id)
            elif rand < 0.9:
                masked_tokens.append(random.randint(0, tokenizer.vocab_size - 1))
            else:
                masked_tokens.append(token_id)
            labels.append(token_id)
        else:
            masked_tokens.append(token_id)
            labels.append(-100)  # Ignore in loss
    
    return masked_tokens, labels
```

**3. Pre-training Loop (PyTorch)**

```python
import torch
from bert_pt import BertEncoder

model = BertEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(epochs):
    for batch in train_loader:
        input_ids, labels = batch
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute MLM loss
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Fine-tuning

### Sequence Classification (Sentiment Analysis)

```neural
network BertClassifier {
  input: (512,)
  
  layers:
    # Load pre-trained BERT encoder
    Embedding(input_dim=30522, output_dim=768, weights="pretrained_bert.h5")
    
    # 12 pre-trained encoder layers
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    # ... (repeat for all 12 layers)
    
    # Classification head
    # Extract [CLS] token representation (first position)
    GlobalAveragePooling1D()  # Or extract [:, 0, :]
    Dense(units=768, activation="tanh")
    Dropout(rate=0.1)
    Output(units=2, activation="softmax")  # Binary sentiment
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=2e-5)  # Lower LR for fine-tuning
  
  train {
    epochs: 3
    batch_size: 16
    validation_split: 0.1
  }
}
```

### Token Classification (Named Entity Recognition)

```neural
network BertNER {
  input: (512,)
  
  layers:
    # Pre-trained BERT
    Embedding(input_dim=30522, output_dim=768, weights="pretrained_bert.h5")
    # ... encoder layers ...
    
    # Token-level classification
    Dropout(rate=0.1)
    Dense(units=num_entity_types, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=3e-5)
  
  train {
    epochs: 4
    batch_size: 16
  }
}
```

## Multi-Backend Support

### TensorFlow Implementation

```python
# Compiled from bert_encoder.neural
import tensorflow as tf

model = tf.keras.models.load_model('bert_tf_model')

# Training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=40,
    batch_size=32
)

# Inference
predictions = model.predict(input_ids)
```

### PyTorch Implementation

```python
# Compiled from bert_encoder.neural
import torch
from bert_pt import BertEncoder

model = BertEncoder()
model.load_state_dict(torch.load('bert_pt_weights.pth'))
model.eval()

# Inference
with torch.no_grad():
    outputs = model(input_ids)
```

### ONNX Export for Production

```bash
# Export TensorFlow model to ONNX
neural compile examples/bert_encoder.neural --backend onnx --output bert.onnx

# Or convert PyTorch model
python -m torch.onnx.export bert_pt_model bert.onnx
```

## Production Deployment

### Optimization Techniques

**1. Mixed Precision Training**

```python
# TensorFlow
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# PyTorch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
```

**2. Gradient Accumulation**

```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**3. Model Distillation**

```neural
# DistilBERT: 6 layers instead of 12
network DistilBertEncoder {
  input: (512,)
  
  layers:
    Embedding(input_dim=30522, output_dim=768)
    LayerNormalization(epsilon=1e-12)
    Dropout(rate=0.1)
    
    # Only 6 layers (40% smaller)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    BertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    
    Dense(units=768, activation="gelu")
    LayerNormalization(epsilon=1e-12)
    Dense(units=30522, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
  
  train {
    epochs: 40
    batch_size: 32
  }
}
```

### Serving with TensorFlow Serving

```bash
# Export SavedModel
neural compile examples/bert_encoder.neural --backend tensorflow --export-saved-model

# Start TensorFlow Serving
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/model,target=/models/bert \
  -e MODEL_NAME=bert \
  tensorflow/serving

# Query the model
curl -X POST http://localhost:8501/v1/models/bert:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[101, 2054, 2003, 15756, 2852, 2140, 102]]}'
```

### Inference Optimization

```python
# PyTorch JIT compilation
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, 'bert_traced.pt')

# Load and use traced model
traced_model = torch.jit.load('bert_traced.pt')
output = traced_model(input_ids)
```

## Best Practices

### Training Tips

1. **Learning Rate Schedule**: Use linear warmup (10,000 steps) then linear decay
2. **Batch Size**: As large as possible (256+ sequences)
3. **Data Quality**: Filter and deduplicate training data
4. **Vocabulary**: Use subword tokenization (WordPiece, BPE)
5. **Regularization**: Dropout 0.1, weight decay 0.01

### Fine-tuning Tips

1. **Lower Learning Rates**: 2e-5, 3e-5, or 5e-5
2. **Fewer Epochs**: 2-4 epochs usually sufficient
3. **Small Batch Sizes**: 8-32 per GPU
4. **Layer-wise LR Decay**: Lower learning rates for early layers
5. **Freeze Early Layers**: If dataset is small, freeze first 6 layers

### Performance Benchmarks

| Task | BERT-Base | BERT-Large |
|------|-----------|------------|
| GLUE (avg) | 82.1 | 84.6 |
| SQuAD 1.1 F1 | 88.5 | 90.9 |
| SQuAD 2.0 F1 | 76.3 | 81.9 |
| MNLI | 84.6 | 86.7 |
| SST-2 | 93.5 | 94.9 |

## Advanced Topics

### RoBERTa Improvements

```neural
# Remove NSP, use dynamic masking, larger batches
network RoBertaEncoder {
  input: (512,)
  
  layers:
    # Same architecture as BERT
    # But trained with:
    # - No NSP objective
    # - Dynamic masking (different each epoch)
    # - Larger batches (8K sequences)
    # - More data (160GB vs 16GB)
    # - BPE tokenization instead of WordPiece
    
    Embedding(input_dim=50265, output_dim=768)  # BPE vocab
    # ... encoder layers ...
}
```

### ALBERT (Parameter Sharing)

```neural
# Share parameters across layers
define SharedBertLayer(num_heads, d_model, d_ff, dropout) {
  # Single layer, shared across all positions
  MultiHeadAttention(num_heads=$num_heads, d_model=$d_model, dropout=$dropout)
  Add()
  LayerNormalization(epsilon=1e-12)
  Dense(units=$d_ff, activation="gelu")
  Dropout(rate=$dropout)
  Dense(units=$d_model, activation="linear")
  Dropout(rate=$dropout)
  Add()
  LayerNormalization(epsilon=1e-12)
}

network ALBERTEncoder {
  input: (512,)
  
  layers:
    Embedding(input_dim=30000, output_dim=128)  # Factorized embedding
    Dense(units=768, activation="linear")  # Project to hidden size
    
    # Reuse same layer 12 times (parameter sharing)
    SharedBertLayer(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    # ... conceptually repeat, but share weights ...
}
```

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Use gradient checkpointing

**2. Slow Convergence**
- Check learning rate schedule
- Verify warmup is applied
- Ensure proper data preprocessing
- Check for label leakage

**3. Poor Fine-tuning Performance**
- Try different learning rates
- Increase epochs (but watch for overfitting)
- Use larger batch sizes
- Check data distribution

## Next Steps

- **Tutorial 2**: [GPT-style Decoder Models](02_transformer_gpt_complete.md)
- **Tutorial 3**: [Seq2Seq Transformers](03_transformer_seq2seq_complete.md)
- **Example Code**: `examples/bert_encoder.neural`
- **API Reference**: See `docs/api/` for detailed API documentation

## Resources

- Original Paper: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- Hugging Face Transformers: https://huggingface.co/transformers/
- Neural DSL Documentation: https://neuraldsl.readthedocs.io/

---

**Last Updated**: 2024
**Maintainer**: Neural DSL Team
