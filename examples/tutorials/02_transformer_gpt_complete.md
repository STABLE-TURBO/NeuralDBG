# Tutorial: Building GPT with Neural DSL

## Overview

This comprehensive tutorial guides you through implementing GPT (Generative Pre-trained Transformer) using Neural DSL. You'll learn how to build decoder-only transformers for autoregressive language modeling and text generation.

## Table of Contents

1. [Introduction to GPT](#introduction-to-gpt)
2. [Architecture Overview](#architecture-overview)
3. [Implementation](#implementation)
4. [Pre-training](#pre-training)
5. [Text Generation](#text-generation)
6. [Fine-tuning](#fine-tuning)
7. [Scaling Laws](#scaling-laws)
8. [Production Deployment](#production-deployment)

## Introduction to GPT

GPT revolutionized language modeling with its decoder-only architecture and massive scale. Key innovations:

- **Autoregressive Generation**: Generate text one token at a time
- **Causal Masking**: Each position can only attend to previous positions
- **Zero-shot Learning**: Perform tasks via prompting without fine-tuning
- **Few-shot Learning**: Learn from examples in the prompt
- **Massive Scale**: Performance improves predictably with model size

### Use Cases

- **Text Generation**: Creative writing, story generation
- **Code Generation**: GitHub Copilot, code completion
- **Dialogue**: Chatbots, conversational AI
- **Translation**: Machine translation without parallel data
- **Summarization**: Abstractive text summarization
- **Question Answering**: Open-domain QA

## Architecture Overview

### Key Components

1. **Token Embeddings**: Map token IDs to dense vectors
2. **Positional Embeddings**: Learned position encodings
3. **Decoder Stack**: 12-96 transformer decoder blocks
4. **Causal Attention**: Masked multi-head self-attention
5. **Language Model Head**: Project to vocabulary for next-token prediction

### GPT Model Variants

| Model | Layers | Hidden Size | Heads | Parameters | Context Length |
|-------|--------|-------------|-------|------------|----------------|
| GPT-2 Small | 12 | 768 | 12 | 117M | 1024 |
| GPT-2 Medium | 24 | 1024 | 16 | 345M | 1024 |
| GPT-2 Large | 36 | 1280 | 20 | 774M | 1024 |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B | 1024 |
| GPT-3 | 96 | 12288 | 96 | 175B | 2048 |

## Implementation

### Core GPT-2 Small Model

See `examples/gpt_decoder.neural` for the complete implementation:

```neural
define CausalAttention(num_heads, d_model, dropout) {
  # Causal attention: cannot see future tokens
  Dense(units=$d_model, activation="linear")  # Query
  Dense(units=$d_model, activation="linear")  # Key
  Dense(units=$d_model, activation="linear")  # Value
  
  # Apply causal mask before softmax
  # mask[i, j] = -inf if j > i
  
  Dropout(rate=$dropout)
  Dense(units=$d_model, activation="linear")  # Output projection
  Dropout(rate=$dropout)
}

define GptBlock(num_heads, d_model, d_ff, dropout) {
  # Pre-LayerNorm architecture (GPT-2+)
  LayerNormalization(epsilon=1e-5)
  CausalAttention(num_heads=$num_heads, d_model=$d_model, dropout=$dropout)
  Add()  # Residual connection
  
  LayerNormalization(epsilon=1e-5)
  Dense(units=$d_ff, activation="gelu")
  Dense(units=$d_model, activation="linear")
  Dropout(rate=$dropout)
  Add()  # Residual connection
}

network GptDecoder {
  input: (1024,)  # Context length
  
  layers:
    # Token + Positional Embeddings
    Embedding(input_dim=50257, output_dim=768)  # BPE vocab
    Dropout(rate=0.1)
    
    # 12 decoder blocks for GPT-2 Small
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    
    # Final layer norm
    LayerNormalization(epsilon=1e-5)
    
    # Language model head
    Dense(units=50257, activation="linear")
    Activation("softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.00025, beta_1=0.9, beta_2=0.95, epsilon=1e-8)
  metrics: ["accuracy"]
  
  train {
    epochs: 1  # Usually train by steps, not epochs
    batch_size: 8
    validation_split: 0.05
    gradient_clip: 1.0
  }
}
```

### Compile for Different Backends

```bash
# TensorFlow
neural compile examples/gpt_decoder.neural --backend tensorflow --output gpt_tf.py

# PyTorch
neural compile examples/gpt_decoder.neural --backend pytorch --output gpt_pt.py

# ONNX for production
neural compile examples/gpt_decoder.neural --backend onnx --output gpt.onnx
```

## Pre-training

### Data Preparation

**1. Byte-Pair Encoding (BPE) Tokenization**

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize text
text = "Neural DSL makes language models easy!"
tokens = tokenizer.encode(text)
# [8199, 1523, 23639, 43, 1838, 3303, 2746, 2562, 0]

# Decode back to text
decoded = tokenizer.decode(tokens)
# "Neural DSL makes language models easy!"

# BPE handles any text (no unknown tokens)
text = "supercalifragilisticexpialidocious"
tokens = tokenizer.tokenize(text)
# ['super', 'cal', 'if', 'rag', 'il', 'istic', 'exp', 'ial', 'id', 'oci', 'ous']
```

**2. Creating Training Data**

```python
import numpy as np

def create_training_sequences(text, tokenizer, max_length=1024):
    """Create training sequences for next-token prediction."""
    # Tokenize entire corpus
    all_tokens = tokenizer.encode(text)
    
    # Split into sequences of max_length
    sequences = []
    for i in range(0, len(all_tokens) - max_length, max_length):
        seq = all_tokens[i:i + max_length]
        sequences.append(seq)
    
    return np.array(sequences)

def prepare_batch(sequences):
    """Prepare input and target sequences."""
    # Input: all tokens except last
    # Target: all tokens except first
    input_ids = sequences[:, :-1]
    target_ids = sequences[:, 1:]
    
    return input_ids, target_ids
```

**3. Pre-training Loop (PyTorch)**

```python
import torch
import torch.nn.functional as F
from gpt_pt import GptDecoder

model = GptDecoder()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2.5e-4,
    betas=(0.9, 0.95),
    eps=1e-8
)

# Cosine learning rate schedule with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

warmup_steps = 2000
total_steps = 1_000_000

def get_lr(step):
    if step < warmup_steps:
        return (step / warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

# Training loop
for step, batch in enumerate(train_loader):
    input_ids, target_ids = batch
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    scheduler.step()
    
    # Log perplexity
    if step % 100 == 0:
        perplexity = torch.exp(loss)
        print(f"Step {step}, Loss: {loss.item():.4f}, PPL: {perplexity.item():.2f}")
```

### Learning Rate Schedule

```python
import matplotlib.pyplot as plt

def visualize_lr_schedule(warmup_steps=2000, total_steps=100000):
    """Visualize the learning rate schedule."""
    steps = np.arange(total_steps)
    lrs = []
    
    for step in steps:
        if step < warmup_steps:
            lr = step / warmup_steps
        else:
            lr = 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        lrs.append(lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate Multiplier')
    plt.title('Cosine Annealing with Linear Warmup')
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    plt.show()
```

## Text Generation

### Autoregressive Generation

```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    """Generate text autoregressively."""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens])
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits for next token
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Append to sequence
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            
            # Stop at EOS token
            if next_token == tokenizer.eos_token_id:
                break
            
            # Truncate if exceeds context length
            if input_ids.shape[1] > 1024:
                input_ids = input_ids[:, -1024:]
    
    return tokenizer.decode(generated)
```

### Sampling Strategies

**1. Temperature Sampling**

```python
def sample_with_temperature(logits, temperature=1.0):
    """Control randomness with temperature."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# temperature = 0.7: More focused, less random
# temperature = 1.0: Sample from model distribution
# temperature = 1.5: More creative, more random
```

**2. Top-k Sampling**

```python
def sample_top_k(logits, k=40):
    """Sample from top-k most likely tokens."""
    # Get top-k tokens
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Zero out logits for tokens not in top-k
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(0, top_k_indices, top_k_logits)
    
    # Sample from filtered distribution
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**3. Top-p (Nucleus) Sampling**

```python
def sample_top_p(logits, p=0.9):
    """Sample from smallest set with cumulative probability >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff index
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Zero out removed tokens
    logits_filtered = logits.clone()
    logits_filtered[sorted_indices[sorted_indices_to_remove]] = float('-inf')
    
    # Sample
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**4. Combined Strategy**

```python
def generate_with_advanced_sampling(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.2
):
    """Generate text with advanced sampling strategies."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens])
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)[0, -1, :]
            
            # Apply repetition penalty
            for token in set(generated):
                logits[token] /= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-p sampling
            next_token = sample_top_p(logits, p=top_p)
            
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated)
```

### Example Generations

```python
# Creative writing
prompt = "Once upon a time in a distant galaxy"
text = generate_with_advanced_sampling(model, tokenizer, prompt, temperature=1.0)

# Technical writing
prompt = "The key features of neural networks are"
text = generate_with_advanced_sampling(model, tokenizer, prompt, temperature=0.7)

# Code generation
prompt = "def fibonacci(n):\n    \"\"\""
text = generate_with_advanced_sampling(model, tokenizer, prompt, temperature=0.2)
```

## Fine-tuning

### Task-Specific Fine-tuning

**1. Text Classification**

```neural
network GptClassifier {
  input: (512,)
  
  layers:
    # Pre-trained GPT
    Embedding(input_dim=50257, output_dim=768, weights="pretrained_gpt.h5")
    
    # 12 decoder blocks
    GptBlock(num_heads=12, d_model=768, d_ff=3072, dropout=0.1)
    # ... (load pre-trained weights)
    
    LayerNormalization(epsilon=1e-5)
    
    # Classification head (use last token representation)
    GlobalAveragePooling1D()
    Dense(units=768, activation="relu")
    Dropout(rate=0.1)
    Output(units=num_classes, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=5e-5)
  
  train {
    epochs: 3
    batch_size: 8
  }
}
```

**2. Dialogue / Chatbot**

```python
def prepare_dialogue_data(conversations):
    """Prepare dialogue data for fine-tuning."""
    formatted = []
    for conv in conversations:
        formatted_conv = ""
        for turn in conv:
            speaker = turn['speaker']  # 'User' or 'Assistant'
            text = turn['text']
            formatted_conv += f"{speaker}: {text}\n"
        formatted.append(formatted_conv)
    return formatted

# Fine-tune on dialogue data
dialogue_data = prepare_dialogue_data(conversations)
# Train with standard next-token prediction
```

### Prompt Engineering

**Zero-shot Classification**

```python
prompt = """Classify the sentiment of the following review:
Review: "This movie was absolutely terrible. Worst film I've ever seen."
Sentiment:"""

# Model completes with "negative"
```

**Few-shot Learning**

```python
prompt = """Translate English to French:
Hello -> Bonjour
Goodbye -> Au revoir
Thank you -> Merci
How are you? ->"""

# Model completes with "Comment allez-vous?"
```

## Scaling Laws

### Performance vs Model Size

GPT performance scales predictably with model size:

```
Loss ∝ N^(-α)
```

Where N is the number of parameters, α ≈ 0.076

| Parameters | Loss | Perplexity | Zero-shot Acc |
|------------|------|------------|---------------|
| 125M | 3.40 | ~30 | ~45% |
| 350M | 3.00 | ~20 | ~52% |
| 1.3B | 2.70 | ~15 | ~58% |
| 6.7B | 2.48 | ~12 | ~63% |
| 175B | 2.30 | ~10 | ~71% |

### Compute Optimal Training

For a compute budget C, optimal allocation:

- **Model Size**: N ∝ C^0.73
- **Data Size**: D ∝ C^0.27

Example: 10x more compute → 5.4x larger model, 2x more data

## Production Deployment

### Optimization Techniques

**1. KV Cache for Faster Generation**

```python
class GPTWithCache:
    def __init__(self, model):
        self.model = model
        self.cache = None
    
    def forward(self, input_ids, use_cache=True):
        if use_cache and self.cache is not None:
            # Only compute for new token
            new_input = input_ids[:, -1:]
            output, self.cache = self.model(new_input, past_key_values=self.cache)
        else:
            output, self.cache = self.model(input_ids)
        
        return output

# Speeds up generation by ~10x
```

**2. Model Quantization**

```python
# PyTorch quantization
import torch.quantization

# Dynamic quantization (8-bit integers)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Reduces model size by ~4x with minimal accuracy loss
```

**3. Distributed Inference**

```python
# Model parallelism for large models
import torch.distributed as dist

def load_model_parallel(model_path, world_size):
    """Load model sharded across multiple GPUs."""
    rank = dist.get_rank()
    
    # Load portion of model on each GPU
    model_shard = load_model_shard(model_path, rank, world_size)
    
    return model_shard

# GPT-3 scale models require model parallelism
```

### Serving with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model once at startup
model = torch.load('gpt_model.pt')
model.eval()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.9

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate text endpoint."""
    text = generate_with_advanced_sampling(
        model,
        tokenizer,
        request.prompt,
        max_length=request.max_length,
        temperature=request.temperature
    )
    return {"generated_text": text}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Monitoring and Logging

```python
import logging
from prometheus_client import Counter, Histogram

# Metrics
generation_requests = Counter('gpt_generation_requests', 'Total generation requests')
generation_latency = Histogram('gpt_generation_latency', 'Generation latency in seconds')

@app.post("/generate")
async def generate(request: GenerationRequest):
    generation_requests.inc()
    
    with generation_latency.time():
        text = generate_with_advanced_sampling(...)
    
    logging.info(f"Generated {len(text)} characters")
    return {"generated_text": text}
```

## Best Practices

### Training

1. **Large Batch Sizes**: Use gradient accumulation for effective batch size of 512+
2. **Learning Rate**: Cosine schedule with linear warmup (2000 steps)
3. **Gradient Clipping**: Clip gradients to norm of 1.0
4. **Weight Decay**: 0.1 on all parameters except biases and layer norms
5. **Data Quality**: High-quality, diverse training data is crucial

### Generation

1. **Temperature**: 0.7-0.9 for coherent text, 1.0-1.2 for creative text
2. **Top-p**: 0.9-0.95 works well for most tasks
3. **Repetition Penalty**: 1.2 prevents repetitive outputs
4. **Length Penalty**: Encourage longer/shorter outputs as needed

### Fine-tuning

1. **Lower Learning Rate**: 1e-5 to 5e-5
2. **Fewer Epochs**: 1-3 epochs usually sufficient
3. **Format**: Use consistent prompt format
4. **Validation**: Monitor perplexity and sample generations

## Troubleshooting

### Common Issues

**1. Repetitive Generation**
- Increase temperature
- Use top-p sampling
- Apply repetition penalty
- Check training data for repetition

**2. Incoherent Output**
- Decrease temperature
- Use lower top-p value
- Fine-tune on higher quality data
- Check tokenization

**3. Slow Inference**
- Implement KV caching
- Use quantization
- Batch multiple requests
- Use ONNX or TensorRT

**4. Out of Memory**
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Use mixed precision training

## Next Steps

- **Tutorial 3**: [Seq2Seq Transformers](03_transformer_seq2seq_complete.md)
- **Tutorial 4**: [Computer Vision with ResNet](04_computer_vision_resnet.md)
- **Example Code**: `examples/gpt_decoder.neural`
- **HPO Integration**: See Tutorial 6 for hyperparameter optimization

## Resources

- Original Papers:
  - [Improving Language Understanding with Unsupervised Learning (GPT)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- Hugging Face Transformers: https://huggingface.co/transformers/
- Neural DSL Docs: https://neuraldsl.readthedocs.io/

---

**Last Updated**: 2024
**Maintainer**: Neural DSL Team
