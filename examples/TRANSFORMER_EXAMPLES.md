# Transformer Architecture Examples

This directory contains comprehensive examples of transformer architectures implemented in Neural DSL, covering all major variants used in modern NLP.

## Available Examples

### 1. **seq2seq_transformer.neural** - Sequence-to-Sequence Translation
Complete encoder-decoder transformer for machine translation tasks.

**Architecture:**
- Full encoder stack (6 layers)
- Full decoder stack (6 layers)
- Multi-head attention (8 heads)
- Position-wise feed-forward networks (d_ff=2048)
- Positional encodings
- 512-dimensional embeddings

**Use Cases:**
- Machine translation (English ↔ French, etc.)
- Text summarization
- Question answering with context
- Any sequence-to-sequence task

**Key Concepts:**
- Encoder-decoder architecture
- Cross-attention mechanism
- Masked decoder attention
- Teacher forcing during training
- Beam search for inference

**Model Size:** ~60M parameters (base configuration)

---

### 2. **bert_encoder.neural** - BERT-Style Encoder-Only Model
Bidirectional encoder for pre-training and fine-tuning on downstream tasks.

**Architecture:**
- 12 encoder layers (BERT-Base)
- 768-dimensional embeddings
- 12 attention heads
- Bidirectional self-attention
- Masked language modeling (MLM) head
- Next sentence prediction (NSP) head

**Use Cases:**
- Masked language model pre-training
- Text classification (sentiment, NLI)
- Token classification (NER, POS tagging)
- Question answering (SQuAD)
- Sentence pair tasks

**Key Concepts:**
- Bidirectional context (can see future tokens)
- Masked token prediction
- [CLS] token for sequence classification
- Segment embeddings for sentence pairs
- Transfer learning via pre-training + fine-tuning

**Model Size:** ~110M parameters (BERT-Base)

---

### 3. **gpt_decoder.neural** - GPT-Style Decoder-Only Model
Autoregressive decoder for causal language modeling and text generation.

**Architecture:**
- 12 decoder layers (GPT-2 Small)
- 768-dimensional embeddings
- 12 attention heads
- Causal (masked) self-attention
- Pre-LayerNorm architecture
- Autoregressive generation

**Use Cases:**
- Language model pre-training
- Text generation
- Zero-shot and few-shot learning
- Prompt-based task completion
- Code generation
- Dialogue systems

**Key Concepts:**
- Decoder-only architecture (no encoder)
- Causal masking (cannot see future)
- Autoregressive generation (one token at a time)
- Next token prediction objective
- Prompt engineering for tasks
- Temperature and nucleus sampling

**Model Size:** ~117M parameters (GPT-2 Small)

---

### 4. **encoder_decoder_transformer.neural** - Full Reference Implementation
Complete annotated implementation of the original "Attention is All You Need" transformer.

**Architecture:**
- 6 encoder layers
- 6 decoder layers
- 8 attention heads
- 512-dimensional embeddings
- 2048-dimensional feed-forward
- Comprehensive documentation of all components

**Use Cases:**
- Educational reference for transformer internals
- Machine translation
- Sequence transduction tasks
- Research baseline

**Key Concepts:**
- Detailed explanation of all components
- Scaled dot-product attention
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Positional encodings (sinusoidal)
- Complete training and inference pipeline

**Model Size:** ~65M parameters (base configuration)

---

## Quick Comparison

| Model | Architecture | Attention | Best For |
|-------|-------------|-----------|----------|
| **Seq2Seq** | Encoder-Decoder | Bidirectional encoder + Masked decoder | Translation, Summarization |
| **BERT** | Encoder-only | Bidirectional (no masking) | Classification, Understanding |
| **GPT** | Decoder-only | Causal (masked) | Generation, Completion |
| **Full Transformer** | Encoder-Decoder | Comprehensive reference | Learning, Research baseline |

---

## Architecture Comparison

### Attention Patterns

**BERT (Encoder-only):**
```
Token 1: attends to [1, 2, 3, 4, 5]
Token 2: attends to [1, 2, 3, 4, 5]
Token 3: attends to [1, 2, 3, 4, 5]
(Bidirectional - sees all tokens)
```

**GPT (Decoder-only):**
```
Token 1: attends to [1]
Token 2: attends to [1, 2]
Token 3: attends to [1, 2, 3]
(Causal - only sees past)
```

**Seq2Seq (Encoder-Decoder):**
```
Encoder: Bidirectional on source
Decoder: Causal on target + Cross-attention to encoder
(Best of both worlds for sequence transduction)
```

---

## Usage Examples

### Compilation

```bash
# Compile to TensorFlow
neural compile examples/seq2seq_transformer.neural --backend tensorflow

# Compile to PyTorch
neural compile examples/gpt_decoder.neural --backend pytorch

# Compile BERT encoder
neural compile examples/bert_encoder.neural --backend tensorflow
```

### Visualization

```bash
# Visualize architecture
neural visualize examples/encoder_decoder_transformer.neural

# Generate architecture diagram
neural visualize examples/seq2seq_transformer.neural --output arch.png
```

### Debugging

```bash
# Launch NeuralDbg debugger
neural debug examples/bert_encoder.neural

# Monitor training in real-time
neural debug examples/gpt_decoder.neural --port 8050
```

---

## Model Configurations

### Scaling Options

All examples include configurations for multiple model sizes:

**Small/Base:**
- Layers: 6-12
- Hidden: 512-768
- Heads: 8-12
- Parameters: ~60-120M

**Medium:**
- Layers: 12-24
- Hidden: 768-1024
- Heads: 12-16
- Parameters: ~300-400M

**Large:**
- Layers: 24-48
- Hidden: 1024-1600
- Heads: 16-25
- Parameters: ~750M-1.5B

**Extra Large (GPT-3):**
- Layers: 96
- Hidden: 12288
- Heads: 96
- Parameters: ~175B

---

## Key Implementation Details

### Embeddings

All models use:
- Token embeddings (learned)
- Positional encodings (learned or sinusoidal)
- Optional segment embeddings (BERT)
- Embedding dropout for regularization

### Attention Mechanism

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Multi-Head Attention:
- Split Q, K, V into h heads
- Apply attention to each head
- Concatenate and project
```

### Position-wise Feed-Forward

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

Typically: d_ff = 4 × d_model
Example: 512 → 2048 → 512
```

### Residual Connections

```
Output = LayerNorm(x + Sublayer(x))

Benefits:
- Gradient flow through network
- Training stability
- Enables deep networks (100+ layers)
```

---

## Training Tips

### Learning Rate Schedules

**Transformer (Original):**
```python
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
warmup_steps = 4000
```

**BERT:**
```python
Linear warmup (10% of steps) → Linear decay to 0
peak_lr = 1e-4
```

**GPT:**
```python
Linear warmup (2000 steps) → Cosine decay to 0
peak_lr = 2.5e-4
```

### Regularization

- **Dropout:** 0.1-0.3 on attention and feed-forward
- **Label smoothing:** 0.1 for translation tasks
- **Weight decay:** 0.01-0.1 (AdamW)
- **Gradient clipping:** Global norm at 1.0-5.0

### Data Requirements

- **Seq2Seq:** 1M+ parallel sentence pairs
- **BERT:** 10GB+ text corpus
- **GPT:** 40GB+ diverse text

---

## Performance Benchmarks

### Translation (WMT14 EN→DE)
- Base Transformer: 27.3 BLEU
- Big Transformer: 28.4 BLEU
- Training: ~12 hours (base) on 8 GPUs

### Language Understanding (GLUE)
- BERT-Base: 79.6 average score
- BERT-Large: 82.1 average score
- Fine-tuning: 2-4 epochs per task

### Language Generation
- GPT-2 (124M): 29.4 perplexity (WebText)
- GPT-2 (1.5B): 18.2 perplexity
- GPT-3 (175B): 10.6 perplexity

---

## Advanced Topics

### Efficient Attention

All examples can be modified for efficient attention:

- **Sparse Attention:** Reduces O(n²) to O(n√n)
- **Linear Attention:** Approximates full attention in O(n)
- **Flash Attention:** Memory-efficient implementation
- **Local Attention:** Window-based attention patterns

### Model Parallelism

For very large models:

- **Data Parallelism:** Split batches across GPUs
- **Pipeline Parallelism:** Split layers across GPUs
- **Tensor Parallelism:** Split matrices across GPUs
- **Mixture of Experts:** Conditional computation

### Optimizations

- **Mixed Precision (FP16):** 2-3x speedup
- **Gradient Checkpointing:** Reduce memory usage
- **KV Caching:** 10x faster generation
- **Distillation:** Smaller models, 95%+ performance

---

## Hyperparameter Optimization

Use HPO features to tune models:

```neural
# Example: Optimize attention heads
define EncoderLayer(num_heads, d_model, d_ff, dropout) {
  MultiHeadAttention(
    num_heads=HPO(choice(4, 8, 12, 16)),
    d_model=$d_model,
    dropout=HPO(range(0.1, 0.3, step=0.05))
  )
  # ... rest of layer
}

# Optimize learning rate
optimizer: Adam(
  learning_rate=HPO(log_range(1e-5, 1e-3))
)
```

---

## References

### Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original transformer paper
   - Encoder-decoder architecture
   - Multi-head attention mechanism

2. **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
   - Encoder-only for language understanding
   - Masked language modeling
   - Transfer learning paradigm

3. **Improving Language Understanding by Generative Pre-Training** (Radford et al., 2018)
   - GPT: Decoder-only for generation
   - Autoregressive pre-training
   - Zero-shot learning

4. **Language Models are Few-Shot Learners** (Brown et al., 2020)
   - GPT-3: Scaling laws
   - In-context learning
   - Emergent abilities

### Resources

- **Hugging Face Transformers:** Pre-trained models and tokenizers
- **The Annotated Transformer:** Line-by-line PyTorch implementation
- **Jay Alammar's Blog:** Visual explanations of transformers
- **Papers with Code:** Latest research and benchmarks

---

## Next Steps

1. **Start Simple:** Begin with the encoder-decoder transformer
2. **Understand Components:** Study each macro definition
3. **Experiment:** Modify hyperparameters, try different tasks
4. **Scale Up:** Increase model size as needed
5. **Optimize:** Apply techniques from advanced topics

For more examples, see:
- `mnist_commented.neural` - CNN basics
- `resnet_block_commented.neural` - Advanced architectures with macros
- `sentiment_analysis_commented.neural` - RNN for text

---

## Support

For questions or issues:
- Check the main `README.md`
- Review `AGENTS.md` for development setup
- Use `neural debug` to inspect models
- Consult the Neural DSL documentation

Happy modeling! 🚀
