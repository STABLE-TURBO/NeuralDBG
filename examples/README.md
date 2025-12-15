# Neural DSL Examples - Complete Guide

Production-ready examples showcasing Neural DSL's capabilities across transformers, computer vision, hyperparameter optimization, and AutoML. All examples support TensorFlow, PyTorch, and ONNX backends.

## 🚀 Quick Start

```bash
# Install Neural DSL with all features
pip install -e ".[full]"

# Try a simple example
neural compile examples/mnist.neural --backend pytorch

# Run hyperparameter optimization
neural hpo examples/mnist_hpo.neural --n-trials 20

# Visualize architecture
neural visualize examples/bert_encoder.neural
```

## 📚 What's Included

### 🤖 **Transformer Architectures** (Production-Ready)

| Model | File | Params | Use Case | Performance |
|-------|------|--------|----------|-------------|
| **BERT** | `bert_encoder.neural` | 110M | MLM, Classification, NER, QA | 82.1 GLUE |
| **GPT-2** | `gpt_decoder.neural` | 117M | Text Gen, Dialogue | 35 PPL |
| **Seq2Seq** | `seq2seq_transformer.neural` | 65M | Translation | 28 BLEU (EN-DE) |

**Tutorials**: Comprehensive guides in `tutorials/01_transformer_bert_complete.md`, `02_transformer_gpt_complete.md`

### 👁️ **Computer Vision** (Production-Ready)

| Model | File | Params | Accuracy | Speed |
|-------|------|--------|----------|-------|
| **ResNet-50** | `computer_vision/resnet50_production.neural` | 25.6M | 76.5% Top-1 | 4-5ms |
| **EfficientNet-B0** | `computer_vision/efficientnet_b0.neural` | 5.3M | 77.3% Top-1 | 2-3ms |

**Features**: Transfer learning, data augmentation, mobile deployment

### 🔧 **Hyperparameter Optimization**

- **Complete HPO**: `hpo_examples/transformer_hpo_complete.neural` with Optuna integration
- **Python Examples**: `hpo_advanced_example.py` with visualization
- **CLI Support**: Multi-objective, distributed, Bayesian optimization

### 🧬 **AutoML and Neural Architecture Search**

- **Comprehensive NAS**: `automl_examples/nas_comprehensive.py`
- **Search Strategies**: Random, Evolutionary, Bayesian
- **Multi-objective**: Accuracy, latency, model size optimization

## 📖 Documentation Structure

```
examples/
├── README.md                          # This file - overview and quick start
├── EXAMPLES_INDEX.md                  # Complete catalog of all examples
├── tutorials/
│   ├── README.md                      # Tutorial navigation
│   ├── 01_transformer_bert_complete.md
│   ├── 02_transformer_gpt_complete.md
│   └── ...
├── computer_vision/
│   ├── resnet50_production.neural
│   └── efficientnet_b0.neural
├── hpo_examples/
│   └── transformer_hpo_complete.neural
├── automl_examples/
│   └── nas_comprehensive.py
└── [50+ example files]
```

## 🎯 Use Case Navigator

### "I want to build a text classifier"
1. Start: `sentiment.neural` (LSTM-based)
2. Upgrade: `bert_encoder.neural` (state-of-the-art)
3. Optimize: `hpo_examples/transformer_hpo_complete.neural`
4. Tutorial: `tutorials/01_transformer_bert_complete.md`

### "I want to generate text"
1. Start: `gpt_decoder.neural` (GPT-2 Small)
2. Scale up: Increase layers/dimensions
3. Fine-tune: For specific tasks (code, dialogue)
4. Tutorial: `tutorials/02_transformer_gpt_complete.md`

### "I need image classification"
1. Start: `mnist.neural` (simple CNN)
2. Production: `computer_vision/resnet50_production.neural`
3. Mobile: `computer_vision/efficientnet_b0.neural`
4. Transfer: Use pre-trained weights

### "I want to optimize hyperparameters"
1. Basic: `mnist_hpo.neural`
2. Advanced: `hpo_examples/transformer_hpo_complete.neural`
3. Python: `hpo_advanced_example.py`
4. Multi-objective: Use `--objectives` flag

### "I want to search for architectures"
1. Start: `automl_examples/nas_comprehensive.py --example basic`
2. Multi-objective: `--example multi`
3. Transformers: `--example transformer`
4. Compare strategies: `--example comparison`

## 🔨 Common Tasks

### Compile to Different Backends

```bash
# PyTorch (recommended for research)
neural compile examples/bert_encoder.neural --backend pytorch

# TensorFlow (good for production)
neural compile examples/bert_encoder.neural --backend tensorflow

# ONNX (best for deployment)
neural compile examples/bert_encoder.neural --backend onnx

# Compile to all backends
neural compile examples/bert_encoder.neural --backend all
```

### Visualize Architecture

```bash
# Generate architecture diagram
neural visualize examples/gpt_decoder.neural --output gpt_arch.png

# Interactive debugging dashboard
neural debug examples/resnet50_production.neural
```

### Run Hyperparameter Optimization

```bash
# Basic HPO (20 trials)
neural hpo examples/mnist_hpo.neural --n-trials 20 --backend pytorch

# Advanced HPO (50 trials, GPU)
neural hpo examples/hpo_examples/transformer_hpo_complete.neural \
  --n-trials 50 --device cuda --backend pytorch

# Multi-objective optimization
neural hpo examples/hpo_examples/transformer_hpo_complete.neural \
  --objectives accuracy:maximize training_time:minimize latency:minimize \
  --n-trials 100 --backend pytorch

# Distributed HPO with Ray
neural hpo examples/hpo_examples/transformer_hpo_complete.neural \
  --distributed ray --num-workers 8 --n-trials 200 --backend pytorch
```

### Run Neural Architecture Search

```bash
# Basic CNN search
python examples/automl_examples/nas_comprehensive.py --example basic

# Multi-objective NAS
python examples/automl_examples/nas_comprehensive.py --example multi

# Transformer architecture search
python examples/automl_examples/nas_comprehensive.py --example transformer

# Compare search strategies
python examples/automl_examples/nas_comprehensive.py --example comparison

# Cross-backend search
python examples/automl_examples/nas_comprehensive.py --example cross
```

## 🎓 Learning Paths

### Beginner Path (1-2 weeks)
1. **Week 1**: Basic examples
   - `mnist.neural` - Understand DSL syntax
   - `mnist_commented.neural` - Detailed explanations
   - `sentiment.neural` - Text classification
   - Compile and run all three

2. **Week 2**: Intermediate concepts
   - `resnet_block_commented.neural` - Residual connections
   - `multihead_attention.neural` - Attention mechanism
   - `mnist_hpo.neural` - Basic HPO
   - Read Tutorial 1 (BERT)

### Intermediate Path (2-4 weeks)
1. **Weeks 1-2**: Transformers
   - `bert_encoder.neural` - Full BERT implementation
   - `gpt_decoder.neural` - GPT architecture
   - Complete Tutorial 1 and 2
   - Fine-tune on custom datasets

2. **Weeks 3-4**: Optimization
   - `hpo_examples/transformer_hpo_complete.neural`
   - `hpo_advanced_example.py` - Python API
   - Run distributed HPO
   - Analyze results with visualizations

### Advanced Path (4+ weeks)
1. **Weeks 1-2**: Computer Vision
   - `computer_vision/resnet50_production.neural`
   - `computer_vision/efficientnet_b0.neural`
   - Transfer learning
   - Mobile deployment

2. **Weeks 3-4**: AutoML
   - `automl_examples/nas_comprehensive.py`
   - Multi-objective optimization
   - Custom search spaces
   - Production deployment

3. **Ongoing**: Production
   - Model serving (TF Serving, TorchServe)
   - Optimization (quantization, pruning)
   - Monitoring and logging
   - A/B testing

## 💡 Best Practices

### 1. Starting a New Project

```bash
# 1. Choose appropriate example
cp examples/bert_encoder.neural my_project.neural

# 2. Modify for your use case
# - Change input dimensions
# - Adjust model size
# - Update output classes

# 3. Compile and test
neural compile my_project.neural --backend pytorch

# 4. Run HPO (optional but recommended)
neural hpo my_project.neural --n-trials 30
```

### 2. Hyperparameter Tuning Strategy

```bash
# Step 1: Quick search (20 trials, wide ranges)
neural hpo my_project.neural --n-trials 20

# Step 2: Focused search (50 trials, narrow ranges based on step 1)
neural hpo my_project.neural --n-trials 50

# Step 3: Fine-tuning (100 trials, Bayesian optimization)
neural hpo my_project.neural --n-trials 100 --sampler tpe
```

### 3. Production Deployment Checklist

- [ ] Optimize model (quantization, pruning)
- [ ] Export to ONNX for maximum compatibility
- [ ] Benchmark inference latency
- [ ] Set up model serving (TF Serving / TorchServe)
- [ ] Implement monitoring and logging
- [ ] Prepare rollback strategy
- [ ] Load testing
- [ ] A/B testing setup

## 📊 Performance Benchmarks

All benchmarks run on NVIDIA V100 GPU unless specified:

### Transformers

| Model | Training (8 GPUs) | Inference (Single GPU) | Memory |
|-------|-------------------|------------------------|--------|
| BERT-Base | ~3 days | 10-20ms | 4GB |
| BERT-Large | ~7 days | 30-40ms | 8GB |
| GPT-2 Small | ~7 days | 50-100ms | 4GB |
| GPT-2 Large | ~14 days | 200-300ms | 12GB |

### Computer Vision

| Model | Training (8 GPUs) | Inference (Single GPU) | Accuracy |
|-------|-------------------|------------------------|----------|
| ResNet-50 | 2-3 days | 4-5ms | 76.5% Top-1 |
| ResNet-101 | 4-5 days | 8-10ms | 77.4% Top-1 |
| EfficientNet-B0 | 2 days | 2-3ms | 77.3% Top-1 |
| EfficientNet-B7 | 7 days | 70ms | 84.4% Top-1 |

### HPO

| Task | Trials | Time (8 GPUs) | Speedup |
|------|--------|---------------|---------|
| CNN (MNIST) | 50 | ~2 hours | 1× |
| ResNet-50 | 50 | ~4 days | 8× (distributed) |
| Transformer | 50 | ~5 days | 8× (distributed) |

### NAS

| Search Space | Trials | Time (8 GPUs) | Best Accuracy |
|--------------|--------|---------------|---------------|
| CNN (CIFAR-10) | 100 | ~2 days | 95%+ |
| Transformer (IMDB) | 100 | ~5 days | 92%+ |
| EfficientNet | 50 | ~3 days | 82%+ |

## 🔗 Additional Resources

### Documentation
- **Main Docs**: `/docs/README.md`
- **API Reference**: `/docs/api/`
- **CLI Reference**: `/docs/cli_reference.md`
- **DSL Syntax**: `/docs/dsl.md`

### Tutorials
- **Tutorial Index**: `tutorials/README.md`
- **BERT Tutorial**: `tutorials/01_transformer_bert_complete.md`
- **GPT Tutorial**: `tutorials/02_transformer_gpt_complete.md`

### Example Catalog
- **Complete Index**: `EXAMPLES_INDEX.md`
- **50+ Examples**: Organized by complexity and use case

### External Resources
- **Papers**: See `tutorials/README.md` for research papers
- **Hugging Face**: Pre-trained transformer weights
- **TensorFlow Hub**: Pre-trained vision models
- **PyTorch Hub**: Model zoo

### Community
- **GitHub**: https://github.com/neuraldsl/neuraldsl
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord**: Real-time community chat

## 🐛 Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Reduce batch size
train { batch_size: 16 }  # Instead of 32 or 64

# Use gradient accumulation
train { 
    batch_size: 8
    gradient_accumulation_steps: 4  # Effective batch size = 32
}

# Enable mixed precision
train { use_mixed_precision: True }
```

**Slow Training**
```bash
# Use GPU
--device cuda

# Increase batch size (if memory allows)
train { batch_size: 128 }

# Enable mixed precision
train { use_mixed_precision: True }

# Use multiple GPUs
train { distributed: True, num_gpus: 8 }
```

**Poor Accuracy**
```neural
# Add regularization
Dropout(rate=0.3)

# Reduce model size (prevent overfitting)
Dense(units=256)  # Instead of 1024

# Increase training data
# Apply data augmentation

# Try different learning rate
optimizer: Adam(learning_rate=0.0001)
```

**Import Errors**
```bash
# Install missing dependencies
pip install -e ".[full]"

# Or specific feature groups
pip install -e ".[hpo]"      # For HPO
pip install -e ".[automl]"   # For AutoML
pip install -e ".[backends]" # For all backends
```

## 🤝 Contributing Examples

We welcome new examples! Guidelines:

1. **Clear Use Case**: Explain what the example demonstrates
2. **Documentation**: Add comments and documentation
3. **Testing**: Ensure it compiles and runs
4. **Performance**: Include expected accuracy/speed
5. **Format**: Follow existing example structure

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## 📝 License

All examples are licensed under MIT License. See [LICENSE](../LICENSE) for details.

---

## 🎉 Quick Reference

### Most Popular Examples

1. **BERT** (`bert_encoder.neural`) - SOTA text understanding
2. **GPT** (`gpt_decoder.neural`) - SOTA text generation
3. **ResNet-50** (`computer_vision/resnet50_production.neural`) - Computer vision
4. **HPO Complete** (`hpo_examples/transformer_hpo_complete.neural`) - Optimization
5. **NAS Comprehensive** (`automl_examples/nas_comprehensive.py`) - Architecture search

### Essential Commands

```bash
# Compile
neural compile <file>.neural --backend pytorch

# Visualize
neural visualize <file>.neural

# HPO
neural hpo <file>.neural --n-trials 50

# Help
neural --help
neural compile --help
neural hpo --help
```

### Next Steps

1. **Try Examples**: Start with `mnist.neural`, progress to transformers
2. **Read Tutorials**: Comprehensive guides in `tutorials/`
3. **Explore HPO**: Optimize your models automatically
4. **Try AutoML**: Let NAS find the best architecture
5. **Deploy**: Production-ready ONNX export

---

**Ready to build state-of-the-art models?** Start with `examples/mnist.neural` and work your way up to transformers!

**Questions?** Check `tutorials/README.md` or open an issue on GitHub.

**Last Updated**: 2024  
**Total Examples**: 50+  
**Backends Supported**: TensorFlow, PyTorch, ONNX  
**Version**: 1.0.0
