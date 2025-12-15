# Neural DSL Tutorials and Examples

Welcome to the comprehensive tutorials and examples for Neural DSL! This guide provides production-ready implementations of state-of-the-art neural network architectures with detailed explanations and best practices.

## 📚 Table of Contents

- [Getting Started](#getting-started)
- [Transformer Architectures](#transformer-architectures)
- [Computer Vision](#computer-vision)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [AutoML and Neural Architecture Search](#automl-and-neural-architecture-search)
- [Multi-Backend Support](#multi-backend-support)
- [Production Deployment](#production-deployment)

## 🚀 Getting Started

### Prerequisites

```bash
# Install Neural DSL with all features
pip install -e ".[full]"

# Or install specific feature groups
pip install -e ".[hpo]"      # Hyperparameter optimization
pip install -e ".[automl]"   # AutoML and NAS
pip install -e ".[backends]" # All ML framework backends
```

### Quick Start Example

```bash
# Compile a simple network
neural compile examples/mnist.neural --backend pytorch

# Visualize architecture
neural visualize examples/bert_encoder.neural

# Run hyperparameter optimization
neural hpo examples/hpo_examples/transformer_hpo_complete.neural \
  --n-trials 50 --backend pytorch
```

## 🤖 Transformer Architectures

### Tutorial 1: BERT (Encoder-Only)

**File**: `01_transformer_bert_complete.md`

Learn to build BERT-style encoder-only transformers for:
- Masked Language Modeling (MLM)
- Sentence Classification
- Named Entity Recognition (NER)
- Question Answering

**Example Code**: `../bert_encoder.neural`

**Key Features**:
- Bidirectional self-attention
- Pre-training and fine-tuning workflows
- Transfer learning
- Production deployment strategies

**Compile and Run**:
```bash
# TensorFlow backend
neural compile examples/bert_encoder.neural --backend tensorflow

# PyTorch backend
neural compile examples/bert_encoder.neural --backend pytorch

# ONNX for production
neural compile examples/bert_encoder.neural --backend onnx
```

**Performance**: 
- BERT-Base: 110M parameters, 82.1 GLUE score
- Training: ~3 days on 16 TPU v3 chips

---

### Tutorial 2: GPT (Decoder-Only)

**File**: `02_transformer_gpt_complete.md`

Build GPT-style decoder-only transformers for:
- Autoregressive text generation
- Code generation
- Dialogue systems
- Zero-shot and few-shot learning

**Example Code**: `../gpt_decoder.neural`

**Key Features**:
- Causal (masked) attention
- Advanced sampling strategies (top-k, top-p, temperature)
- KV caching for fast inference
- Prompt engineering

**Text Generation Example**:
```python
from gpt_pt import GptDecoder
import torch

model = GptDecoder()
model.load_state_dict(torch.load('gpt2_weights.pth'))

# Generate text
prompt = "Once upon a time"
generated = generate_text(model, tokenizer, prompt, max_length=100)
print(generated)
```

**Performance**:
- GPT-2 Small: 117M parameters, ~35 perplexity
- GPT-3: 175B parameters, ~10 perplexity

---

### Tutorial 3: Seq2Seq Transformers

**File**: `03_transformer_seq2seq_complete.md` (to be created)

Complete encoder-decoder transformers for:
- Machine Translation
- Text Summarization
- Speech Recognition
- Image Captioning

**Example Code**: `../seq2seq_transformer.neural`

**Key Features**:
- Encoder-decoder architecture
- Cross-attention mechanism
- Beam search decoding
- BLEU score optimization

**Compile**:
```bash
neural compile examples/seq2seq_transformer.neural --backend pytorch
```

**Performance**:
- WMT14 EN-DE: ~28 BLEU
- WMT14 EN-FR: ~41 BLEU

---

## 👁️ Computer Vision

### Tutorial 4: ResNet-50

**File**: `04_computer_vision_resnet.md` (to be created)

**Example Code**: `../computer_vision/resnet50_production.neural`

Production-ready ResNet-50 for:
- Image Classification
- Feature Extraction
- Transfer Learning
- Object Detection (backbone)

**Architecture Highlights**:
- 50 layers with bottleneck blocks
- Batch normalization
- Skip connections
- 25.6M parameters

**Compile**:
```bash
neural compile examples/computer_vision/resnet50_production.neural \
  --backend pytorch --output resnet50.py
```

**Data Augmentation**:
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Performance**:
- ImageNet Top-1: 76.5%
- ImageNet Top-5: 93.0%
- Inference: 4-5ms per image (GPU)

---

### Tutorial 5: EfficientNet

**File**: `05_computer_vision_efficientnet.md` (to be created)

**Example Code**: `../computer_vision/efficientnet_b0.neural`

Efficient convolutional networks with:
- Compound scaling (width, depth, resolution)
- Mobile inverted bottleneck (MBConv)
- Squeeze-and-excitation blocks
- Optimized for mobile/edge devices

**Model Variants**:
- EfficientNet-B0: 5.3M params, 77.3% top-1
- EfficientNet-B7: 66M params, 84.4% top-1

---

## 🔧 Hyperparameter Optimization

### Tutorial 6: HPO with Optuna

**File**: `06_hpo_comprehensive.md` (to be created)

**Example Code**: `../hpo_examples/transformer_hpo_complete.neural`

Comprehensive HPO covering:
- Bayesian optimization (TPE, GP)
- Multi-objective optimization
- Distributed HPO with Ray Tune
- Parameter importance analysis
- Visualization and reporting

**Basic HPO Example**:
```python
from neural.hpo import optimize_and_return

best_params = optimize_and_return(
    config="examples/hpo_examples/transformer_hpo_complete.neural",
    n_trials=50,
    dataset_name='IMDB',
    backend='pytorch',
    device='cuda',
    sampler='tpe',  # Tree-structured Parzen Estimator
    pruner='median'  # Early stopping
)

print("Best hyperparameters:", best_params)
```

**CLI Usage**:
```bash
# Basic HPO
neural hpo examples/hpo_examples/transformer_hpo_complete.neural \
  --n-trials 50 --dataset IMDB --backend pytorch

# Multi-objective
neural hpo examples/hpo_examples/transformer_hpo_complete.neural \
  --objectives accuracy:maximize training_time:minimize \
  --n-trials 100

# Distributed
neural hpo examples/hpo_examples/transformer_hpo_complete.neural \
  --distributed ray --num-workers 8 --n-trials 200
```

**Search Space Examples**:

```neural
# Categorical choices
Dense(units=HPO(categorical(64, 128, 256, 512)))

# Continuous ranges
Dropout(rate=HPO(range(0.1, 0.5, step=0.05)))

# Log-scale ranges (for learning rates)
optimizer: Adam(learning_rate=HPO(log_range(1e-5, 1e-2)))

# Training hyperparameters
train {
  epochs: HPO(categorical(10, 20, 30))
  batch_size: HPO(categorical(16, 32, 64, 128))
}
```

---

## 🧬 AutoML and Neural Architecture Search

### Tutorial 7: Neural Architecture Search (NAS)

**File**: `07_nas_comprehensive.md` (to be created)

**Example Code**: `../automl_examples/nas_comprehensive.py`

Complete NAS implementations:
- CNN architecture search
- Transformer architecture search
- Multi-objective NAS
- Search strategy comparison
- Cross-backend NAS

**Basic NAS Example**:
```python
from neural.automl import AutoMLEngine, ArchitectureSpace

# Define search space
search_space_config = """
network SearchableCNN {
    input: (32, 32, 3)
    layers:
        Conv2D(
            filters=choice(32, 64, 128, 256),
            kernel_size=choice((3,3), (5,5), (7,7)),
            activation=choice("relu", "elu", "selu")
        )
        # ... more layers ...
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=log_range(1e-4, 1e-2))
}
"""

architecture_space = ArchitectureSpace.from_dsl(search_space_config)

# Initialize AutoML engine
engine = AutoMLEngine(
    search_strategy='evolutionary',
    early_stopping='median',
    backend='pytorch',
    device='cuda'
)

# Run NAS
results = engine.search(
    architecture_space=architecture_space,
    train_data=train_loader,
    val_data=val_loader,
    max_trials=50,
    max_epochs_per_trial=10
)

print("Best architecture:", results['best_architecture'])
print("Best accuracy:", results['best_metrics']['val_acc'])
```

**Search Strategies**:
- **Random Search**: Baseline, good for initial exploration
- **Evolutionary Search**: Population-based, good for multi-objective
- **Bayesian Optimization**: Sample-efficient, good for expensive evaluations
- **Regularized Evolution**: State-of-the-art for NAS

**Run Examples**:
```bash
# Basic CNN NAS
python examples/automl_examples/nas_comprehensive.py --example basic

# Multi-objective NAS
python examples/automl_examples/nas_comprehensive.py --example multi

# Transformer NAS
python examples/automl_examples/nas_comprehensive.py --example transformer

# Compare strategies
python examples/automl_examples/nas_comprehensive.py --example comparison
```

---

## 🔀 Multi-Backend Support

All examples support multiple backends:

### TensorFlow
```bash
neural compile examples/bert_encoder.neural --backend tensorflow
```

### PyTorch
```bash
neural compile examples/gpt_decoder.neural --backend pytorch
```

### ONNX (Production)
```bash
neural compile examples/resnet50_production.neural --backend onnx
```

### Backend Comparison

| Feature | TensorFlow | PyTorch | ONNX |
|---------|------------|---------|------|
| Training | ✅ | ✅ | ❌ |
| Inference | ✅ | ✅ | ✅ |
| Dynamic Shapes | Limited | ✅ | Limited |
| Mobile Deployment | ✅ (TFLite) | ✅ (Mobile) | ✅ |
| Hardware Acceleration | ✅ | ✅ | ✅ |
| Distributed Training | ✅ | ✅ | ❌ |

---

## 🚢 Production Deployment

### TensorFlow Serving

```bash
# Export SavedModel
neural compile examples/bert_encoder.neural \
  --backend tensorflow --export-saved-model

# Start TF Serving
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/model,target=/models/bert \
  -e MODEL_NAME=bert \
  tensorflow/serving
```

### TorchServe

```bash
# Create model archive
torch-model-archiver --model-name gpt2 \
  --version 1.0 \
  --model-file gpt_pt.py \
  --serialized-file gpt2_weights.pth \
  --handler text_handler

# Start TorchServe
torchserve --start --model-store model_store --models gpt2=gpt2.mar
```

### ONNX Runtime

```python
import onnxruntime as ort

# Load model
session = ort.InferenceSession("resnet50.onnx")

# Run inference
outputs = session.run(None, {"input": image_array})
predictions = outputs[0]
```

### TensorRT (NVIDIA GPUs)

```bash
# Convert ONNX to TensorRT
trtexec --onnx=resnet50.onnx \
  --saveEngine=resnet50.trt \
  --fp16  # 2-3× speedup with FP16
```

---

## 📊 Performance Benchmarks

### Transformers

| Model | Parameters | Training Time | Performance |
|-------|------------|---------------|-------------|
| BERT-Base | 110M | 3 days (16 TPUs) | 82.1 GLUE |
| BERT-Large | 340M | 1 week (16 TPUs) | 84.6 GLUE |
| GPT-2 Small | 117M | 1 week (32 TPUs) | 35 PPL |
| GPT-3 | 175B | ~1 month | 10 PPL |

### Computer Vision

| Model | Parameters | Top-1 Acc | Top-5 Acc | Inference (ms) |
|-------|------------|-----------|-----------|----------------|
| ResNet-50 | 25.6M | 76.5% | 93.0% | 4-5 (GPU) |
| EfficientNet-B0 | 5.3M | 77.3% | 93.5% | 3-4 (GPU) |
| EfficientNet-B7 | 66M | 84.4% | 97.1% | 15-20 (GPU) |

---

## 🛠️ Best Practices

### 1. Data Preparation
- Use appropriate tokenization (WordPiece for BERT, BPE for GPT)
- Apply data augmentation for computer vision
- Normalize inputs (ImageNet statistics for vision)
- Handle class imbalance (oversampling, class weights)

### 2. Training
- Start with lower learning rates for fine-tuning (1e-5 to 1e-4)
- Use learning rate schedules (cosine, step decay)
- Apply gradient clipping (1.0 for transformers)
- Monitor train/val metrics for overfitting

### 3. Optimization
- Use mixed precision training (FP16) for 2-3× speedup
- Enable gradient accumulation for larger effective batch sizes
- Implement early stopping to save compute
- Use distributed training for large models

### 4. HPO
- Start with 20-30 trials for initial exploration
- Use 50-100 trials for thorough search
- Apply early stopping/pruning to avoid wasting compute
- Analyze parameter importance after search

### 5. NAS
- Define reasonable search spaces (avoid too large)
- Use evolutionary search for multi-objective
- Use Bayesian optimization for expensive evaluations
- Validate on separate test set

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size
batch_size: 16  # Instead of 64

# Use gradient accumulation
accumulation_steps = 4

# Enable mixed precision
use_mixed_precision: True

# Use gradient checkpointing
gradient_checkpointing: True
```

### Slow Convergence
```neural
# Check learning rate
optimizer: Adam(learning_rate=0.0001)  # Try different values

# Add learning rate warmup
warmup_steps: 2000

# Check batch size
batch_size: 32  # Try larger batches
```

### Poor Validation Accuracy
```neural
# Add regularization
Dropout(rate=0.3)

# Reduce model capacity
Dense(units=256)  # Instead of 1024

# Add data augmentation
# See data augmentation examples above

# Try transfer learning
# Load pre-trained weights
```

---

## 📖 Additional Resources

### Documentation
- [API Reference](../../docs/api/)
- [DSL Syntax Guide](../../docs/dsl.md)
- [CLI Reference](../../docs/cli_reference.md)

### Papers
- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)

### External Libraries
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Optuna](https://optuna.org/)
- [Ray Tune](https://docs.ray.io/en/latest/tune/)

### Community
- GitHub Issues: https://github.com/neuraldsl/neuraldsl/issues
- Discord: https://discord.gg/neuraldsl
- Twitter: @neuraldsl

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](../../CONTRIBUTING.md) for details on:
- Adding new examples
- Improving tutorials
- Reporting bugs
- Suggesting features

---

## 📝 License

All examples and tutorials are licensed under the MIT License. See [LICENSE](../../LICENSE) for details.

---

**Last Updated**: 2024  
**Maintainer**: Neural DSL Team  
**Version**: 1.0.0
