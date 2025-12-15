# Neural DSL Examples Index

Complete catalog of production-ready examples demonstrating Neural DSL capabilities across transformers, computer vision, HPO, and AutoML.

## 🎯 Quick Navigation

- **Beginner**: Start with `mnist.neural` and Tutorial 1 (BERT)
- **Intermediate**: Explore `gpt_decoder.neural` and HPO examples
- **Advanced**: Try AutoML/NAS and multi-objective optimization
- **Production**: See deployment examples and backend comparisons

---

## 📑 Complete Examples List

### 🤖 Transformer Architectures

#### BERT (Encoder-Only)
- **File**: `bert_encoder.neural`
- **Tutorial**: `tutorials/01_transformer_bert_complete.md`
- **Level**: Intermediate
- **Description**: Complete BERT implementation with 12 layers, 768 hidden size, 12 attention heads
- **Use Cases**: MLM, sentence classification, NER, QA
- **Backends**: TensorFlow, PyTorch, ONNX
- **Parameters**: 110M
- **Training Time**: ~3 days on 16 TPUs
- **Performance**: 82.1 GLUE score

**Compile**:
```bash
neural compile examples/bert_encoder.neural --backend pytorch
```

---

#### GPT (Decoder-Only)
- **File**: `gpt_decoder.neural`
- **Tutorial**: `tutorials/02_transformer_gpt_complete.md`
- **Level**: Intermediate
- **Description**: GPT-2 Small with causal attention, autoregressive generation
- **Use Cases**: Text generation, code generation, dialogue
- **Backends**: TensorFlow, PyTorch, ONNX
- **Parameters**: 117M
- **Training Time**: ~1 week on 32 TPUs
- **Performance**: ~35 perplexity

**Compile**:
```bash
neural compile examples/gpt_decoder.neural --backend pytorch
```

**Generate Text**:
```python
from gpt_pt import GptDecoder
generated = generate_text(model, tokenizer, "Once upon a time", max_length=100)
```

---

#### Seq2Seq Transformer
- **File**: `seq2seq_transformer.neural`
- **Tutorial**: `tutorials/03_transformer_seq2seq_complete.md`
- **Level**: Intermediate
- **Description**: Complete encoder-decoder for translation
- **Use Cases**: Machine translation, summarization, speech recognition
- **Backends**: TensorFlow, PyTorch, ONNX
- **Parameters**: ~65M (6 enc + 6 dec layers)
- **Performance**: 28 BLEU (EN-DE), 41 BLEU (EN-FR)

**Compile**:
```bash
neural compile examples/seq2seq_transformer.neural --backend pytorch
```

---

#### Multihead Attention
- **File**: `multihead_attention.neural`
- **Level**: Beginner
- **Description**: Standalone multi-head attention implementation
- **Use Cases**: Understanding attention mechanism
- **Lines**: ~50

---

#### Encoder-Decoder Transformer
- **File**: `encoder_decoder_transformer.neural`
- **Level**: Intermediate
- **Description**: Full transformer with encoder and decoder stacks
- **Use Cases**: Seq2seq tasks

---

### 👁️ Computer Vision

#### ResNet-50
- **File**: `computer_vision/resnet50_production.neural`
- **Tutorial**: `tutorials/04_computer_vision_resnet.md`
- **Level**: Intermediate
- **Description**: Production-ready ResNet-50 with bottleneck blocks
- **Use Cases**: Image classification, feature extraction, transfer learning
- **Backends**: TensorFlow, PyTorch, ONNX
- **Parameters**: 25.6M
- **Training Time**: ~2-3 days on 8 V100 GPUs
- **Performance**: 76.5% top-1 (ImageNet)

**Compile**:
```bash
neural compile examples/computer_vision/resnet50_production.neural \
  --backend pytorch --output resnet50.py
```

**Transfer Learning**:
```python
model = ResNet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)  # Replace final layer
```

---

#### EfficientNet-B0
- **File**: `computer_vision/efficientnet_b0.neural`
- **Tutorial**: `tutorials/05_computer_vision_efficientnet.md`
- **Level**: Advanced
- **Description**: Efficient CNN with compound scaling
- **Use Cases**: Mobile/edge deployment, resource-constrained environments
- **Backends**: TensorFlow, PyTorch, ONNX, TFLite
- **Parameters**: 5.3M
- **Training Time**: ~2 days on 8 V100 GPUs
- **Performance**: 77.3% top-1 (ImageNet)

**Compile**:
```bash
neural compile examples/computer_vision/efficientnet_b0.neural \
  --backend tensorflow --export-tflite
```

**Mobile Deployment**:
- 5× fewer parameters than ResNet-50
- Better accuracy (77.3% vs 76.5%)
- TFLite quantization supported

---

#### ResNet Block (Commented)
- **File**: `resnet_block_commented.neural`
- **Level**: Beginner
- **Description**: Detailed residual block with extensive comments
- **Use Cases**: Learning residual connections

---

### 🔧 Hyperparameter Optimization (HPO)

#### Transformer HPO Complete
- **File**: `hpo_examples/transformer_hpo_complete.neural`
- **Tutorial**: `tutorials/06_hpo_comprehensive.md`
- **Level**: Intermediate
- **Description**: Complete HPO for transformers with Optuna
- **Features**:
  - Architecture search (layers, heads, dimensions)
  - Training hyperparameters (LR, batch size, dropout)
  - Multi-objective optimization
  - Early stopping and pruning
  - Distributed HPO with Ray Tune

**Run HPO**:
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

**Python API**:
```python
from neural.hpo import optimize_and_return

best_params = optimize_and_return(
    config="examples/hpo_examples/transformer_hpo_complete.neural",
    n_trials=50,
    dataset_name='IMDB',
    backend='pytorch',
    sampler='tpe',
    pruner='median'
)
```

---

#### Advanced HPO (v0.2.7)
- **File**: `advanced_hpo_v0.2.7.neural`
- **Level**: Advanced
- **Description**: Advanced HPO features (v0.2.7+)
- **Features**:
  - Bayesian optimization
  - Parameter importance
  - Visualization

---

#### MNIST HPO
- **File**: `mnist_hpo.neural`
- **Level**: Beginner
- **Description**: Simple CNN with HPO on MNIST
- **Use Cases**: Learning HPO basics

**Run**:
```bash
neural hpo examples/mnist_hpo.neural --n-trials 20 --dataset MNIST
```

---

#### HPO Python Examples
- **File**: `hpo_advanced_example.py`
- **Level**: Intermediate
- **Description**: Complete Python examples for HPO
- **Features**:
  - Bayesian optimization with TPE
  - Multi-objective optimization
  - Parameter importance analysis
  - Visualization suite
  - Sampler comparison

**Run**:
```bash
python examples/hpo_advanced_example.py
```

---

### 🧬 AutoML and Neural Architecture Search (NAS)

#### NAS Comprehensive
- **File**: `automl_examples/nas_comprehensive.py`
- **Tutorial**: `tutorials/07_nas_comprehensive.md`
- **Level**: Advanced
- **Description**: Complete NAS implementations
- **Features**:
  - CNN architecture search
  - Transformer architecture search
  - Multi-objective NAS
  - Search strategy comparison
  - Cross-backend NAS

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

# Cross-backend NAS
python examples/automl_examples/nas_comprehensive.py --example cross
```

**Python API**:
```python
from neural.automl import AutoMLEngine, ArchitectureSpace

engine = AutoMLEngine(
    search_strategy='evolutionary',
    early_stopping='median',
    backend='pytorch'
)

results = engine.search(
    architecture_space=architecture_space,
    train_data=train_loader,
    val_data=val_loader,
    max_trials=50,
    max_epochs_per_trial=10
)
```

---

#### AutoML Example
- **File**: `automl_example.py`
- **Level**: Beginner
- **Description**: Basic AutoML demonstration
- **Use Cases**: Getting started with AutoML

---

### 📊 Basic Examples

#### MNIST
- **File**: `mnist.neural`
- **Level**: Beginner
- **Description**: Simple CNN for MNIST digit classification
- **Layers**: Conv2D → Pool → Conv2D → Pool → Dense → Output
- **Parameters**: ~1M
- **Training Time**: ~5 minutes on CPU
- **Performance**: ~99% accuracy

**Compile and Train**:
```bash
neural compile examples/mnist.neural --backend pytorch
python mnist_pt.py --epochs 10
```

---

#### MNIST (Commented)
- **File**: `mnist_commented.neural`
- **Level**: Beginner
- **Description**: MNIST with extensive explanatory comments
- **Use Cases**: Learning Neural DSL syntax

---

#### Tiny Example
- **File**: `tiny.neural`
- **Level**: Beginner
- **Description**: Minimal example (5 lines)
- **Use Cases**: Quick syntax demonstration

---

#### GPU Example
- **File**: `gpu.neural`
- **Level**: Beginner
- **Description**: GPU configuration example
- **Use Cases**: Learning device management

---

### 🎨 Advanced Features

#### Layer Multiplication
- **File**: `layer_multiplication.neural`
- **Level**: Intermediate
- **Description**: Macro-based layer repetition
- **Use Cases**: Building deep networks efficiently

---

#### Nested Layers
- **File**: `nested_layers.neural`
- **Level**: Intermediate
- **Description**: Nested macro definitions
- **Use Cases**: Complex architectural patterns

---

#### Macros
- **File**: `macros.neural`
- **Level**: Intermediate
- **Description**: Reusable macro components
- **Use Cases**: Code organization and reuse

---

#### Sentiment Analysis
- **File**: `sentiment.neural` / `sentiment_analysis_commented.neural`
- **Level**: Intermediate
- **Description**: LSTM-based sentiment classifier
- **Use Cases**: Text classification, NLP basics

---

### 🚀 Advanced Use Cases

#### Positional Encoding
- **File**: `positional_encoding_example.ndsl`
- **Level**: Advanced
- **Description**: Transformer positional encoding implementation
- **Use Cases**: Understanding positional information in transformers

---

#### Export Demo
- **File**: `export_demo.neural`
- **Level**: Intermediate
- **Description**: Model export to various formats
- **Formats**: ONNX, TFLite, TorchScript

---

#### Deployment Example
- **File**: `deployment_example.py`
- **Level**: Advanced
- **Description**: Production deployment patterns
- **Topics**: TF Serving, TorchServe, ONNX Runtime

---

#### Edge Deployment
- **File**: `edge_deployment_example.py`
- **Level**: Advanced
- **Description**: Mobile and edge device deployment
- **Topics**: TFLite, quantization, model optimization

---

### 🧪 Testing and Validation

#### Test HPO Basic
- **File**: `test_hpo_basic.py`
- **Level**: Beginner
- **Description**: Unit tests for HPO functionality

---

#### Validate HPO Examples
- **File**: `validate_hpo_examples.py`
- **Level**: Intermediate
- **Description**: Validation suite for HPO examples

---

### 📈 Profiling and Optimization

#### Profiling Example
- **File**: `profiling_example.py`
- **Level**: Intermediate
- **Description**: Performance profiling tools
- **Topics**: Time, memory, GPU utilization

---

#### GPU Profiling
- **File**: `gpu_profiling_example.py`
- **Level**: Advanced
- **Description**: GPU-specific profiling
- **Topics**: CUDA events, memory profiling

---

#### Comparative Profiling
- **File**: `comparative_profiling_example.py`
- **Level**: Advanced
- **Description**: Compare performance across backends
- **Topics**: TensorFlow vs PyTorch benchmarking

---

#### Distributed Profiling
- **File**: `distributed_profiling_example.py`
- **Level**: Advanced
- **Description**: Multi-GPU/multi-node profiling
- **Topics**: Distributed training analysis

---

### 🔬 Experiment Tracking

#### Experiment Tracking
- **File**: `experiment_tracking_example.py`
- **Level**: Intermediate
- **Description**: Track experiments with MLflow/W&B
- **Topics**: Metrics, artifacts, hyperparameters

---

#### Tracking Example
- **File**: `tracking_example.py`
- **Level**: Intermediate
- **Description**: Basic tracking patterns

---

### 🎯 Specialized Examples

#### Performance Demo
- **File**: `performance_demo.py`
- **Level**: Advanced
- **Description**: Performance optimization techniques

---

#### Benchmarking Demo
- **File**: `benchmarking_demo.py`
- **Level**: Advanced
- **Description**: Comprehensive benchmarking suite

---

#### Shape Propagation Demo
- **File**: `shape_propagation_demo.py`
- **Level**: Intermediate
- **Description**: Shape inference demonstration

---

#### Dashboard Enhanced Usage
- **File**: `dashboard_enhanced_usage.py`
- **Level**: Intermediate
- **Description**: NeuralDbg dashboard features

---

#### AI Examples
- **File**: `ai_examples.py`
- **Level**: Advanced
- **Description**: AI-powered features

---

### 📚 Tutorials

All tutorials are in `tutorials/` directory with detailed explanations:

1. **BERT Complete** (`01_transformer_bert_complete.md`)
2. **GPT Complete** (`02_transformer_gpt_complete.md`)
3. **Seq2Seq Complete** (planned)
4. **ResNet Complete** (planned)
5. **EfficientNet Complete** (planned)
6. **HPO Comprehensive** (planned)
7. **NAS Comprehensive** (planned)

---

## 🎓 Learning Paths

### Path 1: Beginner to Transformers
1. `mnist.neural` - Learn basics
2. `sentiment.neural` - Text classification
3. `bert_encoder.neural` - Full transformer
4. Tutorial 1 - BERT deep dive

### Path 2: Computer Vision
1. `mnist.neural` - Simple CNN
2. `resnet_block_commented.neural` - Residual connections
3. `computer_vision/resnet50_production.neural` - Full ResNet
4. `computer_vision/efficientnet_b0.neural` - Efficient architectures

### Path 3: Hyperparameter Optimization
1. `mnist_hpo.neural` - Basic HPO
2. `hpo_advanced_example.py` - Python API
3. `hpo_examples/transformer_hpo_complete.neural` - Full HPO
4. Tutorial 6 - HPO mastery

### Path 4: AutoML Expert
1. `automl_example.py` - AutoML basics
2. `automl_examples/nas_comprehensive.py` - Full NAS
3. Tutorial 7 - NAS deep dive
4. Multi-objective optimization

---

## 🛠️ Common Commands

### Compilation
```bash
# TensorFlow
neural compile <file>.neural --backend tensorflow

# PyTorch
neural compile <file>.neural --backend pytorch

# ONNX
neural compile <file>.neural --backend onnx

# All backends
neural compile <file>.neural --backend all
```

### Visualization
```bash
# Architecture diagram
neural visualize <file>.neural --output arch.png

# Interactive dashboard
neural debug <file>.neural
```

### HPO
```bash
# Basic
neural hpo <file>.neural --n-trials 50

# Distributed
neural hpo <file>.neural --distributed ray --num-workers 8

# Multi-objective
neural hpo <file>.neural --objectives accuracy:maximize latency:minimize
```

---

## 📊 Performance Comparison

| Model | Parameters | Accuracy | Training Time | Inference |
|-------|------------|----------|---------------|-----------|
| MNIST CNN | 1M | 99% | 5 min (CPU) | <1ms |
| ResNet-50 | 25.6M | 76.5% | 2-3 days (8 GPU) | 4-5ms |
| EfficientNet-B0 | 5.3M | 77.3% | 2 days (8 GPU) | 2-3ms |
| BERT-Base | 110M | 82.1 GLUE | 3 days (16 TPU) | 10-20ms |
| GPT-2 Small | 117M | 35 PPL | 1 week (32 TPU) | 50-100ms |
| GPT-3 | 175B | 10 PPL | 1 month | 2-5s |

---

## 🔗 Additional Resources

- **Documentation**: `/docs/`
- **API Reference**: `/docs/api/`
- **Tutorials**: `/examples/tutorials/`
- **GitHub**: https://github.com/neuraldsl/neuraldsl
- **Discord**: https://discord.gg/neuraldsl

---

## 🤝 Contributing

Add your own examples! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Example template:
```neural
# Description
# Use cases
# Expected performance

network YourModel {
    input: (shape)
    layers:
        # Your architecture
    loss: "loss_function"
    optimizer: Optimizer(params)
    train {
        epochs: 10
        batch_size: 32
    }
}
```

---

**Last Updated**: 2024  
**Total Examples**: 50+  
**Coverage**: Transformers, Computer Vision, HPO, AutoML, Deployment
