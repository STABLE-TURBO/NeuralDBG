# Neural DSL Examples Guide

Comprehensive guide to learning Neural DSL through examples.

## Learning Paths

### 🎓 Beginner Path (New to Deep Learning)

**Goal**: Understand basic neural network concepts

1. **MNIST Classifier** ([mnist_commented.neural](../examples/mnist_commented.neural))
   - Start here: fully annotated CNN
   - Learn: convolution, pooling, activation functions
   - Time: 30 minutes
   - Prerequisites: None

2. **Image Classification Tutorial** ([notebook](../examples/notebooks/image_classification_tutorial.ipynb))
   - Interactive step-by-step guide
   - Build a CNN from scratch
   - Understand: data preprocessing, training loops
   - Time: 1 hour

3. **Simple Sentiment Analysis** ([sentiment.neural](../examples/sentiment.neural))
   - Introduction to sequence modeling
   - Learn: embeddings, LSTM layers
   - Time: 45 minutes

**Next Steps**: Move to Intermediate path or explore more vision examples

---

### 🔧 Intermediate Path (Familiar with Neural Networks)

**Goal**: Master advanced architectures and optimization

1. **Sentiment Analysis with Comments** ([sentiment_analysis_commented.neural](../examples/sentiment_analysis_commented.neural))
   - Deep dive into NLP
   - Learn: bidirectional LSTM, dropout strategies
   - Time: 1 hour

2. **Transformer Tutorial** ([notebook](../examples/notebooks/transformer_nlp_tutorial.ipynb))
   - Modern attention-based architecture
   - Learn: multi-head attention, positional encoding
   - Time: 2 hours

3. **HPO Example** ([mnist_hpo.neural](../examples/mnist_hpo.neural))
   - Automated hyperparameter tuning
   - Learn: search spaces, optimization strategies
   - Time: 1 hour

**Next Steps**: Advanced path or specialize in a domain

---

### 🚀 Advanced Path (Experienced Practitioners)

**Goal**: Production-ready implementations and custom architectures

1. **ResNet with Macros** ([resnet_block_commented.neural](../examples/resnet_block_commented.neural))
   - Reusable architecture blocks
   - Learn: residual connections, batch normalization
   - Time: 2 hours

2. **GAN Tutorial** ([notebook](../examples/notebooks/gan_tutorial.ipynb))
   - Adversarial training
   - Learn: generator/discriminator design, training stability
   - Time: 3 hours

3. **Custom Deployment** ([deployment guide](deployment.md))
   - Export and optimize for production
   - Learn: ONNX, quantization, serving
   - Time: 2 hours

**Next Steps**: Contribute custom layers or architectures

---

## Examples by Domain

### Computer Vision 👁️

#### Basic Classification

**MNIST Digit Classification**
- File: [mnist.neural](../examples/mnist.neural)
- Task: 10-class digit recognition
- Architecture: Simple CNN
- Accuracy: ~99%
- Training time: 5 minutes (CPU)

```bash
neural compile examples/mnist.neural --backend tensorflow
neural visualize examples/mnist.neural
neural debug examples/mnist.neural --dashboard
```

**Fashion-MNIST**
- File: [use_cases/image_classification.neural](../examples/use_cases/image_classification.neural)
- Task: 10-class clothing classification
- Architecture: Deep CNN with batch normalization
- Accuracy: ~92%
- Training time: 15 minutes (GPU)

#### Advanced Vision

**ImageNet Classification**
- File: [use_cases/image_classification.neural](../examples/use_cases/image_classification.neural)
- Task: 1000-class object recognition
- Architecture: ResNet-inspired
- Dataset: ImageNet (requires download)
- Training time: Several hours (multi-GPU)

**Transfer Learning** (Custom implementation)
- Start with pretrained model
- Fine-tune for your dataset
- See: [deployment.md](deployment.md)

---

### Natural Language Processing 📝

#### Text Classification

**Sentiment Analysis (Basic)**
- File: [sentiment.neural](../examples/sentiment.neural)
- Task: 3-class sentiment (positive/neutral/negative)
- Architecture: LSTM-based
- Dataset: IMDB or custom
- Training time: 10 minutes (GPU)

```bash
neural compile examples/sentiment.neural --backend pytorch
neural track init sentiment_exp
neural run examples/sentiment.neural --backend pytorch
```

**Sentiment Analysis (Advanced)**
- File: [sentiment_analysis_commented.neural](../examples/sentiment_analysis_commented.neural)
- Includes: Bidirectional LSTM, dropout, regularization
- Full comments explaining each decision

#### Sequence-to-Sequence

**Transformer Architecture**
- File: [transformer.neural](../examples/transformer.neural)
- Task: General sequence modeling
- Architecture: Multi-head attention, 3 encoder blocks
- Use cases: Translation, summarization, Q&A

```bash
# Visualize attention patterns
neural visualize examples/transformer.neural --attention --format html

# Debug training
neural debug examples/transformer.neural --gradients --dashboard
```

**Encoder-Decoder**
- File: [encoder_decoder_transformer.neural](../examples/encoder_decoder_transformer.neural)
- Full seq2seq with attention
- Good for: Machine translation, text generation

---

### Time Series ⏰

**Forecasting**
- File: [use_cases/time_series.neural](../examples/use_cases/time_series.neural)
- Architecture: CNN + stacked LSTM
- Features: Multi-step prediction, residual learning
- Applications: Stock prices, weather, energy demand

```bash
neural compile examples/use_cases/time_series.neural
neural visualize examples/use_cases/time_series.neural --format png
```

**Practical Usage:**
- Customize input window size
- Adjust prediction horizon
- Add seasonal components

---

### Generative Models 🎨

**GANs**
- File: [use_cases/gan.neural](../examples/use_cases/gan.neural)
- Notebook: [gan_tutorial.ipynb](../examples/notebooks/gan_tutorial.ipynb)
- Learn: Adversarial training, latent space exploration
- Training tips: Balance generator/discriminator, use batch normalization

```bash
# Interactive tutorial
jupyter notebook examples/notebooks/gan_tutorial.ipynb

# Command line
neural compile examples/use_cases/gan.neural --backend tensorflow
```

---

## Features Demonstrated

### Hyperparameter Optimization

**Example**: [mnist_hpo.neural](../examples/mnist_hpo.neural)

Shows how to:
- Define search spaces with `HPO()`
- Use different distributions: `choice()`, `range()`, `log_range()`
- Configure optimization: `search_method: "bayesian"`
- Log and compare results

```yaml
Dense(units=HPO(choice(64, 128, 256)), activation="relu")
Dropout(rate=HPO(range(0.3, 0.7, step=0.1)))
optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
```

Run with:
```bash
neural compile examples/mnist_hpo.neural --hpo --backend tensorflow
neural track compare exp_1 exp_2 exp_3
```

---

### Layer Multiplication and Macros

**Example**: [resnet_block_commented.neural](../examples/resnet_block_commented.neural)

Demonstrates:
- Reusable architecture blocks
- Layer repetition syntax: `Conv2D(32, (3,3))*3`
- Macro definitions
- Residual connections

```yaml
macro ResidualBlock(filters) {
  Conv2D(filters, (3,3), padding="same", activation="relu")
  BatchNormalization()
  Conv2D(filters, (3,3), padding="same")
  Add()  # Residual connection
  Activation("relu")
}

# Use the macro
ResidualBlock(64)*4  # Repeat 4 times
```

---

### Multi-Input/Multi-Output

**Example**: Custom implementation (see advanced docs)

```yaml
network MultiModal {
  inputs:
    image: (224, 224, 3)
    text: (100,)
  
  branches:
    image_branch:
      Conv2D(64, (3,3), "relu")
      GlobalAveragePooling2D()
    
    text_branch:
      Embedding(10000, 128)
      LSTM(128)
  
  merge: Concatenate()
  
  layers:
    Dense(256, "relu")
    Output(10, "softmax")
}
```

---

## Usage Patterns

### Quick Prototyping

```bash
# 1. Start with example
cp examples/mnist.neural my_model.neural

# 2. Edit for your needs
nano my_model.neural

# 3. Quick validation
neural compile my_model.neural --dry-run

# 4. Visualize
neural visualize my_model.neural

# 5. Run
neural run my_model.neural
```

**Time**: 5-10 minutes from idea to working model

---

### Experiment Workflow

```bash
# Set up tracking
neural track init my_experiment

# Run multiple configurations
neural compile model_v1.neural --hpo
neural compile model_v2.neural --hpo
neural compile model_v3.neural --hpo

# Compare results
neural track list
neural track compare exp_v1 exp_v2 exp_v3 --output-dir results/

# Plot metrics
neural track plot exp_v1 --metrics accuracy loss
```

---

### Production Pipeline

```bash
# 1. Develop
neural compile model.neural --backend tensorflow

# 2. Optimize with HPO
neural compile model.neural --hpo --backend tensorflow

# 3. Validate
neural debug model.neural --gradients --dead-neurons --dashboard

# 4. Export optimized model
neural export model.neural --format onnx --optimize --quantize

# 5. Document
neural docs model.neural --pdf

# 6. Deploy (see deployment.md)
```

---

## Interactive Tutorials (Notebooks)

All notebooks are in `examples/notebooks/`:

1. **image_classification_tutorial.ipynb**
   - Build a CNN from scratch
   - Data loading and preprocessing
   - Training and evaluation
   - Visualization techniques

2. **sentiment_analysis_tutorial.ipynb**
   - Text preprocessing
   - Word embeddings
   - LSTM networks
   - Model interpretation

3. **transformer_nlp_tutorial.ipynb**
   - Attention mechanisms
   - Positional encoding
   - Multi-head attention
   - Training transformers

4. **time_series_tutorial.ipynb**
   - Temporal data handling
   - Sequence prediction
   - CNN-LSTM hybrids
   - Forecasting evaluation

5. **gan_tutorial.ipynb**
   - Adversarial training
   - Generator design
   - Discriminator architecture
   - Training stability

**Running Notebooks:**
```bash
pip install jupyter
cd examples/notebooks
jupyter notebook
```

---

## Modifying Examples

### Common Modifications

**Change Input Shape:**
```yaml
# Before
input: (28, 28, 1)

# After (for CIFAR-10)
input: (32, 32, 3)
```

**Add Regularization:**
```yaml
layers:
  Dense(128, "relu")
  Dropout(0.5)          # Add this
  BatchNormalization()  # And this
  Dense(64, "relu")
```

**Adjust Hyperparameters:**
```yaml
train {
  epochs: 50              # More training
  batch_size: 32          # Smaller batches
  validation_split: 0.2   # Hold out 20%
}

optimizer: Adam(
  learning_rate=0.0001,   # Lower learning rate
  beta_1=0.9,             # Momentum
  beta_2=0.999            # Second moment
)
```

**Switch Optimizer:**
```yaml
# Before
optimizer: Adam(learning_rate=0.001)

# After
optimizer: SGD(
  learning_rate=0.01,
  momentum=0.9,
  nesterov=true
)
```

---

## Validation and Testing

**Validate Syntax:**
```bash
neural compile model.neural --dry-run
```

**Check Shapes:**
```bash
neural visualize model.neural
```

**Test Compilation:**
```bash
# Try all backends
neural compile model.neural --backend tensorflow
neural compile model.neural --backend pytorch
neural compile model.neural --backend onnx
```

**Run Tests:**
```bash
cd examples/use_cases
python validate_examples.py
```

---

## Dataset Reference

### Built-in Datasets

Available via Keras/PyTorch:

- **MNIST**: 60k handwritten digits, 28x28 grayscale
- **Fashion-MNIST**: 60k clothing items, 28x28 grayscale
- **CIFAR-10**: 60k natural images, 32x32 RGB, 10 classes
- **CIFAR-100**: 60k natural images, 32x32 RGB, 100 classes
- **IMDB**: 50k movie reviews, sentiment classification

### Custom Datasets

**Image Data:**
```yaml
network CustomImageModel {
  input: (height, width, channels)
  # Your architecture
}
```

Load in Python:
```python
# TensorFlow
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/data',
    image_size=(height, width),
    batch_size=32
)

# PyTorch
from torchvision import datasets, transforms
dataset = datasets.ImageFolder(
    'path/to/data',
    transform=transforms.ToTensor()
)
```

**Text Data:**
```yaml
network CustomTextModel {
  input: (sequence_length,)
  layers:
    Embedding(vocab_size=10000, embedding_dim=128)
    # Your architecture
}
```

---

## Tips and Best Practices

### For Beginners

1. **Start simple**: Use MNIST or Fashion-MNIST
2. **Read comments**: Annotated examples explain every choice
3. **Visualize first**: See architecture before training
4. **Use dry-run**: Preview generated code
5. **Check shapes**: Most errors are shape mismatches

### For Intermediate Users

1. **Experiment with HPO**: Automate hyperparameter search
2. **Try multiple backends**: Compare TensorFlow vs PyTorch
3. **Use tracking**: Log all experiments
4. **Leverage macros**: Reuse architecture blocks
5. **Profile models**: Use debug dashboard

### For Advanced Users

1. **Custom architectures**: Modify generated code
2. **Multi-GPU training**: See distributed guide
3. **Production export**: Optimize for deployment
4. **Contribute examples**: Share your architectures
5. **Benchmark**: Compare with hand-written code

---

## Troubleshooting Examples

**Parser Error:**
```bash
neural compile example.neural --dry-run
# Check: matching braces, comma placement, quotes
```

**Shape Mismatch:**
```bash
neural visualize example.neural
# Look at shape propagation diagram
# Common fix: add Flatten() before Dense layers
```

**Import Error:**
```bash
pip install neural-dsl[backends]  # Install TensorFlow/PyTorch
pip install neural-dsl[visualization]  # For plotting
```

**Slow Training:**
```bash
# Use GPU
neural run model.neural --device gpu

# Reduce batch size (if OOM)
# Edit train block: batch_size: 16

# Use mixed precision (TensorFlow)
# Add to generated code
```

---

## Contributing Examples

Want to add an example? See [CONTRIBUTING.md](../CONTRIBUTING.md)

**Checklist:**
- [ ] .neural file with clear comments
- [ ] Compiles to TensorFlow and PyTorch
- [ ] README entry
- [ ] (Optional) Tutorial notebook
- [ ] Passes validation script
- [ ] CI tests pass

---

## Further Resources

### Documentation
- [DSL Syntax](dsl.md)
- [CLI Reference](cli_reference.md)
- [Getting Started](../GETTING_STARTED.md)
- [Deployment](deployment.md)

### Community
- [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
- [Discord Server](https://discord.gg/KFku4KvS)
- [Issue Tracker](https://github.com/Lemniscate-world/Neural/issues)

### External Resources
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Papers with Code](https://paperswithcode.com/)

---

**Ready to start?** Pick an example from the [beginner path](#-beginner-path-new-to-deep-learning) and dive in!
