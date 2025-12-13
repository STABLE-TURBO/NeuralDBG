# Neural Examples

<p align="center">
  <img src="../docs/images/examples_overview.png" alt="Examples Overview" width="600"/>
</p>

## Overview

This directory contains examples showing different ways to use Neural DSL. We've tried to cover common use cases and model architectures. Each example is validated in CI so they shouldn't break (but if they do, please open an issue).

These examples serve two purposes: documentation for learning the DSL, and starting points for your own models.

## Featured Examples

### Annotated Learning Examples

If you're new to Neural DSL, start here. These examples include extensive comments explaining what's happening and why.

#### Beginner Level
- **[MNIST with Comments](mnist_commented.neural)** - A CNN for digit classification
  - Every layer is explained
  - Shows how shapes transform through the network
  - Includes training configuration
  - Sets realistic expectations about performance

#### Intermediate Level
- **[Sentiment Analysis with Comments](sentiment_analysis_commented.neural)** - LSTM tutorial
  - Text preprocessing explained
  - How LSTM and Bidirectional layers work
  - HPO suggestions included
  - Common mistakes and how to avoid them

#### Advanced Level
- **[ResNet with Macros](resnet_block_commented.neural)** - Reusable architecture patterns
  - Macro definitions for residual blocks
  - How residual connections work
  - Batch normalization best practices
  - Multi-stage architecture design
  - HPO integration for macro parameters
  - Tips for training deep networks

### Quick Reference Examples

These are cleaner versions without all the comments - good for reference once you know what you're doing:

- **[mnist.neural](mnist.neural)** - Basic MNIST classifier
- **[transformer.neural](transformer.neural)** - Transformer architecture
- **[sentiment.neural](sentiment.neural)** - Sentiment analysis
- **[mnist_hpo.neural](mnist_hpo.neural)** - HPO demonstration
- **[gpu.neural](gpu.neural)** - GPU device specification

## Quick Start

### New to Neural DSL?

Start with our tutorial notebooks in [`notebooks/`](notebooks/). They're interactive and easier to follow than reading code:

1. [Image Classification Tutorial](notebooks/image_classification_tutorial.ipynb) - Build your first CNN
2. [Sentiment Analysis Tutorial](notebooks/sentiment_analysis_tutorial.ipynb) - Learn sequence modeling
3. [Transformer Tutorial](notebooks/transformer_nlp_tutorial.ipynb) - Modern NLP
4. [Time Series Tutorial](notebooks/time_series_tutorial.ipynb) - Temporal forecasting
5. [GAN Tutorial](notebooks/gan_tutorial.ipynb) - Generative models

### Looking for Production-Ready Examples?

Check out [`use_cases/`](use_cases/) for complete, validated .neural files:
- `image_classification.neural` - CNN for computer vision
- `sentiment_analysis.neural` - LSTM for NLP
- `transformer_nlp.neural` - Transformer encoder
- `time_series.neural` - CNN-LSTM hybrid
- `gan.neural` - Generator and Discriminator

---

## Directory Structure

```
examples/
├── notebooks/                      # Interactive tutorial notebooks
│   ├── README.md                  # Notebook guide and learning paths
│   ├── image_classification_tutorial.ipynb
│   ├── sentiment_analysis_tutorial.ipynb
│   ├── transformer_nlp_tutorial.ipynb
│   ├── time_series_tutorial.ipynb
│   └── gan_tutorial.ipynb
├── use_cases/                     # Production-ready .neural files
│   ├── README.md                  # Detailed use case documentation
│   ├── image_classification.neural
│   ├── sentiment_analysis.neural
│   ├── transformer_nlp.neural
│   ├── time_series.neural
│   ├── gan.neural
│   └── validate_examples.py       # CI validation script
├── mnist.neural                   # Legacy examples (maintained)
├── sentiment.neural
├── transformer.neural
├── mnist_commented.neural         # Annotated learning examples
├── sentiment_analysis_commented.neural
├── resnet_block_commented.neural
└── README.md                      # This file
```

---

## Example Categories

### 1. Computer Vision

**Image Classification** ([neural](use_cases/image_classification.neural) | [notebook](notebooks/image_classification_tutorial.ipynb))

A deep CNN with batch normalization and dropout. Good for learning the basics of computer vision in Neural DSL.

**Architecture highlights:**
- Multiple convolutional stages
- Batch normalization for training stability
- Dropout for regularization
- Global average pooling

**Use cases:**
- Object recognition
- Medical image analysis
- Quality inspection
- Scene understanding

**Realistic expectations:** On ImageNet-style datasets, expect training to take hours even on a GPU. The example uses a relatively small model, so accuracy will be decent but not state-of-the-art.

---

### 2. Natural Language Processing

**Sentiment Analysis** ([neural](use_cases/sentiment_analysis.neural) | [notebook](notebooks/sentiment_analysis_tutorial.ipynb))

LSTM-based sequence model for text classification. Three sentiment classes: positive, neutral, negative.

**Architecture highlights:**
- Embedding layer for text representation
- LSTM for sequence modeling
- Dropout to prevent overfitting
- Dense output layer

**Transformer NLP** ([neural](use_cases/transformer_nlp.neural) | [notebook](notebooks/transformer_nlp_tutorial.ipynb))

Modern transformer architecture with multi-head attention.

**Architecture highlights:**
- Multi-head attention (8 heads)
- 3 stacked transformer encoder blocks
- Positional encoding
- Layer normalization

**Use cases:**
- Review analysis
- Social media monitoring
- Customer feedback classification
- Document categorization
- Question answering

**Trade-offs:** LSTMs are simpler but transformers generally perform better. Transformers need more data and compute though.

---

### 3. Time Series

**Time Series Forecasting** ([neural](use_cases/time_series.neural) | [notebook](notebooks/time_series_tutorial.ipynb))

Hybrid CNN-LSTM architecture for temporal data.

**Architecture highlights:**
- CNN layers for feature extraction
- Stacked LSTM for temporal modeling
- Supports single-step and multi-step prediction
- Can include external features

**Use cases:**
- Stock prediction (with appropriate skepticism)
- Energy demand forecasting
- Weather prediction
- Anomaly detection
- Traffic prediction

**Important caveat:** Time series forecasting is hard. Don't expect miracles, especially for noisy real-world data. This architecture gives you a reasonable starting point, but you'll likely need domain-specific tuning.

---

### 4. Generative Models

**GAN** ([neural](use_cases/gan.neural) | [notebook](notebooks/gan_tutorial.ipynb))

Generative Adversarial Network with generator and discriminator.

**Architecture highlights:**
- Generator: transforms random noise into images
- Discriminator: distinguishes real from fake
- Adversarial training setup
- Latent space exploration

**Use cases:**
- Data augmentation
- Image synthesis
- Style transfer
- Super resolution
- Creative applications

**Warning:** GANs are notoriously tricky to train. Expect to tune hyperparameters and possibly encounter mode collapse. The example provides a stable starting point, but be prepared for experimentation.

---

## Using the Examples

### Option 1: Interactive Notebooks (Recommended for Learning)

```bash
# Install Jupyter
pip install jupyter

# Start notebook server
cd examples/notebooks
jupyter notebook

# Open any .ipynb file and follow along
```

Notebooks are the best way to learn because you can run cells incrementally and see what's happening.

### Option 2: Command Line (Quick Testing)

```bash
# Compile an example
neural compile examples/use_cases/image_classification.neural --backend tensorflow

# Visualize architecture
neural visualize examples/use_cases/image_classification.neural --format html

# Run with hyperparameter optimization
neural compile examples/use_cases/sentiment_analysis.neural --backend tensorflow --hpo

# Debug with dashboard
neural debug examples/use_cases/transformer_nlp.neural --dashboard --port 8050
```

### Option 3: Programmatic (Integration)

```python
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code

# Load and parse
with open('examples/use_cases/image_classification.neural') as f:
    dsl_code = f.read()

parser = create_parser()
tree = parser.parse(dsl_code)
model_data = ModelTransformer().transform(tree)

# Generate code for different backends
tf_code = generate_code(model_data, 'tensorflow')
pytorch_code = generate_code(model_data, 'pytorch')
```

---

## Validating Examples

All examples are automatically validated in CI. To run validation locally:

```bash
# Run validation script
python examples/use_cases/validate_examples.py

# This checks:
# - Syntax correctness
# - Model structure validity
# - Compilation to TensorFlow
# - Layer compatibility
```

CI runs on every push, so if you see passing checks, the examples work.

---

## Learning Path

### Beginner (New to Deep Learning)

You're new to neural networks in general:

1. Start with the **Image Classification** notebook - CNNs are the easiest to visualize
2. Move to **Sentiment Analysis** notebook - introduces sequences and RNNs
3. Try **Time Series** notebook - applies sequence models to a different domain

Take your time with each one. The goal is understanding, not speed.

### Intermediate (Familiar with Neural Networks)

You know the basics of deep learning:

1. **Transformer NLP** notebook - learn modern architectures
2. **Time Series** notebook - see how to combine CNN and LSTM
3. **GAN** notebook - explore adversarial training

Focus on understanding the architectural choices and trade-offs.

### Advanced (Experienced Practitioners)

You've built neural networks before:

1. Study all .neural files in `use_cases/`
2. Modify architectures for your specific needs
3. Experiment with custom loss functions
4. Optimize for production deployment

At this level, you're probably using these as starting points rather than following them exactly.

---

## Customization Guide

### Modify an Existing Example

```bash
# 1. Copy an example
cp examples/use_cases/image_classification.neural my_model.neural

# 2. Edit with your preferred editor
nano my_model.neural

# 3. Compile and test
neural compile my_model.neural --backend tensorflow

# 4. Visualize to verify changes
neural visualize my_model.neural --format html
```

### Common Modifications

**Change Input Shape:**
```
input: (None, 64, 64, 3)  # Instead of (None, 224, 224, 3)
```

**Add Layers:**
```
layers:
  Conv2D(filters=32, kernel_size=(3,3), activation="relu")
  BatchNormalization()  # Add this for training stability
  Dropout(rate=0.3)     # Add this to prevent overfitting
```

**Adjust Hyperparameters:**
```
train {
  epochs: 100          # More epochs (but watch for overfitting)
  batch_size: 16       # Smaller batches (uses less memory)
  validation_split: 0.3  # More validation data
}
```

---

## Testing Your Changes

```bash
# Quick syntax check (fast)
neural compile my_model.neural --dry-run

# Full compilation
neural compile my_model.neural --backend tensorflow

# Test on multiple backends
neural compile my_model.neural --backend tensorflow
neural compile my_model.neural --backend pytorch

# Run the validation script
python examples/use_cases/validate_examples.py
```

---

## Datasets

### Built-in via Keras/PyTorch

These are easy to use because they're already packaged:
- MNIST (handwritten digits) - classic starting point
- Fashion-MNIST (clothing items) - slightly harder than MNIST
- CIFAR-10/100 (natural images) - more realistic than MNIST
- IMDB (movie reviews) - good for NLP
- Reuters (news articles) - multi-class text classification

### External Sources

For real projects, you'll probably need external data:
- [Kaggle Datasets](https://www.kaggle.com/datasets) - huge variety, active community
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml) - classic datasets
- [Papers with Code](https://paperswithcode.com/datasets) - datasets from recent papers
- [Hugging Face Datasets](https://huggingface.co/datasets) - especially good for NLP

**Tip:** Start with a built-in dataset to verify your architecture works, then move to your real data.

---

## Additional Resources

### Documentation
- [Neural DSL Reference](../docs/DSL.md) - complete language reference
- [Layer Reference](../docs/layers.md) - all available layers and parameters
- [CLI Guide](../docs/CLI.md) - command-line interface
- [AGENTS.md](../AGENTS.md) - development guide

### Tutorials
- [Getting Started](../GETTING_STARTED.md) - installation and first steps
- [Quick Start](../QUICK_START_AUTOMATION.md) - automation features
- [Distribution Guide](../DISTRIBUTION_QUICK_REF.md) - packaging and deployment

### Community
- [GitHub Issues](https://github.com/Lemniscate-SHA-256/Neural/issues) - bug reports and feature requests
- [Discussions](https://github.com/Lemniscate-SHA-256/Neural/discussions) - questions and general discussion

---

## Contributing Examples

We'd love to have more examples, especially for domains we haven't covered.

### Contribution Process

1. **Create Your Example**
   - Write a .neural file with clear, sensible architecture
   - Add comments explaining non-obvious choices
   - Test compilation to both TensorFlow and PyTorch
   - Make sure it actually works (compile and run on sample data)

2. **Add Documentation**
   - Create a tutorial notebook if it's a new domain
   - Update README files
   - Include usage examples and expected results

3. **Validate**
   - Run `validate_examples.py` locally
   - Ensure CI passes on your PR
   - Test on multiple platforms if possible

4. **Submit**
   - Fork the repository
   - Create a feature branch with a descriptive name
   - Submit a pull request with clear description

### Example Contribution Checklist

- [ ] .neural file is syntactically correct
- [ ] Compiles to both TensorFlow and PyTorch
- [ ] Includes comments explaining the architecture
- [ ] README updated with new example
- [ ] (Optional but appreciated) Tutorial notebook created
- [ ] Validation script passes
- [ ] CI tests pass
- [ ] Tested on at least one dataset

---

## Troubleshooting

### Common Issues

**"Parser Error"**

The DSL syntax is wrong. Check:
- All braces are matched `{}`
- Layer parameters use correct syntax
- Commas between parameters
- Quotes around string values

**"Compilation Failed"**

The model structure is invalid. Check:
- Layer types are compatible
- Backend (TensorFlow/PyTorch) is installed
- Shape mismatches between layers

**"Shape Mismatch"**

Output of one layer doesn't match input of next:
- Check input/output shapes carefully
- Use `neural visualize` to see the architecture
- Add flatten layer between conv and dense layers if needed

### Getting Help

1. Check the [documentation](../docs/) first
2. Search [existing issues](https://github.com/Lemniscate-SHA-256/Neural/issues)
3. If you found a bug, open a new issue with:
   - Your .neural code
   - Complete error message
   - Environment details (OS, Python version, backend)
   - Steps to reproduce

Please include enough information that we can actually reproduce the problem. "It doesn't work" isn't helpful.

---

## License

All examples are part of Neural DSL and licensed under the MIT License. See [LICENSE.md](../LICENSE.md) for details.

---

## Acknowledgments

These examples were inspired by:
- TensorFlow and PyTorch official tutorials
- Classic deep learning papers
- Community feedback and real-world use cases
- Our own painful debugging experiences

The annotated examples exist because we wished they existed when we were learning.

---

**Ready to start?** Pick a notebook that matches your experience level and dive in. Don't worry about understanding everything immediately - neural networks take time to really grok.

*Questions? Open an issue or start a discussion on GitHub.*
