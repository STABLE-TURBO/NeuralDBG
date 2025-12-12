# Neural Examples

<p align="center">
  <img src="../docs/images/examples_overview.png" alt="Examples Overview" width="600"/>
</p>

## Overview

This directory contains comprehensive examples demonstrating various use cases and model architectures in Neural DSL. Each example includes detailed documentation, tutorial notebooks, and is validated by CI to ensure correctness.

## 🎯 Quick Start

### New to Neural DSL?
Start with our **tutorial notebooks** in [`notebooks/`](notebooks/):
1. [Image Classification Tutorial](notebooks/image_classification_tutorial.ipynb) - Build your first CNN
2. [Sentiment Analysis Tutorial](notebooks/sentiment_analysis_tutorial.ipynb) - Learn sequence modeling
3. [Transformer Tutorial](notebooks/transformer_nlp_tutorial.ipynb) - Explore modern NLP
4. [Time Series Tutorial](notebooks/time_series_tutorial.ipynb) - Forecast temporal data
5. [GAN Tutorial](notebooks/gan_tutorial.ipynb) - Generate new images

### Looking for Production-Ready Examples?
Check out [`use_cases/`](use_cases/) for complete, validated .neural files:
- `image_classification.neural` - CNN for computer vision
- `sentiment_analysis.neural` - LSTM for NLP
- `transformer_nlp.neural` - Transformer encoder
- `time_series.neural` - CNN-LSTM hybrid
- `gan.neural` - Generator and Discriminator

---

## 📁 Directory Structure

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
└── README.md                      # This file
```

---

## 🚀 Example Categories

### 1. Computer Vision

**Image Classification** ([neural](use_cases/image_classification.neural) | [notebook](notebooks/image_classification_tutorial.ipynb))
- Deep CNN with batch normalization
- Multi-class classification (1000 classes)
- Data augmentation ready
- Transfer learning compatible

**Applications:**
- Object recognition
- Medical image analysis
- Quality inspection
- Scene understanding

---

### 2. Natural Language Processing

**Sentiment Analysis** ([neural](use_cases/sentiment_analysis.neural) | [notebook](notebooks/sentiment_analysis_tutorial.ipynb))
- LSTM-based sequence model
- Embedding layer for text
- Dropout for regularization
- 3-class sentiment classification

**Transformer NLP** ([neural](use_cases/transformer_nlp.neural) | [notebook](notebooks/transformer_nlp_tutorial.ipynb))
- Multi-head attention (8 heads)
- 3 transformer encoder blocks
- Positional encoding
- Modern architecture for NLP

**Applications:**
- Review analysis
- Social media monitoring
- Customer feedback
- Document classification
- Question answering

---

### 3. Time Series

**Time Series Forecasting** ([neural](use_cases/time_series.neural) | [notebook](notebooks/time_series_tutorial.ipynb))
- CNN for feature extraction
- Stacked LSTM for temporal modeling
- Single and multi-step prediction
- Residual analysis

**Applications:**
- Stock prediction
- Energy forecasting
- Weather prediction
- Anomaly detection
- Traffic prediction

---

### 4. Generative Models

**GAN** ([neural](use_cases/gan.neural) | [notebook](notebooks/gan_tutorial.ipynb))
- Generator and Discriminator networks
- Adversarial training
- Latent space exploration
- Image generation from noise

**Applications:**
- Data augmentation
- Image synthesis
- Style transfer
- Super resolution
- Creative applications

---

## 💻 Using the Examples

### Option 1: Interactive Notebooks (Recommended for Learning)

```bash
# Install Jupyter
pip install jupyter

# Start notebook server
cd examples/notebooks
jupyter notebook

# Open any .ipynb file and follow along
```

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

# Generate code
tf_code = generate_code(model_data, 'tensorflow')
pytorch_code = generate_code(model_data, 'pytorch')
```

---

## 🔬 Validating Examples

All examples are automatically validated in CI. To validate locally:

```bash
# Run validation script
python examples/use_cases/validate_examples.py

# This checks:
# - Syntax correctness
# - Model structure
# - Compilation to TensorFlow
# - Layer compatibility
```

CI runs on every push to ensure examples stay up-to-date.

---

## 📚 Learning Path

### Beginner (New to Deep Learning)
1. **Image Classification** notebook - Learn CNN basics
2. **Sentiment Analysis** notebook - Introduction to sequences
3. **Time Series** notebook - Apply to temporal data

### Intermediate (Familiar with Neural Networks)
1. **Transformer NLP** notebook - Modern architectures
2. **Time Series** notebook - Hybrid CNN-LSTM
3. **GAN** notebook - Adversarial training

### Advanced (Experienced Practitioners)
1. Study all .neural files in `use_cases/`
2. Modify architectures for your needs
3. Implement custom loss functions
4. Optimize for production deployment

---

## 🛠️ Customization Guide

### Modify an Existing Example

```bash
# 1. Copy an example
cp examples/use_cases/image_classification.neural my_model.neural

# 2. Edit with your preferred editor
nano my_model.neural

# 3. Compile and test
neural compile my_model.neural --backend tensorflow

# 4. Visualize changes
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
  BatchNormalization()  # Add this
  Dropout(rate=0.3)     # And this
```

**Adjust Hyperparameters:**
```
train {
  epochs: 100          # More epochs
  batch_size: 16       # Smaller batches
  validation_split: 0.3  # More validation data
}
```

---

## 🧪 Testing Your Changes

```bash
# Parse only (fast check)
neural compile my_model.neural --dry-run

# Full compilation
neural compile my_model.neural --backend tensorflow

# Test on multiple backends
neural compile my_model.neural --backend tensorflow
neural compile my_model.neural --backend pytorch

# Validate with CI script
python examples/use_cases/validate_examples.py
```

---

## 📊 Datasets

### Built-in via Keras/PyTorch
- MNIST (handwritten digits)
- Fashion-MNIST (clothing items)
- CIFAR-10/100 (natural images)
- IMDB (movie reviews)
- Reuters (news articles)

### External Sources
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml)
- [Papers with Code](https://paperswithcode.com/datasets)
- [Hugging Face Datasets](https://huggingface.co/datasets)

---

## 🎓 Additional Resources

### Documentation
- [Neural DSL Reference](../docs/DSL.md)
- [Layer Reference](../docs/layers.md)
- [CLI Guide](../docs/CLI.md)
- [AGENTS.md](../AGENTS.md) - Development guide

### Tutorials
- [Getting Started](../GETTING_STARTED.md)
- [Quick Start](../QUICK_START_AUTOMATION.md)
- [Distribution Guide](../DISTRIBUTION_QUICK_REF.md)

### Community
- [GitHub Issues](https://github.com/Lemniscate-SHA-256/Neural/issues)
- [Discussions](https://github.com/Lemniscate-SHA-256/Neural/discussions)

---

## 🤝 Contributing Examples

We welcome new examples! To contribute:

1. **Create Your Example**
   - Write a .neural file with clear architecture
   - Add comprehensive comments
   - Test compilation to multiple backends

2. **Add Documentation**
   - Create a tutorial notebook if appropriate
   - Update README files
   - Include usage examples

3. **Validate**
   - Run `validate_examples.py`
   - Ensure CI passes
   - Test on multiple platforms

4. **Submit**
   - Fork the repository
   - Create a feature branch
   - Submit a pull request

**Example Contribution Checklist:**
- [ ] .neural file is syntactically correct
- [ ] Compiles to TensorFlow and PyTorch
- [ ] Includes comments explaining architecture
- [ ] README updated with new example
- [ ] (Optional) Tutorial notebook created
- [ ] Validation script passes
- [ ] CI tests pass

---

## 🐛 Troubleshooting

### Common Issues

**"Parser Error"**
- Check syntax in .neural file
- Ensure all braces are matched
- Verify layer parameters

**"Compilation Failed"**
- Check layer compatibility
- Verify backend is installed
- Review error message details

**"Shape Mismatch"**
- Check input/output shapes
- Ensure layers are compatible
- Use `neural visualize` to debug

### Getting Help

1. Check [documentation](../docs/)
2. Review [existing issues](https://github.com/Lemniscate-SHA-256/Neural/issues)
3. Open a new issue with:
   - Neural DSL code
   - Error message
   - Environment details
   - Steps to reproduce

---

## 📝 License

All examples are part of Neural DSL and are licensed under the MIT License. See [LICENSE.md](../LICENSE.md) for details.

---

## 🙏 Acknowledgments

These examples were inspired by:
- TensorFlow and PyTorch tutorials
- Deep learning research papers
- Community feedback and contributions
- Real-world use cases

---

**Ready to build amazing neural networks? Pick a notebook and start learning! 🚀**

*Have questions? Open an issue or start a discussion on GitHub.*
