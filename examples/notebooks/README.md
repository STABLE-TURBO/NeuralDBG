# Neural DSL Tutorial Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating end-to-end workflows for common deep learning use cases using Neural DSL.

## 📚 Available Notebooks

### 1. Image Classification Tutorial
**File:** `image_classification_tutorial.ipynb`

Learn how to build, train, and deploy a CNN for image classification.

**Topics Covered:**
- Defining CNN architecture in Neural DSL
- Compiling to TensorFlow/PyTorch
- Training on CIFAR-10 dataset
- Visualizing model architecture
- Making predictions
- Hyperparameter optimization
- Model debugging with NeuralDbg
- Multi-backend export

**Prerequisites:**
- Basic Python knowledge
- Familiarity with CNNs
- TensorFlow or PyTorch (optional)

**Estimated Time:** 30-45 minutes

---

### 2. Sentiment Analysis Tutorial
**File:** `sentiment_analysis_tutorial.ipynb`

Build an LSTM-based sentiment classifier for text data.

**Topics Covered:**
- Text preprocessing and tokenization
- Building LSTM networks
- Training on IMDB reviews
- Embedding layers
- Sequence modeling
- Model evaluation
- Testing custom reviews
- Comparing with baselines

**Prerequisites:**
- Basic NLP concepts
- Understanding of RNNs/LSTMs
- TensorFlow with Keras

**Estimated Time:** 30-40 minutes

---

### 3. Transformer NLP Tutorial
**File:** `transformer_nlp_tutorial.ipynb`

Explore transformer architecture for NLP tasks.

**Topics Covered:**
- Multi-head attention mechanism
- Transformer encoder blocks
- Positional encoding
- Comparing with LSTM models
- Attention visualization
- Performance analysis
- Fine-tuning techniques

**Prerequisites:**
- Understanding of attention mechanisms
- Familiarity with transformer concepts
- Intermediate Python

**Estimated Time:** 45-60 minutes

---

### 4. Time Series Forecasting Tutorial
**File:** `time_series_tutorial.ipynb`

Build CNN-LSTM hybrid models for time series prediction.

**Topics Covered:**
- Time series data preparation
- Sequence generation
- CNN for feature extraction
- LSTM for temporal modeling
- Single-step prediction
- Multi-step forecasting
- Residual analysis
- Model evaluation metrics

**Prerequisites:**
- Basic time series concepts
- NumPy and Pandas
- Matplotlib for visualization

**Estimated Time:** 40-50 minutes

---

### 5. GAN Tutorial
**File:** `gan_tutorial.ipynb`

Implement Generative Adversarial Networks from scratch.

**Topics Covered:**
- GAN architecture (Generator + Discriminator)
- Adversarial training
- Generating images from noise
- Latent space interpolation
- Training stability techniques
- Quality evaluation
- Common pitfalls

**Prerequisites:**
- Understanding of neural networks
- Familiarity with GAN concepts
- Intermediate Python

**Estimated Time:** 60-75 minutes

---

## 🚀 Getting Started

### Installation

1. **Install Neural DSL:**
   ```bash
   pip install neural-dsl
   # Or from source:
   pip install -e .
   ```

2. **Install Jupyter:**
   ```bash
   pip install jupyter notebook
   # Or use JupyterLab:
   pip install jupyterlab
   ```

3. **Install Optional Dependencies:**
   ```bash
   pip install tensorflow torch matplotlib pandas scikit-learn
   ```

### Running Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   # Navigate to examples/notebooks/
   ```

2. **Or use JupyterLab:**
   ```bash
   jupyter lab
   ```

3. **In Google Colab:**
   - Upload notebook to Google Drive
   - Open with Google Colab
   - Install Neural DSL in first cell:
     ```python
     !pip install git+https://github.com/Lemniscate-SHA-256/Neural.git
     ```

### Running in Kaggle

1. Upload notebook to Kaggle
2. Add installation cell:
   ```python
   !pip install neural-dsl
   ```
3. Enable GPU if needed (Settings → Accelerator → GPU)

---

## 📖 Learning Path

### Beginner Path
1. Start with **Image Classification** - most accessible
2. Move to **Sentiment Analysis** - introduces sequence modeling
3. Try **Time Series** - applies similar concepts to different domain

### Intermediate Path
1. **Transformer NLP** - modern architecture
2. **Time Series** - hybrid architectures
3. **GAN** - adversarial training

### Advanced
1. Complete all tutorials in order
2. Modify architectures
3. Apply to custom datasets
4. Explore Neural DSL advanced features

---

## 💡 Tips for Success

### For Beginners
- Read markdown cells carefully
- Run cells sequentially
- Don't skip the "Understanding" sections
- Experiment with small changes
- Use the visualization tools

### For Intermediate Users
- Try different architectures
- Experiment with hyperparameters
- Compare multiple approaches
- Use the debugging tools
- Profile model performance

### For Advanced Users
- Combine different architectures
- Implement custom layers
- Optimize for production
- Benchmark against baselines
- Contribute improvements

---

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Solution: Install missing packages
!pip install <package-name>
```

**2. Memory Errors**
```python
# Solution: Reduce batch size or model size
# In .neural file, change:
batch_size: 32  # instead of 128
```

**3. Training Too Slow**
```python
# Solution: Reduce epochs or use GPU
# Enable GPU in Colab/Kaggle settings
```

**4. Visualization Not Working**
```bash
# Solution: Install graphviz
# Ubuntu/Debian:
sudo apt-get install graphviz
# macOS:
brew install graphviz
# Windows: Download from graphviz.org
```

**5. Neural CLI Not Found**
```bash
# Solution: Ensure proper installation
pip install -e .
# Or add to PATH
export PATH=$PATH:~/.local/bin
```

---

## 🔧 Customization

### Using Your Own Data

1. **Image Classification:**
   - Replace CIFAR-10 with your dataset
   - Adjust input shape in .neural file
   - Modify number of output classes

2. **Text Classification:**
   - Use your text corpus
   - Adjust vocabulary size
   - Update sequence length

3. **Time Series:**
   - Load your time series data
   - Adjust sequence length
   - Modify prediction horizon

### Modifying Architectures

```python
# Example: Add more layers
dsl_code = """
network MyModel {
  input: (None, 28, 28, 1)
  layers:
    Conv2D(filters=64, kernel_size=(3,3), activation="relu")
    # Add your layers here
    MaxPooling2D(pool_size=(2,2))
    # ...
}
"""
```

---

## 📊 Example Datasets

### Built-in (via Keras)
- **MNIST**: Handwritten digits (28×28 grayscale)
- **Fashion-MNIST**: Clothing items (28×28 grayscale)
- **CIFAR-10**: Natural images (32×32 RGB, 10 classes)
- **CIFAR-100**: Natural images (32×32 RGB, 100 classes)
- **IMDB**: Movie reviews (sentiment analysis)

### External Datasets
- **ImageNet**: Large-scale image recognition
- **COCO**: Object detection and segmentation
- **SQuAD**: Question answering
- **WikiText**: Language modeling
- **UCR Time Series**: Various time series tasks

---

## 🎯 Next Steps After Tutorials

1. **Explore More Use Cases:**
   - Object detection
   - Image segmentation
   - Named entity recognition
   - Sequence-to-sequence models
   - Reinforcement learning

2. **Optimize Your Models:**
   - Use hyperparameter optimization
   - Try different architectures
   - Implement data augmentation
   - Use learning rate schedules

3. **Deploy Your Models:**
   - Export to ONNX for production
   - Create REST APIs
   - Deploy to cloud platforms
   - Optimize for mobile/edge

4. **Contribute:**
   - Share your notebooks
   - Report issues
   - Suggest improvements
   - Add new examples

---

## 📝 Feedback

We welcome feedback on these tutorials:
- Open an issue for bugs or unclear sections
- Suggest improvements via pull requests
- Share your success stories
- Request new topics

---

## 🔗 Resources

### Documentation
- [Neural DSL Reference](../../docs/DSL.md)
- [CLI Documentation](../../docs/CLI.md)
- [API Reference](../../docs/API.md)

### Community
- [GitHub Repository](https://github.com/Lemniscate-SHA-256/Neural)
- [Issue Tracker](https://github.com/Lemniscate-SHA-256/Neural/issues)
- [Discussions](https://github.com/Lemniscate-SHA-256/Neural/discussions)

### External Resources
- [Deep Learning Book](http://www.deeplearningbook.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Papers with Code](https://paperswithcode.com/)

---

## 📄 License

These notebooks are part of Neural DSL and are licensed under the MIT License. See [LICENSE.md](../../LICENSE.md) for details.

---

## 🙏 Acknowledgments

These tutorials build upon:
- TensorFlow/Keras documentation
- PyTorch tutorials
- Research papers in deep learning
- Community feedback and contributions

---

**Happy Learning! 🎓**

If you find these tutorials helpful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting issues
- 💡 Suggesting improvements
- 📢 Sharing with others
