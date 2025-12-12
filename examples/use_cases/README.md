# Neural DSL Use Cases

This directory contains end-to-end examples demonstrating common use cases for Neural DSL.

## Available Examples

### 1. Image Classification (`image_classification.neural`)

A comprehensive convolutional neural network for image classification tasks.

**Architecture:**
- Multiple Conv2D blocks with batch normalization
- MaxPooling and dropout for regularization
- Dense layers for classification
- Supports 1000-class classification (ImageNet-style)

**Key Features:**
- Input: (224, 224, 3) RGB images
- 3 convolutional blocks with increasing filters (32→64→128)
- Batch normalization after each conv layer
- Dropout for preventing overfitting
- Suitable for transfer learning

**Usage:**
```bash
# Compile
neural compile image_classification.neural --backend tensorflow

# Visualize
neural visualize image_classification.neural --format html

# Train with HPO
neural compile image_classification.neural --backend tensorflow --hpo --dataset CIFAR10
```

**Applications:**
- Object recognition
- Scene classification
- Medical image analysis
- Quality inspection

---

### 2. Sentiment Analysis (`sentiment_analysis.neural`)

LSTM-based model for text sentiment classification.

**Architecture:**
- Embedding layer for text representation
- Stacked LSTM layers with dropout
- Dense layers for classification
- 3-class output (negative, neutral, positive)

**Key Features:**
- Input: (200,) sequence length
- 20k vocabulary size
- Bidirectional information flow through stacked LSTMs
- Dropout and recurrent dropout for regularization

**Usage:**
```bash
neural compile sentiment_analysis.neural --backend tensorflow
neural visualize sentiment_analysis.neural --format html
```

**Applications:**
- Movie review analysis
- Social media sentiment tracking
- Customer feedback analysis
- Product review classification

---

### 3. Transformer NLP (`transformer_nlp.neural`)

Transformer encoder for advanced NLP tasks.

**Architecture:**
- Multi-head attention mechanism (8 heads)
- 3 transformer encoder blocks
- Feed-forward networks (512 dims)
- Global average pooling for sequence aggregation

**Key Features:**
- Input: (512,) sequence length
- 30k vocabulary with 256-dim embeddings
- Parallel processing advantages over RNNs
- Better long-range dependency modeling

**Usage:**
```bash
neural compile transformer_nlp.neural --backend tensorflow
neural debug transformer_nlp.neural --dashboard --port 8050
```

**Applications:**
- Text classification
- Named entity recognition
- Question answering
- Document classification
- Language understanding tasks

---

### 4. Time Series Prediction (`time_series.neural`)

CNN-LSTM hybrid for time series forecasting.

**Architecture:**
- 1D convolutions for feature extraction
- Stacked LSTM layers for temporal modeling
- Dense layers for prediction
- Linear activation for regression

**Key Features:**
- Input: (100, 1) - 100 timesteps
- Conv1D layers extract local patterns
- LSTM layers capture temporal dependencies
- Suitable for univariate time series

**Usage:**
```bash
neural compile time_series.neural --backend tensorflow
neural run time_series_tensorflow.py --backend tensorflow
```

**Applications:**
- Stock price prediction
- Energy consumption forecasting
- Weather prediction
- Sales forecasting
- Traffic prediction
- Anomaly detection

---

### 5. Generative Adversarial Network (`gan.neural`)

Complete GAN implementation with Generator and Discriminator.

**Generator Architecture:**
- Dense layers with increasing capacity
- Batch normalization for stability
- Tanh activation for output
- Reshape to image dimensions

**Discriminator Architecture:**
- Flattening and dense layers
- Dropout for regularization
- Sigmoid output for binary classification

**Key Features:**
- Generator input: (100,) latent vector
- Generates 28×28 images
- Adversarial training setup
- Binary cross-entropy loss

**Usage:**
```bash
neural compile gan.neural --backend tensorflow
# Note: Training requires custom adversarial loop
```

**Applications:**
- Image generation
- Data augmentation
- Style transfer
- Image-to-image translation
- Super resolution

---

## Validation

All examples are automatically validated in CI. To validate locally:

```bash
python validate_examples.py
```

This will:
- Parse each .neural file
- Verify syntax correctness
- Check model structure
- Attempt compilation to TensorFlow

## Notebooks

Complete tutorial notebooks for each use case are available in `examples/notebooks/`:

- `image_classification_tutorial.ipynb`
- `sentiment_analysis_tutorial.ipynb`
- `transformer_nlp_tutorial.ipynb`
- `time_series_tutorial.ipynb`
- `gan_tutorial.ipynb`

Each notebook includes:
- Detailed explanations
- Code examples
- Visualization
- Training procedures
- Evaluation metrics
- Real-world applications

## Common Patterns

### Compilation
```bash
neural compile <file>.neural --backend <tensorflow|pytorch|onnx>
```

### Visualization
```bash
neural visualize <file>.neural --format <html|png|svg>
```

### Debugging
```bash
neural debug <file>.neural --backend tensorflow --dashboard
```

### Hyperparameter Optimization
```bash
neural compile <file>.neural --backend tensorflow --hpo --dataset <DATASET>
```

## Best Practices

1. **Start Simple**: Begin with smaller models and scale up
2. **Use Batch Normalization**: Improves training stability
3. **Add Dropout**: Prevents overfitting
4. **Visualize First**: Use `neural visualize` before training
5. **Debug Early**: Use NeuralDbg dashboard to catch issues
6. **Optimize Last**: Use HPO after model architecture is finalized

## Extending Examples

To create your own example:

1. Copy an existing .neural file
2. Modify the architecture
3. Update training configuration
4. Validate with `validate_examples.py`
5. Test compilation to multiple backends

## Support

For issues or questions:
- Check the [Neural DSL documentation](../../docs/)
- Review [AGENTS.md](../../AGENTS.md) for development guidelines
- Open an issue on GitHub

## Contributing

We welcome contributions! To add a new example:

1. Create a `.neural` file with clear architecture
2. Add comprehensive comments
3. Update this README
4. Create a corresponding notebook in `examples/notebooks/`
5. Ensure validation passes
6. Submit a pull request

---

**Note**: These examples are designed to be educational and may need adaptation for production use. Consider factors like:
- Dataset-specific preprocessing
- Appropriate batch sizes for your hardware
- Custom loss functions
- Learning rate schedules
- Model checkpointing
- Early stopping
