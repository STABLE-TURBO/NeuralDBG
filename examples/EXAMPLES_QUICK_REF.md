# Examples Quick Reference

Fast reference for working with Neural DSL examples.

## 🚀 Quick Commands

### Compile Examples
```bash
# TensorFlow
neural compile examples/use_cases/<example>.neural --backend tensorflow

# PyTorch
neural compile examples/use_cases/<example>.neural --backend pytorch

# ONNX
neural compile examples/use_cases/<example>.neural --backend onnx

# Dry run (no file output)
neural compile examples/use_cases/<example>.neural --dry-run
```

### Visualize
```bash
# HTML with interactive charts
neural visualize examples/use_cases/<example>.neural --format html

# Static SVG
neural visualize examples/use_cases/<example>.neural --format svg

# PNG image
neural visualize examples/use_cases/<example>.neural --format png
```

### Debug
```bash
# Launch dashboard
neural debug examples/use_cases/<example>.neural --dashboard --port 8050

# Analyze gradients
neural debug examples/use_cases/<example>.neural --gradients

# Check for dead neurons
neural debug examples/use_cases/<example>.neural --dead-neurons
```

### Hyperparameter Optimization
```bash
neural compile examples/use_cases/<example>.neural --backend tensorflow --hpo --dataset CIFAR10
```

---

## 📋 Example Specs

| Example | Input Shape | Output | Layers | Parameters | Use Case |
|---------|-------------|--------|--------|------------|----------|
| image_classification | (224,224,3) | 1000 classes | 19 | ~5M | Object recognition |
| sentiment_analysis | (200,) | 3 classes | 9 | ~2M | Text sentiment |
| transformer_nlp | (512,) | 10 classes | 11 | ~8M | NLP tasks |
| time_series | (100,1) | 1 value | 15 | ~300K | Forecasting |
| gan (generator) | (100,) | (28,28,1) | 8 | ~1M | Image generation |
| gan (discriminator) | (28,28,1) | 1 value | 5 | ~500K | Real/fake classification |

---

## 🎯 Common Tasks

### Start Learning
```bash
# Best first example
jupyter notebook examples/notebooks/image_classification_tutorial.ipynb
```

### Quick Test
```bash
# Parse and check syntax
neural compile examples/use_cases/sentiment_analysis.neural --dry-run

# Compile and inspect output
neural compile examples/use_cases/sentiment_analysis.neural --backend tensorflow
cat sentiment_analysis_tensorflow.py
```

### Production Use
```bash
# Compile to multiple backends
neural compile examples/use_cases/image_classification.neural --backend tensorflow -o model_tf.py
neural compile examples/use_cases/image_classification.neural --backend pytorch -o model_pt.py
neural compile examples/use_cases/image_classification.neural --backend onnx -o model.onnx
```

### Validation
```bash
# Validate all examples
python examples/use_cases/validate_examples.py

# Run CI checks locally
pytest tests/test_examples.py
```

---

## 🔧 Customization Patterns

### Change Dataset Size
```python
# Original
input: (None, 224, 224, 3)

# CIFAR-10 size
input: (None, 32, 32, 3)

# Custom size
input: (None, 128, 128, 3)
```

### Add Regularization
```python
# After any layer, add:
Dropout(rate=0.3)
BatchNormalization()
```

### Adjust Training
```python
train {
  epochs: 50              # More/fewer epochs
  batch_size: 32          # Adjust for memory
  validation_split: 0.2   # Validation size
}
```

### Change Optimizer
```python
# Adam with custom learning rate
optimizer: Adam(learning_rate=0.0001)

# SGD with momentum
optimizer: SGD(learning_rate=0.01, momentum=0.9)

# RMSprop
optimizer: RMSprop(learning_rate=0.001)
```

---

## 📊 Performance Tips

### Memory Optimization
- Reduce `batch_size` in train block
- Use smaller input dimensions
- Reduce number of filters/units
- Use gradient checkpointing (backend-specific)

### Speed Optimization
- Increase `batch_size` if memory allows
- Use GPU: `neural run --device gpu`
- Enable mixed precision (backend-specific)
- Use simpler architectures for prototyping

### Accuracy Improvement
- More epochs with early stopping
- Data augmentation (in training code)
- Ensemble multiple models
- Hyperparameter optimization: `--hpo`

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Parser error | Check syntax, missing braces, commas |
| Shape mismatch | Verify input shape and layer compatibility |
| Out of memory | Reduce batch_size or model size |
| Slow training | Reduce epochs, use smaller dataset |
| Poor accuracy | More data, better architecture, HPO |
| Import error | Install backend: `pip install tensorflow` |

---

## 📝 Cheat Sheet

### File Extensions
- `.neural` - Neural DSL model definition
- `.ipynb` - Jupyter notebook tutorial
- `.py` - Generated Python code
- `.onnx` - ONNX model export

### Key Directories
- `examples/notebooks/` - Interactive tutorials
- `examples/use_cases/` - Production examples
- `.neural_cache/` - Cached visualizations

### Environment Variables
```bash
export NEURAL_SKIP_WELCOME=1      # Skip welcome message
export NEURAL_FORCE_CPU=1         # Force CPU mode
export TF_CPP_MIN_LOG_LEVEL=3    # Suppress TF logs
```

### Common Flags
- `--verbose` - Detailed output
- `--cpu` - Force CPU mode
- `--no-animations` - Disable spinners
- `--output` - Custom output path
- `--dry-run` - Parse only, no output

---

## 🔗 Quick Links

- [Full Documentation](../docs/)
- [Main README](README.md)
- [Use Cases README](use_cases/README.md)
- [Notebooks README](notebooks/README.md)
- [AGENTS.md](../AGENTS.md)
- [GitHub Issues](https://github.com/Lemniscate-SHA-256/Neural/issues)

---

## 💡 Tips

1. **Always visualize first**: `neural visualize` before training
2. **Start small**: Test with fewer epochs initially
3. **Use notebooks**: Better for learning and experimentation
4. **Check validation**: Run `validate_examples.py` after changes
5. **Read errors carefully**: Parser provides line/column numbers
6. **Try multiple backends**: Compare TensorFlow, PyTorch, ONNX
7. **Use HPO wisely**: Good for fine-tuning, not exploration
8. **Profile first**: Use `--debug` to understand bottlenecks

---

**Need help? Check the full documentation or open an issue!**
