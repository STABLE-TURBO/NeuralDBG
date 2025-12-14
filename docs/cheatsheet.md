# Neural DSL Cheat Sheet

Quick reference for common tasks and syntax.

## Installation

```bash
# Minimal
pip install neural-dsl

# Full features
pip install neural-dsl[full]

# Selective
pip install neural-dsl[backends,hpo,dashboard]
```

## Basic Commands

```bash
# Compile
neural compile model.neural

# Visualize
neural visualize model.neural

# Debug
neural debug model.neural --dashboard

# Run
neural run model.neural

# Export
neural export model.neural --format onnx

# Clean
neural clean --yes

# Help
neural --help
neural compile --help
```

## Basic Model Structure

```yaml
network ModelName {
  input: (height, width, channels)
  
  layers:
    LayerType(params)
    LayerType(params)
  
  loss: "loss_function"
  optimizer: OptimizerName(params)
  
  train {
    epochs: 10
    batch_size: 32
  }
}
```

## Common Layers

### Convolutional

```yaml
# 2D Convolution
Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same")

# Pooling
MaxPooling2D(pool_size=(2,2))
AveragePooling2D(pool_size=(2,2))
GlobalAveragePooling2D()

# Upsampling
UpSampling2D(size=(2,2))
Conv2DTranspose(filters=32, kernel_size=(3,3))
```

### Dense

```yaml
# Fully connected
Dense(units=128, activation="relu")

# Output layer
Output(units=10, activation="softmax")  # Classification
Output(units=1, activation="sigmoid")   # Binary classification
Output(units=1, activation="linear")    # Regression
```

### Recurrent

```yaml
# LSTM
LSTM(units=128, return_sequences=false)

# GRU
GRU(units=128, return_sequences=false)

# Bidirectional
Bidirectional(LSTM(128))
```

### Regularization

```yaml
# Dropout
Dropout(rate=0.5)

# Batch normalization
BatchNormalization()

# L1/L2 regularization (in layer params)
Dense(128, kernel_regularizer="l2")
```

### Other

```yaml
# Flatten
Flatten()

# Embedding
Embedding(vocab_size=10000, embedding_dim=128)

# Reshape
Reshape(target_shape=(28, 28, 1))

# Concatenate
Concatenate()

# Add (residual)
Add()
```

## Transformer Layers

```yaml
# Multi-head attention
MultiHeadAttention(num_heads=8, key_dim=64)

# Transformer encoder
TransformerEncoder(num_heads=8, ff_dim=512, num_blocks=3)

# Positional encoding (manual)
PositionalEncoding(max_len=512, embedding_dim=256)
```

## Activation Functions

```yaml
"relu"
"sigmoid"
"tanh"
"softmax"
"elu"
"selu"
"leaky_relu"
"swish"
```

## Loss Functions

```yaml
# Classification
loss: "categorical_crossentropy"
loss: "sparse_categorical_crossentropy"
loss: "binary_crossentropy"

# Regression
loss: "mse"  # Mean Squared Error
loss: "mae"  # Mean Absolute Error
loss: "huber"

# Custom
loss: "custom_loss_function"
```

## Optimizers

```yaml
# Adam (most common)
optimizer: Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# SGD with momentum
optimizer: SGD(learning_rate=0.01, momentum=0.9, nesterov=true)

# RMSprop
optimizer: RMSprop(learning_rate=0.001, rho=0.9)

# AdamW
optimizer: AdamW(learning_rate=0.001, weight_decay=0.01)
```

## Learning Rate Schedules

```yaml
# Exponential decay
optimizer: Adam(
  learning_rate=ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    decay_rate=0.96
  )
)

# Step decay
optimizer: Adam(
  learning_rate=StepDecay(
    initial_learning_rate=0.1,
    drop=0.5,
    epochs_drop=10
  )
)
```

## Hyperparameter Optimization

```yaml
# Define search space
layers:
  Dense(units=HPO(choice(64, 128, 256)), activation="relu")
  Dropout(rate=HPO(range(0.3, 0.7, step=0.1)))

# Optimizer HPO
optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))

# Training HPO
train {
  epochs: 20
  batch_size: HPO(choice(16, 32, 64))
  search_method: "bayesian"  # or "random", "grid"
}
```

### HPO Functions

```yaml
HPO(choice(val1, val2, val3))           # Discrete choices
HPO(range(min, max, step=0.1))          # Continuous range
HPO(log_range(min, max))                # Log-scale range
HPO(int_range(min, max))                # Integer range
```

## Training Configuration

```yaml
train {
  epochs: 10
  batch_size: 32
  validation_split: 0.2
  shuffle: true
  callbacks: ["early_stopping", "model_checkpoint"]
  early_stopping_patience: 5
  reduce_lr_patience: 3
}
```

## Layer Multiplication

```yaml
# Repeat a layer
Conv2D(32, (3,3), "relu")*3  # Repeat 3 times

# Repeat a block
layers:
  Conv2D(32, (3,3), "relu")
  MaxPooling2D((2,2))
  * 4  # Repeat the block 4 times
```

## Macros (Reusable Blocks)

```yaml
# Define macro
macro ConvBlock(filters) {
  Conv2D(filters, (3,3), padding="same", activation="relu")
  BatchNormalization()
  MaxPooling2D((2,2))
}

# Use macro
network Model {
  input: (224, 224, 3)
  layers:
    ConvBlock(32)
    ConvBlock(64)
    ConvBlock(128)
    Flatten()
    Dense(10, "softmax")
}
```

## Multi-Input Models

```yaml
network MultiInputModel {
  inputs:
    image: (224, 224, 3)
    metadata: (10,)
  
  branches:
    image_branch:
      Conv2D(32, (3,3), "relu")
      Flatten()
    
    metadata_branch:
      Dense(64, "relu")
  
  merge: Concatenate()
  
  layers:
    Dense(128, "relu")
    Output(10, "softmax")
}
```

## Command Options

### Compile

```bash
neural compile model.neural \
  --backend tensorflow \
  --output model.py \
  --dry-run \
  --hpo
```

### Visualize

```bash
neural visualize model.neural \
  --format html \
  --attention \
  --data input.npy
```

### Debug

```bash
neural debug model.neural \
  --gradients \
  --dead-neurons \
  --anomalies \
  --dashboard \
  --port 8050
```

### Export

```bash
neural export model.neural \
  --format onnx \
  --optimize \
  --quantize \
  --quantization-type int8
```

### Track

```bash
# Initialize
neural track init experiment_name

# Log data
neural track log -p '{"lr": 0.001}' -m '{"acc": 0.95}'

# List experiments
neural track list

# Show details
neural track show exp_id

# Compare
neural track compare exp1 exp2 exp3

# Plot metrics
neural track plot exp_id --metrics accuracy loss
```

## Common Patterns

### CNN for Images

```yaml
network ImageCNN {
  input: (224, 224, 3)
  layers:
    Conv2D(32, (3,3), "relu")
    MaxPooling2D((2,2))
    Conv2D(64, (3,3), "relu")
    MaxPooling2D((2,2))
    Conv2D(128, (3,3), "relu")
    GlobalAveragePooling2D()
    Dense(256, "relu")
    Dropout(0.5)
    Output(1000, "softmax")
  optimizer: Adam(learning_rate=0.001)
  loss: "categorical_crossentropy"
}
```

### LSTM for Text

```yaml
network TextLSTM {
  input: (100,)
  layers:
    Embedding(vocab_size=10000, embedding_dim=128)
    LSTM(128)
    Dense(64, "relu")
    Dropout(0.5)
    Output(3, "softmax")
  optimizer: Adam(learning_rate=0.001)
  loss: "categorical_crossentropy"
}
```

### Transformer

```yaml
network Transformer {
  input: (512,)
  layers:
    Embedding(vocab_size=10000, embedding_dim=256)
    TransformerEncoder(num_heads=8, ff_dim=512, num_blocks=3)
    GlobalAveragePooling1D()
    Dense(128, "relu")
    Output(10, "softmax")
  optimizer: Adam(learning_rate=0.001)
  loss: "sparse_categorical_crossentropy"
}
```

### ResNet Block

```yaml
macro ResidualBlock(filters) {
  # First path
  Conv2D(filters, (3,3), padding="same", activation="relu")
  BatchNormalization()
  Conv2D(filters, (3,3), padding="same")
  BatchNormalization()
  
  # Skip connection
  Add()
  Activation("relu")
}

network ResNet {
  input: (224, 224, 3)
  layers:
    Conv2D(64, (7,7), strides=(2,2), padding="same")
    MaxPooling2D((3,3), strides=(2,2))
    ResidualBlock(64)*3
    ResidualBlock(128)*4
    ResidualBlock(256)*6
    ResidualBlock(512)*3
    GlobalAveragePooling2D()
    Output(1000, "softmax")
}
```

### Autoencoder

```yaml
network Autoencoder {
  input: (28, 28, 1)
  
  # Encoder
  layers:
    Conv2D(32, (3,3), "relu", padding="same")
    MaxPooling2D((2,2))
    Conv2D(64, (3,3), "relu", padding="same")
    MaxPooling2D((2,2))
  
  # Latent representation
  Flatten()
  Dense(128, "relu")
  
  # Decoder
  Dense(7*7*64, "relu")
  Reshape((7, 7, 64))
  Conv2DTranspose(64, (3,3), "relu", padding="same")
  UpSampling2D((2,2))
  Conv2DTranspose(32, (3,3), "relu", padding="same")
  UpSampling2D((2,2))
  Conv2D(1, (3,3), "sigmoid", padding="same")
  
  optimizer: Adam(learning_rate=0.001)
  loss: "binary_crossentropy"
}
```

## Tips and Tricks

### Shape Debugging

```bash
# Always visualize first
neural visualize model.neural

# Use dry-run to check
neural compile model.neural --dry-run

# Debug mode for detailed info
neural debug model.neural --step
```

### Performance Optimization

```yaml
# Use batch normalization
BatchNormalization()

# Use appropriate pooling
GlobalAveragePooling2D()  # Instead of Flatten() when possible

# Learning rate schedules
optimizer: Adam(learning_rate=ExponentialDecay(...))

# Data augmentation (in generated code)
# Early stopping
train {
  callbacks: ["early_stopping"]
  early_stopping_patience: 5
}
```

### Common Fixes

```yaml
# Shape mismatch? Add Flatten()
Flatten()
Dense(128, "relu")

# Overfitting? Add regularization
Dropout(0.5)
BatchNormalization()
Dense(128, kernel_regularizer="l2")

# Vanishing gradients? Try:
- BatchNormalization()
- Different activation: "relu", "elu", "selu"
- Residual connections (Add())
- Lower learning rate

# Exploding gradients? Try:
- Gradient clipping (in optimizer)
- Batch normalization
- Lower learning rate
- Smaller network
```

## Environment Variables

```bash
# Skip welcome message
export NEURAL_SKIP_WELCOME=1

# Force CPU mode
export NEURAL_FORCE_CPU=1

# Suppress TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=3
```

## Quick Workflows

### Prototype Workflow

```bash
cp examples/mnist.neural my_model.neural
nano my_model.neural
neural compile my_model.neural --dry-run
neural visualize my_model.neural
neural run my_model.neural
```

### Experiment Workflow

```bash
neural track init my_exp
neural compile model.neural --hpo
neural track list
neural track compare exp1 exp2 exp3
```

### Deploy Workflow

```bash
neural compile model.neural --backend tensorflow
neural export model.neural --format onnx --optimize
neural docs model.neural --pdf
neural clean --yes
```

## File Extensions

- `.neural` - Neural DSL model file
- `.nr` - Alternative Neural DSL extension
- `_tensorflow.py` - Generated TensorFlow code
- `_pytorch.py` - Generated PyTorch code
- `.onnx` - ONNX model export

## Common Errors

```bash
# Parser error
-> Check syntax, matching braces, quotes

# Shape mismatch
-> Run: neural visualize model.neural
-> Add Flatten() before Dense layers

# Import error
-> Install backend: pip install neural-dsl[backends]

# Command not found
-> Use: python -m neural.cli compile model.neural

# Graphviz error
-> Install: sudo apt-get install graphviz
```

## Resources

- **Docs**: [docs/](.)
- **Examples**: [examples/](../examples/)
- **CLI**: [cli_reference.md](cli_reference.md)
- **DSL**: [dsl.md](dsl.md)
- **GitHub**: https://github.com/Lemniscate-world/Neural
- **Discord**: https://discord.gg/KFku4KvS

---

**Print this cheat sheet** for quick reference while coding!

Save as: `neural_cheatsheet.pdf` via your browser's print function.
