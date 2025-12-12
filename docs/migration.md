# Migration Guide

This guide helps you migrate your Neural DSL code between versions and from other frameworks.

## Table of Contents

- [Version Migrations](#version-migrations)
  - [Migrating to v0.3.0](#migrating-to-v030)
  - [Migrating to v0.2.9](#migrating-to-v029)
  - [Migrating to v0.2.7-v0.2.8](#migrating-to-v027-v028)
  - [Migrating to v0.2.5-v0.2.6](#migrating-to-v025-v026)
  - [Migrating from v0.1.x](#migrating-from-v01x)
- [Framework Migrations](#framework-migrations)
  - [From TensorFlow/Keras](#from-tensorflowkeras)
  - [From PyTorch](#from-pytorch)
  - [From ONNX](#from-onnx)
- [Best Practices](#best-practices)

---

## Version Migrations

### Migrating to v0.3.0

**Release Date:** In Development  
**Status:** Preview

#### New Features

1. **🤖 AI-Powered Development**
   - Natural language to DSL conversion
   - Multi-language support

2. **🔄 Automation System**
   - Automated releases and maintenance
   - Blog post generation

#### Breaking Changes

**None** - v0.3.0 is fully backward compatible with v0.2.x

#### New Capabilities

**AI-Powered Model Creation:**

```python
from neural.ai import NaturalLanguageProcessor

nlp = NaturalLanguageProcessor()

# Create model from natural language
dsl_code = nlp.process("Create a CNN for MNIST with 32 filters")

# Or use the CLI
# neural ai "Create a CNN for MNIST with 32 filters"
```

**Automated Workflows:**

```bash
# Automated release (maintainers only)
python scripts/automation/master_automation.py --task release

# Automated blog post generation
python scripts/automation/master_automation.py --task blog
```

#### Migration Steps

1. **Update Neural DSL:**
   ```bash
   pip install --upgrade neural-dsl
   ```

2. **Enable AI features (optional):**
   ```bash
   # Install AI dependencies
   pip install neural-dsl[ai]
   
   # Configure LLM provider (optional)
   export OPENAI_API_KEY=your_key
   # Or use free Ollama: ollama pull llama2
   ```

3. **No DSL changes required** - existing models work as-is

---

### Migrating to v0.2.9

**Release Date:** May 5, 2025

#### New Features

1. **Aquarium IDE Integration**
   - Desktop IDE for visual model design
   - Real-time shape propagation

2. **Enhanced Dashboard UI**
   - Improved dark theme
   - Better visualization

#### Breaking Changes

**None**

#### Improvements

- Better code quality and consistency
- Enhanced error messages
- Improved documentation

#### Migration Steps

1. **Update Neural DSL:**
   ```bash
   pip install neural-dsl==0.2.9
   ```

2. **Try Aquarium IDE (optional):**
   ```bash
   # Download from releases page
   # https://github.com/Lemniscate-world/Aquarium/releases
   ```

3. **Verify existing models:**
   ```bash
   neural compile your_model.neural --dry-run
   ```

---

### Migrating to v0.2.7-v0.2.8

**Release Dates:** April 16-30, 2025

#### New Features

1. **Enhanced HPO Support**
   - Conv2D kernel_size HPO tracking
   - Better ExponentialDecay parameter handling

2. **Cloud Integration Improvements**
   - Kaggle, Colab, AWS SageMaker support
   - Interactive shell for cloud platforms

#### Breaking Changes

**HPO log_range parameter naming:**

```yaml
# ❌ Old (v0.2.6 and earlier)
optimizer: Adam(learning_rate=HPO(log_range(low=1e-4, high=1e-2)))

# ✅ New (v0.2.7+)
optimizer: Adam(learning_rate=HPO(log_range(min=1e-4, max=1e-2)))
```

Or simply use positional arguments:
```yaml
optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
```

#### Migration Steps

1. **Update installation:**
   ```bash
   pip install neural-dsl==0.2.8
   ```

2. **Update HPO parameters:**
   ```bash
   # Use this script to update your DSL files:
   sed -i 's/log_range(low=/log_range(min=/g' *.neural
   sed -i 's/log_range(high=/log_range(max=/g' *.neural
   ```

3. **Test models:**
   ```bash
   neural compile model.neural --backend tensorflow
   ```

4. **Enable cloud features (optional):**
   ```bash
   pip install neural-dsl[cloud]
   ```

---

### Migrating to v0.2.5-v0.2.6

**Release Dates:** March 24 - April 6, 2025

#### New Features

1. **Multi-Framework HPO**
   - HPO works across TensorFlow and PyTorch
   - Unified HPO configuration

2. **Enhanced Dashboard**
   - Dark theme support
   - Better visualization components

#### Breaking Changes

**Optimizer HPO syntax standardization:**

```yaml
# ❌ Old - quotes sometimes required
optimizer: Adam(learning_rate="HPO(log_range(1e-4, 1e-2))")

# ✅ New - no quotes needed
optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
```

#### Migration Steps

1. **Update to v0.2.6:**
   ```bash
   pip install neural-dsl==0.2.6
   ```

2. **Remove quotes from HPO parameters:**
   ```bash
   # Find and update manually or use:
   sed -i 's/"HPO(/HPO(/g' *.neural
   sed -i 's/)"/)/g' *.neural
   ```

3. **Validate HPO configurations:**
   ```bash
   neural hpo model.neural --validate
   ```

---

### Migrating from v0.1.x

**v0.1.x Release Date:** February 21-24, 2025

#### Major Changes

v0.2.x introduces significant improvements over v0.1.x:

1. **Semantic Validation**
2. **Macro System**
3. **Better Error Messages**
4. **HPO Support**
5. **Cloud Integration**

#### Breaking Changes

**1. Validation is stricter:**

```yaml
# ❌ No longer allowed
Dense(units=-5)          # Negative units
Dropout(rate=1.5)        # Rate > 1
Conv2D(filters=0)        # Zero filters

# ✅ Must fix
Dense(units=5)
Dropout(rate=0.5)
Conv2D(filters=32)
```

**2. Layer parameter names standardized:**

```yaml
# ❌ Old v0.1.x (some variations worked)
Dense(n_units=128)
Conv2D(n_filters=32)

# ✅ v0.2.x (strict naming)
Dense(units=128)
Conv2D(filters=32)
```

**3. Training configuration structure:**

```yaml
# ❌ Old v0.1.x
network MyModel {
  epochs: 10
  batch_size: 32
}

# ✅ v0.2.x (nested train block)
network MyModel {
  train {
    epochs: 10
    batch_size: 32
  }
}
```

#### Migration Steps

1. **Backup existing DSL files:**
   ```bash
   cp -r models/ models_backup/
   ```

2. **Update to v0.2.x:**
   ```bash
   pip uninstall neural-dsl
   pip install neural-dsl==0.2.9
   ```

3. **Update DSL files:**

   **a) Fix training config structure:**
   ```bash
   # Manual update required - wrap training params in train {}
   ```

   **b) Fix parameter names:**
   ```bash
   sed -i 's/n_units=/units=/g' *.neural
   sed -i 's/n_filters=/filters=/g' *.neural
   ```

   **c) Validate all parameters:**
   ```bash
   for file in *.neural; do
     echo "Validating $file..."
     neural compile "$file" --dry-run
   done
   ```

4. **Test compilation:**
   ```bash
   neural compile model.neural --backend tensorflow
   python model_tensorflow.py  # Test execution
   ```

---

## Framework Migrations

### From TensorFlow/Keras

Converting existing TensorFlow/Keras code to Neural DSL.

#### Example: Sequential Model

**TensorFlow/Keras:**
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, 
          epochs=15, 
          batch_size=64, 
          validation_split=0.2)
```

**Neural DSL:**
```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  
  layers:
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
    MaxPooling2D(pool_size=(2, 2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  
  train {
    epochs: 15
    batch_size: 64
    validation_split: 0.2
  }
}
```

#### Example: Functional API

**TensorFlow/Keras Functional API:**
```python
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
branch1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
branch2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')(x)
x = tf.keras.layers.Concatenate()([branch1, branch2])
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

**Neural DSL (with macros for branches):**
```yaml
define ConvBranch(filters, kernel_size) {
  Conv2D(filters=$filters, kernel_size=$kernel_size, activation="relu")
}

network BranchingModel {
  input: (28, 28, 1)
  
  layers:
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
    MaxPooling2D(pool_size=(2, 2))
    Parallel {
      ConvBranch(filters=64, kernel_size=(3, 3))
      ConvBranch(filters=64, kernel_size=(5, 5))
    }
    Concatenate()
    Flatten()
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
}
```

#### Example: Custom Training Loop

**TensorFlow:**
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

for epoch in range(15):
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

**Neural DSL (automatic):**
```yaml
network MyModel {
  input: (28, 28, 1)
  layers:
    # Your layers here
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 15
    batch_size: 64
  }
}
```

**Note:** Neural DSL generates the training loop automatically. For custom loops, use generated code as starting point.

### From PyTorch

Converting PyTorch code to Neural DSL.

#### Example: PyTorch Module

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(15):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

**Neural DSL:**
```yaml
network MNISTNet {
  input: (1, 28, 28)  # PyTorch format (channels-first)
  
  layers:
    Conv2D(filters=32, kernel_size=(3, 3))
    Activation("relu")
    MaxPooling2D(pool_size=(2, 2))
    Flatten()
    Dense(units=128)
    Activation("relu")
    Dropout(rate=0.5)
    Output(units=10)
  
  loss: "cross_entropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 15
    batch_size: 64
  }
}
```

**Compile to PyTorch:**
```bash
neural compile mnist.neural --backend pytorch --output mnist_torch.py
```

#### Layer Name Mapping

| PyTorch | Neural DSL |
|---------|------------|
| `nn.Linear` | `Dense` |
| `nn.Conv2d` | `Conv2D` |
| `nn.MaxPool2d` | `MaxPooling2D` |
| `nn.ReLU()` | `Activation("relu")` |
| `nn.Dropout` | `Dropout` |
| `nn.BatchNorm2d` | `BatchNormalization` |
| `nn.LSTM` | `LSTM` |
| `nn.Embedding` | `Embedding` |
| `nn.Flatten()` | `Flatten` |

### From ONNX

Converting ONNX models to Neural DSL for retraining or modification.

#### Steps:

1. **Export ONNX model structure:**
   ```python
   import onnx
   
   model = onnx.load("model.onnx")
   print(onnx.helper.printable_graph(model.graph))
   ```

2. **Manually convert to DSL:**
   - Map ONNX operators to Neural DSL layers
   - Reconstruct network structure
   - Add training configuration

3. **Compile and verify:**
   ```bash
   neural compile converted.neural --backend tensorflow
   ```

**Note:** Automatic ONNX → Neural DSL conversion is planned for future releases.

---

## Best Practices

### Version Management

1. **Pin versions in production:**
   ```bash
   # requirements.txt
   neural-dsl==0.2.9
   tensorflow==2.12.0
   ```

2. **Use virtual environments:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install neural-dsl==0.2.9
   ```

3. **Test before upgrading:**
   ```bash
   # Create test environment
   python -m venv test_env
   source test_env/bin/activate
   pip install neural-dsl==0.3.0
   
   # Run your models
   neural compile model.neural --dry-run
   ```

### Migration Testing

1. **Validate DSL syntax:**
   ```bash
   neural compile model.neural --dry-run
   ```

2. **Compare outputs:**
   ```bash
   # Old version
   neural compile model.neural --backend tensorflow -o old_model.py
   
   # New version
   neural compile model.neural --backend tensorflow -o new_model.py
   
   # Diff the outputs
   diff old_model.py new_model.py
   ```

3. **Test training:**
   ```python
   # Quick convergence test
   # Train for 2 epochs and check loss decreases
   ```

### Documentation

1. **Track DSL version:**
   ```yaml
   # Add comment at top of DSL files
   # Neural DSL version: 0.2.9
   # Created: 2025-05-01
   # Backend: TensorFlow 2.12
   
   network MyModel {
     # ...
   }
   ```

2. **Document changes:**
   ```yaml
   # CHANGELOG.neural
   # v2 (2025-05-15): Updated to Neural DSL 0.2.9
   #   - Changed HPO log_range parameters (low/high → min/max)
   #   - Added cloud integration support
   # v1 (2025-04-01): Initial version with Neural DSL 0.2.6
   ```

### Rollback Plan

1. **Keep old virtual environment:**
   ```bash
   # Don't delete old venv immediately
   mv .venv .venv_v0.2.8_backup
   python -m venv .venv
   ```

2. **Maintain multiple versions:**
   ```bash
   # Use pyenv or conda
   conda create -n neural_v028 python=3.9
   conda activate neural_v028
   pip install neural-dsl==0.2.8
   
   conda create -n neural_v029 python=3.9
   conda activate neural_v029
   pip install neural-dsl==0.2.9
   ```

3. **Git version control:**
   ```bash
   git tag v0.2.8-models
   git commit -m "Upgrade to Neural DSL 0.2.9"
   # Can rollback with: git checkout v0.2.8-models
   ```

---

## Quick Reference

### Version Compatibility Matrix

| Neural DSL | Python | TensorFlow | PyTorch | Key Features |
|------------|--------|------------|---------|--------------|
| 0.3.0 | 3.8+ | 2.10+ | 1.13+ | AI-powered, Automation |
| 0.2.9 | 3.8+ | 2.10+ | 1.13+ | Aquarium IDE, Enhanced UI |
| 0.2.8 | 3.8+ | 2.10+ | 1.13+ | Cloud integration |
| 0.2.7 | 3.8+ | 2.10+ | 1.13+ | Enhanced HPO |
| 0.2.6 | 3.8+ | 2.10+ | 1.13+ | Multi-framework HPO |
| 0.2.5 | 3.8+ | 2.10+ | 1.13+ | HPO support |
| 0.1.x | 3.8+ | 2.10+ | 1.13+ | Initial release |

### Update Commands Quick Reference

```bash
# Check current version
neural --version

# Update to latest
pip install --upgrade neural-dsl

# Update to specific version
pip install neural-dsl==0.2.9

# Install with all features
pip install neural-dsl[full]

# Validate after update
neural compile model.neural --dry-run
```

---

## Getting Help

- **Documentation:** [docs/](../docs/)
- **Examples:** [examples/](../examples/)
- **Discord:** [Join our community](https://discord.gg/KFku4KvS)
- **Issues:** [Report migration issues](https://github.com/Lemniscate-world/Neural/issues)

For version-specific migration help, check the [CHANGELOG.md](../CHANGELOG.md) for detailed release notes.
