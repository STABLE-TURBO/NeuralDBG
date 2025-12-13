# Neural Aquarium - Quick Start Guide

Get started with Neural Aquarium IDE in 5 minutes!

## Installation

Make sure Neural DSL is installed with dashboard dependencies:

```bash
pip install -e ".[full]"
```

Or install just the dashboard dependencies:

```bash
pip install dash dash-bootstrap-components plotly
```

## Launch Aquarium

```bash
python -m neural.aquarium.aquarium
```

The IDE will open at `http://localhost:8052`

## Your First Model

### Step 1: Write DSL Code

In the DSL Editor, enter:

```neural
network SimpleClassifier {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
```

### Step 2: Validate

Click **"Parse DSL"** to validate your code. You should see a green success message.

### Step 3: Configure

In the Runner panel:
- Backend: **TensorFlow**
- Dataset: **MNIST**
- Epochs: **5**
- Batch Size: **32**

### Step 4: Compile

Click **"Compile"** button. Watch the console for compilation logs.

### Step 5: Run

Click **"Run"** button. Training will start and you'll see:
- Real-time logs in the console
- Training progress (epochs, loss, accuracy)
- Metrics visualization (when available)

### Step 6: Export (Optional)

After training:
1. Click **"Export Script"**
2. Choose a filename (e.g., `my_mnist_model.py`)
3. Select export location (e.g., `./my_models`)
4. Click **"Export"**

Your trained model script is now saved!

## Tips

- **Load Examples**: Click "Load Example" for pre-built models
- **Stop Training**: Use the "Stop" button to terminate long runs
- **Clear Console**: Click "Clear" to reset the output view
- **Open in IDE**: Click "Open in IDE" to edit scripts externally

## Common Tasks

### Training with Different Datasets

```neural
# For CIFAR10
network CIFAR10Model {
    input: (None, 32, 32, 3)
    # ... layers ...
}
```

Select "CIFAR10" from the dataset dropdown.

### Using PyTorch Backend

1. Write your DSL model
2. Select "PyTorch" from backend dropdown
3. Compile and run as usual

### Custom Training Parameters

Adjust in the Runner panel:
- **Epochs**: Number of training iterations
- **Batch Size**: Samples per gradient update
- **Validation Split**: Fraction of data for validation

### Saving Model Weights

Check the "Save model weights" option in Runner Options.

## Next Steps

- Explore the **Debugger** tab for NeuralDbg integration
- Check the **Visualization** tab for model architecture
- Read the **Documentation** tab for DSL syntax reference
- Try the examples in the repository under `examples/`

## Need Help?

- Check the full README: `neural/aquarium/README.md`
- Review Neural DSL documentation: `docs/dsl.md`
- Open an issue on GitHub for bugs or feature requests

Happy modeling! 🚀
