# Getting Started with Neural DSL

Welcome to Neural DSL. This guide will help you get started, and it's okay if some parts feel confusing at first—building neural networks takes time to learn, and we'll walk through it together.

## What is Neural DSL, Really?

Neural DSL is a tool that lets you define neural network architectures using a simplified language instead of writing lots of framework-specific code. Think of it as a blueprint language for neural networks that can then generate working code for TensorFlow, PyTorch, or ONNX.

If you're coming from pure Python ML frameworks, this might feel like an extra abstraction at first. The benefit is that you can switch between frameworks or experiment with architectures without rewriting everything from scratch.

## Installation (This Should Take About 5 Minutes)

```bash
pip install neural-dsl
```

**Common issue**: If you get a `pip not found` error, you might need to use `pip3` instead of `pip`, or you may need to install Python first.

**Another common issue**: If you're on Windows and see permission errors, try running your terminal as administrator, or add `--user` to the install command: `pip install --user neural-dsl`

## Your First Model: A Step-by-Step Walkthrough

Let's build a simple image classifier. We'll use the MNIST dataset as an example because it's a common starting point, though you might find the dimensions confusing at first—that's normal.

### Option 1: Writing DSL Code Directly

Create a file called `my_model.neural` with the following content:

```
network MyModel {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

**Let's break this down** (because the syntax might not be obvious):

- `input: (28, 28, 1)` - This means images that are 28x28 pixels with 1 color channel (grayscale). The dimensions go (height, width, channels).
- `Conv2D(32, (3, 3), "relu")` - A convolutional layer with 32 filters, each 3x3 pixels, using ReLU activation.
- `MaxPooling2D((2, 2))` - Reduces the spatial dimensions by taking the maximum value in each 2x2 region.
- `Flatten()` - Converts the 2D feature maps into a 1D vector so we can feed it to regular dense layers.
- `Dense(128, "relu")` - A fully connected layer with 128 neurons.
- `Output(10, "softmax")` - The final layer with 10 outputs (one for each digit 0-9), using softmax for probability distribution.

**This might be confusing at first**: The order of layers matters. If you try to use `Dense` before `Flatten`, you'll get shape errors. This is because dense layers expect 1D input, but convolutional layers output 2D feature maps.

Now compile your model to actual Python code:

```bash
neural compile my_model.neural --backend tensorflow --output my_model.py
```

This generates a Python file that you can run. You can change `tensorflow` to `pytorch` or `onnx` if you prefer those frameworks.

**Common mistake**: Forgetting to specify the `--output` filename. Without it, the output goes to stdout (your terminal), which is useful for checking but not for saving.

### Option 2: Using the AI Assistant (Experimental)

If writing DSL syntax feels like learning yet another language, you can try the AI assistant:

```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for MNIST classification")
print(result['dsl_code'])

# Save the generated DSL to a file
with open("my_model.neural", "w") as f:
    f.write(result['dsl_code'])
```

**Heads up**: The `use_llm=False` parameter means it uses rule-based generation, not an actual large language model. It's faster and doesn't require API keys, but it's also more limited in what it understands. If you want more flexibility, you can set `use_llm=True`, but you'll need to configure API credentials.

## Running Your Model

After compiling, you have a Python file with your model. Running it is straightforward:

```bash
python my_model.py
```

**Wait, why doesn't it do anything?** The generated code defines the model architecture, but it doesn't automatically train or evaluate it. You'll need to add training code yourself, or use the model in your own scripts. This is intentional—Neural DSL focuses on the architecture definition, leaving training details to you.

If this feels incomplete, you're not wrong. Neural DSL is about defining architectures quickly, not providing a complete ML pipeline. You'll still need to write or integrate training loops, data loading, etc.

## Common Tasks (The Things You'll Actually Want to Do)

### Visualizing Your Model Architecture

Sometimes you want to see what your model looks like before running it:

```bash
neural visualize my_model.neural --format png
```

This creates a diagram showing how layers connect. It's helpful when you're debugging why shapes don't match up.

**Troubleshooting**: If this command fails with a Graphviz error, you need to install Graphviz separately. On Ubuntu/Debian: `sudo apt-get install graphviz`. On Mac: `brew install graphviz`. On Windows, download the installer from graphviz.org and add it to your PATH.

### Debugging Shape Mismatches

Shape errors are probably the most common frustration in neural networks. Neural DSL has a debugging tool:

```bash
neural debug my_model.neural
```

This opens a dashboard that shows how tensor shapes change through each layer. If you get shape errors during compilation, this tool shows you exactly where dimensions don't line up.

### Generating Documentation

You can auto-generate documentation for your model:

```bash
neural docs my_model.neural --output model.md
```

This creates a markdown file describing your architecture. Useful when you need to share model details with others or just remind yourself what you built three months ago.

## Honest Talk: When This Gets Confusing

### Shape Propagation Issues

Neural DSL tries to validate that your layer dimensions work together. Sometimes you'll see errors like "Expected 2D input but got 3D". This usually means:

1. You forgot to `Flatten()` before a dense layer
2. You're trying to use a convolutional layer after flattening (convolutions need 2D inputs)
3. The input dimensions don't match what the first layer expects

The `neural debug` command is your friend here. It shows the exact shape at each layer.

### Backend Differences

The same DSL can compile to TensorFlow, PyTorch, or ONNX, but there are subtle differences in how each framework works. If code works in one backend but not another, it's not necessarily your fault—frameworks have different conventions and features.

**From experience**: Stick with one backend while learning. TensorFlow is generally the most tested in Neural DSL.

### The Learning Curve

If you're new to both neural networks AND this DSL, you're learning two things at once. That's hard. Consider learning basic neural networks in plain TensorFlow or PyTorch first, then coming back to Neural DSL once you understand the concepts. The DSL will make more sense when you know what you're abstracting over.

## Troubleshooting Based on Real Questions

**"I installed it but the `neural` command isn't found"**

This usually means the Python scripts directory isn't in your PATH. Try `python -m neural.cli` instead of just `neural`. If that works, the issue is your PATH. On Linux/Mac, add `~/.local/bin` to PATH. On Windows, check your Python Scripts directory.

**"My model compiles but has terrible accuracy"**

Neural DSL defines architecture, not training quality. If your model trains but performs poorly, the issue is likely hyperparameters, data preprocessing, or architecture choices—not the DSL itself. You'd have the same problem writing the model directly in TensorFlow.

**"Can I use my own custom layers?"**

Not directly in the DSL syntax. You can compile to Python and then modify the generated code, but at that point you're back to writing framework code. Neural DSL works best with standard layer types.

**"Why would I use this instead of just writing TensorFlow?"**

Valid question. Benefits: faster prototyping, backend-agnostic models, automatic documentation and visualization. Downsides: less control, learning curve, another abstraction layer. If you're doing very custom research, plain framework code might be better. If you're building standard architectures or need to switch backends, Neural DSL helps.

## Advanced Features (For Later)

Once you're comfortable with basics, there's more:

- **AI-Powered Development**: Use natural language to generate models (see [AI Integration Guide](docs/ai_integration_guide.md))
- **Hyperparameter Optimization**: Automated tuning with Optuna (see [HPO docs](neural/hpo/README.md))
- **AutoML and NAS**: Automated architecture search (see [AutoML docs](neural/automl/README.md))
- **NeuralDbg Dashboard**: Real-time training visualization and debugging
- **Cloud Integrations**: Deploy to various ML platforms

Don't feel pressure to learn these immediately. Get comfortable with basic model definition first.

## Learning Resources

**Start here:**
- [DSL Documentation](docs/dsl.md) - Complete syntax reference
- [Examples directory](examples/) - Working model examples
  - `mnist.neural` - Simple image classifier
  - `sentiment.neural` - Text classification
  - `transformer.neural` - Attention-based model (more advanced)

**Official guides:**
- [AI Integration Guide](docs/ai_integration_guide.md) - Natural language model generation
- [Automation Guide](AUTOMATION_GUIDE.md) - Automated workflows
- [Contributing Guide](CONTRIBUTING.md) - If you want to improve Neural DSL itself

## Getting Help

**If you're stuck:**
- Check the [examples/](examples/) directory to see if there's a similar model
- Search GitHub issues—someone else might have had the same problem
- Open a new GitHub issue if you found a bug
- Start a GitHub discussion for general questions

**Set realistic expectations**: This is an open-source project. Response times vary, and some features might be incomplete or experimental. That's okay—part of learning is working through imperfect tools.

## What's Next?

1. Try modifying one of the example models in the `examples/` directory
2. Compile it to different backends and see how the generated code differs
3. Experiment with adding or removing layers
4. When something breaks (and it will), use `neural debug` to understand why

Building neural networks takes practice. Start simple, experiment, and don't get discouraged when things don't work the first time. That's normal.
