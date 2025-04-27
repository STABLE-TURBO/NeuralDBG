# Neural DSL v0.2.8: Cloud Integration & Automated Issue Management

*April 30, 2025*

We're excited to announce the release of Neural DSL v0.2.8, which brings significant improvements to cloud integration capabilities, interactive shell features for cloud platforms, and enhanced GitHub workflows for automated issue management.

## What's New in v0.2.8

### Cloud Integration Improvements

One of the most significant improvements in v0.2.8 is the enhanced support for running Neural in cloud environments like Kaggle, Google Colab, and AWS SageMaker. You can now seamlessly:

- Run Neural DSL models directly in cloud notebooks
- Connect to cloud platforms from your local terminal
- Visualize models and debug them remotely
- Leverage cloud GPUs for faster training

```bash
# Connect to a cloud platform
neural cloud connect kaggle

# Execute a Neural DSL file on Kaggle
neural cloud execute kaggle my_model.neural

# Run Neural in cloud mode with remote access
neural cloud run --setup-tunnel
```

### Interactive Shell for Cloud Platforms

We've improved the cloud connect command to properly spawn an interactive CLI interface when connecting to cloud platforms. This makes it easier to:

- Manage your models across different cloud environments
- Run commands interactively without reconnecting
- Monitor training progress in real-time
- Debug models running in the cloud

```bash
# Start an interactive shell connected to Kaggle
neural cloud connect kaggle --interactive

# The interactive shell provides a familiar Neural CLI experience
# but executes commands on the cloud platform
```

### Automated Issue Management

We've enhanced our GitHub workflows for automatically creating and closing issues based on test results. This helps us:

- Track and fix bugs more efficiently
- Ensure that fixed issues are properly closed
- Maintain a cleaner issue tracker
- Provide better visibility into the development process

## Getting Started with v0.2.8

You can install Neural DSL v0.2.8 using pip:

```bash
pip install neural-dsl==0.2.8
```

Or upgrade from a previous version:

```bash
pip install --upgrade neural-dsl
```

## Example: Running Neural in Google Colab

Here's a complete example that demonstrates the new cloud features in v0.2.8:

```python
# Install Neural DSL in your Colab notebook
!pip install neural-dsl==0.2.8

# Import the cloud module
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()
print(f"Detected environment: {executor.environment}")
print(f"GPU available: {executor.is_gpu_available}")

# Define a model
dsl_code = """
network MnistCNN {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Dense(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
"""

# Compile and run the model
model_path = executor.compile_model(dsl_code, backend='tensorflow')
results = executor.run_model(model_path, dataset='MNIST')

# Start the NeuralDbg dashboard with ngrok tunnel
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard URL: {dashboard_info['tunnel_url']}")
```

## What's Next?

We're continuously working to improve Neural DSL and make it more powerful and user-friendly. In upcoming releases, we plan to:

- Further enhance the NeuralPaper.ai integration for better model visualization and annotation
- Expand PyTorch support to match TensorFlow capabilities
- Improve documentation with more examples and tutorials
- Add support for more advanced HPO techniques

Stay tuned for more updates, and as always, we welcome your feedback and contributions!

## Get Involved

- GitHub: [https://github.com/Lemniscate-world/Neural](https://github.com/Lemniscate-world/Neural)
- Documentation: [https://github.com/Lemniscate-world/Neural/blob/main/docs/dsl.md](https://github.com/Lemniscate-world/Neural/blob/main/docs/dsl.md)
- Discord: [https://discord.gg/KFku4KvS](https://discord.gg/KFku4KvS)

Happy coding with Neural DSL!
