---
title: "Neural DSL v0.2.8: Cloud Integration & Automated Issue Management"
published: true
description: "Neural DSL v0.2.8 brings enhanced cloud integration capabilities, interactive shell features for cloud platforms, and improved GitHub workflows for automated issue management."
tags: machinelearning, python, neuralnetworks, cloudcomputing
cover_image: https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b
---

# Neural DSL v0.2.8: Cloud Integration & Automated Issue Management

![Neural DSL Logo](https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b)

We're excited to announce the release of Neural DSL v0.2.8, which brings significant improvements to cloud integration capabilities, interactive shell features for cloud platforms, and enhanced GitHub workflows for automated issue management.

## 🚀 Spotlight Feature: Cloud Integration Improvements

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

## 💻 Interactive Shell for Cloud Platforms

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

## 🔄 Automated Issue Management

We've enhanced our GitHub workflows for automatically creating and closing issues based on test results. This helps us:

- Track and fix bugs more efficiently
- Ensure that fixed issues are properly closed
- Maintain a cleaner issue tracker
- Provide better visibility into the development process

## 📝 Example: Running Neural in Google Colab

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

## ✨ Other Improvements

- **Version References**: Updated version references across documentation to ensure consistency
- **CLI Debug Messages**: Further reduced debug logs when starting the Neural CLI for a cleaner user experience
- **Documentation**: Enhanced README with more detailed explanations of cloud integration features
- **Dependency Management**: Refined dependency specifications for better compatibility across environments
- **Release Workflow**: Streamlined the release process with better automation for version updates

## 🐛 Bug Fixes

- Fixed issues with the cloud connect command to properly spawn an interactive CLI interface
- Updated version references across documentation to ensure consistency
- Further reduced debug logs when starting the Neural CLI for a cleaner user experience
- Refined dependency specifications for better compatibility across environments

## 📦 Installation

```bash
pip install neural-dsl==0.2.8
```

Or upgrade from a previous version:

```bash
pip install --upgrade neural-dsl
```

## 🔗 Links

- [GitHub Repository](https://github.com/Lemniscate-world/Neural)
- [Documentation](https://github.com/Lemniscate-world/Neural/blob/main/docs/dsl.md)
- [Discord Community](https://discord.gg/KFku4KvS)

## 🙏 Support Us

If you find Neural DSL useful, please consider giving us a star on GitHub ⭐ and sharing this project with your friends and colleagues. The more developers we reach, the more likely we are to build something truly revolutionary together!
