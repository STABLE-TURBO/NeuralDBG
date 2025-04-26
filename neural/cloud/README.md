# Neural DSL Cloud Integration

This module provides tools and utilities for running Neural DSL in cloud environments like Kaggle, Google Colab, and AWS SageMaker.

## Features

- **Cloud Environment Detection**: Automatically detect Kaggle, Colab, SageMaker, and other cloud environments
- **GPU Detection**: Check for GPU availability and configure Neural DSL accordingly
- **Remote Dashboard Access**: Access NeuralDbg and No-Code interfaces through ngrok tunnels
- **Simplified API**: Run Neural DSL commands with a simple Python API
- **Example Notebooks**: Ready-to-use notebooks for Kaggle and Colab
- **Interactive Shell**: Command-line interface for executing Neural DSL commands on cloud platforms
- **Jupyter-like Notebook Interface**: Web-based notebook interface for executing Neural DSL code on cloud platforms
- **Remote Execution**: Execute Neural DSL files on cloud platforms from a local terminal

## Installation

### Option 1: Direct Installation

```python
# In your Kaggle/Colab notebook
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git
```

### Option 2: Installation Script

```python
# In your Kaggle/Colab notebook
!curl -s https://raw.githubusercontent.com/Lemniscate-SHA-256/Neural/main/neural/cloud/install_neural.py | python
```

## Quick Start

### Option 1: Using the Python API

```python
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

# Compile the model
model_path = executor.compile_model(dsl_code, backend='tensorflow')

# Run the model
results = executor.run_model(model_path, dataset='MNIST', epochs=5)

# Start the NeuralDbg dashboard with ngrok tunnel
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard URL: {dashboard_info['tunnel_url']}")
```

### Option 2: Using the Interactive Shell

```bash
# Connect to Kaggle with an interactive shell
neural cloud connect kaggle --interactive

# In the shell, you can run commands like:
neural-cloud> run my_model.neural --backend tensorflow
neural-cloud> visualize my_model.neural
neural-cloud> debug my_model.neural --setup-tunnel
neural-cloud> shell ls -la
neural-cloud> python print("Hello from Kaggle!")
```

### Option 3: Using the Notebook Interface

```bash
# Connect to Kaggle with a notebook interface
neural cloud connect kaggle --notebook --port 8888

# This will open a Jupyter-like notebook interface in your browser
# where you can execute Neural DSL code on Kaggle
```

### Option 4: Remote Execution from Terminal

```bash
# Execute a Neural DSL file on Kaggle
neural cloud execute kaggle my_model.neural

# Run Neural in cloud mode with remote access
neural cloud run --setup-tunnel
```

## Example Notebooks

- [Neural DSL on Kaggle](examples/neural_kaggle_example.ipynb)
- [Neural DSL on Google Colab](examples/neural_colab_example.ipynb)

## Components

### `install_neural.py`

A script for installing Neural DSL in cloud environments.

```python
!python -c "$(curl -s https://raw.githubusercontent.com/Lemniscate-SHA-256/Neural/main/neural/cloud/install_neural.py)"
```

### `cloud_execution.py`

The main module for executing Neural DSL in cloud environments.

```python
from neural.cloud.cloud_execution import CloudExecutor

executor = CloudExecutor()
```

### `remote_connection.py`

Module for connecting to cloud platforms from a local terminal.

```python
from neural.cloud.remote_connection import RemoteConnection

remote = RemoteConnection()
remote.connect_to_kaggle()
```

### `interactive_shell.py`

Interactive shell for executing Neural DSL commands on cloud platforms.

```python
from neural.cloud.interactive_shell import start_interactive_shell

start_interactive_shell('kaggle')
```

### `notebook_interface.py`

Jupyter-like notebook interface for executing Neural DSL code on cloud platforms.

```python
from neural.cloud.notebook_interface import start_notebook_interface

start_notebook_interface('kaggle', port=8888)
```

### `sagemaker_integration.py`

AWS SageMaker integration for Neural DSL.

```python
from neural.cloud.sagemaker_integration import SageMakerHandler

handler = SageMakerHandler()
```

## Dashboard Access

When running in cloud environments, the dashboards (NeuralDbg and No-Code interface) are not directly accessible through localhost. The `CloudExecutor` class sets up ngrok tunnels to make these dashboards accessible:

```python
# Start the NeuralDbg dashboard
dashboard_info = executor.start_debug_dashboard(dsl_code, setup_tunnel=True)
print(f"Dashboard URL: {dashboard_info['tunnel_url']}")

# Start the No-Code interface
nocode_info = executor.start_nocode_interface(setup_tunnel=True)
print(f"No-Code Interface URL: {nocode_info['tunnel_url']}")
```

## GPU Support

The `CloudExecutor` automatically detects GPU availability and configures Neural DSL accordingly:

```python
executor = CloudExecutor()
print(f"GPU available: {executor.is_gpu_available}")
```

## Cleanup

When you're done, you can clean up temporary files and processes:

```python
executor.cleanup()
```

## Requirements

- Neural DSL
- pyngrok (for tunneling)
- IPython (for notebook integration)
