"""
Jupyter-like Notebook Interface for Neural Cloud Integration

This module provides a web-based notebook interface for executing Neural DSL code on cloud platforms.
"""

import os
import sys
import json
import time
import tempfile
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralNotebook:
    """Jupyter-like notebook interface for Neural Cloud Integration."""

    def __init__(self, platform: str, remote_connection=None, port: int = 8888, quiet: bool = False):
        """
        Initialize the notebook interface.

        Args:
            platform: The cloud platform ('kaggle', 'colab', 'sagemaker')
            remote_connection: An existing RemoteConnection object
            port: The port to run the notebook server on
            quiet: Whether to suppress output
        """
        self.platform = platform
        self.port = port
        self.temp_dir = Path(tempfile.mkdtemp(prefix="neural_notebook_"))
        self.notebook_path = self.temp_dir / "neural_notebook.ipynb"
        self.kernel_id = None
        self.notebook_name = None
        self.server_process = None
        self.quiet = quiet

        # Initialize the remote connection
        if remote_connection:
            self.remote = remote_connection
        else:
            try:
                from neural.cloud.remote_connection import RemoteConnection
                self.remote = RemoteConnection()
            except ImportError:
                logger.error("Remote connection module not found")
                raise ImportError("Remote connection module not found")

        # Connect to the platform
        self._connect_to_platform()

        # Create a kernel or notebook
        self._create_execution_environment()

        # Create the notebook file
        self._create_notebook_file()

    def _connect_to_platform(self):
        """Connect to the cloud platform."""
        logger.info(f"Connecting to {self.platform}...")

        if self.platform.lower() == 'kaggle':
            result = self.remote.connect_to_kaggle()
        elif self.platform.lower() == 'colab':
            result = self.remote.connect_to_colab()
        elif self.platform.lower() == 'sagemaker':
            result = self.remote.connect_to_sagemaker()
        else:
            logger.error(f"Unsupported platform: {self.platform}")
            raise ValueError(f"Unsupported platform: {self.platform}")

        if not result['success']:
            logger.error(f"Failed to connect to {self.platform}: {result.get('error', 'Unknown error')}")
            raise ConnectionError(f"Failed to connect to {self.platform}: {result.get('error', 'Unknown error')}")

        logger.info(f"Successfully connected to {self.platform}")

    def _create_execution_environment(self):
        """Create a kernel or notebook for execution."""
        logger.info(f"Creating execution environment on {self.platform}...")

        if self.platform.lower() == 'kaggle':
            # Create a kernel
            self.kernel_id = self.remote.create_kaggle_kernel(f"neural-notebook-{int(time.time())}")
            if not self.kernel_id:
                logger.error("Failed to create Kaggle kernel")
                raise RuntimeError("Failed to create Kaggle kernel")

            logger.info(f"Created Kaggle kernel: {self.kernel_id}")

            # Initialize the kernel with Neural DSL
            init_code = """
# Install Neural DSL
!pip install git+https://github.com/Lemniscate-world/Neural.git

# Import the cloud module
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()
print(f"Detected environment: {executor.environment}")
print(f"GPU available: {executor.is_gpu_available}")

# Define helper functions
def run_dsl(dsl_code, backend='tensorflow', dataset='MNIST', epochs=5):
    # Compile the model
    model_path = executor.compile_model(dsl_code, backend=backend)
    print(f"Model compiled to: {model_path}")

    # Run the model
    results = executor.run_model(model_path, dataset=dataset, epochs=epochs)
    print(f"Model execution results: {results}")

    return model_path, results

def visualize_model(dsl_code, output_format='png'):
    # Visualize the model
    viz_path = executor.visualize_model(dsl_code, output_format=output_format)
    print(f"Model visualization saved to: {viz_path}")

    return viz_path

def debug_model(dsl_code, backend='tensorflow', setup_tunnel=True):
    # Start the NeuralDbg dashboard
    dashboard_info = executor.start_debug_dashboard(dsl_code, backend=backend, setup_tunnel=setup_tunnel)
    print(f"Dashboard URL: {dashboard_info['dashboard_url']}")
    if dashboard_info.get('tunnel_url'):
        print(f"Tunnel URL: {dashboard_info['tunnel_url']}")

    return dashboard_info

print("Neural DSL is ready to use!")
"""

            result = self.remote.execute_on_kaggle(self.kernel_id, init_code)
            if not result['success']:
                logger.error(f"Failed to initialize Kaggle kernel: {result.get('error', 'Unknown error')}")
                raise RuntimeError(f"Failed to initialize Kaggle kernel: {result.get('error', 'Unknown error')}")

            logger.info("Kaggle kernel initialized with Neural DSL")

        elif self.platform.lower() == 'sagemaker':
            # Create a notebook instance
            self.notebook_name = self.remote.create_sagemaker_notebook(f"neural-notebook-{int(time.time())}")
            if not self.notebook_name:
                logger.error("Failed to create SageMaker notebook instance")
                raise RuntimeError("Failed to create SageMaker notebook instance")

            logger.info(f"Created SageMaker notebook instance: {self.notebook_name}")

            # Initialize the notebook with Neural DSL
            init_code = """
# Install Neural DSL
!pip install git+https://github.com/Lemniscate-world/Neural.git

# Import the cloud module
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()
print(f"Detected environment: {executor.environment}")
print(f"GPU available: {executor.is_gpu_available}")

# Define helper functions
def run_dsl(dsl_code, backend='tensorflow', dataset='MNIST', epochs=5):
    # Compile the model
    model_path = executor.compile_model(dsl_code, backend=backend)
    print(f"Model compiled to: {model_path}")

    # Run the model
    results = executor.run_model(model_path, dataset=dataset, epochs=epochs)
    print(f"Model execution results: {results}")

    return model_path, results

def visualize_model(dsl_code, output_format='png'):
    # Visualize the model
    viz_path = executor.visualize_model(dsl_code, output_format=output_format)
    print(f"Model visualization saved to: {viz_path}")

    return viz_path

def debug_model(dsl_code, backend='tensorflow', setup_tunnel=True):
    # Start the NeuralDbg dashboard
    dashboard_info = executor.start_debug_dashboard(dsl_code, backend=backend, setup_tunnel=setup_tunnel)
    print(f"Dashboard URL: {dashboard_info['dashboard_url']}")
    if dashboard_info.get('tunnel_url'):
        print(f"Tunnel URL: {dashboard_info['tunnel_url']}")

    return dashboard_info

print("Neural DSL is ready to use!")
"""

            result = self.remote.execute_on_sagemaker(self.notebook_name, init_code)
            if not result['success']:
                logger.error(f"Failed to initialize SageMaker notebook: {result.get('error', 'Unknown error')}")
                raise RuntimeError(f"Failed to initialize SageMaker notebook: {result.get('error', 'Unknown error')}")

            logger.info("SageMaker notebook initialized with Neural DSL")

        elif self.platform.lower() == 'colab':
            logger.error("Colab notebook interface is not supported yet")
            raise NotImplementedError("Colab notebook interface is not supported yet")

    def _create_notebook_file(self):
        """Create the notebook file."""
        logger.info("Creating notebook file...")

        # Create a basic notebook structure
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Neural DSL Cloud Notebook\n",
                        "\n",
                        "This notebook allows you to run Neural DSL code on cloud platforms.\n",
                        "\n",
                        "## Platform: " + self.platform
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Define a simple model\n",
                        "dsl_code = \"\"\"\n",
                        "network MnistCNN {\n",
                        "    input: (28, 28, 1)\n",
                        "    layers:\n",
                        "        Conv2D(32, (3, 3), \"relu\")\n",
                        "        MaxPooling2D((2, 2))\n",
                        "        Flatten()\n",
                        "        Dense(128, \"relu\")\n",
                        "        Dense(10, \"softmax\")\n",
                        "    loss: \"categorical_crossentropy\"\n",
                        "    optimizer: Adam(learning_rate=0.001)\n",
                        "}\n",
                        "\"\"\"\n",
                        "\n",
                        "print(dsl_code)"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Compile and Run the Model\n",
                        "\n",
                        "Use the `run_dsl` function to compile and run the model."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Compile and run the model\n",
                        "model_path, results = run_dsl(dsl_code, backend='tensorflow', dataset='MNIST', epochs=5)"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Visualize the Model\n",
                        "\n",
                        "Use the `visualize_model` function to visualize the model."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Visualize the model\n",
                        "viz_path = visualize_model(dsl_code, output_format='png')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Debug the Model\n",
                        "\n",
                        "Use the `debug_model` function to start the NeuralDbg dashboard."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Debug the model\n",
                        "dashboard_info = debug_model(dsl_code, backend='tensorflow', setup_tunnel=True)"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        # Write the notebook to a file
        with open(self.notebook_path, 'w') as f:
            json.dump(notebook, f, indent=2)

        logger.info(f"Notebook file created: {self.notebook_path}")

    def start_notebook_server(self):
        """Start the notebook server."""
        logger.info(f"Starting notebook server on port {self.port}...")

        try:
            # Check if jupyter is installed
            import subprocess
            try:
                subprocess.run(["jupyter", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info("Installing jupyter...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "jupyter"])

            # Start the notebook server
            cmd = [
                "jupyter", "notebook",
                "--no-browser",
                f"--port={self.port}",
                f"--notebook-dir={self.temp_dir}"
            ]

            self.server_process = subprocess.Popen(cmd)

            # Wait for the server to start
            time.sleep(2)

            # Open the notebook in the browser
            import webbrowser
            webbrowser.open(f"http://localhost:{self.port}/notebooks/neural_notebook.ipynb")

            logger.info(f"Notebook server started on port {self.port}")

            # Set up a proxy for executing cells on the cloud platform
            self._setup_cell_execution_proxy()

            return True
        except Exception as e:
            logger.error(f"Failed to start notebook server: {e}")
            return False

    def _setup_cell_execution_proxy(self):
        """Set up a proxy for executing cells on the cloud platform."""
        logger.info("Setting up cell execution proxy...")

        # Create a custom kernel.json file
        kernel_dir = self.temp_dir / "kernels" / "neural_cloud"
        kernel_dir.mkdir(parents=True, exist_ok=True)

        kernel_json = {
            "argv": [
                sys.executable,
                "-m", "neural.cloud.notebook_kernel",
                "-f", "{connection_file}",
                "--platform", self.platform
            ],
            "display_name": f"Neural Cloud ({self.platform})",
            "language": "python"
        }

        with open(kernel_dir / "kernel.json", 'w') as f:
            json.dump(kernel_json, f, indent=2)

        # Install the kernel
        import subprocess
        subprocess.run([
            "jupyter", "kernelspec", "install", "--user", str(kernel_dir)
        ], check=True)

        logger.info("Cell execution proxy set up")

    def execute_cell(self, cell_code: str) -> Dict[str, Any]:
        """
        Execute a cell on the cloud platform.

        Args:
            cell_code: The cell code to execute

        Returns:
            Dictionary with execution results
        """
        logger.info("Executing cell...")

        if self.platform.lower() == 'kaggle':
            if not self.kernel_id:
                return {'success': False, 'error': "No Kaggle kernel available"}
            return self.remote.execute_on_kaggle(self.kernel_id, cell_code)
        elif self.platform.lower() == 'sagemaker':
            if not self.notebook_name:
                return {'success': False, 'error': "No SageMaker notebook available"}
            return self.remote.execute_on_sagemaker(self.notebook_name, cell_code)
        else:
            return {'success': False, 'error': f"Unsupported platform: {self.platform}"}

    def stop_notebook_server(self):
        """Stop the notebook server."""
        logger.info("Stopping notebook server...")

        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None

        logger.info("Notebook server stopped")

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")

        # Stop the notebook server
        self.stop_notebook_server()

        # Delete the kernel or notebook
        if self.platform.lower() == 'kaggle' and self.kernel_id:
            if self.remote.delete_kaggle_kernel(self.kernel_id):
                logger.info(f"Deleted Kaggle kernel: {self.kernel_id}")
            else:
                logger.warning(f"Failed to delete Kaggle kernel: {self.kernel_id}")

        elif self.platform.lower() == 'sagemaker' and self.notebook_name:
            if self.remote.delete_sagemaker_notebook(self.notebook_name):
                logger.info(f"Deleted SageMaker notebook: {self.notebook_name}")
            else:
                logger.warning(f"Failed to delete SageMaker notebook: {self.notebook_name}")

        # Clean up the remote connection
        self.remote.cleanup()

        # Remove the temporary directory
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory: {e}")

        logger.info("Cleanup complete")


def start_notebook_interface(platform: str, remote_connection=None, port: int = 8888, quiet: bool = False):
    """
    Start a notebook interface for the specified cloud platform.

    Args:
        platform: The cloud platform ('kaggle', 'colab', 'sagemaker')
        remote_connection: An existing RemoteConnection object
        port: The port to run the notebook server on
        quiet: Whether to suppress output
    """
    # Configure logging to be less verbose if quiet mode is enabled
    if quiet:
        logging.basicConfig(level=logging.ERROR)

    notebook = NeuralNotebook(platform, remote_connection, port)

    try:
        if notebook.start_notebook_server():
            if not quiet:
                print(f"Notebook server started on port {port}")
                print(f"Open your browser to http://localhost:{port}/notebooks/neural_notebook.ipynb")
                print("Press Ctrl+C to stop the server")

            # Keep the process running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                if not quiet:
                    print("\nStopping notebook server...")
                notebook.cleanup()
                if not quiet:
                    print("Notebook server stopped")
        elif not quiet:
            print("Failed to start notebook server")
    except Exception as e:
        if not quiet:
            print(f"Error: {e}")
        notebook.cleanup()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Missing platform argument")
        print("Usage: python notebook_interface.py <platform> [<port>]")
        sys.exit(1)

    platform = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888

    start_notebook_interface(platform, port=port)
