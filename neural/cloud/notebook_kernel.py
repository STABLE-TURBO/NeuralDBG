"""
Jupyter Kernel for Neural Cloud Integration

This module provides a Jupyter kernel for executing Neural DSL code on cloud platforms.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='neural_cloud_kernel.log'
)
logger = logging.getLogger(__name__)

try:
    from ipykernel.kernelbase import Kernel
except ImportError:
    logger.error("ipykernel not found. Please install it with 'pip install ipykernel'")
    sys.exit(1)

class NeuralCloudKernel(Kernel):
    """Jupyter kernel for Neural Cloud Integration."""

    implementation = 'neural_cloud'
    implementation_version = '0.1'
    language = 'python'
    language_version = '3.8'
    language_info = {
        'name': 'python',
        'mimetype': 'text/x-python',
        'file_extension': '.py',
        'pygments_lexer': 'ipython3',
        'codemirror_mode': {
            'name': 'python',
            'version': 3
        }
    }
    banner = "Neural Cloud Kernel - Execute Neural DSL code on cloud platforms"

    def __init__(self, **kwargs):
        """Initialize the kernel."""
        super().__init__(**kwargs)
        self.platform = kwargs.get('platform', 'kaggle')
        self.remote = None
        self.kernel_id = None
        self.notebook_name = None

        # Initialize the remote connection
        self._initialize_remote_connection()

    def _initialize_remote_connection(self):
        """Initialize the remote connection."""
        logger.info(f"Initializing remote connection to {self.platform}...")

        try:
            from neural.cloud.remote_connection import RemoteConnection
            self.remote = RemoteConnection()

            # Connect to the platform
            if self.platform.lower() == 'kaggle':
                result = self.remote.connect_to_kaggle()
                if result['success']:
                    # Create a kernel
                    import time
                    self.kernel_id = self.remote.create_kaggle_kernel(f"neural-jupyter-kernel-{int(time.time())}")
                    if not self.kernel_id:
                        logger.error("Failed to create Kaggle kernel")
                        return

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
                        return

                    logger.info("Kaggle kernel initialized with Neural DSL")
                else:
                    logger.error(f"Failed to connect to Kaggle: {result.get('error', 'Unknown error')}")

            elif self.platform.lower() == 'sagemaker':
                result = self.remote.connect_to_sagemaker()
                if result['success']:
                    # Create a notebook instance
                    import time
                    self.notebook_name = self.remote.create_sagemaker_notebook(f"neural-jupyter-kernel-{int(time.time())}")
                    if not self.notebook_name:
                        logger.error("Failed to create SageMaker notebook instance")
                        return

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
                        return

                    logger.info("SageMaker notebook initialized with Neural DSL")
                else:
                    logger.error(f"Failed to connect to SageMaker: {result.get('error', 'Unknown error')}")

            elif self.platform.lower() == 'colab':
                logger.error("Colab kernel is not supported yet")

            else:
                logger.error(f"Unsupported platform: {self.platform}")

        except Exception as e:
            logger.error(f"Failed to initialize remote connection: {e}")

    def do_execute(self, code: str, silent: bool, store_history=True,
                  user_expressions=None, allow_stdin=False) -> Dict[str, Any]:
        """
        Execute code on the cloud platform.

        Args:
            code: The code to execute
            silent: Whether to suppress output
            store_history: Whether to store the execution in history
            user_expressions: User expressions to evaluate
            allow_stdin: Whether to allow stdin

        Returns:
            Dictionary with execution results
        """
        logger.info("Executing code...")

        if not self.remote:
            error_msg = "Remote connection not initialized"
            logger.error(error_msg)
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': 'ConnectionError',
                'evalue': error_msg,
                'traceback': [error_msg]
            }

        try:
            # Execute the code on the cloud platform
            if self.platform.lower() == 'kaggle' and self.kernel_id:
                result = self.remote.execute_on_kaggle(self.kernel_id, code)
            elif self.platform.lower() == 'sagemaker' and self.notebook_name:
                result = self.remote.execute_on_sagemaker(self.notebook_name, code)
            else:
                error_msg = f"Execution environment not initialized for {self.platform}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'execution_count': self.execution_count,
                    'ename': 'EnvironmentError',
                    'evalue': error_msg,
                    'traceback': [error_msg]
                }

            # Process the result
            if result['success']:
                if not silent:
                    # Send the output to the client
                    stream_content = {'name': 'stdout', 'text': result['output']}
                    self.send_response(self.iopub_socket, 'stream', stream_content)

                return {
                    'status': 'ok',
                    'execution_count': self.execution_count,
                    'payload': [],
                    'user_expressions': {}
                }
            else:
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"Execution failed: {error_msg}")
                return {
                    'status': 'error',
                    'execution_count': self.execution_count,
                    'ename': 'ExecutionError',
                    'evalue': error_msg,
                    'traceback': [error_msg]
                }

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': [str(e)]
            }

    def do_shutdown(self, restart: bool) -> Dict[str, Any]:
        """
        Shutdown the kernel.

        Args:
            restart: Whether to restart the kernel

        Returns:
            Dictionary with shutdown status
        """
        logger.info(f"Shutting down kernel (restart={restart})...")

        # Clean up resources
        if self.remote:
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

        return {'status': 'ok', 'restart': restart}


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Cloud Kernel')
    parser.add_argument('-f', '--connection-file', help='Connection file')
    parser.add_argument('--platform', default='kaggle', choices=['kaggle', 'colab', 'sagemaker'],
                        help='Cloud platform')
    args = parser.parse_args()

    # Start the kernel
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=NeuralCloudKernel, platform=args.platform)
