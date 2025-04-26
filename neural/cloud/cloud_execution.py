"""
Cloud Execution Module for Neural DSL
This module provides functions to run Neural DSL in cloud environments like Kaggle and Google Colab.
"""

import os
import sys
import subprocess
import importlib
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Try to import Neural modules
try:
    from neural.parser.parser import ModelTransformer, create_parser
    from neural.code_generation.code_generator import generate_code
    from neural.shape_propagation.shape_propagator import ShapePropagator
    from neural.cli.utils import print_info, print_success, print_error, print_warning
    NEURAL_IMPORTED = True
except ImportError:
    NEURAL_IMPORTED = False

class CloudExecutor:
    """Class for executing Neural DSL in cloud environments."""

    def __init__(self, environment: str = None):
        """
        Initialize the cloud executor.

        Args:
            environment: The cloud environment ('kaggle', 'colab', or None for auto-detect)
        """
        self.environment = environment or self._detect_environment()
        self.is_gpu_available = self._check_gpu_availability()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="neural_"))

        # Initialize Neural components if available
        if NEURAL_IMPORTED:
            self.parser = create_parser('network')
            self.transformer = ModelTransformer()
            self.propagator = ShapePropagator(debug=False)

        # Set environment variables
        os.environ['NEURAL_CLOUD_ENV'] = self.environment
        if not self.is_gpu_available:
            os.environ['NEURAL_FORCE_CPU'] = '1'

    def _detect_environment(self) -> str:
        """Detect the cloud environment we're running in."""
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return 'kaggle'
        if 'COLAB_GPU' in os.environ:
            return 'colab'
        if 'SM_MODEL_DIR' in os.environ:
            return 'sagemaker'
        return 'unknown'

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available in the current environment."""
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def compile_model(self,
                     dsl_code: str,
                     backend: str = 'tensorflow',
                     output_file: Optional[str] = None) -> str:
        """
        Compile a Neural DSL model to code.

        Args:
            dsl_code: The Neural DSL code
            backend: The target backend ('tensorflow', 'pytorch', 'jax')
            output_file: Optional output file path

        Returns:
            The path to the generated code file
        """
        if not NEURAL_IMPORTED:
            raise ImportError("Neural DSL is not installed. Run the installation script first.")

        # Parse the DSL code
        tree = self.parser.parse(dsl_code)
        model_data = self.transformer.transform(tree)

        # Generate code
        code = generate_code(model_data, backend)

        # Save to file
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self.temp_dir / f"model_{backend}.py"

        output_path.write_text(code)
        print_success(f"Model compiled to {output_path}")

        return str(output_path)

    def run_model(self,
                 model_file: str,
                 dataset: str = 'MNIST',
                 epochs: int = 5,
                 batch_size: int = 32) -> Dict[str, Any]:
        """
        Run a compiled model.

        Args:
            model_file: Path to the compiled model file
            dataset: Dataset to use ('MNIST', 'CIFAR10', etc.)
            epochs: Number of epochs to train
            batch_size: Batch size for training

        Returns:
            Dictionary with results
        """
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env['NEURAL_DATASET'] = dataset
        env['NEURAL_EPOCHS'] = str(epochs)
        env['NEURAL_BATCH_SIZE'] = str(batch_size)

        # Run the model
        try:
            result = subprocess.run(
                [sys.executable, model_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                check=True
            )
            print_success("Model execution completed successfully")
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.CalledProcessError as e:
            print_error(f"Model execution failed: {e}")
            return {
                'success': False,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'error': str(e)
            }

    def visualize_model(self,
                       dsl_code: str,
                       output_format: str = 'png',
                       output_file: Optional[str] = None) -> str:
        """
        Visualize a Neural DSL model.

        Args:
            dsl_code: The Neural DSL code
            output_format: Output format ('png', 'svg', 'html')
            output_file: Optional output file path

        Returns:
            The path to the generated visualization file
        """
        if not NEURAL_IMPORTED:
            raise ImportError("Neural DSL is not installed. Run the installation script first.")

        # Parse the DSL code
        tree = self.parser.parse(dsl_code)
        model_data = self.transformer.transform(tree)

        # Determine output path
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self.temp_dir / f"model_visualization.{output_format}"

        # Import visualization module
        try:
            from neural.visualization.visualizer import visualize_model
            visualize_model(model_data, str(output_path), output_format)
            print_success(f"Model visualization saved to {output_path}")
            return str(output_path)
        except ImportError:
            print_error("Visualization module not available")
            raise

    def setup_ngrok_tunnel(self, port: int = 8050) -> Optional[str]:
        """
        Set up an ngrok tunnel for accessing dashboards from cloud environments.

        Args:
            port: The local port to expose

        Returns:
            The public URL or None if setup failed
        """
        try:
            # Try to import pyngrok
            try:
                from pyngrok import ngrok
            except ImportError:
                print_info("Installing pyngrok...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
                from pyngrok import ngrok

            # Start the tunnel
            public_url = ngrok.connect(port).public_url
            print_success(f"Dashboard available at: {public_url}")
            return public_url
        except Exception as e:
            print_error(f"Failed to set up ngrok tunnel: {e}")
            return None

    def start_debug_dashboard(self,
                             dsl_code: str,
                             backend: str = 'tensorflow',
                             setup_tunnel: bool = True) -> Dict[str, Any]:
        """
        Start the NeuralDbg dashboard for a model.

        Args:
            dsl_code: The Neural DSL code
            backend: The target backend
            setup_tunnel: Whether to set up an ngrok tunnel

        Returns:
            Dictionary with dashboard information
        """
        # Save DSL to temporary file
        temp_file = self.temp_dir / "debug_model.neural"
        temp_file.write_text(dsl_code)

        # Start NeuralDbg in background
        cmd = ["python", "-m", "neural.cli", "debug", str(temp_file), "--backend", backend]
        process = subprocess.Popen(cmd)

        # Set up tunnel if requested
        tunnel_url = None
        if setup_tunnel:
            tunnel_url = self.setup_ngrok_tunnel(8050)

        return {
            "session_id": f"debug_{process.pid}",
            "dashboard_url": tunnel_url or "http://localhost:8050",
            "process_id": process.pid,
            "tunnel_url": tunnel_url
        }

    def start_nocode_interface(self,
                              port: int = 8051,
                              setup_tunnel: bool = True) -> Dict[str, Any]:
        """
        Start the Neural No-Code interface.

        Args:
            port: The port to run the interface on
            setup_tunnel: Whether to set up an ngrok tunnel

        Returns:
            Dictionary with interface information
        """
        # Start No-Code interface in background
        cmd = ["python", "-m", "neural.cli", "no-code", "--port", str(port)]
        process = subprocess.Popen(cmd)

        # Set up tunnel if requested
        tunnel_url = None
        if setup_tunnel:
            tunnel_url = self.setup_ngrok_tunnel(port)

        return {
            "session_id": f"nocode_{process.pid}",
            "interface_url": tunnel_url or f"http://localhost:{port}",
            "process_id": process.pid,
            "tunnel_url": tunnel_url
        }

    def cleanup(self):
        """Clean up temporary files and processes."""
        import shutil
        import signal

        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print_warning(f"Failed to clean up temporary directory: {e}")

        # Clean up ngrok tunnels if pyngrok is installed
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except ImportError:
            pass
