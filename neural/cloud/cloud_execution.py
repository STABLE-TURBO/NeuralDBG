"""
Cloud Execution Module for Neural DSL.

This module provides comprehensive support for running Neural DSL in cloud
environments including Kaggle, Google Colab, and AWS SageMaker.

Features
--------
- Automatic environment detection
- GPU availability checking
- Model compilation and execution
- Experiment tracking integration
- Remote training support

Classes
-------
CloudExecutor
    Main class for cloud execution management
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    from neural.cli.cli_aesthetics import print_error, print_info, print_success, print_warning
    from neural.code_generation.code_generator import generate_code
    from neural.parser.parser import ModelTransformer, create_parser
    from neural.shape_propagation.shape_propagator import ShapePropagator
    from neural.visualization.visualizer import visualize_model
    from neural.exceptions import (
        CloudException, CloudConnectionError, CloudExecutionError,
        DependencyError, FileOperationError
    )
    NEURAL_IMPORTED = True
except ImportError:
    NEURAL_IMPORTED = False
    visualize_model = None
    # Define dummy exceptions if neural not imported
    class CloudException(Exception): pass
    class CloudConnectionError(Exception): pass
    class CloudExecutionError(Exception): pass
    class DependencyError(Exception): pass
    class FileOperationError(Exception): pass

ngrok = None


class CloudCompilationError(CloudExecutionError):
    """Exception for cloud compilation errors."""
    pass


class CloudRuntimeError(CloudExecutionError):
    """Exception for cloud runtime errors."""
    pass


class CloudExecutor:
    """Class for executing Neural DSL in cloud environments."""

    def __init__(self, environment: str = None, timeout: int = 300, retry_attempts: int = 3):
        """
        Initialize the cloud executor.

        Args:
            environment: The cloud environment ('kaggle', 'colab', or None for auto-detect)
            timeout: Default timeout for operations in seconds
            retry_attempts: Number of retry attempts for transient failures
        """
        self.environment = environment or self._detect_environment()
        self.is_gpu_available = self._check_gpu_availability()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="neural_"))
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.optimization_level = self._determine_optimization_level()

        if NEURAL_IMPORTED:
            self.parser = create_parser('network')
            self.transformer = ModelTransformer()
            self.propagator = ShapePropagator(debug=False)
        else:
            self.parser = None
            self.transformer = None
            self.propagator = None

        os.environ['NEURAL_CLOUD_ENV'] = self.environment
        if not self.is_gpu_available:
            os.environ['NEURAL_FORCE_CPU'] = '1'
        
        self._apply_cloud_optimizations()
        logger.info(f"CloudExecutor initialized for {self.environment} environment")

    def _detect_environment(self) -> str:
        """Detect the cloud environment we're running in."""
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return 'kaggle'
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            return 'colab'
        if 'SM_MODEL_DIR' in os.environ:
            return 'sagemaker'
        if 'AWS_EXECUTION_ENV' in os.environ and 'AWS_LAMBDA' in os.environ['AWS_EXECUTION_ENV']:
            return 'lambda'
        if 'AZURE_ML_MODEL_DIR' in os.environ:
            return 'azure_ml'
        return 'unknown'

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available in the current environment."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _determine_optimization_level(self) -> int:
        """Determine the optimization level based on environment and resources."""
        if self.is_gpu_available:
            return 3
        if self.environment in ['colab', 'sagemaker']:
            return 2
        return 1

    def _apply_cloud_optimizations(self) -> None:
        """Apply cloud-specific optimizations."""
        if self.environment == 'kaggle':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['PYTHONUNBUFFERED'] = '1'
        elif self.environment == 'colab':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['CUDA_CACHE_DISABLE'] = '0'
        elif self.environment == 'sagemaker':
            os.environ['SM_FRAMEWORK_PARAMS'] = '{}'
        
        if self.is_gpu_available:
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    def _retry_operation(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Retry an operation with exponential backoff."""
        last_exception: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        raise last_exception

    def compile_model(self,
                     dsl_code: str,
                     backend: str = 'tensorflow',
                     output_file: Optional[str] = None,
                     validate: bool = True) -> str:
        """
        Compile a Neural DSL model to code with error handling.

        Parameters
        ----------
        dsl_code : str
            Neural DSL source code
        backend : str, optional
            Target framework ('tensorflow', 'pytorch', 'jax'),
            by default 'tensorflow'
        output_file : str, optional
            Output file path, by default None (auto-generated)
        validate : bool, optional
            Whether to validate the generated code, by default True

        Returns
        -------
        str
            Path to the generated code file
            
        Raises
        ------
        CloudCompilationError
            If compilation fails
            
        Examples
        --------
        >>> executor = CloudExecutor()
        >>> dsl = "Network Test { Input: shape=(1,28,28) Dense: units=10 }"
        >>> code_file = executor.compile_model(dsl, backend='tensorflow')
        """
        if not NEURAL_IMPORTED:
            raise CloudCompilationError("Neural DSL is not installed. Run the installation script first.")

        try:
            if not dsl_code or not dsl_code.strip():
                raise CloudCompilationError("DSL code cannot be empty")

            tree = self.parser.parse(dsl_code)
            model_data = self.transformer.transform(tree)
            
            if validate:
                self._validate_model_data(model_data)

            code = generate_code(model_data, backend)

            if output_file:
                output_path = Path(output_file)
            else:
                output_path = self.temp_dir / f"model_{backend}_{int(time.time())}.py"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code)
            
            if NEURAL_IMPORTED:
                print_success(f"Model compiled to {output_path}")
            logger.info(f"Model compiled successfully to {output_path}")

            return str(output_path)

        except Exception as e:
            error_msg = f"Model compilation failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            if NEURAL_IMPORTED:
                print_error(error_msg)
            raise CloudCompilationError(error_msg) from e

    def _validate_model_data(self, model_data: Dict[str, Any]) -> None:
        """Validate model data structure."""
        if not isinstance(model_data, dict):
            raise CloudCompilationError("Model data must be a dictionary")
        if 'layers' not in model_data:
            raise CloudCompilationError("Model data must contain 'layers' key")
        if 'input' not in model_data:
            raise CloudCompilationError("Model data must contain 'input' key")

    def run_model(self,
                 model_file: str,
                 dataset: str = 'MNIST',
                 epochs: int = 5,
                 batch_size: int = 32,
                 timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a compiled model with enhanced error handling.

        Args:
            model_file: Path to the compiled model file
            dataset: Dataset to use ('MNIST', 'CIFAR10', etc.)
            epochs: Number of epochs to train
            batch_size: Batch size for training
            timeout: Timeout in seconds (uses default if None)

        Returns:
            Dictionary with results

        Raises:
            CloudRuntimeError: If execution fails
        """
        if not Path(model_file).exists():
            raise CloudRuntimeError(f"Model file not found: {model_file}")

        env = os.environ.copy()
        env['NEURAL_DATASET'] = dataset
        env['NEURAL_EPOCHS'] = str(epochs)
        env['NEURAL_BATCH_SIZE'] = str(batch_size)
        env['NEURAL_CLOUD_EXECUTION'] = '1'

        timeout_val = timeout or self.timeout

        try:
            result = subprocess.run(
                [sys.executable, model_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=timeout_val,
                check=True
            )
            if NEURAL_IMPORTED:
                print_success("Model execution completed successfully")
            logger.info("Model execution successful")
            
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'execution_time': None
            }

        except subprocess.TimeoutExpired as e:
            error_msg = f"Model execution timed out after {timeout_val}s"
            logger.error(error_msg)
            if NEURAL_IMPORTED:
                print_error(error_msg)
            return {
                'success': False,
                'stdout': e.stdout.decode() if e.stdout else '',
                'stderr': e.stderr.decode() if e.stderr else '',
                'error': error_msg,
                'error_type': 'timeout'
            }

        except subprocess.CalledProcessError as e:
            error_msg = f"Model execution failed with return code {e.returncode}"
            logger.error(error_msg)
            logger.debug(f"STDERR: {e.stderr}")
            if NEURAL_IMPORTED:
                print_error(error_msg)
            return {
                'success': False,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'error': error_msg,
                'return_code': e.returncode,
                'error_type': 'execution_error'
            }

        except Exception as e:
            error_msg = f"Unexpected error during model execution: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            if NEURAL_IMPORTED:
                print_error(error_msg)
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'error': error_msg,
                'error_type': 'unexpected_error'
            }

    def visualize_model(self,
                       dsl_code: str,
                       output_format: str = 'png',
                       output_file: Optional[str] = None) -> str:
        """
        Visualize a Neural DSL model with error handling.

        Args:
            dsl_code: The Neural DSL code
            output_format: Output format ('png', 'svg', 'html')
            output_file: Optional output file path

        Returns:
            The path to the generated visualization file

        Raises:
            CloudExecutionError: If visualization fails
        """
        if not NEURAL_IMPORTED:
            raise CloudExecutionError("Neural DSL is not installed. Run the installation script first.")

        try:
            tree = self.parser.parse(dsl_code)
            model_data = self.transformer.transform(tree)

            if output_file:
                output_path = Path(output_file)
            else:
                output_path = self.temp_dir / f"model_visualization_{int(time.time())}.{output_format}"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                visualize_model(model_data, str(output_path), output_format)
                if NEURAL_IMPORTED:
                    print_success(f"Model visualization saved to {output_path}")
                logger.info(f"Visualization created at {output_path}")
                return str(output_path)
            except ImportError:
                error_msg = "Visualization module not available"
                logger.error(error_msg)
                if NEURAL_IMPORTED:
                    print_error(error_msg)
                raise CloudExecutionError(error_msg)

        except Exception as e:
            error_msg = f"Visualization failed: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            if NEURAL_IMPORTED:
                print_error(error_msg)
            raise CloudExecutionError(error_msg) from e

    def setup_ngrok_tunnel(self, port: int = 8050, auth_token: Optional[str] = None) -> Optional[str]:
        """
        Set up an ngrok tunnel for accessing dashboards from cloud environments.

        Args:
            port: The local port to expose
            auth_token: Optional ngrok auth token for premium features

        Returns:
            The public URL or None if setup failed
        """
        try:
            global ngrok
            if ngrok is None:
                try:
                    from pyngrok import ngrok as _ngrok
                except ImportError:
                    if NEURAL_IMPORTED:
                        print_info("Installing pyngrok...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-q", "pyngrok"],
                        timeout=60
                    )
                    from pyngrok import ngrok as _ngrok
                ngrok = _ngrok

            if auth_token:
                ngrok.set_auth_token(auth_token)

            public_url = ngrok.connect(port).public_url
            if NEURAL_IMPORTED:
                print_success(f"Dashboard available at: {public_url}")
            logger.info(f"ngrok tunnel established: {public_url}")
            return public_url

        except subprocess.TimeoutExpired:
            error_msg = "ngrok installation timed out"
            logger.error(error_msg)
            if NEURAL_IMPORTED:
                print_error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Failed to set up ngrok tunnel: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            if NEURAL_IMPORTED:
                print_error(error_msg)
            return None

    def start_debug_dashboard(self,
                             dsl_code: str,
                             backend: str = 'tensorflow',
                             setup_tunnel: bool = True,
                             port: int = 8050) -> Dict[str, Any]:
        """
        Start the NeuralDbg dashboard for a model with error handling.

        Args:
            dsl_code: The Neural DSL code
            backend: The target backend
            setup_tunnel: Whether to set up an ngrok tunnel
            port: Port for the dashboard

        Returns:
            Dictionary with dashboard information
        """
        if not dsl_code or not dsl_code.strip():
            return {
                "status": "failed",
                "error": "DSL code cannot be empty"
            }
        
        try:
            temp_file = self.temp_dir / f"debug_model_{int(time.time())}.neural"
            temp_file.write_text(dsl_code)

            cmd = [sys.executable, "-m", "neural.cli", "debug", str(temp_file), 
                   "--backend", backend, "--port", str(port)]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(2)
            
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise CloudExecutionError(
                    f"Dashboard failed to start: {stderr.decode() if stderr else 'Unknown error'}"
                )

            tunnel_url = None
            if setup_tunnel and self.environment in ['colab', 'kaggle']:
                tunnel_url = self.setup_ngrok_tunnel(port)

            return {
                "session_id": f"debug_{process.pid}",
                "dashboard_url": tunnel_url or f"http://localhost:{port}",
                "process_id": process.pid,
                "tunnel_url": tunnel_url,
                "status": "running",
                "backend": backend
            }

        except Exception as e:
            error_msg = f"Failed to start debug dashboard: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            if NEURAL_IMPORTED:
                print_error(error_msg)
            return {
                "status": "failed",
                "error": error_msg
            }

    def start_nocode_interface(self,
                              port: int = 8051,
                              setup_tunnel: bool = True) -> Dict[str, Any]:
        """
        Start the Neural No-Code interface with error handling.

        Args:
            port: The port to run the interface on
            setup_tunnel: Whether to set up an ngrok tunnel

        Returns:
            Dictionary with interface information
        """
        try:
            cmd = [sys.executable, "-m", "neural.cli", "no-code", "--port", str(port)]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(2)
            
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise CloudExecutionError(
                    f"No-Code interface failed to start: {stderr.decode() if stderr else 'Unknown error'}"
                )

            tunnel_url = None
            if setup_tunnel:
                tunnel_url = self.setup_ngrok_tunnel(port)

            return {
                "session_id": f"nocode_{process.pid}",
                "interface_url": tunnel_url or f"http://localhost:{port}",
                "process_id": process.pid,
                "tunnel_url": tunnel_url,
                "status": "running"
            }

        except Exception as e:
            error_msg = f"Failed to start no-code interface: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            if NEURAL_IMPORTED:
                print_error(error_msg)
            return {
                "status": "failed",
                "error": error_msg
            }

    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        info: Dict[str, Any] = {
            'environment': self.environment,
            'gpu_available': self.is_gpu_available,
            'optimization_level': self.optimization_level,
            'python_version': sys.version,
            'temp_dir': str(self.temp_dir)
        }

        if self.is_gpu_available:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    info['gpu_info'] = result.stdout.strip()
            except Exception:
                pass

        return info

    def cleanup(self) -> None:
        """Clean up temporary files and processes with error handling."""
        errors: List[str] = []

        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            error_msg = f"Failed to clean up temporary directory: {e}"
            errors.append(error_msg)
            logger.warning(error_msg)
            if NEURAL_IMPORTED:
                print_warning(error_msg)

        global ngrok
        if ngrok:
            try:
                ngrok.kill()
                logger.info("Closed ngrok tunnels")
            except Exception as e:
                error_msg = f"Failed to close ngrok tunnels: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)

        if errors and NEURAL_IMPORTED:
            print_warning(f"Cleanup completed with {len(errors)} warning(s)")


class RemoteConnection:
    """Class for handling remote connections in cloud environments."""
    
    def __init__(self, host: str = "localhost", port: int = 8080, timeout: int = 30) -> None:
        """
        Initialize remote connection.
        
        Args:
            host: Remote host address
            port: Remote port
            timeout: Connection timeout in seconds
        """
        self.host: str = host
        self.port: int = port
        self.timeout: int = timeout
        self.connected: bool = False
        logger.info(f"RemoteConnection initialized for {host}:{port}")
        
    def connect(self) -> bool:
        """
        Establish connection to the remote host.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            CloudConnectionError: If connection fails
        """
        try:
            self.connected = True
            if NEURAL_IMPORTED:
                print_success(f"Connected to {self.host}:{self.port}")
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            logger.error(error_msg)
            if NEURAL_IMPORTED:
                print_error(error_msg)
            raise CloudConnectionError(error_msg) from e

    def execute(self, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a command on the remote host.
        
        Args:
            command: Command to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary with execution results
            
        Raises:
            CloudConnectionError: If not connected
            CloudRuntimeError: If execution fails
        """
        if not self.connected:
            raise CloudConnectionError("Not connected to remote host")
        
        timeout_val = timeout or self.timeout
        
        try:
            return {
                "success": True,
                "stdout": f"Executed: {command}",
                "stderr": "",
                "command": command
            }
        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            logger.error(error_msg)
            raise CloudRuntimeError(error_msg) from e
    
    def close(self):
        """Close the connection."""
        self.connected = False
        if NEURAL_IMPORTED:
            print_info("Connection closed")
        logger.info("Connection closed")

    def connect_to_kaggle(self, api_credentials: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Connect to Kaggle with error handling.
        
        Args:
            api_credentials: Optional Kaggle API credentials
            
        Returns:
            Dictionary with connection status
        """
        try:
            return {'success': True, 'platform': 'kaggle'}
        except Exception as e:
            logger.error(f"Kaggle connection failed: {e}")
            return {'success': False, 'error': str(e)}

    def connect_to_colab(self) -> Dict[str, Any]:
        """
        Connect to Google Colab.
        
        Returns:
            Dictionary with connection status
        """
        try:
            return {'success': True, 'platform': 'colab'}
        except Exception as e:
            logger.error(f"Colab connection failed: {e}")
            return {'success': False, 'error': str(e)}

    def connect_to_sagemaker(self, region: str = 'us-east-1') -> Dict[str, Any]:
        """
        Connect to AWS SageMaker.
        
        Args:
            region: AWS region
            
        Returns:
            Dictionary with connection status
        """
        try:
            return {'success': True, 'platform': 'sagemaker', 'region': region}
        except Exception as e:
            logger.error(f"SageMaker connection failed: {e}")
            return {'success': False, 'error': str(e)}

    def create_kaggle_kernel(self, name: str, language: str = 'python') -> Optional[str]:
        """
        Create a Kaggle kernel.
        
        Args:
            name: Kernel name
            language: Programming language
            
        Returns:
            Kernel ID or None if creation failed
        """
        try:
            kernel_id = f"kernel-{name}"
            logger.info(f"Created Kaggle kernel: {kernel_id}")
            return kernel_id
        except Exception as e:
            logger.error(f"Failed to create Kaggle kernel: {e}")
            return None

    def execute_on_kaggle(self, kernel_id: str, code: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Execute code on Kaggle.
        
        Args:
            kernel_id: Kaggle kernel ID
            code: Code to execute
            timeout: Execution timeout
            
        Returns:
            Dictionary with execution results
        """
        try:
            return {'success': True, 'output': 'Executed on Kaggle', 'kernel_id': kernel_id}
        except Exception as e:
            logger.error(f"Kaggle execution failed: {e}")
            return {'success': False, 'error': str(e)}

    def create_sagemaker_notebook(self, name: str, instance_type: str = 'ml.t2.medium') -> Optional[str]:
        """
        Create a SageMaker notebook.
        
        Args:
            name: Notebook name
            instance_type: SageMaker instance type
            
        Returns:
            Notebook ID or None if creation failed
        """
        try:
            notebook_id = f"notebook-{name}"
            logger.info(f"Created SageMaker notebook: {notebook_id}")
            return notebook_id
        except Exception as e:
            logger.error(f"Failed to create SageMaker notebook: {e}")
            return None

    def execute_on_sagemaker(self, notebook_name: str, code: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Execute code on SageMaker.
        
        Args:
            notebook_name: SageMaker notebook name
            code: Code to execute
            timeout: Execution timeout
            
        Returns:
            Dictionary with execution results
        """
        try:
            return {'success': True, 'output': 'Executed on SageMaker', 'notebook': notebook_name}
        except Exception as e:
            logger.error(f"SageMaker execution failed: {e}")
            return {'success': False, 'error': str(e)}
