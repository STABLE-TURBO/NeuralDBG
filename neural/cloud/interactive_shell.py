"""
Interactive Shell for Neural Cloud Integration

This module provides an interactive shell for executing Neural DSL commands on cloud platforms.
"""

import os
import sys
import cmd
import time
import json
import tempfile
import logging
import readline
import shlex
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging - redirect to file to avoid cluttering the console
log_file = os.path.join(tempfile.gettempdir(), "neural_cloud_shell.log")
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.propagate = False  # Don't propagate to root logger (console)

class NeuralCloudShell(cmd.Cmd):
    """Interactive shell for Neural Cloud Integration."""

    intro = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                              ┃
┃   ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗      ██████╗██╗      ██████╗ ┃
┃   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     ██╔════╝██║     ██╔═══██╗┃
┃   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     ██║     ██║     ██║   ██║┃
┃   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ██║     ██║     ██║   ██║┃
┃   ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗╚██████╗███████╗╚██████╔╝┃
┃   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝ ╚═════╝ ┃
┃                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

🌟 Interactive shell for Neural Cloud Integration 🌟

Available commands:
  • run <file> [--backend <backend>] [--dataset <dataset>] [--epochs <epochs>]
  • visualize <file> [--format <format>]
  • debug <file> [--backend <backend>] [--no-tunnel]
  • shell <command>
  • python <code>
  • history [<count>]
  • help or ?
  • exit or quit

"""
    prompt = '🔮 neural-cloud> '

    def __init__(self, platform: str, remote_connection=None, quiet: bool = False):
        """
        Initialize the shell.

        Args:
            platform: The cloud platform ('kaggle', 'colab', 'sagemaker')
            remote_connection: An existing RemoteConnection object
            quiet: Whether to suppress output
        """
        super().__init__()
        self.platform = platform
        self.temp_dir = Path(tempfile.mkdtemp(prefix="neural_cloud_shell_"))
        self.history_file = self.temp_dir / "history.json"
        self.history = []
        self.kernel_id = None
        self.notebook_name = None
        self.quiet = quiet

        # Initialize the remote connection
        if remote_connection:
            self.remote = remote_connection
        else:
            try:
                from neural.cloud.remote_connection import RemoteConnection
                self.remote = RemoteConnection()
            except ImportError:
                if not self.quiet:
                    print("Error: Remote connection module not found.")
                sys.exit(1)

        # Connect to the platform
        self._connect_to_platform()

        # Create a kernel or notebook
        self._create_execution_environment()

        # Load history
        self._load_history()

    def _connect_to_platform(self):
        """Connect to the cloud platform."""
        if not self.quiet:
            print(f"Connecting to {self.platform}...")

        if self.platform.lower() == 'kaggle':
            result = self.remote.connect_to_kaggle()
        elif self.platform.lower() == 'colab':
            result = self.remote.connect_to_colab()
        elif self.platform.lower() == 'sagemaker':
            result = self.remote.connect_to_sagemaker()
        else:
            if not self.quiet:
                print(f"Error: Unsupported platform: {self.platform}")
            sys.exit(1)

        if not result['success']:
            if not self.quiet:
                print(f"Error: Failed to connect to {self.platform}: {result.get('error', 'Unknown error')}")
            sys.exit(1)

        if not self.quiet:
            print(f"Successfully connected to {self.platform}!")

    def _create_execution_environment(self):
        """Create a kernel or notebook for execution."""
        if not self.quiet:
            print(f"Creating execution environment on {self.platform}...")

        if self.platform.lower() == 'kaggle':
            # Create a kernel with a unique name
            import uuid
            unique_name = f"neural-shell-{uuid.uuid4().hex[:8]}"

            # Try multiple times with different names if needed
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                if not self.quiet:
                    print(f"Attempt {attempt}/{max_attempts} to create Kaggle kernel...")
                self.kernel_id = self.remote.create_kaggle_kernel(unique_name)
                if self.kernel_id:
                    break

                # Try a different name
                unique_name = f"neural-shell-{uuid.uuid4().hex[:8]}"

            if not self.kernel_id:
                if not self.quiet:
                    print("Error: Failed to create Kaggle kernel after multiple attempts")
                    print("Please check your Kaggle API credentials and try again")
                    print("You can set up your credentials by running 'kaggle config' in your terminal")
                sys.exit(1)

            if not self.quiet:
                print(f"Created Kaggle kernel: {self.kernel_id}")

            # Initialize the kernel with Neural DSL
            init_code = """
# Install Neural DSL
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git

# Import the cloud module
try:
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
except Exception as e:
    print(f"Error initializing Neural DSL: {e}")
    print("You can still use shell commands and Python code")
"""

            # Try to initialize the kernel
            max_attempts = 2  # Reduced to 2 attempts for better user experience
            for attempt in range(1, max_attempts + 1):
                if not self.quiet:
                    print(f"Attempt {attempt}/{max_attempts} to initialize Kaggle kernel...")
                result = self.remote.execute_on_kaggle(self.kernel_id, init_code)
                if result['success']:
                    break

                # Wait a bit before retrying
                time.sleep(5)

            # Even if initialization failed, we can still provide a useful experience
            # The shell commands will work, and we'll handle Python code execution
            if not result.get('success', False):
                if not self.quiet:
                    print(f"Note: Kaggle kernel initialization had issues: {result.get('error', 'Unknown error')}")
                    print("This is normal and expected with the current Kaggle API limitations.")
                    print("You can still use the following commands:")
                    print("  - shell <command>: Execute shell commands")
                    print("  - python <code>: Execute Python code")
                    print("  - run <file>: Attempt to run a Neural DSL file (may not work)")
                    print("  - visualize <file>: Attempt to visualize a model (may not work)")
                    print("  - debug <file>: Attempt to debug a model (may not work)")
                    print("  - history: Show command history")
                    print("  - exit/quit: Exit the shell")
            elif not self.quiet:
                print("Kaggle kernel initialized with Neural DSL")

        elif self.platform.lower() == 'sagemaker':
            # Create a notebook instance
            self.notebook_name = self.remote.create_sagemaker_notebook(f"neural-cloud-shell-{int(time.time())}")
            if not self.notebook_name:
                if not self.quiet:
                    print("Error: Failed to create SageMaker notebook instance")
                sys.exit(1)

            if not self.quiet:
                print(f"Created SageMaker notebook instance: {self.notebook_name}")

            # Initialize the notebook with Neural DSL
            init_code = """
# Install Neural DSL
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git

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
                if not self.quiet:
                    print(f"Error: Failed to initialize SageMaker notebook: {result.get('error', 'Unknown error')}")
                sys.exit(1)

            if not self.quiet:
                print("SageMaker notebook initialized with Neural DSL")

        elif self.platform.lower() == 'colab':
            if not self.quiet:
                print("Colab interactive shell is not supported yet")
            sys.exit(1)

    def _load_history(self):
        """Load command history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

    def _save_history(self):
        """Save command history."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    def _add_to_history(self, command: str, result: Dict[str, Any]):
        """Add a command to history."""
        self.history.append({
            'timestamp': time.time(),
            'command': command,
            'success': result.get('success', False),
            'output': result.get('output', '')
        })
        self._save_history()

    def _execute_dsl(self, dsl_code: str, backend: str = 'tensorflow',
                    dataset: str = 'MNIST', epochs: int = 5) -> Dict[str, Any]:
        """
        Execute DSL code on the cloud platform.

        Args:
            dsl_code: The DSL code to execute
            backend: The backend to use
            dataset: The dataset to use
            epochs: The number of epochs to train

        Returns:
            Dictionary with execution results
        """
        execution_code = f"""
# Define the model
dsl_code = \"\"\"
{dsl_code}
\"\"\"

# Run the model
model_path, results = run_dsl(dsl_code, backend='{backend}', dataset='{dataset}', epochs={epochs})
"""

        if self.platform.lower() == 'kaggle':
            if not self.kernel_id:
                return {'success': False, 'error': "No Kaggle kernel available"}
            return self.remote.execute_on_kaggle(self.kernel_id, execution_code)
        elif self.platform.lower() == 'sagemaker':
            if not self.notebook_name:
                return {'success': False, 'error': "No SageMaker notebook available"}
            return self.remote.execute_on_sagemaker(self.notebook_name, execution_code)
        else:
            return {'success': False, 'error': f"Unsupported platform: {self.platform}"}

    def _visualize_dsl(self, dsl_code: str, output_format: str = 'png') -> Dict[str, Any]:
        """
        Visualize DSL code on the cloud platform.

        Args:
            dsl_code: The DSL code to visualize
            output_format: The output format

        Returns:
            Dictionary with visualization results
        """
        visualization_code = f"""
# Define the model
dsl_code = \"\"\"
{dsl_code}
\"\"\"

# Visualize the model
viz_path = visualize_model(dsl_code, output_format='{output_format}')
"""

        if self.platform.lower() == 'kaggle':
            if not self.kernel_id:
                return {'success': False, 'error': "No Kaggle kernel available"}
            return self.remote.execute_on_kaggle(self.kernel_id, visualization_code)
        elif self.platform.lower() == 'sagemaker':
            if not self.notebook_name:
                return {'success': False, 'error': "No SageMaker notebook available"}
            return self.remote.execute_on_sagemaker(self.notebook_name, visualization_code)
        else:
            return {'success': False, 'error': f"Unsupported platform: {self.platform}"}

    def _debug_dsl(self, dsl_code: str, backend: str = 'tensorflow',
                  setup_tunnel: bool = True) -> Dict[str, Any]:
        """
        Debug DSL code on the cloud platform.

        Args:
            dsl_code: The DSL code to debug
            backend: The backend to use
            setup_tunnel: Whether to set up a tunnel

        Returns:
            Dictionary with debugging results
        """
        debug_code = f"""
# Define the model
dsl_code = \"\"\"
{dsl_code}
\"\"\"

# Debug the model
dashboard_info = debug_model(dsl_code, backend='{backend}', setup_tunnel={str(setup_tunnel).lower()})
"""

        if self.platform.lower() == 'kaggle':
            if not self.kernel_id:
                return {'success': False, 'error': "No Kaggle kernel available"}
            return self.remote.execute_on_kaggle(self.kernel_id, debug_code)
        elif self.platform.lower() == 'sagemaker':
            if not self.notebook_name:
                return {'success': False, 'error': "No SageMaker notebook available"}
            return self.remote.execute_on_sagemaker(self.notebook_name, debug_code)
        else:
            return {'success': False, 'error': f"Unsupported platform: {self.platform}"}

    def _execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a shell command on the cloud platform.

        Args:
            command: The command to execute

        Returns:
            Dictionary with execution results
        """
        execution_code = f"""
# Execute shell command
!{command}
"""

        if self.platform.lower() == 'kaggle':
            if not self.kernel_id:
                return {'success': False, 'error': "No Kaggle kernel available"}
            return self.remote.execute_on_kaggle(self.kernel_id, execution_code)
        elif self.platform.lower() == 'sagemaker':
            if not self.notebook_name:
                return {'success': False, 'error': "No SageMaker notebook available"}
            return self.remote.execute_on_sagemaker(self.notebook_name, execution_code)
        else:
            return {'success': False, 'error': f"Unsupported platform: {self.platform}"}

    def _execute_python(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code on the cloud platform.

        Args:
            code: The Python code to execute

        Returns:
            Dictionary with execution results
        """
        if self.platform.lower() == 'kaggle':
            if not self.kernel_id:
                return {'success': False, 'error': "No Kaggle kernel available"}
            return self.remote.execute_on_kaggle(self.kernel_id, code)
        elif self.platform.lower() == 'sagemaker':
            if not self.notebook_name:
                return {'success': False, 'error': "No SageMaker notebook available"}
            return self.remote.execute_on_sagemaker(self.notebook_name, code)
        else:
            return {'success': False, 'error': f"Unsupported platform: {self.platform}"}

    def do_run(self, arg):
        """
        Run a Neural DSL file on the cloud platform.

        Usage: run <file> [--backend <backend>] [--dataset <dataset>] [--epochs <epochs>]
        """
        args = shlex.split(arg)
        if not args:
            print("Error: Missing file argument")
            return

        file_path = args[0]
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return

        # Parse options
        backend = 'tensorflow'
        dataset = 'MNIST'
        epochs = 5

        i = 1
        while i < len(args):
            if args[i] == '--backend' and i + 1 < len(args):
                backend = args[i + 1]
                i += 2
            elif args[i] == '--dataset' and i + 1 < len(args):
                dataset = args[i + 1]
                i += 2
            elif args[i] == '--epochs' and i + 1 < len(args):
                try:
                    epochs = int(args[i + 1])
                except ValueError:
                    print(f"Error: Invalid epochs value: {args[i + 1]}")
                    return
                i += 2
            else:
                print(f"Error: Unknown option: {args[i]}")
                return

        # Read the file
        try:
            with open(file_path, 'r') as f:
                dsl_code = f.read()
        except Exception as e:
            print(f"Error: Failed to read file: {e}")
            return

        print(f"Running {file_path} on {self.platform} with backend={backend}, dataset={dataset}, epochs={epochs}...")

        # Execute the DSL code
        result = self._execute_dsl(dsl_code, backend=backend, dataset=dataset, epochs=epochs)

        # Add to history
        self._add_to_history(f"run {arg}", result)

        # Print the result
        if result['success']:
            print("Execution successful!")
            print("\nOutput:")
            print(result['output'])
        else:
            print(f"Execution failed: {result.get('error', 'Unknown error')}")

    def do_visualize(self, arg):
        """
        Visualize a Neural DSL file on the cloud platform.

        Usage: visualize <file> [--format <format>]
        """
        args = shlex.split(arg)
        if not args:
            print("Error: Missing file argument")
            return

        file_path = args[0]
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return

        # Parse options
        output_format = 'png'

        i = 1
        while i < len(args):
            if args[i] == '--format' and i + 1 < len(args):
                output_format = args[i + 1]
                i += 2
            else:
                print(f"Error: Unknown option: {args[i]}")
                return

        # Read the file
        try:
            with open(file_path, 'r') as f:
                dsl_code = f.read()
        except Exception as e:
            print(f"Error: Failed to read file: {e}")
            return

        print(f"Visualizing {file_path} on {self.platform} with format={output_format}...")

        # Visualize the DSL code
        result = self._visualize_dsl(dsl_code, output_format=output_format)

        # Add to history
        self._add_to_history(f"visualize {arg}", result)

        # Print the result
        if result['success']:
            print("Visualization successful!")
            print("\nOutput:")
            print(result['output'])
        else:
            print(f"Visualization failed: {result.get('error', 'Unknown error')}")

    def do_debug(self, arg):
        """
        Debug a Neural DSL file on the cloud platform.

        Usage: debug <file> [--backend <backend>] [--no-tunnel]
        """
        args = shlex.split(arg)
        if not args:
            print("Error: Missing file argument")
            return

        file_path = args[0]
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return

        # Parse options
        backend = 'tensorflow'
        setup_tunnel = True

        i = 1
        while i < len(args):
            if args[i] == '--backend' and i + 1 < len(args):
                backend = args[i + 1]
                i += 2
            elif args[i] == '--no-tunnel':
                setup_tunnel = False
                i += 1
            else:
                print(f"Error: Unknown option: {args[i]}")
                return

        # Read the file
        try:
            with open(file_path, 'r') as f:
                dsl_code = f.read()
        except Exception as e:
            print(f"Error: Failed to read file: {e}")
            return

        print(f"Debugging {file_path} on {self.platform} with backend={backend}, setup_tunnel={setup_tunnel}...")

        # Debug the DSL code
        result = self._debug_dsl(dsl_code, backend=backend, setup_tunnel=setup_tunnel)

        # Add to history
        self._add_to_history(f"debug {arg}", result)

        # Print the result
        if result['success']:
            print("Debugging session started!")
            print("\nOutput:")
            print(result['output'])
        else:
            print(f"Debugging failed: {result.get('error', 'Unknown error')}")

    def do_shell(self, arg):
        """
        Execute a shell command on the cloud platform.

        Usage: shell <command>
        """
        if not arg:
            print("Error: Missing command argument")
            return

        # For Kaggle, we'll execute the command locally if it's a simple command
        # This provides a better user experience since the Kaggle API has limitations
        if self.platform.lower() == 'kaggle':
            try:
                print(f"Executing shell command locally: {arg}")

                # Execute the command locally
                import subprocess
                result = subprocess.run(arg, shell=True, capture_output=True, text=True)

                # Add to history
                self._add_to_history(f"shell {arg}", {
                    'success': result.returncode == 0,
                    'output': result.stdout if result.returncode == 0 else result.stderr
                })

                # Print the result
                if result.returncode == 0:
                    print("Command executed successfully!")
                    print("\nOutput:")
                    print(result.stdout)
                else:
                    print(f"Command execution failed with exit code {result.returncode}")
                    print("\nError output:")
                    print(result.stderr)

                return
            except Exception as e:
                print(f"Error executing command locally: {e}")
                print("Falling back to cloud execution...")

        print(f"Executing shell command on {self.platform}: {arg}")

        # Execute the command
        result = self._execute_command(arg)

        # Add to history
        self._add_to_history(f"shell {arg}", result)

        # Print the result
        if result['success']:
            print("Command executed successfully!")
            print("\nOutput:")
            print(result['output'])
        else:
            print(f"Command execution failed: {result.get('error', 'Unknown error')}")

    def do_python(self, arg):
        """
        Execute Python code on the cloud platform.

        Usage: python <code>
        """
        if not arg:
            print("Error: Missing code argument")
            return

        # For Kaggle, we'll execute the code locally if it's simple code
        # This provides a better user experience since the Kaggle API has limitations
        if self.platform.lower() == 'kaggle':
            try:
                print(f"Executing Python code locally...")

                # Execute the code locally
                import sys
                from io import StringIO

                # Capture stdout
                old_stdout = sys.stdout
                redirected_output = StringIO()
                sys.stdout = redirected_output

                # Execute the code
                try:
                    exec(arg)
                    success = True
                    error_msg = None
                except Exception as e:
                    success = False
                    error_msg = str(e)

                # Restore stdout
                sys.stdout = old_stdout

                # Get the output
                output = redirected_output.getvalue()

                # Add to history
                self._add_to_history(f"python {arg}", {
                    'success': success,
                    'output': output if success else error_msg
                })

                # Print the result
                if success:
                    print("Code executed successfully!")
                    if output:
                        print("\nOutput:")
                        print(output)
                else:
                    print(f"Code execution failed: {error_msg}")

                return
            except Exception as e:
                print(f"Error executing code locally: {e}")
                print("Falling back to cloud execution...")

        print(f"Executing Python code on {self.platform}...")

        # Execute the code
        result = self._execute_python(arg)

        # Add to history
        self._add_to_history(f"python {arg}", result)

        # Print the result
        if result['success']:
            print("Code executed successfully!")
            print("\nOutput:")
            print(result['output'])
        else:
            print(f"Code execution failed: {result.get('error', 'Unknown error')}")

    def do_history(self, arg):
        """
        Show command history.

        Usage: history [<count>]
        """
        count = 10
        if arg:
            try:
                count = int(arg)
            except ValueError:
                print(f"Error: Invalid count value: {arg}")
                return

        if not self.history:
            print("No history available")
            return

        print(f"Last {min(count, len(self.history))} commands:")
        for i, entry in enumerate(self.history[-count:]):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))
            status = "✓" if entry['success'] else "✗"
            print(f"{i+1}. [{timestamp}] {status} {entry['command']}")

    def do_exit(self, _):
        """Exit the shell."""
        return self._cleanup()

    def do_quit(self, _):
        """Exit the shell."""
        return self._cleanup()

    def do_EOF(self, _):
        """Exit the shell on Ctrl+D."""
        print()  # Add a newline
        return self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        if not self.quiet:
            print("Cleaning up...")

        # Delete the kernel or notebook
        if self.platform.lower() == 'kaggle' and self.kernel_id:
            if self.remote.delete_kaggle_kernel(self.kernel_id):
                if not self.quiet:
                    print(f"Deleted Kaggle kernel: {self.kernel_id}")
            elif not self.quiet:
                print(f"Failed to delete Kaggle kernel: {self.kernel_id}")

        elif self.platform.lower() == 'sagemaker' and self.notebook_name:
            if self.remote.delete_sagemaker_notebook(self.notebook_name):
                if not self.quiet:
                    print(f"Deleted SageMaker notebook: {self.notebook_name}")
            elif not self.quiet:
                print(f"Failed to delete SageMaker notebook: {self.notebook_name}")

        # Clean up the remote connection
        self.remote.cleanup()

        if not self.quiet:
            print("Goodbye!")
        return True

    def emptyline(self):
        """Do nothing on empty line."""
        pass

    def default(self, line):
        """Handle unknown commands."""
        if not self.quiet:
            print(f"Unknown command: {line}")
            print("Type 'help' or '?' to list available commands.")


def start_interactive_shell(platform: str, remote_connection=None, quiet: bool = False):
    """
    Start an interactive shell for the specified cloud platform.

    Args:
        platform: The cloud platform ('kaggle', 'colab', 'sagemaker')
        remote_connection: An existing RemoteConnection object
        quiet: Whether to suppress output
    """
    # Create the shell with the quiet parameter
    shell = NeuralCloudShell(platform, remote_connection, quiet=quiet)

    # Start the shell
    shell.cmdloop()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Missing platform argument")
        print("Usage: python interactive_shell.py <platform>")
        sys.exit(1)

    platform = sys.argv[1]
    start_interactive_shell(platform)
