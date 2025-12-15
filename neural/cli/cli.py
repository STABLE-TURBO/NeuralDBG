#!/usr/bin/env python
"""
Main CLI implementation for Neural using Click.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np
from lark import exceptions

from .cli_aesthetics import (
    Colors,
    Spinner,
    animate_neural_network,
    print_command_header,
    print_error,
    print_help_command,
    print_info,
    print_neural_logo,
    print_success,
    print_warning,
    progress_bar,
)
from .cpu_mode import is_cpu_mode, set_cpu_mode
from .lazy_imports import (
    code_generator as code_generator_module,
)
from .lazy_imports import (
    experiment_tracker as experiment_tracker_module,
)
from .lazy_imports import get_module
from .lazy_imports import hpo as hpo_module
from .lazy_imports import jax, optuna, tensorflow, torch
from .lazy_imports import shape_propagator as shape_propagator_module
from .lazy_imports import tensor_flow as tensor_flow_module
from .version import __version__
from .welcome_message import show_welcome_message


# Optional debugging dependency
try:
    import pysnooper

    _HAS_PYSNOOPER = True
except ImportError:
    pysnooper = None
    _HAS_PYSNOOPER = False

def configure_logging(verbose: bool = False) -> None:
    """Configure logging levels based on verbosity."""
    # Set environment variables to suppress debug messages from dependencies
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
    os.environ['MPLBACKEND'] = 'Agg'          # Non-interactive matplotlib backend

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Configure Neural logger
    neural_logger = logging.getLogger('neural')
    neural_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s" if verbose else "%(levelname)s: %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    neural_logger.handlers = [handler]

    # Ensure all neural submodules use the same log level
    for logger_name in ['neural.parser', 'neural.code_generation', 'neural.hpo']:
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(logging.WARNING if not verbose else logging.DEBUG)
        module_logger.handlers = [handler]
        module_logger.propagate = False

    # Silence noisy libraries
    for logger_name in [
        'graphviz', 'matplotlib', 'tensorflow', 'jax', 'tf', 'absl',
        'pydot', 'PIL', 'torch', 'urllib3', 'requests', 'h5py',
        'filelock', 'numba', 'asyncio', 'parso', 'werkzeug',
        'matplotlib.font_manager', 'matplotlib.ticker', 'optuna',
        'dash', 'plotly', 'ipykernel', 'traitlets', 'click'
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        logging.getLogger(logger_name).propagate = False

# Create logger
logger = logging.getLogger(__name__)

# Supported datasets
SUPPORTED_DATASETS = {"MNIST", "CIFAR10", "CIFAR100", "ImageNet"}

# Input validation functions
def sanitize_file_path(file_path: str, allow_absolute: bool = True) -> str:
    """
    Sanitize and validate a file path to prevent path traversal attacks.
    
    Args:
        file_path: The file path to sanitize
        allow_absolute: Whether to allow absolute paths
        
    Returns:
        Sanitized file path
        
    Raises:
        ValueError: If the path is invalid or contains malicious patterns
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")
    
    # Remove null bytes
    file_path = file_path.replace('\0', '')
    
    # Normalize the path for cross-platform compatibility
    normalized = os.path.normpath(file_path)
    
    # Check for path traversal attempts
    if '..' in normalized.split(os.sep):
        raise ValueError(f"Path traversal detected in: {file_path}")
    
    # Check for absolute paths if not allowed
    if not allow_absolute and os.path.isabs(normalized):
        raise ValueError(f"Absolute paths not allowed: {file_path}")
    
    # Check for suspicious patterns
    suspicious_patterns = [r'\.\./', r'\.\.$', r'^\.\.', r'~/', r'\$\{', r'%[A-Z_]+%']
    for pattern in suspicious_patterns:
        if re.search(pattern, file_path):
            raise ValueError(f"Suspicious pattern detected in path: {file_path}")
    
    return normalized

def validate_port(port: int, min_port: int = 1024, max_port: int = 65535) -> int:
    """
    Validate a port number.
    
    Args:
        port: Port number to validate
        min_port: Minimum allowed port (default: 1024 to avoid privileged ports)
        max_port: Maximum allowed port
        
    Returns:
        Validated port number
        
    Raises:
        ValueError: If port is invalid
    """
    if not isinstance(port, int):
        try:
            port = int(port)
        except (TypeError, ValueError):
            raise ValueError(f"Port must be an integer, got: {type(port).__name__}")
    
    if port < min_port or port > max_port:
        raise ValueError(f"Port must be between {min_port} and {max_port}, got: {port}")
    
    return port

def validate_backend(backend: str) -> str:
    """
    Validate and normalize a backend name.
    
    Args:
        backend: Backend name to validate
        
    Returns:
        Normalized backend name
        
    Raises:
        ValueError: If backend is invalid
    """
    if not backend or not isinstance(backend, str):
        raise ValueError("Backend must be a non-empty string")
    
    backend = backend.lower().strip()
    valid_backends = {'tensorflow', 'pytorch', 'onnx', 'jax'}
    
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend '{backend}'. Supported: {', '.join(sorted(valid_backends))}")
    
    return backend

def validate_dataset_name(dataset: str) -> str:
    """
    Validate a dataset name.
    
    Args:
        dataset: Dataset name to validate
        
    Returns:
        Validated dataset name
        
    Raises:
        ValueError: If dataset name is invalid
    """
    if not dataset or not isinstance(dataset, str):
        raise ValueError("Dataset name must be a non-empty string")
    
    # Remove any potentially dangerous characters
    dataset = dataset.strip()
    
    # Check for alphanumeric and basic separators only
    if not re.match(r'^[a-zA-Z0-9_-]+$', dataset):
        raise ValueError(f"Dataset name contains invalid characters: {dataset}")
    
    # Check length
    if len(dataset) > 100:
        raise ValueError(f"Dataset name too long (max 100 characters): {dataset}")
    
    return dataset

def validate_json_input(json_str: str, max_size: int = 1024 * 1024) -> dict:
    """
    Validate and parse JSON input safely.
    
    Args:
        json_str: JSON string to parse
        max_size: Maximum allowed size in bytes
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If JSON is invalid or too large
    """
    if not json_str or not isinstance(json_str, str):
        raise ValueError("JSON input must be a non-empty string")
    
    if len(json_str.encode('utf-8')) > max_size:
        raise ValueError(f"JSON input too large (max {max_size} bytes)")
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
    
    if not isinstance(data, dict):
        raise ValueError("JSON must be a dictionary/object")
    
    return data

def sanitize_experiment_name(name: str) -> str:
    """
    Sanitize an experiment or model name.
    
    Args:
        name: Name to sanitize
        
    Returns:
        Sanitized name
        
    Raises:
        ValueError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError("Name must be a non-empty string")
    
    # Remove leading/trailing whitespace
    name = name.strip()
    
    # Check length
    if len(name) < 1 or len(name) > 200:
        raise ValueError(f"Name length must be between 1 and 200 characters, got: {len(name)}")
    
    # Allow alphanumeric, spaces, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9_\- ]+$', name):
        raise ValueError(f"Name contains invalid characters: {name}")
    
    return name

# Global CLI context
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--cpu', is_flag=True, help='Force CPU mode')
@click.option('--no-animations', is_flag=True, help='Disable animations and spinners')
@click.version_option(version=__version__, prog_name="Neural")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, cpu: bool, no_animations: bool) -> None:
    """Neural CLI: A compiler-like interface for .neural and .nr files."""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NO_ANIMATIONS'] = no_animations
    ctx.obj['CPU_MODE'] = cpu

    configure_logging(verbose)

    if cpu:
        set_cpu_mode()
        logger.info("Running in CPU mode")

    # Show welcome message if not disabled
    if not os.environ.get('NEURAL_SKIP_WELCOME') and not hasattr(cli, '_welcome_shown'):
        show_welcome_message()
        setattr(cli, '_welcome_shown', True)
    elif not show_welcome_message():
        print_neural_logo(__version__)

@cli.command()
@click.pass_context
def help(ctx: click.Context) -> None:
    """Show help for commands."""
    print_help_command(ctx, cli.commands)

@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--backend', '-b', default='tensorflow', help='Target backend', type=click.Choice(['tensorflow', 'pytorch', 'onnx', 'jax'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
@click.option('--output', '-o', default=None, help='Output file path (defaults to <file>_<backend>.py)')
@click.option('--dry-run', is_flag=True, help='Preview generated code without writing to file')
@click.option('--hpo', is_flag=True, help='Enable hyperparameter optimization')
@click.option('--auto-flatten-output', is_flag=True, help='Auto-insert Flatten before Dense/Output when input is rank>2')
@click.pass_context
def compile(ctx, file: str, backend: str, dataset: str, output: Optional[str], dry_run: bool, hpo: bool, auto_flatten_output: bool):
    """Compile a .neural or .nr file into an executable Python script."""
    print_command_header("compile")
    
    # Sanitize and normalize file path for cross-platform compatibility
    try:
        file = os.path.normpath(file)
    except Exception as e:
        print_error(f"Invalid file path: {str(e)}")
        sys.exit(1)
    
    backend = backend.lower()
    print_info(f"Compiling {file} for {backend} backend")

    # Validate file type
    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}. Supported: .neural, .nr, .rnr")
        print_info("Hint: Use '.neural' or '.nr' extension for network definitions")
        sys.exit(1)

    # Parse the Neural DSL file
    with Spinner("Parsing Neural DSL file") as spinner:
        if ctx.obj.get('NO_ANIMATIONS'):
            spinner.stop()
        try:
            from neural.parser.parser import create_parser, ModelTransformer
            from neural.exceptions import DSLValidationError
            
            parser_instance = create_parser(start_rule=start_rule)
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = parser_instance.parse(content)
            model_data = ModelTransformer().transform(tree)
        except DSLValidationError as e:
            print_error(f"Parsing failed: {str(e)}")
            if hasattr(e, 'line') and hasattr(e, 'column') and e.line is not None:
                lines = content.split('\n')
                line_num = int(e.line) - 1
                if 0 <= line_num < len(lines):
                    print(f"\nLine {e.line}: {lines[line_num]}")
                    print(f"{' ' * max(0, int(e.column) - 1)}^")
            sys.exit(1)
        except (exceptions.UnexpectedCharacters, exceptions.UnexpectedToken) as e:
            print_error(f"Parsing failed: {str(e)}")
            if hasattr(e, 'line') and hasattr(e, 'column') and e.line is not None:
                lines = content.split('\n')
                line_num = int(e.line) - 1
                if 0 <= line_num < len(lines):
                    print(f"\nLine {e.line}: {lines[line_num]}")
                    print(f"{' ' * max(0, int(e.column) - 1)}^")
            sys.exit(1)
        except (PermissionError, IOError) as e:
            print_error(f"Failed to read {file}: {str(e)}")
            print_info("Check file permissions and try again")
            sys.exit(1)
        except ImportError as e:
            print_error(f"Failed to import parser: {str(e)}")
            print_info("Ensure neural parser dependencies are installed: pip install -e .")
            sys.exit(1)

    # Run HPO if requested
    if hpo:
        print_info("Running hyperparameter optimization")
        if dataset not in SUPPORTED_DATASETS:
            print_warning(f"Dataset '{dataset}' may not be supported. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}")

        try:
            optimize_and_return = get_module(hpo_module).optimize_and_return
            generate_optimized_dsl = get_module(code_generator_module).generate_optimized_dsl
            with Spinner("Optimizing hyperparameters") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                best_params = optimize_and_return(content, n_trials=3, dataset_name=dataset, backend=backend)
            print_success("Hyperparameter optimization complete!")
            print(f"\n{Colors.CYAN}Best Parameters:{Colors.ENDC}")
            for param, value in best_params.items():
                print(f"  {Colors.BOLD}{param}:{Colors.ENDC} {value}")
            with Spinner("Generating optimized DSL code") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                content = generate_optimized_dsl(content, best_params)
        except ImportError as e:
            print_error(f"HPO dependencies not available: {str(e)}")
            print_info("Install HPO dependencies: pip install -e \".[hpo]\"")
            sys.exit(1)
        except Exception as e:
            print_error(f"HPO failed: {str(e)}")
            if ctx.obj.get('VERBOSE'):
                import traceback
                traceback.print_exc()
            sys.exit(1)

    # Generate code
    try:
        generate_code = get_module(code_generator_module).generate_code
        with Spinner(f"Generating {backend} code") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            code = generate_code(model_data, backend, auto_flatten_output=auto_flatten_output)
    except ImportError as e:
        print_error(f"Code generator not available: {str(e)}")
        print_info("Install code generation dependencies: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print_error(f"Code generation failed: {str(e)}")
        if ctx.obj.get('VERBOSE'):
            import traceback
            traceback.print_exc()
        print_info(f"Hint: Check your model definition for {backend}-specific constraints")
        sys.exit(1)

    # Output the generated code
    if dry_run:
        print_info("Generated code (dry run)")
        print(f"\n{Colors.CYAN}{'='*50}{Colors.ENDC}")
        print(code)
        print(f"{Colors.CYAN}{'='*50}{Colors.ENDC}")
        print_success("Dry run complete! No files were created.")
    else:
        output_file = output or f"{os.path.splitext(file)[0]}_{backend}.py"
        output_file = os.path.normpath(output_file)
        try:
            with Spinner(f"Writing code to {output_file}") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(code)
            print_success(f"Compilation successful!")
            print(f"\n{Colors.CYAN}Output:{Colors.ENDC}")
            print(f"  {Colors.BOLD}File:{Colors.ENDC} {output_file}")
            print(f"  {Colors.BOLD}Backend:{Colors.ENDC} {backend}")
            print(f"  {Colors.BOLD}Size:{Colors.ENDC} {len(code)} bytes")
            if not ctx.obj.get('NO_ANIMATIONS'):
                print("\nNeural network structure:")
                animate_neural_network(2)
        except (PermissionError, IOError) as e:
            print_error(f"Failed to write to {output_file}: {str(e)}")
            print_info("Check directory permissions and disk space")
            sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--gradients', is_flag=True, help='Analyze gradient flow')
@click.option('--dead-neurons', is_flag=True, help='Detect dead neurons')
@click.option('--anomalies', is_flag=True, help='Detect training anomalies')
@click.option('--step', is_flag=True, help='Enable step debugging mode')
@click.option('--backend', '-b', default='tensorflow', help='Backend for runtime', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
@click.option('--dashboard', '-d', is_flag=True, help='Start the NeuralDbg dashboard')
@click.option('--port', default=8050, help='Port for the dashboard server')
@click.pass_context
def debug(ctx: click.Context, file: str, gradients: bool, dead_neurons: bool, anomalies: bool, step: bool, backend: str, dataset: str, dashboard: bool, port: int) -> None:
    """Debug a neural network model with NeuralDbg."""
    print_command_header("debug")
    
    # Input validation and path normalization
    try:
        file = os.path.normpath(file)
        backend = validate_backend(backend)
        dataset = validate_dataset_name(dataset)
        port = validate_port(port)
    except ValueError as e:
        print_error(f"Invalid input: {str(e)}")
        sys.exit(1)
    
    print_info(f"Debugging {file} with NeuralDbg (backend: {backend})")

    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}. Supported: .neural, .nr, .rnr")
        sys.exit(1)

    if dataset not in SUPPORTED_DATASETS:
        print_warning(f"Dataset '{dataset}' may not be supported. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}")

    # Parse the Neural DSL file
    try:
        from neural.parser.parser import create_parser, ModelTransformer
        with Spinner("Parsing Neural DSL file") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            parser_instance = create_parser(start_rule=start_rule)
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = parser_instance.parse(content)
            model_data = ModelTransformer().transform(tree)
    except (exceptions.LarkError, IOError, PermissionError) as e:
        print_error(f"Processing {file} failed: {str(e)}")
        sys.exit(1)
    except ImportError as e:
        print_error(f"Parser not available: {str(e)}")
        print_info("Install dependencies: pip install -e .")
        sys.exit(1)

    # Shape propagation
    try:
        ShapePropagator = get_module(shape_propagator_module).ShapePropagator
        with Spinner("Propagating shapes through the network") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            propagator = ShapePropagator(debug=True)
            input_shape = model_data['input']['shape']
            for layer in model_data['layers']:
                input_shape = propagator.propagate(input_shape, layer, backend)
            trace_data = propagator.get_trace()
    except ImportError as e:
        print_error(f"Shape propagator not available: {str(e)}")
        print_info("Install dependencies: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print_error(f"Shape propagation failed: {str(e)}")
        sys.exit(1)

    print_success("Model analysis complete!")

    # Collect real metrics if any debugging mode is enabled or dashboard is requested
    if gradients or dead_neurons or anomalies or dashboard:
        print_info("Collecting real metrics by training the model...")

        try:
            # Import the real metrics collector
            from neural.metrics.real_metrics import collect_real_metrics

            # Collect real metrics
            trace_data = collect_real_metrics(model_data, trace_data, backend, dataset)

            print_success("Real metrics collected successfully!")

        except ImportError:
            print_warning("Real metrics collection module not found. Using simulated metrics.")
            # Generate simulated metrics
            for entry in trace_data:
                layer_type = entry.get('layer', '')

                # Gradient flow metrics
                if 'Conv' in layer_type:
                    entry['grad_norm'] = np.random.uniform(0.3, 0.7)
                elif 'Dense' in layer_type or 'Output' in layer_type:
                    entry['grad_norm'] = np.random.uniform(0.5, 1.0)
                elif 'Pool' in layer_type:
                    entry['grad_norm'] = np.random.uniform(0.1, 0.3)
                else:
                    entry['grad_norm'] = np.random.uniform(0.2, 0.5)

                # Dead neuron metrics
                if 'ReLU' in layer_type or 'Conv' in layer_type:
                    entry['dead_ratio'] = np.random.uniform(0.05, 0.2)
                elif 'Dense' in layer_type:
                    entry['dead_ratio'] = np.random.uniform(0.01, 0.1)
                else:
                    entry['dead_ratio'] = np.random.uniform(0.0, 0.05)

                # Activation metrics
                if 'ReLU' in layer_type:
                    entry['mean_activation'] = np.random.uniform(0.3, 0.7)
                elif 'Sigmoid' in layer_type:
                    entry['mean_activation'] = np.random.uniform(0.4, 0.6)
                elif 'Tanh' in layer_type:
                    entry['mean_activation'] = np.random.uniform(-0.3, 0.3)
                elif 'Softmax' in layer_type or 'Output' in layer_type:
                    entry['mean_activation'] = np.random.uniform(0.1, 0.3)
                else:
                    entry['mean_activation'] = np.random.uniform(0.2, 0.8)

                # Anomaly detection
                if np.random.random() > 0.9:
                    entry['anomaly'] = True
                    if np.random.random() > 0.5:
                        entry['mean_activation'] = np.random.uniform(5.0, 15.0)
                    else:
                        entry['mean_activation'] = np.random.uniform(0.0001, 0.01)
                else:
                    entry['anomaly'] = False

            print_success("Simulated metrics generated successfully!")

        except Exception as e:
            print_error(f"Failed to collect metrics: {str(e)}")

            # Fallback to basic simulated metrics
            for entry in trace_data:
                entry['grad_norm'] = np.random.uniform(0.1, 1.0)
                entry['dead_ratio'] = np.random.uniform(0.0, 0.3)
                entry['mean_activation'] = np.random.uniform(0.3, 0.8)
                entry['anomaly'] = np.random.random() > 0.8

    # Display metrics in the console
    if gradients:
        print(f"\n{Colors.CYAN}Gradient Flow Analysis{Colors.ENDC}")
        for entry in trace_data:
            print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: grad_norm = {entry.get('grad_norm', 'N/A')}")

    if dead_neurons:
        print(f"\n{Colors.CYAN}Dead Neuron Detection{Colors.ENDC}")
        for entry in trace_data:
            print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: dead_ratio = {entry.get('dead_ratio', 'N/A')}")

    if anomalies:
        print(f"\n{Colors.CYAN}Anomaly Detection{Colors.ENDC}")
        anomaly_found = False
        for entry in trace_data:
            if entry.get('anomaly', False):
                print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: mean_activation = {entry.get('mean_activation', 'N/A')}, anomaly = {entry.get('anomaly', False)}")
                anomaly_found = True
        if not anomaly_found:
            print("  No anomalies detected")

    if step:
        print(f"\n{Colors.CYAN}Step Debugging Mode{Colors.ENDC}")
        print_info("Stepping through network layer by layer...")
        propagator = ShapePropagator(debug=True)
        input_shape = model_data['input']['shape']
        for i, layer in enumerate(model_data['layers']):
            input_shape = propagator.propagate(input_shape, layer, backend)
            print(f"\n{Colors.BOLD}Step {i+1}/{len(model_data['layers'])}{Colors.ENDC}: {layer['type']}")
            print(f"  Output Shape: {input_shape}")
            if 'params' in layer and layer['params']:
                print(f"  Parameters: {layer['params']}")
            if not ctx.obj.get('NO_ANIMATIONS') and click.confirm("Continue?", default=True):
                continue
            else:
                print_info("Debugging paused by user")
                break

    # Start the dashboard if requested
    if dashboard:
        try:
            print_info(f"Starting NeuralDbg dashboard on port {port}...")
            print_info(f"Dashboard URL: http://localhost:{port}")
            print_info("Press Ctrl+C to stop the dashboard")

            # Import the dashboard module
            try:
                from neural.dashboard import dashboard as dashboard_module
            except ImportError:
                # Try alternative import
                import neural.dashboard.dashboard as dashboard_module

            # Update the dashboard data using the update function
            if hasattr(dashboard_module, 'update_dashboard_data'):
                dashboard_module.update_dashboard_data(model_data, trace_data, backend)
                print_info("Dashboard data updated. Starting server...")
            else:
                print_warning("update_dashboard_data function not found, starting with default data")

            # Run the dashboard server
            if hasattr(dashboard_module, 'app'):
                dashboard_module.app.run_server(debug=False, host="localhost", port=port)
            else:
                print_error("Dashboard app not found in module")
                sys.exit(1)
        except ImportError as e:
            print_error(f"Dashboard module not found: {str(e)}")
            print_info("Make sure the dashboard dependencies are installed: pip install -e \".[dashboard]\"")
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to start dashboard: {str(e)}")
            if ctx.obj.get('VERBOSE'):
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        print_success("Debug session completed!")
        print_info("To start the dashboard, use the --dashboard flag")
        if not ctx.obj.get('NO_ANIMATIONS'):
            animate_neural_network(2)


if __name__ == '__main__':
    cli()
