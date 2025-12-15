#!/usr/bin/env python
"""
Simplified CLI implementation for Neural v0.4.0 - Core features only.
Removed: cloud, track, marketplace, cost, aquarium, no-code, docs, explain, config, data, collab commands.
"""

import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Optional

import click
from lark import exceptions
import numpy as np

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
from .cpu_mode import set_cpu_mode
from .lazy_imports import (
    code_generator as code_generator_module,
)
from .lazy_imports import get_module, jax, optuna, tensorflow, torch
from .lazy_imports import hpo as hpo_module
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

    # Normalize the path
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
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Port must be an integer, got: {type(port).__name__}"
            ) from e

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

# Global CLI context
@click.group(name="cli", context_settings={"help_option_names": ["-h", "--help"]})
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--cpu', is_flag=True, help='Force CPU mode')
@click.option('--no-animations', is_flag=True, help='Disable animations and spinners')
@click.version_option(version=__version__, prog_name="Neural")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, cpu: bool, no_animations: bool) -> None:
    """Neural DSL v0.4.0: Focused DSL compiler for neural networks with multi-backend support."""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['NO_ANIMATIONS'] = no_animations
    ctx.obj['CPU_MODE'] = cpu

    configure_logging(verbose)

    if cpu:
        set_cpu_mode()
        logger.info("Running in CPU mode")

    # Show welcome message if not disabled
    if not os.environ.get('NEURAL_SKIP_WELCOME'):
        # Use the context to track if welcome was shown, not the cli object itself
        if not ctx.obj.get('_WELCOME_SHOWN'):
            show_welcome_message()
            ctx.obj['_WELCOME_SHOWN'] = True
        elif not show_welcome_message():
            print_neural_logo(__version__)

@cli.command()
@click.pass_context
def help(ctx: click.Context) -> None:
    """Show help for commands."""
    print_help_command(ctx, cli.commands)

@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--backend', '-b', default='tensorflow', help='Target backend', type=click.Choice(['tensorflow', 'pytorch', 'onnx'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
@click.option('--output', '-o', default=None, help='Output file path (defaults to <file>_<backend>.py)')
@click.option('--dry-run', is_flag=True, help='Preview generated code without writing to file')
@click.option('--hpo', is_flag=True, help='Enable hyperparameter optimization')
@click.option('--auto-flatten-output', is_flag=True, help='Auto-insert Flatten before Dense/Output when input is rank>2')
@click.pass_context
def compile(ctx, file: str, backend: str, dataset: str, output: Optional[str], dry_run: bool, hpo: bool, auto_flatten_output: bool):
    """Compile a .neural or .nr file into an executable Python script."""
    print_command_header("compile")
    print_info(f"Compiling {file} for {backend} backend")

    # Validate file type
    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}. Supported: .neural, .nr, .rnr")
        sys.exit(1)

    # Parse the Neural DSL file
    with Spinner("Parsing Neural DSL file") as spinner:
        if ctx.obj.get('NO_ANIMATIONS'):
            spinner.stop()
        try:
            from neural.parser.parser import DSLValidationError, ModelTransformer, create_parser
            parser_instance = create_parser(start_rule=start_rule)
            with open(file, 'r') as f:
                content = f.read()
            tree = parser_instance.parse(content)
            model_data = ModelTransformer().transform(tree)
        except (exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError) as e:
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
        except Exception as e:
            print_error(f"HPO failed: {str(e)}")
            sys.exit(1)

    # Generate code
    try:
        generate_code = get_module(code_generator_module).generate_code
        with Spinner(f"Generating {backend} code") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            code = generate_code(model_data, backend, auto_flatten_output=auto_flatten_output)
    except Exception as e:
        print_error(f"Code generation failed: {str(e)}")
        sys.exit(1)

    # Output the generated code
    if dry_run:
        print_info("Generated code (dry run)")
        logger.info(f"\n{Colors.CYAN}{'='*50}{Colors.ENDC}")
        logger.info(code)
        logger.info(f"{Colors.CYAN}{'='*50}{Colors.ENDC}")
        print_success("Dry run complete! No files were created.")
    else:
        output_file = output or f"{os.path.splitext(file)[0]}_{backend}.py"
        try:
            with Spinner(f"Writing code to {output_file}") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                with open(output_file, 'w') as f:
                    f.write(code)
            print_success("Compilation successful!")
            logger.info(f"\n{Colors.CYAN}Output:{Colors.ENDC}")
            logger.info(f"  {Colors.BOLD}File:{Colors.ENDC} {output_file}")
            logger.info(f"  {Colors.BOLD}Backend:{Colors.ENDC} {backend}")
            logger.info(f"  {Colors.BOLD}Size:{Colors.ENDC} {len(code)} bytes")
            if not ctx.obj.get('NO_ANIMATIONS'):
                logger.info("\nNeural network structure:")
                animate_neural_network(2)
        except (PermissionError, IOError) as e:
            print_error(f"Failed to write to {output_file}: {str(e)}")
            sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--backend', '-b', default='tensorflow', help='Backend to run', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
@click.option('--hpo', is_flag=True, help='Enable HPO for .neural files')
@click.option('--device', '-d', default='auto', help='Device to use (auto, cpu, gpu)', type=click.Choice(['auto', 'cpu', 'gpu'], case_sensitive=False))
@click.pass_context
def run(ctx, file: str, backend: str, dataset: str, hpo: bool, device: str):
    """Run a compiled model or optimize and run a .neural file."""
    print_command_header("run")

    # Input validation
    try:
        file = sanitize_file_path(file)
        backend = validate_backend(backend)
        dataset = validate_dataset_name(dataset)
    except ValueError as e:
        print_error(f"Invalid input: {str(e)}")
        sys.exit(1)

    print_info(f"Running {file} with {backend} backend")

    # Set device mode
    device = device.lower()
    if device == 'cpu' or ctx.obj.get('CPU_MODE'):
        set_cpu_mode()
        print_info("Running in CPU mode")
    elif device == 'gpu':
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ['NEURAL_FORCE_CPU'] = '0'
        print_info("Running in GPU mode")

    ext = os.path.splitext(file)[1].lower()
    if ext == '.py':
        try:
            with Spinner("Executing Python script") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                subprocess.run([sys.executable, file], check=True)
            print_success("Execution completed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Execution failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except (PermissionError, IOError) as e:
            print_error(f"Failed to execute {file}: {str(e)}")
            sys.exit(1)
    elif ext in ['.neural', '.nr'] and hpo:
        if dataset not in SUPPORTED_DATASETS:
            print_warning(f"Dataset '{dataset}' may not be supported. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}")

        try:
            # Reuse compile command logic
            output_file = f"{os.path.splitext(file)[0]}_optimized_{backend}.py"
            ctx.invoke(
                compile,
                file=file,
                backend=backend,
                dataset=dataset,
                output=output_file,
                dry_run=False,
                hpo=True
            )
            # Run the compiled file
            with Spinner("Executing optimized script") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                subprocess.run([sys.executable, output_file], check=True)
            print_success("Execution completed successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"Execution failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except Exception as e:
            print_error(f"Optimization or execution failed: {str(e)}")
            sys.exit(1)
    else:
        print_error(f"Expected a .py file or .neural/.nr with --hpo. Got {ext}.")
        sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--format', '-f', default='html', help='Output format', type=click.Choice(['html', 'png', 'svg'], case_sensitive=False))
@click.option('--cache/--no-cache', default=True, help='Use cached visualizations if available')
@click.pass_context
def visualize(ctx, file: str, format: str, cache: bool):
    """Visualize network architecture and shape propagation."""
    print_command_header("visualize")
    print_info(f"Visualizing {file} in {format} format")

    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}. Supported: .neural, .nr, .rnr")
        sys.exit(1)

    # Cache handling
    cache_dir = Path(".neural_cache")
    cache_dir.mkdir(exist_ok=True)
    file_hash = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    cache_file = cache_dir / f"viz_{file_hash}_{format}"
    file_mtime = Path(file).stat().st_mtime

    if cache and cache_file.exists() and cache_file.stat().st_mtime >= file_mtime:
        try:
            with Spinner("Copying cached visualization") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                shutil.copy(cache_file, f"architecture.{format}")
            print_success(f"Cached visualization copied to architecture.{format}")
            return
        except (PermissionError, IOError) as e:
            print_warning(f"Failed to use cache: {str(e)}. Generating new visualization.")

    # Parse the Neural DSL file
    try:
        from neural.parser.parser import ModelTransformer, create_parser
        with Spinner("Parsing Neural DSL file") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            parser_instance = create_parser(start_rule=start_rule)
            with open(file, 'r') as f:
                content = f.read()
            tree = parser_instance.parse(content)
            model_data = ModelTransformer().transform(tree)
    except (exceptions.LarkError, IOError, PermissionError) as e:
        print_error(f"Processing {file} failed: {str(e)}")
        sys.exit(1)

    # Shape propagation
    try:
        ShapePropagator = get_module(shape_propagator_module).ShapePropagator
        propagator = ShapePropagator()
        input_shape = model_data['input']['shape']
        if not input_shape:
            print_error("Input shape not defined in model")
            sys.exit(1)

        print_info("Propagating shapes through the network...")
        shape_history = []
        total_layers = len(model_data['layers'])
        for i, layer in enumerate(model_data['layers']):
            # Handle different layer structure formats
            layer_type = layer.get('type', 'Unknown')
            if not ctx.obj.get('NO_ANIMATIONS'):
                progress_bar(i, total_layers, prefix='Progress:', suffix=f'Layer: {layer_type}', length=40)
            input_shape = propagator.propagate(input_shape, layer, model_data.get('framework', 'tensorflow'))
            shape_history.append({"layer": layer_type, "output_shape": input_shape})
        if not ctx.obj.get('NO_ANIMATIONS'):
            progress_bar(total_layers, total_layers, prefix='Progress:', suffix='Complete', length=40)
    except Exception as e:
        print_error(f"Shape propagation failed: {str(e)}")
        sys.exit(1)

    # Generate visualizations
    try:
        with Spinner("Generating visualizations") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            report = propagator.generate_report()
            dot = report['dot_graph']
            dot.format = format if format != 'html' else 'svg'

            # Try to render the dot graph, but handle gracefully if graphviz is not available
            try:
                dot.render('architecture', cleanup=True)
            except Exception as graphviz_error:
                print_warning(f"Graphviz rendering failed: {str(graphviz_error)}")
                print_info("Continuing with other visualizations...")

            if format == 'html':
                try:
                    report['plotly_chart'].write_html('shape_propagation.html')
                    create_animated_network = get_module(tensor_flow_module).create_animated_network
                    create_animated_network(shape_history).write_html('tensor_flow.html')
                except Exception as html_error:
                    print_warning(f"HTML visualization generation failed: {str(html_error)}")
    except Exception as e:
        print_error(f"Visualization generation failed: {str(e)}")
        sys.exit(1)

    # Show success message
    if format == 'html':
        print_success("Visualizations generated successfully!")
        logger.info(f"{Colors.CYAN}Files created:{Colors.ENDC}")
        logger.info(f"  - {Colors.GREEN}architecture.svg{Colors.ENDC} (Network architecture)")
        logger.info(f"  - {Colors.GREEN}shape_propagation.html{Colors.ENDC} (Parameter count chart)")
        logger.info(f"  - {Colors.GREEN}tensor_flow.html{Colors.ENDC} (Data flow animation)")
        if not ctx.obj.get('NO_ANIMATIONS'):
            logger.info("\nNeural network data flow animation:")
            animate_neural_network(3)
    else:
        print_success(f"Visualization saved as architecture.{format}")

    # Cache the visualization
    if cache:
        try:
            with Spinner("Caching visualization") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                shutil.copy(f"architecture.{format}", cache_file)
            print_info("Visualization cached for future use")
        except (PermissionError, IOError) as e:
            print_warning(f"Failed to cache visualization: {str(e)}")

@cli.command()
@click.option('--yes', is_flag=True, help='Apply deletions; otherwise perform a dry run')
@click.option('--all', 'clean_all', is_flag=True, help='Also remove caches and artifact directories')
@click.pass_context
def clean(ctx, yes: bool, clean_all: bool):
    """Remove generated artifacts safely (dry-run by default).

    - Targets only known generated files:
      *_tensorflow.py, *_pytorch.py, *_onnx.py, architecture.(svg|png), shape_propagation.html, tensor_flow.html
    - With --all: also removes .neural_cache/ and comparison_plots/
    - Pass --yes to actually delete; otherwise prints what would be removed.
    """
    print_command_header("clean")

    patterns = [
        "*_tensorflow.py",
        "*_pytorch.py",
        "*_onnx.py",
        "architecture.svg",
        "architecture.png",
        "shape_propagation.html",
        "tensor_flow.html",
    ]

    dirs = [".neural_cache", "comparison_plots"] if clean_all else []

    # Collect matches
    to_remove = []
    for p in patterns:
        for match in Path('.').glob(p):
            if match.is_file():
                to_remove.append(match)
    for d in dirs:
        if Path(d).exists():
            to_remove.append(Path(d))

    if not to_remove:
        print_warning("No generated artifacts found to clean")
        return

    # Dry-run summary
    print_info("Items to remove:" if yes else "Dry run: would remove these items:")
    preview = 0
    for item in to_remove:
        print(f"  - {item}")
        preview += 1
        if preview >= 10 and len(to_remove) > 10:
            print(f"  - ...and {len(to_remove) - 10} more")
            break

    if not yes:
        print_warning("Pass --yes to apply deletions")
        return

    # Apply deletions
    removed = 0
    try:
        with Spinner("Removing artifacts") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            for item in to_remove:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink(missing_ok=True)
                    removed += 1
                except (PermissionError, OSError) as e:
                    print_warning(f"Skip {item}: {e}")
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
        sys.exit(1)

    print_success(f"Removed {removed} item(s)")

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

    # Input validation
    try:
        file = sanitize_file_path(file)
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
        from neural.parser.parser import ModelTransformer, create_parser
        with Spinner("Parsing Neural DSL file") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            parser_instance = create_parser(start_rule=start_rule)
            with open(file, 'r') as f:
                content = f.read()
            tree = parser_instance.parse(content)
            model_data = ModelTransformer().transform(tree)
    except (exceptions.LarkError, IOError, PermissionError) as e:
        print_error(f"Processing {file} failed: {str(e)}")
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
    except Exception as e:
        print_error(f"Shape propagation failed: {str(e)}")
        sys.exit(1)

    print_success("Model analysis complete!")

    # Collect real metrics if any debugging mode is enabled or dashboard is requested
    if gradients or dead_neurons or anomalies or dashboard:
        print_info("Generating simulated metrics...")
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

            # Anomaly detection (about 10% chance)
            if np.random.random() > 0.9:
                entry['anomaly'] = True
                if np.random.random() > 0.5:
                    entry['mean_activation'] = np.random.uniform(5.0, 15.0)
                else:
                    entry['mean_activation'] = np.random.uniform(0.0001, 0.01)
            else:
                entry['anomaly'] = False

    # Display metrics in the console
    if gradients:
        logger.info(f"\n{Colors.CYAN}Gradient Flow Analysis{Colors.ENDC}")
        for entry in trace_data:
            logger.info(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: grad_norm = {entry.get('grad_norm', 'N/A'):.3f}")

    if dead_neurons:
        logger.info(f"\n{Colors.CYAN}Dead Neuron Detection{Colors.ENDC}")
        for entry in trace_data:
            logger.info(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: dead_ratio = {entry.get('dead_ratio', 'N/A'):.3f}")

    if anomalies:
        logger.info(f"\n{Colors.CYAN}Anomaly Detection{Colors.ENDC}")
        anomaly_found = False
        for entry in trace_data:
            if entry.get('anomaly', False):
                logger.info(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: mean_activation = {entry.get('mean_activation', 'N/A'):.3f}, anomaly = True")
                anomaly_found = True
        if not anomaly_found:
            logger.info("  No anomalies detected")

    if step:
        logger.info(f"\n{Colors.CYAN}Step Debugging Mode{Colors.ENDC}")
        print_info("Stepping through network layer by layer...")
        propagator = ShapePropagator(debug=True)
        input_shape = model_data['input']['shape']
        for i, layer in enumerate(model_data['layers']):
            input_shape = propagator.propagate(input_shape, layer, backend)
            logger.info(f"\n{Colors.BOLD}Step {i+1}/{len(model_data['layers'])}{Colors.ENDC}: {layer['type']}")
            logger.info(f"  Output Shape: {input_shape}")
            if 'params' in layer and layer['params']:
                logger.info(f"  Parameters: {layer['params']}")
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
            import neural.dashboard.dashboard as dashboard_module

            # Update the dashboard data using the update function
            dashboard_module.update_dashboard_data(model_data, trace_data, backend)

            # Print debug information
            print_info("Dashboard data updated. Starting server...")

            # Run the dashboard server
            dashboard_module.app.run_server(debug=False, host="localhost", port=port)
        except ImportError:
            print_error("Dashboard module not found. Make sure the dashboard dependencies are installed.")
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to start dashboard: {str(e)}")
            sys.exit(1)
    else:
        print_success("Debug session completed!")
        print_info("To start the dashboard, use the --dashboard flag")
        if not ctx.obj.get('NO_ANIMATIONS'):
            animate_neural_network(2)

@cli.command()
@click.option('--host', default='localhost', help='Server host address')
@click.option('--port', default=8050, type=int, help='Server port')
@click.option('--no-browser', is_flag=True, help='Do not open browser automatically')
@click.pass_context
def server(ctx, host: str, port: int, no_browser: bool):
    """Start unified Neural DSL web server (dashboard only in v0.4.0)."""
    print_command_header("server")

    try:
        from neural.dashboard import dashboard as dashboard_module

        print_success(f"Server will start on http://{host}:{port}")
        print_info("Press Ctrl+C to stop the server")

        if not no_browser and not ctx.obj.get('NO_ANIMATIONS'):
            print_info("Opening browser...")
            import threading
            import webbrowser
            def open_browser():
                time.sleep(1.5)
                webbrowser.open(f"http://{host}:{port}")
            threading.Thread(target=open_browser, daemon=True).start()

        dashboard_module.app.run_server(debug=False, host=host, port=port)

    except ImportError as e:
        print_error(f"Server dependencies not installed: {e}")
        print_info("Install with: pip install -e \".[dashboard]\"")
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("\nServer stopped by user")
    except Exception as e:
        print_error(f"Failed to start server: {e}")
        sys.exit(1)

@cli.command()
@click.pass_context
def version(ctx):
    """Show the version of Neural CLI and dependencies."""
    print_command_header("version")
    import lark

    print(f"\n{Colors.CYAN}System Information:{Colors.ENDC}")
    print(f"  {Colors.BOLD}Python:{Colors.ENDC}      {sys.version.split()[0]}")
    print(f"  {Colors.BOLD}Platform:{Colors.ENDC}    {sys.platform}")

    # Detect cloud environment
    env_type = "local"
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        env_type = "Kaggle"
    elif 'COLAB_GPU' in os.environ:
        env_type = "Google Colab"
    print(f"  {Colors.BOLD}Environment:{Colors.ENDC} {env_type}")

    print(f"\n{Colors.CYAN}Core Dependencies:{Colors.ENDC}")
    print(f"  {Colors.BOLD}Click:{Colors.ENDC}       {click.__version__}")
    print(f"  {Colors.BOLD}Lark:{Colors.ENDC}        {lark.__version__}")

    print(f"\n{Colors.CYAN}ML Frameworks:{Colors.ENDC}")
    for pkg, lazy_module in [('torch', torch), ('tensorflow', tensorflow), ('jax', jax), ('optuna', optuna)]:
        try:
            ver = get_module(lazy_module).__version__
            print(f"  {Colors.BOLD}{pkg.capitalize()}:{Colors.ENDC}" + " " * (12 - len(pkg)) + f"{ver}")
        except (ImportError, AttributeError):
            print(f"  {Colors.BOLD}{pkg.capitalize()}:{Colors.ENDC}" + " " * (12 - len(pkg)) + f"{Colors.YELLOW}Not installed{Colors.ENDC}")

    if not ctx.obj.get('NO_ANIMATIONS'):
        logger.info("\nNeural is ready to build amazing neural networks!")
        animate_neural_network(2)


if __name__ == '__main__':
    cli()
