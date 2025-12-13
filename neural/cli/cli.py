#!/usr/bin/env python
"""
Main CLI implementation for Neural using Click.
"""

import hashlib
import json
import logging
import os
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
            from neural.parser.parser import create_parser, ModelTransformer, DSLValidationError
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
    output_file = output or f"{os.path.splitext(file)[0]}_{backend}.py"
    if dry_run:
        print_info("Generated code (dry run)")
        print(f"\n{Colors.CYAN}{'='*50}{Colors.ENDC}")
        print(code)
        print(f"{Colors.CYAN}{'='*50}{Colors.ENDC}")
    else:
        try:
            with Spinner(f"Writing code to {output_file}") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                with open(output_file, 'w') as f:
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
            sys.exit(1)
@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--output', '-o', default='model.md', help='Output Markdown file')
@click.option('--pdf', is_flag=True, help='Also export to PDF via Pandoc if available')
@click.pass_context
def docs(ctx, file: str, output: str, pdf: bool):
    """Generate Markdown (and optionally PDF) documentation with math and shapes."""
    print_command_header("docs")
    
    # Input validation
    try:
        file = sanitize_file_path(file)
        output = sanitize_file_path(output)
    except ValueError as e:
        print_error(f"Invalid input: {str(e)}")
        sys.exit(1)
    
    print_info(f"Generating documentation for {file}")

    # Parse the Neural DSL file (same start rule detection as compile)
    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}. Supported: .neural, .nr, .rnr")
        sys.exit(1)

    try:
        from neural.parser.parser import create_parser, ModelTransformer
        parser_instance = create_parser(start_rule=start_rule)
        with open(file, 'r') as f:
            content = f.read()
        tree = parser_instance.parse(content)
        model_data = ModelTransformer().transform(tree)
    except Exception as e:
        print_error(f"Parsing failed: {str(e)}")
        sys.exit(1)

    try:
        from neural.docgen.docgen import generate_markdown
        md = generate_markdown(model_data)
        with open(output, 'w', encoding='utf-8') as f:
            f.write(md)
        print_success(f"Wrote Markdown to {output}")

        if pdf:
            import shutil as _shutil
            pandoc = _shutil.which('pandoc')
            if not pandoc:
                print_warning("Pandoc not found; skipping PDF export")
            else:
                pdf_out = os.path.splitext(output)[0] + '.pdf'
                try:
                    subprocess.run([pandoc, output, '-o', pdf_out], check=True)
                    print_success(f"Wrote PDF to {pdf_out}")
                except subprocess.CalledProcessError as e:
                    print_warning(f"Pandoc failed with exit code {e.returncode}")
    except Exception as e:
        print_error(f"Doc generation failed: {str(e)}")
        sys.exit(1)



####Â RUN COMMAND #####

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
@click.option('--attention', is_flag=True, help='Visualize attention weights for transformer models')
@click.option('--backend', '-b', default='tensorflow', help='Backend for attention visualization', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--data', '-d', type=click.Path(exists=True), help='Input data file for attention visualization (.npy format)')
@click.option('--tokens', help='Comma-separated token labels for attention visualization')
@click.option('--layer', help='Specific attention layer to visualize')
@click.option('--head', type=int, help='Specific attention head to visualize')
@click.pass_context
def visualize(ctx, file: str, format: str, cache: bool, attention: bool, backend: str, data: Optional[str], tokens: Optional[str], layer: Optional[str], head: Optional[int]):
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
        from neural.parser.parser import create_parser, ModelTransformer
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
        print(f"{Colors.CYAN}Files created:{Colors.ENDC}")
        print(f"  - {Colors.GREEN}architecture.svg{Colors.ENDC} (Network architecture)")
        print(f"  - {Colors.GREEN}shape_propagation.html{Colors.ENDC} (Parameter count chart)")
        print(f"  - {Colors.GREEN}tensor_flow.html{Colors.ENDC} (Data flow animation)")
        if not ctx.obj.get('NO_ANIMATIONS'):
            print("\nNeural network data flow animation:")
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
    
    # Attention visualization for transformer models
    if attention:
        print_info("Visualizing attention weights...")
        
        try:
            from neural.explainability.attention_visualizer import AttentionVisualizer
            
            # Parse tokens if provided
            token_list = None
            if tokens:
                token_list = [t.strip() for t in tokens.split(',')]
            
            # Load or generate input data
            if data:
                input_data = np.load(data)
                print_info(f"Loaded input data: {input_data.shape}")
            else:
                print_warning("No input data provided. Generating synthetic data based on model input shape.")
                input_shape = model_data['input']['shape']
                if input_shape:
                    batch_size = 1
                    input_data = np.random.randn(batch_size, *input_shape).astype(np.float32)
                    print_info(f"Generated synthetic data with shape: {input_data.shape}")
                else:
                    print_error("Cannot generate synthetic data without input shape in model")
                    sys.exit(1)
            
            # Create output directory for attention visualizations
            attention_output = 'attention_outputs'
            os.makedirs(attention_output, exist_ok=True)
            
            # Use the AttentionVisualizer
            visualizer = AttentionVisualizer(model=None, backend=backend)
            
            with Spinner("Extracting and visualizing attention weights") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                
                results = visualizer.visualize_from_dsl(
                    dsl_file=file,
                    input_data=input_data,
                    backend=backend,
                    layer_name=layer,
                    head_index=head,
                    tokens=token_list,
                    output_dir=attention_output
                )
            
            if results['attention_weights']:
                print_success(f"Attention visualization complete!")
                print(f"\n{Colors.CYAN}Attention Analysis:{Colors.ENDC}")
                print(f"  {Colors.BOLD}Layers with attention:{Colors.ENDC} {len(results['attention_weights'])}")
                
                for layer_name, weights in results['attention_weights'].items():
                    print(f"  {Colors.BOLD}{layer_name}:{Colors.ENDC} {weights.shape}")
                    
                    # Analyze attention patterns
                    analysis = visualizer.analyze_attention_patterns(weights)
                    print(f"    - Num heads: {analysis['num_heads']}")
                    print(f"    - Avg entropy: {analysis['avg_entropy']:.3f}")
                    print(f"    - Avg max attention: {analysis['avg_max_attention']:.3f}")
                    print(f"    - Diagonal strength: {analysis['avg_diagonal_strength']:.3f}")
                
                print(f"\n{Colors.CYAN}Output files:{Colors.ENDC}")
                print(f"  - {Colors.GREEN}{attention_output}/attention_heatmap.png{Colors.ENDC}")
                
                for layer_name in results['attention_weights'].keys():
                    heads_file = f"{attention_output}/attention_heads_{layer_name}.png"
                    if os.path.exists(heads_file):
                        print(f"  - {Colors.GREEN}{heads_file}{Colors.ENDC}")
                
                # Create interactive visualization if plotly is available
                try:
                    interactive_path = visualizer.create_interactive_visualization(
                        results['attention_weights'],
                        tokens=token_list,
                        output_path=os.path.join(attention_output, 'attention_interactive.html')
                    )
                    if interactive_path:
                        print(f"  - {Colors.GREEN}{interactive_path}{Colors.ENDC} (interactive)")
                except Exception as e:
                    logger.debug(f"Could not create interactive visualization: {e}")
            else:
                print_warning("No attention layers found in the model")
                print_info("Make sure your model contains TransformerEncoder, MultiHeadAttention, or similar attention layers")
        
        except ImportError as e:
            print_error(f"Missing dependencies for attention visualization: {str(e)}")
            print_info("Install visualization dependencies: pip install matplotlib seaborn plotly")
            sys.exit(1)
        except Exception as e:
            print_error(f"Attention visualization failed: {str(e)}")
            if ctx.obj.get('VERBOSE'):
                import traceback
                traceback.print_exc()
            sys.exit(1)

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
        print("\nNeural is ready to build amazing neural networks!")
        animate_neural_network(2)

@cli.group()
@click.pass_context
def cloud(ctx: click.Context) -> None:
    """Commands for cloud integration."""
    pass

@cli.group()
@click.pass_context
def track(ctx):
    """Commands for experiment tracking."""
    pass

@track.command('init')
@click.argument('experiment_name', required=False)
@click.option('--base-dir', default='neural_experiments', help='Base directory for storing experiment data')
@click.option('--integration', type=click.Choice(['mlflow', 'wandb', 'tensorboard']), help='External tracking tool to use')
@click.option('--project-name', default='neural', help='Project name for W&B')
@click.option('--tracking-uri', default=None, help='MLflow tracking URI')
@click.option('--log-dir', default='runs/neural', help='TensorBoard log directory')
@click.pass_context
def track_init(ctx, experiment_name, base_dir, integration, project_name, tracking_uri, log_dir):
    """Initialize experiment tracking."""
    print_command_header("track init")

    try:
        # Import experiment tracker
        ExperimentManager = get_module(experiment_tracker_module).ExperimentManager
        create_integration = get_module(experiment_tracker_module).create_integration

        # Create experiment manager
        manager = ExperimentManager(base_dir=base_dir)

        # Create experiment
        experiment = manager.create_experiment(experiment_name=experiment_name)

        # Create integration if requested
        if integration:
            if integration == 'mlflow':
                integration_instance = create_integration('mlflow', experiment_name=experiment.experiment_name, tracking_uri=tracking_uri)
            elif integration == 'wandb':
                integration_instance = create_integration('wandb', experiment_name=experiment.experiment_name, project_name=project_name)
            elif integration == 'tensorboard':
                integration_instance = create_integration('tensorboard', experiment_name=experiment.experiment_name, log_dir=log_dir)

            # Store integration info in experiment metadata
            experiment.metadata['integration'] = {
                'type': integration,
                'config': {
                    'project_name': project_name if integration == 'wandb' else None,
                    'tracking_uri': tracking_uri if integration == 'mlflow' else None,
                    'log_dir': log_dir if integration == 'tensorboard' else None
                }
            }
            experiment._save_metadata()

        # Save experiment ID to a file for easy access
        with open('.neural_experiment', 'w') as f:
            f.write(experiment.experiment_id)

        print_success(f"Initialized experiment: {experiment.experiment_name} (ID: {experiment.experiment_id})")
        print(f"\n{Colors.CYAN}Experiment Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Name:{Colors.ENDC}      {experiment.experiment_name}")
        print(f"  {Colors.BOLD}ID:{Colors.ENDC}        {experiment.experiment_id}")
        print(f"  {Colors.BOLD}Directory:{Colors.ENDC} {experiment.experiment_dir}")
        if integration:
            print(f"  {Colors.BOLD}Integration:{Colors.ENDC} {integration}")

    except Exception as e:
        print_error(f"Failed to initialize experiment tracking: {str(e)}")
        sys.exit(1)

@track.command('log')
@click.option('--experiment-id', default=None, help='Experiment ID (defaults to the current experiment)')
@click.option('--hyperparameters', '-p', help='Hyperparameters as JSON string')
@click.option('--hyperparameters-file', '-f', type=click.Path(exists=True), help='Path to JSON file with hyperparameters')
@click.option('--metrics', '-m', help='Metrics as JSON string')
@click.option('--metrics-file', type=click.Path(exists=True), help='Path to JSON file with metrics')
@click.option('--step', type=int, default=None, help='Step or epoch number for metrics')
@click.option('--artifact', type=click.Path(exists=True), help='Path to artifact file')
@click.option('--artifact-name', help='Name for the artifact (defaults to filename)')
@click.option('--model', type=click.Path(exists=True), help='Path to model file')
@click.option('--framework', default='unknown', help='Framework used for the model')
@click.pass_context
def track_log(ctx, experiment_id, hyperparameters, hyperparameters_file, metrics, metrics_file, step, artifact, artifact_name, model, framework):
    """Log data to an experiment."""
    print_command_header("track log")

    try:
        # Get experiment ID from file if not provided
        if not experiment_id and os.path.exists('.neural_experiment'):
            with open('.neural_experiment', 'r') as f:
                experiment_id = f.read().strip()

        if not experiment_id:
            print_error("No experiment ID provided and no current experiment found")
            print_info("Initialize an experiment first with 'neural track init'")
            sys.exit(1)

        # Import experiment tracker
        ExperimentManager = get_module(experiment_tracker_module).ExperimentManager

        # Get experiment
        manager = ExperimentManager()
        experiment = manager.get_experiment(experiment_id)

        if not experiment:
            print_error(f"Experiment not found: {experiment_id}")
            sys.exit(1)

        # Log hyperparameters
        if hyperparameters or hyperparameters_file:
            if hyperparameters_file:
                with open(hyperparameters_file, 'r') as f:
                    hyperparams = json.load(f)
            else:
                hyperparams = json.loads(hyperparameters)

            experiment.log_hyperparameters(hyperparams)
            print_success(f"Logged {len(hyperparams)} hyperparameters")

        # Log metrics
        if metrics or metrics_file:
            if metrics_file:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            else:
                metrics_data = json.loads(metrics)

            experiment.log_metrics(metrics_data, step=step)
            print_success(f"Logged {len(metrics_data)} metrics" + (f" at step {step}" if step is not None else ""))

        # Log artifact
        if artifact:
            experiment.log_artifact(artifact, artifact_name=artifact_name)
            print_success(f"Logged artifact: {artifact}")

        # Log model
        if model:
            experiment.log_model(model, framework=framework)
            print_success(f"Logged {framework} model: {model}")

    except Exception as e:
        print_error(f"Failed to log data: {str(e)}")
        sys.exit(1)

@track.command('list')
@click.option('--base-dir', default='neural_experiments', help='Base directory for experiments')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def track_list(ctx, base_dir, format):
    """List all experiments."""
    print_command_header("track list")

    try:
        # Import experiment tracker
        ExperimentManager = get_module(experiment_tracker_module).ExperimentManager

        # List experiments
        manager = ExperimentManager(base_dir=base_dir)
        experiments = manager.list_experiments()

        if not experiments:
            print_warning("No experiments found")
            return

        if format == 'json':
            import json
            print(json.dumps(experiments, indent=2))
        else:
            print(f"\n{Colors.CYAN}Experiments:{Colors.ENDC}")
            print(f"  {Colors.BOLD}{'Name':<20} {'ID':<10} {'Status':<10} {'Start Time':<20}{Colors.ENDC}")
            print(f"  {'-' * 60}")
            for exp in experiments:
                name = exp['experiment_name'][:18] + '..' if len(exp['experiment_name']) > 20 else exp['experiment_name']
                status = exp['status']
                status_color = Colors.GREEN if status == 'completed' else Colors.YELLOW if status == 'running' else Colors.RED if status == 'failed' else Colors.ENDC
                start_time = exp['start_time'].split('T')[0] + ' ' + exp['start_time'].split('T')[1][:8] if 'T' in exp['start_time'] else exp['start_time']
                print(f"  {name:<20} {exp['experiment_id']:<10} {status_color}{status:<10}{Colors.ENDC} {start_time:<20}")

    except Exception as e:
        print_error(f"Failed to list experiments: {str(e)}")
        sys.exit(1)

@track.command('show')
@click.argument('experiment_id')
@click.option('--base-dir', default='neural_experiments', help='Base directory for experiments')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def track_show(ctx, experiment_id, base_dir, format):
    """Show details of an experiment."""
    print_command_header("track show")

    try:
        # Import experiment tracker
        ExperimentManager = get_module(experiment_tracker_module).ExperimentManager

        # Get experiment
        manager = ExperimentManager(base_dir=base_dir)
        experiment = manager.get_experiment(experiment_id)

        if not experiment:
            print_error(f"Experiment not found: {experiment_id}")
            sys.exit(1)

        # Generate summary
        summary_path = experiment.save_experiment_summary()

        # Load summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        if format == 'json':
            print(json.dumps(summary, indent=2))
        else:
            print(f"\n{Colors.CYAN}Experiment Details:{Colors.ENDC}")
            print(f"  {Colors.BOLD}Name:{Colors.ENDC}      {summary['experiment_name']}")
            print(f"  {Colors.BOLD}ID:{Colors.ENDC}        {summary['experiment_id']}")
            print(f"  {Colors.BOLD}Status:{Colors.ENDC}    {summary['metadata']['status']}")
            print(f"  {Colors.BOLD}Start Time:{Colors.ENDC} {summary['metadata']['start_time']}")
            if 'end_time' in summary['metadata']:
                print(f"  {Colors.BOLD}End Time:{Colors.ENDC}   {summary['metadata']['end_time']}")

            if summary['hyperparameters']:
                print(f"\n{Colors.CYAN}Hyperparameters:{Colors.ENDC}")
                for name, value in summary['hyperparameters'].items():
                    print(f"  {Colors.BOLD}{name}:{Colors.ENDC} {value}")

            if summary['metrics']['latest']:
                print(f"\n{Colors.CYAN}Latest Metrics:{Colors.ENDC}")
                for name, value in summary['metrics']['latest'].items():
                    print(f"  {Colors.BOLD}{name}:{Colors.ENDC} {value}")

            if summary['metrics']['best']:
                print(f"\n{Colors.CYAN}Best Metrics:{Colors.ENDC}")
                for name, info in summary['metrics']['best'].items():
                    print(f"  {Colors.BOLD}{name}:{Colors.ENDC} {info['value']} (step {info['step']})")

            if summary['artifacts']:
                print(f"\n{Colors.CYAN}Artifacts:{Colors.ENDC}")
                for artifact in summary['artifacts']:
                    print(f"  - {artifact}")

    except Exception as e:
        print_error(f"Failed to show experiment details: {str(e)}")
        sys.exit(1)

@track.command('plot')
@click.argument('experiment_id')
@click.option('--base-dir', default='neural_experiments', help='Base directory for experiments')
@click.option('--metrics', '-m', multiple=True, help='Metrics to plot (plots all if not specified)')
@click.option('--output', '-o', default='metrics.png', help='Output file path')
@click.pass_context
def track_plot(ctx, experiment_id, base_dir, metrics, output):
    """Plot metrics from an experiment."""
    print_command_header("track plot")

    try:
        # Import experiment tracker
        ExperimentManager = get_module(experiment_tracker_module).ExperimentManager

        # Get experiment
        manager = ExperimentManager(base_dir=base_dir)
        experiment = manager.get_experiment(experiment_id)

        if not experiment:
            print_error(f"Experiment not found: {experiment_id}")
            sys.exit(1)

        # Plot metrics
        metrics_list = list(metrics) if metrics else None
        fig = experiment.plot_metrics(metric_names=metrics_list)

        # Save figure
        fig.savefig(output)

        print_success(f"Metrics plot saved to {output}")

    except Exception as e:
        print_error(f"Failed to plot metrics: {str(e)}")
        sys.exit(1)

@track.command('compare')
@click.argument('experiment_ids', nargs=-1, required=True)
@click.option('--base-dir', default='neural_experiments', help='Base directory for experiments')
@click.option('--metrics', '-m', multiple=True, help='Metrics to compare (compares all if not specified)')
@click.option('--output-dir', '-o', default='comparison_plots', help='Output directory for plots')
@click.pass_context
def track_compare(ctx, experiment_ids, base_dir, metrics, output_dir):
    """Compare multiple experiments."""
    print_command_header("track compare")

    try:
        # Import experiment tracker
        ExperimentManager = get_module(experiment_tracker_module).ExperimentManager

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get experiments
        manager = ExperimentManager(base_dir=base_dir)

        # Compare experiments
        metrics_list = list(metrics) if metrics else None
        plots = manager.compare_experiments(experiment_ids, metric_names=metrics_list)

        if not plots:
            print_warning("No plots generated")
            return

        # Save plots
        for name, fig in plots.items():
            output_path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(output_path)

        print_success(f"Comparison plots saved to {output_dir}/")
        print(f"Generated {len(plots)} plots:")
        for name in plots.keys():
            print(f"  - {name}.png")

    except Exception as e:
        print_error(f"Failed to compare experiments: {str(e)}")
        sys.exit(1)

@cloud.command('run')
@click.option('--setup-tunnel', is_flag=True, help='Set up an ngrok tunnel for remote access')
@click.option('--port', default=8051, help='Port for the No-Code interface')
@click.pass_context
def cloud_run(ctx: click.Context, setup_tunnel: bool, port: int) -> None:
    """Run Neural in cloud environments (Kaggle, Colab, etc.)."""
    print_command_header("cloud run")

    # Detect environment
    env_type = "unknown"
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        env_type = "Kaggle"
    elif 'COLAB_GPU' in os.environ:
        env_type = "Google Colab"
    elif 'SM_MODEL_DIR' in os.environ:
        env_type = "AWS SageMaker"

    print_info(f"Detected cloud environment: {env_type}")

    # Check for GPU
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        has_gpu = result.returncode == 0
    except FileNotFoundError:
        has_gpu = False

    print_info(f"GPU available: {has_gpu}")

    # Import cloud module
    try:
        with Spinner("Initializing cloud environment") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()

            # Try to import the cloud module
            try:
                from neural.cloud.cloud_execution import CloudExecutor
            except ImportError:
                print_warning("Cloud module not found. Installing required dependencies...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
                from neural.cloud.cloud_execution import CloudExecutor

            # Initialize the cloud executor
            executor = CloudExecutor(environment=env_type)

            # Set up ngrok tunnel if requested
            if setup_tunnel:
                tunnel_url = executor.setup_ngrok_tunnel(port)
                if tunnel_url:
                    print_success(f"Tunnel established at: {tunnel_url}")
                else:
                    print_error("Failed to set up tunnel")

            # Start the No-Code interface
            nocode_info = executor.start_nocode_interface(port=port, setup_tunnel=setup_tunnel)

            print_success("Neural is now running in cloud mode!")
            print(f"\n{Colors.CYAN}Cloud Information:{Colors.ENDC}")
            print(f"  {Colors.BOLD}Environment:{Colors.ENDC} {env_type}")
            print(f"  {Colors.BOLD}GPU:{Colors.ENDC}         {'Available' if has_gpu else 'Not available'}")
            print(f"  {Colors.BOLD}Interface:{Colors.ENDC}   {nocode_info['interface_url']}")

            if setup_tunnel and nocode_info.get('tunnel_url'):
                print(f"  {Colors.BOLD}Tunnel URL:{Colors.ENDC}  {nocode_info['tunnel_url']}")

            print(f"\n{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}")

            # Keep the process running
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print_info("\nShutting down...")
                executor.cleanup()
                print_success("Neural cloud environment stopped")

    except Exception as e:
        print_error(f"Failed to initialize cloud environment: {str(e)}")
        sys.exit(1)

@cloud.command('connect')
@click.argument('platform', type=click.Choice(['kaggle', 'colab', 'sagemaker'], case_sensitive=False))
@click.option('--interactive', '-i', is_flag=True, help='Start an interactive shell')
@click.option('--notebook', '-n', is_flag=True, help='Start a Jupyter-like notebook interface')
@click.option('--port', default=8888, help='Port for the notebook server (only with --notebook)')
@click.option('--quiet', '-q', is_flag=True, help='Reduce output verbosity')
@click.pass_context
def cloud_connect(ctx, platform: str, interactive: bool, notebook: bool, port: int, quiet: bool):
    """Connect to a cloud platform."""
    # Configure logging to be less verbose
    if quiet:
        import logging
        logging.basicConfig(level=logging.ERROR)

    # Create a more aesthetic header
    if not quiet:
        platform_emoji = {
            'kaggle': 'ðŸ†',
            'colab': 'ðŸ§ª',
            'sagemaker': 'â˜ï¸'
        }.get(platform.lower(), 'ðŸŒ')

        print("\n" + "â”€" * 60)
        print(f"  {platform_emoji}  Neural Cloud Connect: {platform.upper()}")
        print("â”€" * 60 + "\n")

    try:
        # Import the remote connection module
        with Spinner("Connecting to cloud platform", quiet=quiet) as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()

            # Try to import the remote connection module
            try:
                from neural.cloud.remote_connection import RemoteConnection
            except ImportError:
                if not quiet:
                    print_warning("Installing required dependencies...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "boto3", "kaggle"],
                    stdout=subprocess.DEVNULL if quiet else None,
                    stderr=subprocess.DEVNULL if quiet else None
                )
                from neural.cloud.remote_connection import RemoteConnection

            # Initialize the remote connection
            remote = RemoteConnection()

            # Connect to the platform
            if platform.lower() == 'kaggle':
                result = remote.connect_to_kaggle()
            elif platform.lower() == 'colab':
                result = remote.connect_to_colab()
            elif platform.lower() == 'sagemaker':
                result = remote.connect_to_sagemaker()
            else:
                print_error(f"Unsupported platform: {platform}")
                sys.exit(1)

            if result['success']:
                if not quiet:
                    print_success(result['message'])

                # Start interactive shell if requested
                if interactive and notebook:
                    if not quiet:
                        print_warning("Both --interactive and --notebook specified. Using --interactive.")

                if interactive:
                    try:
                        # Use the more aesthetic script if not in quiet mode
                        if not quiet:
                            import subprocess
                            import os

                            # Get the path to the script
                            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                      "cloud", "run_interactive_shell.py")

                            # Run the script
                            subprocess.run([sys.executable, script_path, platform])
                            return  # Exit after the script finishes
                        else:
                            # Use the regular function in quiet mode
                            from neural.cloud.interactive_shell import start_interactive_shell
                            start_interactive_shell(platform, remote, quiet=quiet)
                    except ImportError:
                        print_error("Interactive shell module not found")
                        sys.exit(1)
                    except Exception as e:
                        print_error(f"Failed to start interactive shell: {e}")
                        sys.exit(1)
                elif notebook:
                    try:
                        from neural.cloud.notebook_interface import start_notebook_interface
                        if not quiet:
                            print_info(f"Starting notebook interface for {platform} on port {port}...")
                        # Pass the port and quiet parameters
                        start_notebook_interface(platform, remote, port, quiet=quiet)
                    except ImportError:
                        print_error("Notebook interface module not found")
                        sys.exit(1)
            else:
                print_error(f"Failed to connect: {result.get('error', 'Unknown error')}")
                sys.exit(1)

    except Exception as e:
        print_error(f"Failed to connect to {platform}: {str(e)}")
        sys.exit(1)

@cloud.command('execute')
@click.argument('platform', type=click.Choice(['kaggle', 'colab', 'sagemaker'], case_sensitive=False))
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--name', help='Name for the kernel/notebook')
@click.pass_context
def cloud_execute(ctx: click.Context, platform: str, file: str, name: Optional[str]) -> None:
    """Execute a Neural DSL file on a cloud platform."""
    print_command_header(f"cloud execute: {platform}")

    try:
        # Read the file
        with open(file, 'r') as f:
            dsl_code = f.read()

        # Import the remote connection module
        with Spinner("Executing on cloud platform") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()

            # Try to import the remote connection module
            try:
                from neural.cloud.remote_connection import RemoteConnection
            except ImportError:
                print_warning("Remote connection module not found. Installing required dependencies...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "kaggle"])
                from neural.cloud.remote_connection import RemoteConnection

            # Initialize the remote connection
            remote = RemoteConnection()

            # Generate a name if not provided
            if not name:
                import hashlib
                name = f"neural-{hashlib.md5(dsl_code.encode()).hexdigest()[:8]}"

            # Execute on the platform
            if platform.lower() == 'kaggle':
                # Connect to Kaggle
                result = remote.connect_to_kaggle()
                if not result['success']:
                    print_error(f"Failed to connect to Kaggle: {result.get('error', 'Unknown error')}")
                    sys.exit(1)

                # Create a kernel
                kernel_id = remote.create_kaggle_kernel(name)
                if not kernel_id:
                    print_error("Failed to create Kaggle kernel")
                    sys.exit(1)

                print_info(f"Created Kaggle kernel: {kernel_id}")

                # Generate code to execute
                execution_code = f"""
# Install Neural DSL
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git

# Import the cloud module
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()
print(f"Detected environment: {{executor.environment}}")
print(f"GPU available: {{executor.is_gpu_available}}")

# Define the model
dsl_code = \"\"\"
{dsl_code}
\"\"\"

# Compile the model
model_path = executor.compile_model(dsl_code, backend='tensorflow')
print(f"Model compiled to: {{model_path}}")

# Run the model
results = executor.run_model(model_path, dataset='MNIST', epochs=5)
print(f"Model execution results: {{results}}")

# Visualize the model
viz_path = executor.visualize_model(dsl_code, output_format='png')
print(f"Model visualization saved to: {{viz_path}}")
"""

                # Execute the code
                print_info("Executing on Kaggle...")
                result = remote.execute_on_kaggle(kernel_id, execution_code)

                if result['success']:
                    print_success("Execution completed successfully")
                    print("\nOutput:")
                    print(result['output'])
                else:
                    print_error(f"Execution failed: {result.get('error', 'Unknown error')}")
                    sys.exit(1)

                # Clean up
                remote.delete_kaggle_kernel(kernel_id)

            elif platform.lower() == 'sagemaker':
                # Connect to SageMaker
                result = remote.connect_to_sagemaker()
                if not result['success']:
                    print_error(f"Failed to connect to SageMaker: {result.get('error', 'Unknown error')}")
                    sys.exit(1)

                # Create a notebook instance
                notebook_name = remote.create_sagemaker_notebook(name)
                if not notebook_name:
                    print_error("Failed to create SageMaker notebook instance")
                    sys.exit(1)

                print_info(f"Created SageMaker notebook instance: {notebook_name}")

                # Generate code to execute
                execution_code = f"""
# Install Neural DSL
!pip install git+https://github.com/Lemniscate-SHA-256/Neural.git

# Import the cloud module
from neural.cloud.cloud_execution import CloudExecutor

# Initialize the cloud executor
executor = CloudExecutor()
print(f"Detected environment: {{executor.environment}}")
print(f"GPU available: {{executor.is_gpu_available}}")

# Define the model
dsl_code = \"\"\"
{dsl_code}
\"\"\"

# Compile the model
model_path = executor.compile_model(dsl_code, backend='tensorflow')
print(f"Model compiled to: {{model_path}}")

# Run the model
results = executor.run_model(model_path, dataset='MNIST', epochs=5)
print(f"Model execution results: {{results}}")

# Visualize the model
viz_path = executor.visualize_model(dsl_code, output_format='png')
print(f"Model visualization saved to: {{viz_path}}")
"""

                # Execute the code
                print_info("Executing on SageMaker...")
                result = remote.execute_on_sagemaker(notebook_name, execution_code)

                if result['success']:
                    print_success("Execution completed successfully")
                    print("\nOutput:")
                    print(result['output'])
                else:
                    print_error(f"Execution failed: {result.get('error', 'Unknown error')}")
                    sys.exit(1)

                # Clean up
                remote.delete_sagemaker_notebook(notebook_name)

            elif platform.lower() == 'colab':
                print_error("Colab execution from terminal is not supported yet")
                sys.exit(1)

            else:
                print_error(f"Unsupported platform: {platform}")
                sys.exit(1)

    except Exception as e:
        print_error(f"Failed to execute on {platform}: {str(e)}")
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
        from neural.parser.parser import create_parser, ModelTransformer
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
                    # Convolutional layers typically have moderate gradients
                    entry['grad_norm'] = np.random.uniform(0.3, 0.7)
                elif 'Dense' in layer_type or 'Output' in layer_type:
                    # Dense layers can have larger gradients
                    entry['grad_norm'] = np.random.uniform(0.5, 1.0)
                elif 'Pool' in layer_type:
                    # Pooling layers have smaller gradients
                    entry['grad_norm'] = np.random.uniform(0.1, 0.3)
                else:
                    # Other layers
                    entry['grad_norm'] = np.random.uniform(0.2, 0.5)

                # Dead neuron metrics
                if 'ReLU' in layer_type or 'Conv' in layer_type:
                    # ReLU and Conv layers can have dead neurons
                    entry['dead_ratio'] = np.random.uniform(0.05, 0.2)
                elif 'Dense' in layer_type:
                    # Dense layers typically have fewer dead neurons
                    entry['dead_ratio'] = np.random.uniform(0.01, 0.1)
                else:
                    # Other layers
                    entry['dead_ratio'] = np.random.uniform(0.0, 0.05)

                # Activation metrics
                if 'ReLU' in layer_type:
                    # ReLU activations are typically positive
                    entry['mean_activation'] = np.random.uniform(0.3, 0.7)
                elif 'Sigmoid' in layer_type:
                    # Sigmoid activations are between 0 and 1
                    entry['mean_activation'] = np.random.uniform(0.4, 0.6)
                elif 'Tanh' in layer_type:
                    # Tanh activations are between -1 and 1
                    entry['mean_activation'] = np.random.uniform(-0.3, 0.3)
                elif 'Softmax' in layer_type or 'Output' in layer_type:
                    # Softmax activations sum to 1
                    entry['mean_activation'] = np.random.uniform(0.1, 0.3)
                else:
                    # Other layers
                    entry['mean_activation'] = np.random.uniform(0.2, 0.8)

                # Anomaly detection
                # Simulate anomalies in some layers (about 10% chance)
                if np.random.random() > 0.9:
                    entry['anomaly'] = True
                    # Anomalous activations are either very high or very low
                    if np.random.random() > 0.5:
                        entry['mean_activation'] = np.random.uniform(5.0, 15.0)  # Very high
                    else:
                        entry['mean_activation'] = np.random.uniform(0.0001, 0.01)  # Very low
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

@cli.command(name='no-code')
@click.option('--port', default=8051, help='Web interface port', type=int)
@click.pass_context
def no_code(ctx: click.Context, port: int) -> None:
    """Launch the no-code interface for building models."""
    print_command_header("no-code")
    print_info("Launching the Neural no-code interface...")

    # Lazy load dashboard
    try:
        from .lazy_imports import dash
        with Spinner("Loading dashboard components") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            app = get_module(dash).get_app()
        print_success("Dashboard ready!")
        print(f"\n{Colors.CYAN}Server Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}URL:{Colors.ENDC}         http://localhost:{port}")
        print(f"  {Colors.BOLD}Interface:{Colors.ENDC}   Neural No-Code Builder")
        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}")
        app.run_server(debug=False, host="localhost", port=port)
    except (ImportError, AttributeError, Exception) as e:
        print_error(f"Failed to launch no-code interface: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("Server stopped by user")

@cli.command()
@click.argument('model_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--method', '-m', default='shap', help='Explanation method', type=click.Choice(['shap', 'lime', 'saliency', 'attention', 'feature_importance', 'counterfactual', 'all'], case_sensitive=False))
@click.option('--backend', '-b', default='tensorflow', help='Model backend', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--data', '-d', type=click.Path(exists=True), help='Input data file (numpy .npy format)')
@click.option('--output', '-o', default='explanations', help='Output directory for explanations')
@click.option('--num-samples', type=int, default=10, help='Number of samples to explain')
@click.option('--generate-model-card', is_flag=True, help='Generate a model card')
@click.pass_context
def explain(ctx, model_path: str, method: str, backend: str, data: Optional[str], output: str, num_samples: int, generate_model_card: bool):
    """Explain model predictions using various interpretability methods."""
    print_command_header("explain")
    print_info(f"Explaining model: {model_path}")
    
    import os
    os.makedirs(output, exist_ok=True)
    
    try:
        if backend == 'tensorflow':
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
        elif backend == 'pytorch':
            import torch
            model = torch.load(model_path)
        else:
            print_error(f"Unsupported backend: {backend}")
            sys.exit(1)
        
        print_success(f"Loaded {backend} model")
        
        if data:
            import numpy as np
            input_data = np.load(data)
            print_info(f"Loaded input data: {input_data.shape}")
            
            if len(input_data) > num_samples:
                input_data = input_data[:num_samples]
                print_info(f"Using first {num_samples} samples")
        else:
            print_warning("No input data provided. Using synthetic data for demonstration.")
            import numpy as np
            if backend == 'tensorflow':
                input_shape = model.input_shape[1:]
            else:
                input_shape = tuple(model.parameters()).__next__().shape[1:]
            input_data = np.random.randn(num_samples, *input_shape).astype(np.float32)
        
        from neural.explainability import ModelExplainer
        
        with Spinner("Initializing explainer") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            explainer = ModelExplainer(
                model=model,
                backend=backend,
                task_type='classification'
            )
        
        print_success("Explainer initialized")
        
        if method == 'shap' or method == 'all':
            print_info("Generating SHAP explanations...")
            try:
                with Spinner("Computing SHAP values") as spinner:
                    if ctx.obj.get('NO_ANIMATIONS'):
                        spinner.stop()
                    shap_results = explainer.explain_prediction(input_data[0], method='shap')
                
                print_success("SHAP explanations generated")
                print(f"  {Colors.CYAN}SHAP values shape:{Colors.ENDC} {shap_results['shap_values'].shape if hasattr(shap_results['shap_values'], 'shape') else 'N/A'}")
            except Exception as e:
                print_warning(f"SHAP explanation failed: {str(e)}")
        
        if method == 'lime' or method == 'all':
            print_info("Generating LIME explanations...")
            try:
                with Spinner("Computing LIME explanations") as spinner:
                    if ctx.obj.get('NO_ANIMATIONS'):
                        spinner.stop()
                    lime_results = explainer.explain_prediction(input_data[0], method='lime')
                
                print_success("LIME explanations generated")
                if 'feature_weights' in lime_results:
                    print(f"  {Colors.CYAN}Top features:{Colors.ENDC}")
                    for feature, weight in list(lime_results['feature_weights'].items())[:5]:
                        print(f"    - {feature}: {weight:.4f}")
            except Exception as e:
                print_warning(f"LIME explanation failed: {str(e)}")
        
        if method == 'saliency' or method == 'all':
            print_info("Generating saliency maps...")
            try:
                with Spinner("Computing saliency maps") as spinner:
                    if ctx.obj.get('NO_ANIMATIONS'):
                        spinner.stop()
                    saliency_results = explainer.explain_prediction(input_data[0], method='saliency')
                
                output_path = os.path.join(output, 'saliency_map.png')
                from neural.explainability.saliency_maps import SaliencyMapGenerator
                saliency_gen = SaliencyMapGenerator(model, backend)
                saliency_gen.visualize(
                    saliency_results['saliency_map'],
                    input_data[0],
                    output_path=output_path
                )
                print_success(f"Saliency map saved to {output_path}")
            except Exception as e:
                print_warning(f"Saliency map generation failed: {str(e)}")
        
        if method == 'attention' or method == 'all':
            print_info("Visualizing attention weights...")
            try:
                with Spinner("Extracting attention weights") as spinner:
                    if ctx.obj.get('NO_ANIMATIONS'):
                        spinner.stop()
                    attention_results = explainer.visualize_attention(input_data[0])
                
                if attention_results['attention_weights']:
                    print_success(f"Extracted attention from {len(attention_results['attention_weights'])} layers")
                else:
                    print_warning("No attention layers found in model")
            except Exception as e:
                print_warning(f"Attention visualization failed: {str(e)}")
        
        if method == 'feature_importance' or method == 'all':
            print_info("Ranking feature importance...")
            try:
                with Spinner("Computing feature importance") as spinner:
                    if ctx.obj.get('NO_ANIMATIONS'):
                        spinner.stop()
                    importance_results = explainer.rank_features(input_data, method='gradient')
                
                print_success("Feature importance computed")
                print(f"  {Colors.CYAN}Top 5 features:{Colors.ENDC}")
                for i, (feature, score) in enumerate(importance_results['rankings'][:5]):
                    print(f"    {i+1}. {feature}: {score:.4f}")
                
                output_path = os.path.join(output, 'feature_importance.png')
                from neural.explainability.feature_importance import FeatureImportanceRanker
                ranker = FeatureImportanceRanker(model, backend)
                ranker.plot_importance(
                    importance_results['importance_scores'],
                    top_k=20,
                    output_path=output_path
                )
                print_success(f"Feature importance plot saved to {output_path}")
            except Exception as e:
                print_warning(f"Feature importance ranking failed: {str(e)}")
        
        if method == 'counterfactual' or method == 'all':
            print_info("Generating counterfactual explanations...")
            try:
                with Spinner("Computing counterfactuals") as spinner:
                    if ctx.obj.get('NO_ANIMATIONS'):
                        spinner.stop()
                    cf_results = explainer.generate_counterfactuals(input_data[0], num_samples=3)
                
                print_success(f"Generated {len(cf_results['counterfactuals'])} counterfactuals")
                print(f"  {Colors.CYAN}Distances:{Colors.ENDC} {[f'{d:.3f}' for d in cf_results['distances']]}")
                
                output_path = os.path.join(output, 'counterfactuals.png')
                from neural.explainability.counterfactual import CounterfactualGenerator
                cf_gen = CounterfactualGenerator(model, backend)
                cf_gen.visualize_counterfactuals(
                    cf_results['original_input'],
                    cf_results['counterfactuals'],
                    output_path=output_path
                )
                print_success(f"Counterfactual visualization saved to {output_path}")
            except Exception as e:
                print_warning(f"Counterfactual generation failed: {str(e)}")
        
        if generate_model_card:
            print_info("Generating model card...")
            try:
                from neural.explainability.model_card import ModelCardGenerator
                
                model_info = {
                    'model_name': os.path.basename(model_path),
                    'framework': backend,
                    'model_type': 'Neural Network',
                    'model_details': {
                        'model_path': model_path,
                        'backend': backend
                    }
                }
                
                card_gen = ModelCardGenerator()
                card_path = os.path.join(output, 'model_card.md')
                card_gen.generate(model_info, card_path)
                print_success(f"Model card saved to {card_path}")
            except Exception as e:
                print_warning(f"Model card generation failed: {str(e)}")
        
        print_success("Explanation completed!")
        print(f"\n{Colors.CYAN}Output directory:{Colors.ENDC} {output}")
        
    except ImportError as e:
        print_error(f"Missing dependencies: {str(e)}")
        print_info("Install explainability dependencies: pip install shap lime scikit-image")
        sys.exit(1)
    except Exception as e:
        print_error(f"Explanation failed: {str(e)}")
        if ctx.obj.get('VERBOSE'):
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command(name='aquarium')
@click.option('--base-dir', default='neural_experiments', help='Base directory containing experiments', type=str)
@click.option('--port', default=8053, help='Port to run the dashboard on', type=int)
@click.option('--host', default='127.0.0.1', help='Host to run the dashboard on', type=str)
@click.option('--debug', is_flag=True, help='Run in debug mode')
@click.pass_context
def aquarium(ctx, base_dir: str, port: int, host: str, debug: bool):
    """Launch the Aquarium experiment tracking dashboard."""
    print_command_header("aquarium")
    print_info("Launching the Aquarium experiment tracking dashboard...")

    try:
        with Spinner("Loading Aquarium components") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            from neural.tracking import launch_aquarium as launch_aquarium_func
        
        print_success("Aquarium ready!")
        print(f"\n{Colors.CYAN}Server Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}URL:{Colors.ENDC}         http://{host}:{port}")
        print(f"  {Colors.BOLD}Base Dir:{Colors.ENDC}    {base_dir}")
        print(f"  {Colors.BOLD}Interface:{Colors.ENDC}   Aquarium Experiment Tracker")
        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}")
        
        launch_aquarium_func(base_dir=base_dir, port=port, host=host, debug=debug)
    except (ImportError, AttributeError, Exception) as e:
        print_error(f"Failed to launch Aquarium: {str(e)}")
        print_info("Make sure dash and related dependencies are installed:")
        print_info("  pip install dash dash-bootstrap-components plotly")
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("Server stopped by user")

# Import and register monitoring commands
try:
    from neural.monitoring.cli_commands import monitor
    cli.add_command(monitor)
except ImportError:
    pass

# Import and register teams commands
try:
    from neural.teams.cli_commands import teams
    cli.add_command(teams)
except ImportError:
    pass

@cli.group()
@click.pass_context
def marketplace(ctx):
    """Commands for model marketplace operations."""
    pass

@marketplace.command('search')
@click.argument('query', required=False)
@click.option('--author', help='Filter by author')
@click.option('--tags', help='Filter by tags (comma-separated)')
@click.option('--license', help='Filter by license')
@click.option('--limit', default=10, type=int, help='Maximum number of results')
@click.pass_context
def marketplace_search(ctx, query, author, tags, license, limit):
    """Search for models in the marketplace."""
    print_command_header("marketplace search")

    try:
        from neural.marketplace import ModelRegistry, SemanticSearch

        with Spinner("Initializing marketplace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            registry = ModelRegistry()
            search = SemanticSearch(registry)

        # Build filters
        filters = {}
        if author:
            filters['author'] = author
        if tags:
            filters['tags'] = [t.strip() for t in tags.split(',')]
        if license:
            filters['license'] = license

        # Search
        if query:
            with Spinner(f"Searching for '{query}'") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                results = search.search(query, limit=limit, filters=filters)

            print_success(f"Found {len(results)} models")
            print(f"\n{Colors.CYAN}Search Results:{Colors.ENDC}\n")

            for i, (model_id, similarity, model) in enumerate(results, 1):
                print(f"{Colors.BOLD}{i}. {model['name']}{Colors.ENDC} by {model['author']}")
                print(f"   ID: {model_id}")
                print(f"   Similarity: {similarity:.2f}")
                print(f"   Description: {model['description'][:80]}...")
                print(f"   Tags: {', '.join(model.get('tags', []))}")
                print(f"   Downloads: {model.get('downloads', 0)} | License: {model.get('license', 'MIT')}")
                print()
        else:
            # List all models with filters
            with Spinner("Loading models") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                models = registry.list_models(
                    author=author,
                    tags=filters.get('tags'),
                    limit=limit
                )

            # Apply license filter manually
            if license:
                models = [m for m in models if m.get('license') == license]

            print_success(f"Found {len(models)} models")
            print(f"\n{Colors.CYAN}Models:{Colors.ENDC}\n")

            for i, model in enumerate(models, 1):
                print(f"{Colors.BOLD}{i}. {model['name']}{Colors.ENDC} by {model['author']}")
                print(f"   ID: {model['id']}")
                print(f"   Description: {model['description'][:80]}...")
                print(f"   Tags: {', '.join(model.get('tags', []))}")
                print(f"   Downloads: {model.get('downloads', 0)} | License: {model.get('license', 'MIT')}")
                print()

    except Exception as e:
        print_error(f"Search failed: {str(e)}")
        sys.exit(1)

@marketplace.command('download')
@click.argument('model_id')
@click.option('--output', '-o', default='.', help='Output directory')
@click.pass_context
def marketplace_download(ctx, model_id, output):
    """Download a model from the marketplace."""
    print_command_header("marketplace download")
    print_info(f"Downloading model: {model_id}")

    try:
        from neural.marketplace import ModelRegistry

        with Spinner("Initializing marketplace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            registry = ModelRegistry()

        with Spinner("Downloading model") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            file_path = registry.download_model(model_id, output)

        print_success(f"Model downloaded successfully!")
        print(f"\n{Colors.CYAN}Download Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}File:{Colors.ENDC}     {file_path}")
        print(f"  {Colors.BOLD}Location:{Colors.ENDC} {os.path.abspath(output)}")

    except (ValueError, FileNotFoundError) as e:
        print_error(f"Download failed: {str(e)}")
        sys.exit(1)

@marketplace.command('publish')
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--name', required=True, help='Model name')
@click.option('--author', required=True, help='Author name')
@click.option('--description', default='', help='Model description')
@click.option('--license', default='MIT', help='License (MIT, Apache-2.0, GPL-3.0, etc.)')
@click.option('--tags', help='Tags (comma-separated)')
@click.option('--version', default='1.0.0', help='Model version')
@click.pass_context
def marketplace_publish(ctx, model_path, name, author, description, license, tags, version):
    """Publish a model to the marketplace."""
    print_command_header("marketplace publish")
    print_info(f"Publishing model: {name}")

    try:
        from neural.marketplace import ModelRegistry

        with Spinner("Initializing marketplace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            registry = ModelRegistry()

        tags_list = [t.strip() for t in tags.split(',') if t.strip()] if tags else []

        with Spinner("Uploading model") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            model_id = registry.upload_model(
                name=name,
                author=author,
                model_path=model_path,
                description=description,
                license=license,
                tags=tags_list,
                version=version
            )

        print_success(f"Model published successfully!")
        print(f"\n{Colors.CYAN}Publication Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Model ID:{Colors.ENDC}  {model_id}")
        print(f"  {Colors.BOLD}Name:{Colors.ENDC}      {name}")
        print(f"  {Colors.BOLD}Author:{Colors.ENDC}    {author}")
        print(f"  {Colors.BOLD}Version:{Colors.ENDC}   {version}")
        print(f"  {Colors.BOLD}License:{Colors.ENDC}   {license}")
        if tags_list:
            print(f"  {Colors.BOLD}Tags:{Colors.ENDC}      {', '.join(tags_list)}")

    except (FileNotFoundError, Exception) as e:
        print_error(f"Publication failed: {str(e)}")
        sys.exit(1)

@marketplace.command('info')
@click.argument('model_id')
@click.pass_context
def marketplace_info(ctx, model_id):
    """Get information about a model."""
    print_command_header("marketplace info")

    try:
        from neural.marketplace import ModelRegistry

        with Spinner("Initializing marketplace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            registry = ModelRegistry()

        model_info = registry.get_model_info(model_id)
        stats = registry.get_usage_stats(model_id)

        print(f"\n{Colors.CYAN}Model Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}ID:{Colors.ENDC}          {model_info['id']}")
        print(f"  {Colors.BOLD}Name:{Colors.ENDC}        {model_info['name']}")
        print(f"  {Colors.BOLD}Author:{Colors.ENDC}      {model_info['author']}")
        print(f"  {Colors.BOLD}Version:{Colors.ENDC}     {model_info.get('version', '1.0.0')}")
        print(f"  {Colors.BOLD}License:{Colors.ENDC}     {model_info.get('license', 'MIT')}")
        print(f"  {Colors.BOLD}Framework:{Colors.ENDC}   {model_info.get('framework', 'neural-dsl')}")

        print(f"\n{Colors.CYAN}Description:{Colors.ENDC}")
        print(f"  {model_info.get('description', 'No description available')}")

        if model_info.get('tags'):
            print(f"\n{Colors.CYAN}Tags:{Colors.ENDC}")
            print(f"  {', '.join(model_info['tags'])}")

        print(f"\n{Colors.CYAN}Statistics:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Downloads:{Colors.ENDC}  {stats.get('downloads', 0)}")
        print(f"  {Colors.BOLD}Views:{Colors.ENDC}      {stats.get('views', 0)}")

        print(f"\n{Colors.CYAN}Dates:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Uploaded:{Colors.ENDC}   {model_info['uploaded_at']}")
        print(f"  {Colors.BOLD}Updated:{Colors.ENDC}    {model_info['updated_at']}")

    except ValueError as e:
        print_error(f"Model not found: {str(e)}")
        sys.exit(1)

@marketplace.command('list')
@click.option('--author', help='Filter by author')
@click.option('--tags', help='Filter by tags (comma-separated)')
@click.option('--sort-by', default='uploaded_at', type=click.Choice(['uploaded_at', 'downloads', 'name']), help='Sort by field')
@click.option('--limit', default=20, type=int, help='Maximum number of results')
@click.pass_context
def marketplace_list(ctx, author, tags, sort_by, limit):
    """List all models in the marketplace."""
    print_command_header("marketplace list")

    try:
        from neural.marketplace import ModelRegistry

        with Spinner("Loading models") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            registry = ModelRegistry()

        tags_list = [t.strip() for t in tags.split(',') if t.strip()] if tags else None

        models = registry.list_models(
            author=author,
            tags=tags_list,
            sort_by=sort_by,
            limit=limit
        )

        print_success(f"Found {len(models)} models")
        print(f"\n{Colors.CYAN}Models:{Colors.ENDC}\n")

        for i, model in enumerate(models, 1):
            print(f"{Colors.BOLD}{i}. {model['name']}{Colors.ENDC} by {model['author']}")
            print(f"   ID: {model['id']}")
            print(f"   Version: {model.get('version', '1.0.0')} | License: {model.get('license', 'MIT')}")
            print(f"   Downloads: {model.get('downloads', 0)}")
            if model.get('tags'):
                print(f"   Tags: {', '.join(model['tags'][:5])}")
            print()

    except Exception as e:
        print_error(f"Failed to list models: {str(e)}")
        sys.exit(1)

@marketplace.command('web')
@click.option('--port', default=8052, type=int, help='Web interface port')
@click.option('--host', default='localhost', help='Host address')
@click.pass_context
def marketplace_web(ctx, port, host):
    """Launch the marketplace web interface."""
    print_command_header("marketplace web")
    print_info("Launching the Neural Marketplace web interface...")

    try:
        from neural.marketplace.web_ui import MarketplaceUI

        with Spinner("Initializing marketplace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            ui = MarketplaceUI()

        print_success("Marketplace ready!")
        print(f"\n{Colors.CYAN}Server Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}URL:{Colors.ENDC}         http://{host}:{port}/marketplace")
        print(f"  {Colors.BOLD}Interface:{Colors.ENDC}   Neural Marketplace")
        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}")

        ui.run(host=host, port=port, debug=False)

    except ImportError as e:
        print_error(f"Failed to launch marketplace: {str(e)}")
        print_info("Install required dependencies: pip install dash dash-bootstrap-components")
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("\nServer stopped by user")

@marketplace.command('hub-upload')
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('repo_id')
@click.option('--name', required=True, help='Model name')
@click.option('--description', default='', help='Model description')
@click.option('--license', default='mit', help='License')
@click.option('--tags', help='Tags (comma-separated)')
@click.option('--private', is_flag=True, help='Create private repository')
@click.pass_context
def marketplace_hub_upload(ctx, model_path, repo_id, name, description, license, tags, private):
    """Upload a model to HuggingFace Hub."""
    print_command_header("marketplace hub-upload")
    print_info(f"Uploading model to HuggingFace Hub: {repo_id}")

    try:
        from neural.marketplace import HuggingFaceIntegration

        with Spinner("Initializing HuggingFace integration") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            hf = HuggingFaceIntegration()

        tags_list = [t.strip() for t in tags.split(',') if t.strip()] if tags else []

        with Spinner("Uploading to HuggingFace Hub") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            result = hf.upload_to_hub(
                model_path=model_path,
                repo_id=repo_id,
                model_name=name,
                description=description,
                license=license,
                tags=tags_list,
                private=private
            )

        print_success(f"Model uploaded to HuggingFace Hub!")
        print(f"\n{Colors.CYAN}Upload Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Repository:{Colors.ENDC} {result['repo_id']}")
        print(f"  {Colors.BOLD}URL:{Colors.ENDC}        {result['url']}")

    except ImportError:
        print_error("HuggingFace Hub integration not available")
        print_info("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print_error(f"Upload failed: {str(e)}")
        sys.exit(1)

@marketplace.command('hub-download')
@click.argument('repo_id')
@click.argument('filename')
@click.option('--output', '-o', default='.', help='Output directory')
@click.option('--revision', default='main', help='Git revision')
@click.pass_context
def marketplace_hub_download(ctx, repo_id, filename, output, revision):
    """Download a model from HuggingFace Hub."""
    print_command_header("marketplace hub-download")
    print_info(f"Downloading from HuggingFace Hub: {repo_id}/{filename}")

    try:
        from neural.marketplace import HuggingFaceIntegration

        with Spinner("Initializing HuggingFace integration") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            hf = HuggingFaceIntegration()

        with Spinner("Downloading from HuggingFace Hub") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            file_path = hf.download_from_hub(
                repo_id=repo_id,
                filename=filename,
                output_dir=output,
                revision=revision
            )

        print_success(f"Model downloaded successfully!")
        print(f"\n{Colors.CYAN}Download Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}File:{Colors.ENDC}     {file_path}")
        print(f"  {Colors.BOLD}Location:{Colors.ENDC} {os.path.abspath(output)}")

    except ImportError:
        print_error("HuggingFace Hub integration not available")
        print_info("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print_error(f"Download failed: {str(e)}")
        sys.exit(1)

@cli.group()
@click.pass_context
def cost(ctx):
    """Commands for cost optimization and tracking."""
    pass

@cost.command('estimate')
@click.option('--provider', type=click.Choice(['aws', 'gcp', 'azure']), required=True, help='Cloud provider')
@click.option('--instance', required=True, help='Instance type name')
@click.option('--hours', type=float, required=True, help='Training hours')
@click.option('--storage-gb', type=float, default=100.0, help='Storage in GB')
@click.option('--spot/--on-demand', default=True, help='Use spot instances')
@click.pass_context
def cost_estimate(ctx, provider, instance, hours, storage_gb, spot):
    """Estimate training costs."""
    print_command_header("cost estimate")
    
    try:
        from neural.cost import CostEstimator, CloudProvider
        
        estimator = CostEstimator()
        cloud_provider = CloudProvider(provider)
        
        with Spinner("Calculating cost estimate") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            
            estimate = estimator.estimate_cost(
                provider=cloud_provider,
                instance_name=instance,
                training_hours=hours,
                storage_gb=storage_gb,
                use_spot=spot
            )
        
        print_success("Cost estimation complete!")
        print(f"\n{Colors.CYAN}Cost Breakdown:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Provider:{Colors.ENDC}      {estimate.provider.value}")
        print(f"  {Colors.BOLD}Instance:{Colors.ENDC}      {estimate.instance_type.name}")
        print(f"  {Colors.BOLD}Duration:{Colors.ENDC}      {estimate.estimated_hours:.2f} hours")
        print(f"\n  {Colors.BOLD}Compute (On-Demand):{Colors.ENDC} ${estimate.on_demand_cost:.2f}")
        print(f"  {Colors.BOLD}Compute (Spot):{Colors.ENDC}      ${estimate.spot_cost:.2f}")
        print(f"  {Colors.BOLD}Storage:{Colors.ENDC}             ${estimate.storage_cost:.2f}")
        print(f"  {Colors.BOLD}Data Transfer:{Colors.ENDC}       ${estimate.data_transfer_cost:.2f}")
        print(f"\n  {Colors.GREEN}{Colors.BOLD}Total (Spot):{Colors.ENDC}        ${estimate.total_spot_cost:.2f}")
        print(f"  {Colors.BOLD}Potential Savings:{Colors.ENDC}   ${estimate.potential_savings:.2f}")
        
    except Exception as e:
        print_error(f"Cost estimation failed: {str(e)}")
        sys.exit(1)

@cost.command('compare')
@click.option('--gpu-count', type=int, default=1, help='Number of GPUs')
@click.option('--hours', type=float, required=True, help='Training hours')
@click.option('--max-cost', type=float, help='Maximum acceptable cost')
@click.pass_context
def cost_compare(ctx, gpu_count, hours, max_cost):
    """Compare costs across cloud providers."""
    print_command_header("cost compare")
    print_info(f"Comparing costs for {gpu_count} GPU(s), {hours:.1f} hours")
    
    try:
        from neural.cost import CostEstimator
        
        estimator = CostEstimator()
        
        with Spinner("Analyzing provider costs") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            
            estimates = estimator.compare_providers(
                gpu_count=gpu_count,
                training_hours=hours
            )
        
        if max_cost:
            estimates = [e for e in estimates if e.total_spot_cost <= max_cost]
        
        print_success(f"Found {len(estimates)} matching configurations")
        print(f"\n{Colors.CYAN}{'Provider':<10} {'Instance':<25} {'Spot Cost':<12} {'Savings':<10}{Colors.ENDC}")
        print(f"{'-' * 60}")
        
        for est in estimates[:10]:
            savings_pct = (est.potential_savings / est.total_on_demand_cost * 100) if est.total_on_demand_cost > 0 else 0
            print(f"{est.provider.value:<10} {est.instance_type.name:<25} ${est.total_spot_cost:<11.2f} {savings_pct:<9.1f}%")
        
    except Exception as e:
        print_error(f"Cost comparison failed: {str(e)}")
        sys.exit(1)

@cost.command('dashboard')
@click.option('--port', type=int, default=8052, help='Dashboard port')
@click.pass_context
def cost_dashboard(ctx, port):
    """Launch interactive cost dashboard."""
    print_command_header("cost dashboard")
    print_info(f"Starting cost dashboard on port {port}")
    
    try:
        from neural.cost.dashboard import create_dashboard
        
        dashboard = create_dashboard(port=port)
        print_success(f"Dashboard running at http://localhost:{port}")
        print(f"{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}")
        
        dashboard.run(debug=False)
        
    except ImportError as e:
        print_error("Dashboard dependencies not available")
        print_info("Install with: pip install neural-dsl[dashboard]")
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("\nDashboard stopped by user")

@cli.group()
@click.pass_context
def data(ctx):
    """Commands for data versioning and lineage tracking."""
    pass

@data.command('version')
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--version', '-v', default=None, help='Version name (auto-generated if not provided)')
@click.option('--metadata', '-m', help='Metadata as JSON string')
@click.option('--tags', '-t', multiple=True, help='Tags for this version')
@click.option('--no-copy', is_flag=True, help='Do not copy data (just track)')
@click.option('--base-dir', default='.neural_data', help='Base directory for data storage')
@click.pass_context
def data_version(ctx, dataset_path, version, metadata, tags, no_copy, base_dir):
    """Create a new version of a dataset."""
    print_command_header("data version")
    print_info(f"Creating version for dataset: {dataset_path}")
    
    try:
        from neural.data import DatasetVersionManager
        
        manager = DatasetVersionManager(base_dir=base_dir)
        
        metadata_dict = json.loads(metadata) if metadata else {}
        tags_list = list(tags) if tags else []
        
        with Spinner("Creating dataset version") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            
            dataset_version = manager.create_version(
                dataset_path=dataset_path,
                version=version,
                metadata=metadata_dict,
                tags=tags_list,
                copy_data=not no_copy,
            )
        
        print_success("Dataset version created!")
        print(f"\n{Colors.CYAN}Version Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Version:{Colors.ENDC}    {dataset_version.version}")
        print(f"  {Colors.BOLD}Checksum:{Colors.ENDC}   {dataset_version.checksum[:16]}...")
        print(f"  {Colors.BOLD}Created:{Colors.ENDC}    {dataset_version.created_at}")
        if tags_list:
            print(f"  {Colors.BOLD}Tags:{Colors.ENDC}       {', '.join(tags_list)}")
    
    except Exception as e:
        print_error(f"Failed to create dataset version: {str(e)}")
        sys.exit(1)

@data.command('list')
@click.option('--tags', '-t', multiple=True, help='Filter by tags')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.option('--base-dir', default='.neural_data', help='Base directory for data storage')
@click.pass_context
def data_list(ctx, tags, format, base_dir):
    """List all dataset versions."""
    print_command_header("data list")
    
    try:
        from neural.data import DatasetVersionManager
        
        manager = DatasetVersionManager(base_dir=base_dir)
        versions = manager.list_versions(tags=list(tags) if tags else None)
        
        if not versions:
            print_warning("No dataset versions found")
            return
        
        if format == 'json':
            print(json.dumps([v.to_dict() for v in versions], indent=2))
        else:
            print(f"\n{Colors.CYAN}Dataset Versions:{Colors.ENDC}")
            print(f"  {Colors.BOLD}{'Version':<20} {'Checksum':<20} {'Created':<20} {'Tags':<20}{Colors.ENDC}")
            print(f"  {'-' * 80}")
            for v in versions:
                version_str = v.version[:18] + '..' if len(v.version) > 20 else v.version
                checksum_str = v.checksum[:18] + '..' if len(v.checksum) > 20 else v.checksum
                created_str = v.created_at.split('T')[0] + ' ' + v.created_at.split('T')[1][:8] if 'T' in v.created_at else v.created_at[:20]
                tags_str = ', '.join(v.tags[:3]) if v.tags else ''
                if len(v.tags) > 3:
                    tags_str += '...'
                print(f"  {version_str:<20} {checksum_str:<20} {created_str:<20} {tags_str:<20}")
    
    except Exception as e:
        print_error(f"Failed to list dataset versions: {str(e)}")
        sys.exit(1)

@data.command('validate')
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--rules', '-r', multiple=True, help='Specific rules to apply')
@click.option('--save', is_flag=True, help='Save validation results')
@click.option('--name', default='dataset', help='Dataset name for saved results')
@click.option('--base-dir', default='.neural_data', help='Base directory for data storage')
@click.pass_context
def data_validate(ctx, dataset_path, rules, save, name, base_dir):
    """Validate dataset quality."""
    print_command_header("data validate")
    print_info(f"Validating dataset: {dataset_path}")
    
    try:
        from neural.data import DataQualityValidator
        import numpy as np
        
        validator = DataQualityValidator(base_dir=base_dir)
        
        try:
            if dataset_path.endswith('.npy'):
                data = np.load(dataset_path)
            elif dataset_path.endswith('.npz'):
                data = np.load(dataset_path)['data']
            else:
                try:
                    import pandas as pd
                    data = pd.read_csv(dataset_path)
                except ImportError:
                    data = np.loadtxt(dataset_path)
        except Exception as e:
            print_warning(f"Could not load data for validation: {str(e)}")
            print_info("Proceeding with validation rules check only")
            data = None
        
        rules_list = list(rules) if rules else None
        
        with Spinner("Validating dataset") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            
            if save and data is not None:
                results = validator.validate_and_save(data, name, rules_list)
            elif data is not None:
                results = validator.validate(data, rules_list)
            else:
                results = []
        
        if results:
            print_success("Validation complete!")
            print(f"\n{Colors.CYAN}Validation Results:{Colors.ENDC}")
            
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            
            print(f"  {Colors.BOLD}Total Rules:{Colors.ENDC} {total}")
            print(f"  {Colors.BOLD}Passed:{Colors.ENDC}      {Colors.GREEN}{passed}{Colors.ENDC}")
            print(f"  {Colors.BOLD}Failed:{Colors.ENDC}      {Colors.RED}{total - passed}{Colors.ENDC}")
            
            print(f"\n{Colors.CYAN}Details:{Colors.ENDC}")
            for result in results:
                status_color = Colors.GREEN if result.passed else Colors.RED
                status = "PASS" if result.passed else "FAIL"
                print(f"  {status_color}[{status}]{Colors.ENDC} {result.rule_name}: {result.message}")
        else:
            print_warning("No validation results")
            print_info("Available rules:")
            for rule_name in validator.list_rules():
                print(f"  - {rule_name}")
    
    except Exception as e:
        print_error(f"Validation failed: {str(e)}")
        sys.exit(1)

@data.command('lineage')
@click.argument('graph_name')
@click.option('--trace', help='Node ID to trace lineage from')
@click.option('--visualize', '-v', is_flag=True, help='Generate lineage visualization')
@click.option('--output', '-o', help='Output path for visualization')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.option('--base-dir', default='.neural_data', help='Base directory for data storage')
@click.pass_context
def data_lineage(ctx, graph_name, trace, visualize, output, format, base_dir):
    """Show or visualize data lineage."""
    print_command_header("data lineage")
    
    try:
        from neural.data import LineageTracker
        
        tracker = LineageTracker(base_dir=base_dir)
        graph = tracker.get_graph(graph_name)
        
        if not graph:
            print_error(f"Lineage graph not found: {graph_name}")
            print_info("Available graphs:")
            for name in tracker.list_graphs():
                print(f"  - {name}")
            sys.exit(1)
        
        if trace:
            with Spinner("Tracing lineage") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                
                lineage = tracker.get_full_lineage(graph_name, trace)
            
            print_success(f"Lineage traced for node: {trace}")
            
            if format == 'json':
                print(json.dumps({
                    "upstream": [n.to_dict() for n in lineage['upstream']],
                    "downstream": [n.to_dict() for n in lineage['downstream']],
                }, indent=2))
            else:
                print(f"\n{Colors.CYAN}Upstream Dependencies:{Colors.ENDC}")
                for node in lineage['upstream']:
                    print(f"  {Colors.BOLD}{node.node_type}:{Colors.ENDC} {node.name} ({node.node_id})")
                
                print(f"\n{Colors.CYAN}Downstream Consumers:{Colors.ENDC}")
                for node in lineage['downstream']:
                    print(f"  {Colors.BOLD}{node.node_type}:{Colors.ENDC} {node.name} ({node.node_id})")
        
        if visualize:
            with Spinner("Generating lineage visualization") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                
                viz_path = tracker.visualize_lineage(graph_name, output)
            
            print_success(f"Lineage visualization saved: {viz_path}")
        
        if not trace and not visualize:
            print_info(f"Lineage graph: {graph_name}")
            print(f"\n{Colors.CYAN}Graph Statistics:{Colors.ENDC}")
            print(f"  {Colors.BOLD}Total Nodes:{Colors.ENDC}  {len(graph.nodes)}")
            print(f"  {Colors.BOLD}Total Edges:{Colors.ENDC}  {len(graph.edges)}")
            
            node_types = {}
            for node in graph.nodes.values():
                node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
            
            print(f"\n{Colors.CYAN}Node Types:{Colors.ENDC}")
            for ntype, count in node_types.items():
                print(f"  {Colors.BOLD}{ntype}:{Colors.ENDC} {count}")
    
    except Exception as e:
        print_error(f"Failed to process lineage: {str(e)}")
        sys.exit(1)

@cli.group()
@click.pass_context
def collab(ctx):
    """Commands for collaborative editing."""
    pass

@collab.command('create')
@click.argument('workspace_name')
@click.option('--description', '-d', help='Workspace description')
@click.option('--user-id', '-u', required=True, help='Your user ID')
@click.option('--base-dir', default='neural_workspaces', help='Base directory for workspaces')
@click.pass_context
def collab_create(ctx, workspace_name, description, user_id, base_dir):
    """Create a new collaborative workspace."""
    print_command_header("collab create")
    print_info(f"Creating workspace: {workspace_name}")

    try:
        from neural.collaboration import WorkspaceManager

        with Spinner("Creating workspace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            manager = WorkspaceManager(base_dir=base_dir)
            workspace = manager.create_workspace(workspace_name, user_id, description)

        print_success("Workspace created successfully!")
        print(f"\n{Colors.CYAN}Workspace Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}ID:{Colors.ENDC}          {workspace.workspace_id}")
        print(f"  {Colors.BOLD}Name:{Colors.ENDC}        {workspace.name}")
        print(f"  {Colors.BOLD}Owner:{Colors.ENDC}       {workspace.owner}")
        print(f"  {Colors.BOLD}Directory:{Colors.ENDC}   {workspace.workspace_dir}")
        if description:
            print(f"  {Colors.BOLD}Description:{Colors.ENDC} {description}")

    except Exception as e:
        print_error(f"Failed to create workspace: {str(e)}")
        sys.exit(1)

@collab.command('join')
@click.argument('workspace_id')
@click.option('--user-id', '-u', required=True, help='Your user ID')
@click.option('--username', '-n', required=True, help='Your username')
@click.option('--host', default='localhost', help='Collaboration server host')
@click.option('--port', default=8080, type=int, help='Collaboration server port')
@click.pass_context
def collab_join(ctx, workspace_id, user_id, username, host, port):
    """Join a collaborative workspace."""
    print_command_header("collab join")
    print_info(f"Connecting to workspace: {workspace_id}")

    try:
        import asyncio
        import json

        try:
            import websockets
        except ImportError:
            print_error("websockets package required")
            print_info("Install with: pip install websockets")
            sys.exit(1)

        async def connect_to_workspace():
            uri = f"ws://{host}:{port}"
            
            with Spinner("Connecting to collaboration server") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                
                try:
                    async with websockets.connect(uri) as websocket:
                        auth_message = {
                            'type': 'auth',
                            'workspace_id': workspace_id,
                            'user_id': user_id,
                            'username': username
                        }
                        
                        await websocket.send(json.dumps(auth_message))
                        
                        response = await websocket.recv()
                        data = json.loads(response)
                        
                        if data.get('type') == 'auth_success':
                            print_success(f"Connected to workspace successfully!")
                            print(f"\n{Colors.CYAN}Connection Information:{Colors.ENDC}")
                            print(f"  {Colors.BOLD}Workspace ID:{Colors.ENDC} {workspace_id}")
                            print(f"  {Colors.BOLD}Username:{Colors.ENDC}     {username}")
                            print(f"  {Colors.BOLD}Client ID:{Colors.ENDC}    {data.get('client_id')}")
                            print(f"\n{Colors.YELLOW}Press Ctrl+C to disconnect{Colors.ENDC}\n")
                            
                            while True:
                                message = await websocket.recv()
                                msg_data = json.loads(message)
                                
                                if msg_data.get('type') == 'user_joined':
                                    print(f"{Colors.GREEN}✓{Colors.ENDC} {msg_data.get('username')} joined the workspace")
                                elif msg_data.get('type') == 'user_left':
                                    print(f"{Colors.YELLOW}✗{Colors.ENDC} {msg_data.get('username')} left the workspace")
                                elif msg_data.get('type') == 'edit':
                                    print(f"{Colors.CYAN}✎{Colors.ENDC} {msg_data.get('username')} made an edit")
                        else:
                            print_error(f"Authentication failed: {data.get('message')}")
                
                except websockets.exceptions.ConnectionRefused:
                    print_error(f"Could not connect to server at {host}:{port}")
                    print_info("Make sure the collaboration server is running")
                    print_info("Start server with: neural collab server")
        
        asyncio.run(connect_to_workspace())

    except KeyboardInterrupt:
        print_info("\nDisconnected from workspace")
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")
        sys.exit(1)

@collab.command('server')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=8080, type=int, help='Server port')
@click.pass_context
def collab_server(ctx, host, port):
    """Start the collaboration server."""
    print_command_header("collab server")
    print_info(f"Starting collaboration server on {host}:{port}")

    try:
        from neural.collaboration import CollaborationServer

        print_success("Collaboration server starting...")
        print(f"\n{Colors.CYAN}Server Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Host:{Colors.ENDC} {host}")
        print(f"  {Colors.BOLD}Port:{Colors.ENDC} {port}")
        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}\n")

        server = CollaborationServer(host=host, port=port)
        server.start()

    except ImportError:
        print_error("Collaboration module not available")
        print_info("Install dependencies: pip install websockets")
        sys.exit(1)
    except Exception as e:
        print_error(f"Server failed: {str(e)}")
        sys.exit(1)

@collab.command('list')
@click.option('--user-id', '-u', help='Filter by user ID')
@click.option('--base-dir', default='neural_workspaces', help='Base directory for workspaces')
@click.pass_context
def collab_list(ctx, user_id, base_dir):
    """List collaborative workspaces."""
    print_command_header("collab list")

    try:
        from neural.collaboration import WorkspaceManager

        with Spinner("Loading workspaces") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            manager = WorkspaceManager(base_dir=base_dir)
            workspaces = manager.list_workspaces(user_id=user_id)

        if not workspaces:
            print_warning("No workspaces found")
            return

        print_success(f"Found {len(workspaces)} workspace(s)")
        print(f"\n{Colors.CYAN}Workspaces:{Colors.ENDC}\n")

        for ws in workspaces:
            print(f"{Colors.BOLD}{ws.name}{Colors.ENDC}")
            print(f"  ID:          {ws.workspace_id}")
            print(f"  Owner:       {ws.owner}")
            print(f"  Members:     {len(ws.members)}")
            print(f"  Files:       {len(ws.files)}")
            print(f"  Created:     {ws.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print()

    except Exception as e:
        print_error(f"Failed to list workspaces: {str(e)}")
        sys.exit(1)

@collab.command('info')
@click.argument('workspace_id')
@click.option('--base-dir', default='neural_workspaces', help='Base directory for workspaces')
@click.pass_context
def collab_info(ctx, workspace_id, base_dir):
    """Show workspace information."""
    print_command_header("collab info")

    try:
        from neural.collaboration import WorkspaceManager

        with Spinner("Loading workspace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            manager = WorkspaceManager(base_dir=base_dir)
            workspace = manager.get_workspace(workspace_id)

        if not workspace:
            print_error(f"Workspace not found: {workspace_id}")
            sys.exit(1)

        print(f"\n{Colors.CYAN}Workspace Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}ID:{Colors.ENDC}          {workspace.workspace_id}")
        print(f"  {Colors.BOLD}Name:{Colors.ENDC}        {workspace.name}")
        print(f"  {Colors.BOLD}Owner:{Colors.ENDC}       {workspace.owner}")
        print(f"  {Colors.BOLD}Directory:{Colors.ENDC}   {workspace.workspace_dir}")
        print(f"  {Colors.BOLD}Created:{Colors.ENDC}     {workspace.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  {Colors.BOLD}Updated:{Colors.ENDC}     {workspace.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if workspace.metadata.get('description'):
            print(f"\n{Colors.CYAN}Description:{Colors.ENDC}")
            print(f"  {workspace.metadata['description']}")

        print(f"\n{Colors.CYAN}Members ({len(workspace.members)}):{Colors.ENDC}")
        for member in workspace.members:
            role = workspace.get_role(member)
            print(f"  - {member} ({role})")

        if workspace.files:
            print(f"\n{Colors.CYAN}Files ({len(workspace.files)}):{Colors.ENDC}")
            for filename in workspace.files:
                print(f"  - {filename}")

    except Exception as e:
        print_error(f"Failed to get workspace info: {str(e)}")
        sys.exit(1)

@collab.command('sync')
@click.argument('workspace_id')
@click.option('--user-id', '-u', required=True, help='Your user ID')
@click.option('--base-dir', default='neural_workspaces', help='Base directory for workspaces')
@click.pass_context
def collab_sync(ctx, workspace_id, user_id, base_dir):
    """Sync workspace with version control."""
    print_command_header("collab sync")
    print_info(f"Synchronizing workspace: {workspace_id}")

    try:
        from neural.collaboration import WorkspaceManager, GitIntegration, SyncManager

        with Spinner("Loading workspace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            manager = WorkspaceManager(base_dir=base_dir)
            workspace = manager.get_workspace(workspace_id)

        if not workspace:
            print_error(f"Workspace not found: {workspace_id}")
            sys.exit(1)

        if not workspace.has_member(user_id):
            print_error("You are not a member of this workspace")
            sys.exit(1)

        with Spinner("Initializing Git") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            git = GitIntegration(workspace.workspace_dir)
            
            if not git.is_repo():
                git.init_repo()
                print_info("Initialized Git repository")

        with Spinner("Checking workspace status") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            status = git.get_status()

        if status['modified'] or status['untracked']:
            print(f"\n{Colors.CYAN}Workspace Status:{Colors.ENDC}")
            
            if status['modified']:
                print(f"\n  Modified files:")
                for f in status['modified']:
                    print(f"    {Colors.YELLOW}M{Colors.ENDC} {f}")
            
            if status['untracked']:
                print(f"\n  Untracked files:")
                for f in status['untracked']:
                    print(f"    {Colors.YELLOW}?{Colors.ENDC} {f}")
            
            if click.confirm("\nCommit changes?", default=True):
                git.add_files(['.'])
                commit_msg = click.prompt("Commit message", default="Update workspace")
                git.commit(commit_msg, author_name=user_id, author_email=f"{user_id}@neural.local")
                print_success("Changes committed")
        else:
            print_info("No changes to commit")

    except Exception as e:
        print_error(f"Sync failed: {str(e)}")
        sys.exit(1)

@collab.command('add-member')
@click.argument('workspace_id')
@click.argument('member_user_id')
@click.option('--role', default='member', type=click.Choice(['viewer', 'member', 'admin']), help='Member role')
@click.option('--owner-id', '-o', required=True, help='Your user ID (must be owner)')
@click.option('--base-dir', default='neural_workspaces', help='Base directory for workspaces')
@click.pass_context
def collab_add_member(ctx, workspace_id, member_user_id, role, owner_id, base_dir):
    """Add a member to workspace."""
    print_command_header("collab add-member")

    try:
        from neural.collaboration import WorkspaceManager

        with Spinner("Loading workspace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            manager = WorkspaceManager(base_dir=base_dir)
            workspace = manager.get_workspace(workspace_id)

        if not workspace:
            print_error(f"Workspace not found: {workspace_id}")
            sys.exit(1)

        if workspace.owner != owner_id:
            print_error("Only workspace owner can add members")
            sys.exit(1)

        workspace.add_member(member_user_id, role)
        manager.update_workspace(workspace)

        print_success(f"Added {member_user_id} to workspace as {role}")

    except Exception as e:
        print_error(f"Failed to add member: {str(e)}")
        sys.exit(1)

@collab.command('remove-member')
@click.argument('workspace_id')
@click.argument('member_user_id')
@click.option('--owner-id', '-o', required=True, help='Your user ID (must be owner)')
@click.option('--base-dir', default='neural_workspaces', help='Base directory for workspaces')
@click.pass_context
def collab_remove_member(ctx, workspace_id, member_user_id, owner_id, base_dir):
    """Remove a member from workspace."""
    print_command_header("collab remove-member")

    try:
        from neural.collaboration import WorkspaceManager

        with Spinner("Loading workspace") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            manager = WorkspaceManager(base_dir=base_dir)
            workspace = manager.get_workspace(workspace_id)

        if not workspace:
            print_error(f"Workspace not found: {workspace_id}")
            sys.exit(1)

        if workspace.owner != owner_id:
            print_error("Only workspace owner can remove members")
            sys.exit(1)

        workspace.remove_member(member_user_id)
        manager.update_workspace(workspace)

        print_success(f"Removed {member_user_id} from workspace")

    except Exception as e:
        print_error(f"Failed to remove member: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
