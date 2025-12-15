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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['MPLBACKEND'] = 'Agg'

    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    neural_logger = logging.getLogger('neural')
    neural_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s" if verbose else "%(levelname)s: %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    neural_logger.handlers = [handler]

    for logger_name in ['neural.parser', 'neural.code_generation', 'neural.hpo']:
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(logging.WARNING if not verbose else logging.DEBUG)
        module_logger.handlers = [handler]
        module_logger.propagate = False

    for logger_name in [
        'graphviz', 'matplotlib', 'tensorflow', 'jax', 'tf', 'absl',
        'pydot', 'PIL', 'torch', 'urllib3', 'requests', 'h5py',
        'filelock', 'numba', 'asyncio', 'parso', 'werkzeug',
        'matplotlib.font_manager', 'matplotlib.ticker', 'optuna',
        'dash', 'plotly', 'ipykernel', 'traitlets', 'click'
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        logging.getLogger(logger_name).propagate = False

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"MNIST", "CIFAR10", "CIFAR100", "ImageNet"}

def sanitize_file_path(file_path: str, allow_absolute: bool = True) -> str:
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")
    
    file_path = file_path.replace('\0', '')
    normalized = os.path.normpath(file_path)
    
    if '..' in normalized.split(os.sep):
        raise ValueError(f"Path traversal detected in: {file_path}")
    
    if not allow_absolute and os.path.isabs(normalized):
        raise ValueError(f"Absolute paths not allowed: {file_path}")
    
    suspicious_patterns = [r'\.\./', r'\.\.$', r'^\.\.', r'~/', r'\$\{', r'%[A-Z_]+%']
    for pattern in suspicious_patterns:
        if re.search(pattern, file_path):
            raise ValueError(f"Suspicious pattern detected in path: {file_path}")
    
    return normalized

def validate_port(port: int, min_port: int = 1024, max_port: int = 65535) -> int:
    if not isinstance(port, int):
        try:
            port = int(port)
        except (TypeError, ValueError):
            raise ValueError(f"Port must be an integer, got: {type(port).__name__}")
    
    if port < min_port or port > max_port:
        raise ValueError(f"Port must be between {min_port} and {max_port}, got: {port}")
    
    return port

def validate_backend(backend: str) -> str:
    if not backend or not isinstance(backend, str):
        raise ValueError("Backend must be a non-empty string")
    
    backend = backend.lower().strip()
    valid_backends = {'tensorflow', 'pytorch', 'onnx', 'jax'}
    
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend '{backend}'. Supported: {', '.join(sorted(valid_backends))}")
    
    return backend

def validate_dataset_name(dataset: str) -> str:
    if not dataset or not isinstance(dataset, str):
        raise ValueError("Dataset name must be a non-empty string")
    
    dataset = dataset.strip()
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', dataset):
        raise ValueError(f"Dataset name contains invalid characters: {dataset}")
    
    if len(dataset) > 100:
        raise ValueError(f"Dataset name too long (max 100 characters): {dataset}")
    
    return dataset

def validate_json_input(json_str: str, max_size: int = 1024 * 1024) -> dict:
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
    if not name or not isinstance(name, str):
        raise ValueError("Name must be a non-empty string")
    
    name = name.strip()
    
    if len(name) < 1 or len(name) > 200:
        raise ValueError(f"Name length must be between 1 and 200 characters, got: {len(name)}")
    
    if not re.match(r'^[a-zA-Z0-9_\- ]+$', name):
        raise ValueError(f"Name contains invalid characters: {name}")
    
    return name

@click.group(name="cli", context_settings={"help_option_names": ["-h", "--help"]})
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

    if not os.environ.get('NEURAL_SKIP_WELCOME'):
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

    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}. Supported: .neural, .nr, .rnr")
        sys.exit(1)

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

    if hpo:
        print_info("Running hyperparameter optimization")
        if dataset not in SUPPORTED_DATASETS:
            print_warning(f"Dataset '{dataset}' may not be supported. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}")

        try:
            from neural.hpo.hpo import optimize_and_return
            from neural.code_generation.code_generator import generate_optimized_dsl
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

    try:
        from neural.code_generation.code_generator import generate_code
        with Spinner(f"Generating {backend} code") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            code = generate_code(model_data, backend, auto_flatten_output=auto_flatten_output)
    except Exception as e:
        print_error(f"Code generation failed: {str(e)}")
        sys.exit(1)

    if dry_run:
        print_info("Generated code (dry run)")
        print(f"\n{Colors.CYAN}{'='*50}{Colors.ENDC}")
        print(code)
        print(f"{Colors.CYAN}{'='*50}{Colors.ENDC}")
        print_success("Dry run complete! No files were created.")
    else:
        output_file = output or f"{os.path.splitext(file)[0]}_{backend}.py"
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
    
    try:
        file = sanitize_file_path(file)
        output = sanitize_file_path(output)
    except ValueError as e:
        print_error(f"Invalid input: {str(e)}")
        sys.exit(1)
    
    print_info(f"Generating documentation for {file}")

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
    
    try:
        file = sanitize_file_path(file)
        backend = validate_backend(backend)
        dataset = validate_dataset_name(dataset)
    except ValueError as e:
        print_error(f"Invalid input: {str(e)}")
        sys.exit(1)
    
    print_info(f"Running {file} with {backend} backend")

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

    try:
        from neural.shape_propagation.shape_propagator import ShapePropagator
        propagator = ShapePropagator()
        input_shape = model_data['input']['shape']
        if not input_shape:
            print_error("Input shape not defined in model")
            sys.exit(1)

        print_info("Propagating shapes through the network...")
        shape_history = []
        total_layers = len(model_data['layers'])
        for i, layer in enumerate(model_data['layers']):
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

    try:
        with Spinner("Generating visualizations") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            report = propagator.generate_report()
            dot = report['dot_graph']
            dot.format = format if format != 'html' else 'svg'

            try:
                dot.render('architecture', cleanup=True)
            except Exception as graphviz_error:
                print_warning(f"Graphviz rendering failed: {str(graphviz_error)}")
                print_info("Continuing with other visualizations...")

            if format == 'html':
                try:
                    report['plotly_chart'].write_html('shape_propagation.html')
                    from neural.dashboard.tensor_flow import create_animated_network
                    create_animated_network(shape_history).write_html('tensor_flow.html')
                except Exception as html_error:
                    print_warning(f"HTML visualization generation failed: {str(html_error)}")
    except Exception as e:
        print_error(f"Visualization generation failed: {str(e)}")
        sys.exit(1)

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

    if cache:
        try:
            with Spinner("Caching visualization") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                shutil.copy(f"architecture.{format}", cache_file)
            print_info("Visualization cached for future use")
        except (PermissionError, IOError) as e:
            print_warning(f"Failed to cache visualization: {str(e)}")
    
    if attention:
        print_info("Visualizing attention weights...")
        
        try:
            from neural.explainability.attention_visualizer import AttentionVisualizer
            
            token_list = None
            if tokens:
                token_list = [t.strip() for t in tokens.split(',')]
            
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
            
            attention_output = 'attention_outputs'
            os.makedirs(attention_output, exist_ok=True)
            
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
    """Remove generated artifacts safely (dry-run by default)."""
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
@click.option('--host', default='localhost', help='Server host address')
@click.option('--port', default=8050, type=int, help='Server port')
@click.option('--no-browser', is_flag=True, help='Do not open browser automatically')
@click.option('--features', multiple=True, help='Specific features to enable (e.g., debug, nocode, monitoring)')
@click.pass_context
def server(ctx, host: str, port: int, no_browser: bool, features: tuple):
    """Start unified Neural DSL web server (dashboard, builder, monitoring)."""
    print_command_header("server")
    
    try:
        from neural.server import start_server
        from neural.config import get_config
        
        config = get_config()
        
        if features:
            print_info(f"Starting server with features: {', '.join(features)}")
            feature_list = list(features)
        else:
            enabled = config.get_all_enabled_features()
            print_info(f"Starting server with enabled features: {', '.join(enabled)}")
            feature_list = None
        
        print_success(f"Server will start on http://{host}:{port}")
        print_info("Press Ctrl+C to stop the server")
        
        if not no_browser and not ctx.obj.get('NO_ANIMATIONS'):
            print_info("Opening browser...")
            import webbrowser
            import threading
            def open_browser():
                time.sleep(1.5)
                webbrowser.open(f"http://{host}:{port}")
            threading.Thread(target=open_browser, daemon=True).start()
        
        start_server(host=host, port=port, debug=False, features=feature_list)
        
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
    for pkg, module_name in [('torch', 'torch'), ('tensorflow', 'tensorflow'), ('jax', 'jax'), ('optuna', 'optuna')]:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            ver = mod.__version__
            print(f"  {Colors.BOLD}{pkg.capitalize()}:{Colors.ENDC}" + " " * (12 - len(pkg)) + f"{ver}")
        except (ImportError, AttributeError):
            print(f"  {Colors.BOLD}{pkg.capitalize()}:{Colors.ENDC}" + " " * (12 - len(pkg)) + f"{Colors.YELLOW}Not installed{Colors.ENDC}")

    if not ctx.obj.get('NO_ANIMATIONS'):
        print("\nNeural is ready to build amazing neural networks!")
        animate_neural_network(2)

# Import monitoring commands if available
try:
    from neural.monitoring.cli_commands import monitor
    cli.add_command(monitor)
except ImportError:
    pass

# Dynamic command imports
_dynamic_imports = [
    ('neural.cli.cloud_commands', 'cloud'),
    ('neural.cli.config_commands', 'config'),
    ('neural.cli.track_commands', 'track'),
    ('neural.cli.debug_commands', 'debug'),
    ('neural.cli.no_code_commands', 'no_code'),
    ('neural.cli.explain_commands', 'explain'),
    ('neural.cli.cost_commands', 'cost'),
    ('neural.cli.data_commands', 'data'),
]

for module_name, command_name in _dynamic_imports:
    try:
        import importlib
        mod = importlib.import_module(module_name)
        if hasattr(mod, command_name):
            cli.add_command(getattr(mod, command_name))
    except ImportError:
        pass


if __name__ == '__main__':
    cli()
