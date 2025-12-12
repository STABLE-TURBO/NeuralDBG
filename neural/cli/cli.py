#!/usr/bin/env python
"""
Main CLI implementation for Neural using Click.
"""
from __future__ import annotations

import os
import sys
import subprocess
import click
import logging
import hashlib
import shutil
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional
from lark import exceptions
# Optional debugging dependency
try:
    import pysnooper
    _HAS_PYSNOOPER = True
except ImportError:
    pysnooper = None
    _HAS_PYSNOOPER = False

# Import CLI aesthetics
from .cli_aesthetics import (
    print_neural_logo, print_command_header, print_success,
    print_error, print_warning, print_info, Spinner,
    progress_bar, animate_neural_network, Colors,
    print_help_command
)

# Import welcome message
from .welcome_message import show_welcome_message

# Import version
from .version import __version__

# Import CPU mode
from .cpu_mode import set_cpu_mode, is_cpu_mode

# Import lazy loaders
from .lazy_imports import (
    shape_propagator as shape_propagator_module,
    tensor_flow as tensor_flow_module,
    hpo as hpo_module,
    code_generator as code_generator_module,
    experiment_tracker as experiment_tracker_module,
    get_module,
    tensorflow, torch, jax, optuna
)

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

# Global CLI context
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--cpu', is_flag=True, help='Force CPU mode')
@click.option('--no-animations', is_flag=True, help='Disable animations and spinners')
@click.version_option(version=__version__, prog_name="Neural")
@click.pass_context
def cli(ctx, verbose: bool, cpu: bool, no_animations: bool):
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
def help(ctx):
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



#### RUN COMMAND #####

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
def cloud(ctx):
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
def cloud_run(ctx, setup_tunnel: bool, port: int):
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
            'kaggle': '🏆',
            'colab': '🧪',
            'sagemaker': '☁️'
        }.get(platform.lower(), '🌐')

        print("\n" + "─" * 60)
        print(f"  {platform_emoji}  Neural Cloud Connect: {platform.upper()}")
        print("─" * 60 + "\n")

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
def cloud_execute(ctx, platform: str, file: str, name: str):
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
@click.option('--backend', '-b', default='tensorflow', help='Backend framework', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--format', '-f', 'export_format', default='onnx', help='Export format', type=click.Choice(['onnx', 'tflite', 'torchscript', 'savedmodel'], case_sensitive=False))
@click.option('--output', '-o', default=None, help='Output path (defaults based on format)')
@click.option('--optimize', is_flag=True, help='Apply optimization passes')
@click.option('--quantize', is_flag=True, help='Apply quantization (TFLite only)')
@click.option('--quantization-type', default='int8', type=click.Choice(['int8', 'float16', 'dynamic']), help='Quantization type (TFLite only)')
@click.option('--deployment', type=click.Choice(['torchserve', 'tfserving', 'none'], case_sensitive=False), default='none', help='Generate deployment configs')
@click.option('--model-name', default='model', help='Name for deployed model')
@click.pass_context
def export(ctx, file: str, backend: str, export_format: str, output: Optional[str], optimize: bool, quantize: bool, quantization_type: str, deployment: str, model_name: str):
    """Export a model for deployment in various formats."""
    print_command_header("export")
    print_info(f"Exporting {file} to {export_format} format")
    
    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}. Supported: .neural, .nr, .rnr")
        sys.exit(1)
    
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
    except Exception as e:
        print_error(f"Parsing failed: {str(e)}")
        sys.exit(1)
    
    try:
        from neural.code_generation.export import ModelExporter
        exporter = ModelExporter(model_data, backend)
        
        if output is None:
            format_extensions = {
                'onnx': '.onnx',
                'tflite': '.tflite',
                'torchscript': '.pt',
                'savedmodel': '_saved_model'
            }
            output = f"{os.path.splitext(file)[0]}{format_extensions[export_format]}"
        
        with Spinner(f"Exporting to {export_format}") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            
            if export_format == 'onnx':
                output_path = exporter.export_onnx(output, optimize=optimize)
            elif export_format == 'tflite':
                if backend != 'tensorflow':
                    print_error("TFLite export requires TensorFlow backend")
                    sys.exit(1)
                output_path = exporter.export_tflite(
                    output,
                    quantize=quantize,
                    quantization_type=quantization_type
                )
            elif export_format == 'torchscript':
                if backend != 'pytorch':
                    print_error("TorchScript export requires PyTorch backend")
                    sys.exit(1)
                output_path = exporter.export_torchscript(output)
            elif export_format == 'savedmodel':
                if backend != 'tensorflow':
                    print_error("SavedModel export requires TensorFlow backend")
                    sys.exit(1)
                output_path = exporter.export_savedmodel(output)
        
        print_success(f"Model exported successfully!")
        print(f"\n{Colors.CYAN}Export Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}Format:{Colors.ENDC}   {export_format}")
        print(f"  {Colors.BOLD}Output:{Colors.ENDC}   {output_path}")
        print(f"  {Colors.BOLD}Backend:{Colors.ENDC}  {backend}")
        if optimize:
            print(f"  {Colors.BOLD}Optimized:{Colors.ENDC} Yes")
        if quantize:
            print(f"  {Colors.BOLD}Quantized:{Colors.ENDC} Yes ({quantization_type})")
        
        if deployment != 'none':
            print_info(f"Generating {deployment} deployment configs...")
            deploy_dir = f"{os.path.splitext(file)[0]}_deployment"
            
            if deployment == 'torchserve':
                config_path, model_store = exporter.create_torchserve_config(
                    output_path, model_name, deploy_dir
                )
                scripts = exporter.generate_deployment_scripts(deploy_dir, 'torchserve')
            elif deployment == 'tfserving':
                config_path = exporter.create_tfserving_config(
                    output_path, model_name, deploy_dir
                )
                scripts = exporter.generate_deployment_scripts(deploy_dir, 'tfserving')
            
            print_success("Deployment configs generated!")
            print(f"\n{Colors.CYAN}Deployment Files:{Colors.ENDC}")
            print(f"  {Colors.BOLD}Directory:{Colors.ENDC} {deploy_dir}")
            print(f"  {Colors.BOLD}Config:{Colors.ENDC}    {config_path}")
            for script in scripts:
                print(f"  {Colors.BOLD}Script:{Colors.ENDC}    {script}")
    
    except Exception as e:
        print_error(f"Export failed: {str(e)}")
        import traceback
        if ctx.obj.get('VERBOSE'):
            traceback.print_exc()
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
def debug(ctx, file: str, gradients: bool, dead_neurons: bool, anomalies: bool, step: bool, backend: str, dataset: str, dashboard: bool, port: int):
    """Debug a neural network model with NeuralDbg."""
    print_command_header("debug")
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
def no_code(ctx, port: int):
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


@cli.command(name='experiments')
@click.option('--port', default=8052, help='Web interface port', type=int)
@click.option('--base-dir', default='neural_experiments', help='Experiments directory', type=click.Path())
@click.option('--host', default='127.0.0.1', help='Server host')
@click.pass_context
def experiments(ctx, port: int, base_dir: str, host: str):
    """Launch the experiment comparison UI."""
    print_command_header("experiments")
    print_info("Loading experiment tracking system...")
    
    try:
        from neural.tracking import ExperimentManager, ExperimentComparisonUI
        
        with Spinner("Initializing experiment manager") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            manager = ExperimentManager(base_dir=base_dir)
            experiments_list = manager.list_experiments()
        
        print_success(f"Found {len(experiments_list)} experiments in {base_dir}")
        
        if len(experiments_list) > 0:
            print(f"\n{Colors.CYAN}Recent Experiments:{Colors.ENDC}")
            for exp in experiments_list[:5]:
                status_color = Colors.GREEN if exp['status'] == 'completed' else Colors.YELLOW
                print(f"  {status_color}●{Colors.ENDC} {exp['experiment_name']} ({exp['experiment_id']}) - {exp['status']}")
            if len(experiments_list) > 5:
                print(f"  {Colors.CYAN}... and {len(experiments_list) - 5} more{Colors.ENDC}")
        else:
            print_warning("No experiments found. Run some experiments first!")
        
        print(f"\n{Colors.CYAN}Server Information:{Colors.ENDC}")
        print(f"  {Colors.BOLD}URL:{Colors.ENDC}         http://{host}:{port}")
        print(f"  {Colors.BOLD}Interface:{Colors.ENDC}   Experiment Comparison UI")
        print(f"  {Colors.BOLD}Base Dir:{Colors.ENDC}    {base_dir}")
        print(f"\n{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}\n")
        
        with Spinner("Starting web server") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            ui = ExperimentComparisonUI(manager=manager, port=port)
        
        ui.run(debug=False, host=host)
        
    except ImportError as e:
        print_error(f"Failed to load experiment tracking: {str(e)}")
        print_info("Install required packages: pip install dash dash-bootstrap-components plotly")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to launch experiment comparison UI: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("Server stopped by user")


if __name__ == '__main__':
    cli()
