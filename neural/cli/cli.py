#!/usr/bin/env python
"""
Main CLI implementation for Neural using Click.
"""

import os
import sys
import subprocess
import click
import logging
import hashlib
import shutil
import time
from pathlib import Path
from typing import Optional
from lark import exceptions
import pysnooper

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
    get_module,
    tensorflow, torch, jax, optuna
)

def configure_logging(verbose=False):
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
@click.pass_context
def compile(ctx, file: str, backend: str, dataset: str, output: Optional[str], dry_run: bool, hpo: bool):
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
            code = generate_code(model_data, backend)
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

#### RUN COMMAND #####

@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--backend', '-b', default='tensorflow', help='Backend to run', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
@click.option('--hpo', is_flag=True, help='Enable HPO for .neural files')
@click.option('--device', '-d', default='auto', help='Device to use (auto, cpu, gpu)', type=click.Choice(['auto', 'cpu', 'gpu'], case_sensitive=False))
@click.pass_context
@pysnooper.snoop()
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
            if not ctx.obj.get('NO_ANIMATIONS'):
                progress_bar(i, total_layers, prefix='Progress:', suffix=f'Layer: {layer["type"]}', length=40)
            input_shape = propagator.propagate(input_shape, layer, model_data.get('framework', 'tensorflow'))
            shape_history.append({"layer": layer['type'], "output_shape": input_shape})
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
            dot.render('architecture', cleanup=True)
            if format == 'html':
                report['plotly_chart'].write_html('shape_propagation.html')
                create_animated_network = get_module(tensor_flow_module).create_animated_network
                create_animated_network(shape_history).write_html('tensor_flow.html')
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
@click.pass_context
def clean(ctx):
    """Remove generated files (e.g., .py, .png, .svg, .html, cache)."""
    print_command_header("clean")
    print_info("Cleaning up generated files...")

    extensions = ['.py', '.png', '.svg', '.html']
    removed_files = []

    try:
        with Spinner("Scanning for generated files") as spinner:
            if ctx.obj.get('NO_ANIMATIONS'):
                spinner.stop()
            for file in os.listdir('.'):
                if any(file.endswith(ext) for ext in extensions):
                    os.remove(file)
                    removed_files.append(file)
    except (PermissionError, OSError) as e:
        print_error(f"Failed to remove files: {str(e)}")
        sys.exit(1)

    if removed_files:
        print_success(f"Removed {len(removed_files)} generated files")
        for file in removed_files[:5]:
            print(f"  - {file}")
        if len(removed_files) > 5:
            print(f"  - ...and {len(removed_files) - 5} more")

    if os.path.exists(".neural_cache"):
        try:
            with Spinner("Removing cache directory") as spinner:
                if ctx.obj.get('NO_ANIMATIONS'):
                    spinner.stop()
                shutil.rmtree(".neural_cache")
            print_success("Removed cache directory")
        except (PermissionError, OSError) as e:
            print_error(f"Failed to remove cache directory: {str(e)}")
            sys.exit(1)

    if not removed_files and not os.path.exists(".neural_cache"):
        print_warning("No files to clean")

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
@click.option('--gradients', is_flag=True, help='Analyze gradient flow')
@click.option('--dead-neurons', is_flag=True, help='Detect dead neurons')
@click.option('--anomalies', is_flag=True, help='Detect training anomalies')
@click.option('--step', is_flag=True, help='Enable step debugging mode')
@click.option('--backend', '-b', default='tensorflow', help='Backend for runtime', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
@click.pass_context
def debug(ctx, file: str, gradients: bool, dead_neurons: bool, anomalies: bool, step: bool, backend: str, dataset: str):
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

    # Debugging modes
    if gradients:
        print(f"\n{Colors.CYAN}Gradient Flow Analysis{Colors.ENDC}")
        print_warning("Gradient flow analysis is simulated")
        for entry in trace_data:
            print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: mean_activation = {entry.get('mean_activation', 'N/A')}")

    if dead_neurons:
        print(f"\n{Colors.CYAN}Dead Neuron Detection{Colors.ENDC}")
        print_warning("Dead neuron detection is simulated")
        for entry in trace_data:
            print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: active_ratio = {entry.get('active_ratio', 'N/A')}")

    if anomalies:
        print(f"\n{Colors.CYAN}Anomaly Detection{Colors.ENDC}")
        print_warning("Anomaly detection is simulated")
        anomaly_found = False
        for entry in trace_data:
            if 'anomaly' in entry:
                print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: {entry['anomaly']}")
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

    print_success("Debug session completed!")
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

if __name__ == '__main__':
    cli()
