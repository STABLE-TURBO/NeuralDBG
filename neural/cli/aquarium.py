"""
CLI command for launching the Aquarium experiment tracking dashboard.
"""

import click

from neural.tracking import launch_aquarium


@click.command()
@click.option(
    "--base-dir",
    default="neural_experiments",
    help="Base directory containing experiments",
    type=str,
)
@click.option(
    "--port",
    default=8053,
    help="Port to run the dashboard on",
    type=int,
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to run the dashboard on",
    type=str,
)
@click.option(
    "--debug",
    is_flag=True,
    help="Run in debug mode",
)
def aquarium(base_dir, port, host, debug):
    """
    Launch the Aquarium experiment tracking dashboard.

    Aquarium provides a comprehensive web interface for viewing, comparing,
    and exporting Neural experiments. Features include:

    - Interactive experiment list with filtering and sorting
    - Detailed experiment views with metrics, hyperparameters, and artifacts
    - Side-by-side experiment comparison with visualization
    - Export to MLflow, Weights & Biases, and TensorBoard

    Example:

        neural aquarium --port 8053

        neural aquarium --base-dir ./my_experiments --port 8080
    """
    click.echo("Starting Aquarium Experiment Tracking Dashboard...")
    click.echo(f"Base directory: {base_dir}")
    click.echo(f"URL: http://{host}:{port}")
    click.echo()

    launch_aquarium(base_dir=base_dir, port=port, host=host, debug=debug)


if __name__ == "__main__":
    aquarium()
