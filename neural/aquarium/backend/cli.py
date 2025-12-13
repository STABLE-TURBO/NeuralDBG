"""CLI for running the backend bridge server."""

import logging

import click
import uvicorn

from .config import settings
from .server import create_app

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Neural DSL Backend Bridge CLI."""
    pass


@cli.command()
@click.option("--host", default=settings.host, help="Host to bind to")
@click.option("--port", default=settings.port, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--log-level", default=settings.log_level, help="Log level")
def serve(host: str, port: int, reload: bool, log_level: str):
    """Start the backend bridge server."""
    click.echo(f"Starting {settings.app_name} v{settings.version}")
    click.echo(f"Server: http://{host}:{port}")
    click.echo(f"API Docs: http://{host}:{port}/docs")

    uvicorn.run(
        "neural.aquarium.backend.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )


@cli.command()
def info():
    """Display backend bridge information."""
    click.echo(f"Application: {settings.app_name}")
    click.echo(f"Version: {settings.version}")
    click.echo(f"Default Host: {settings.host}")
    click.echo(f"Default Port: {settings.port}")
    click.echo(f"Log Level: {settings.log_level}")


if __name__ == "__main__":
    cli()
