#!/usr/bin/env python
"""
Neural CLI entry point.
This script is installed as the 'neural' command when the package is installed.
It sets up the environment to suppress debug messages before importing the CLI.
"""

import os
import sys

# Set environment variables to suppress debug messages from dependencies
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['PYTHONWARNINGS'] = 'ignore'    # Suppress Python warnings
os.environ['MPLBACKEND'] = 'Agg'           # Non-interactive matplotlib backend

# Optionally suppress stderr if explicitly requested via env var.
# Set NEURAL_SUPPRESS_STDERR=1 to enable. Default is off to avoid hiding CLI output.
if os.environ.get('NEURAL_SUPPRESS_STDERR') == '1':
    try:
        # Save the original stderr
        original_stderr = sys.stderr
        # Open /dev/null for writing
        null_fd = open(os.devnull, 'w')
        # Replace stderr with /dev/null
        sys.stderr = null_fd
        # Register a cleanup function to restore stderr when the program exits
        import atexit
        def restore_stderr():
            sys.stderr = original_stderr
            null_fd.close()
        atexit.register(restore_stderr)
    except Exception:
        # If anything goes wrong, just continue with the original stderr
        pass

# Import the CLI after setting up the environment
from neural.cli import cli

if __name__ == '__main__':
    # Run the CLI
    cli()
