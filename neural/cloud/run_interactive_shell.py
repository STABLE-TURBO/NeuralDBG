#!/usr/bin/env python3
"""
Run the Neural Cloud Interactive Shell with a more aesthetic interface.
"""

import os
import sys
import argparse
import time
from pathlib import Path

def print_ascii_art():
    """Print a more aesthetic ASCII art header."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗                        ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║                        ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║                        ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║                        ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗                   ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝                   ║
║                                                                          ║
║   ██████╗██╗      ██████╗ ██╗   ██╗██████╗                              ║
║  ██╔════╝██║     ██╔═══██╗██║   ██║██╔══██╗                             ║
║  ██║     ██║     ██║   ██║██║   ██║██║  ██║                             ║
║  ██║     ██║     ██║   ██║██║   ██║██║  ██║                             ║
║  ╚██████╗███████╗╚██████╔╝╚██████╔╝██████╔╝                             ║
║   ╚═════╝╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝                              ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

def print_loading_bar(platform):
    """Print a loading bar with platform-specific emoji."""
    platform_emoji = {
        'kaggle': '🏆',
        'colab': '🧪',
        'sagemaker': '☁️'
    }.get(platform.lower(), '🌐')

    print(f"\n{platform_emoji} Connecting to {platform.upper()}...")

    # Print a loading bar
    bar_length = 40
    for i in range(bar_length + 1):
        progress = i / bar_length
        bar = '█' * i + '░' * (bar_length - i)
        percentage = int(progress * 100)

        # Use carriage return to update the same line
        sys.stdout.write(f"\r[{bar}] {percentage}%")
        sys.stdout.flush()
        time.sleep(0.02)  # Adjust speed as needed

    print("\n")

def main():
    """Run the Neural Cloud Interactive Shell."""
    parser = argparse.ArgumentParser(description='Run the Neural Cloud Interactive Shell')
    parser.add_argument('platform', choices=['kaggle', 'colab', 'sagemaker'],
                        help='The cloud platform to connect to')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    if not args.quiet:
        print_ascii_art()
        print_loading_bar(args.platform)

    # Import the interactive shell module
    try:
        from neural.cloud.interactive_shell import start_interactive_shell

        # Import the remote connection module
        try:
            from neural.cloud.remote_connection import RemoteConnection
            remote = RemoteConnection()
        except ImportError:
            if not args.quiet:
                print("Installing required dependencies...")
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "boto3", "kaggle"],
                stdout=subprocess.DEVNULL if args.quiet else None,
                stderr=subprocess.DEVNULL if args.quiet else None
            )
            from neural.cloud.remote_connection import RemoteConnection
            remote = RemoteConnection()

        # Start the interactive shell
        start_interactive_shell(args.platform, remote, quiet=args.quiet)

    except ImportError:
        print("Error: Neural Cloud modules not found.")
        print("Please make sure Neural is installed correctly.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == '__main__':
    main()
