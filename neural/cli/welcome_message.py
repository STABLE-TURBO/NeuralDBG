"""
Welcome message module for Neural CLI.
Displays a welcome message the first time the CLI is run.
"""

import logging
import os
from pathlib import Path
import sys

from .cli_aesthetics import Colors, animate_neural_network, print_error, print_neural_logo


logger = logging.getLogger(__name__)

WELCOME_MESSAGE = f"""
{Colors.CYAN}Welcome to Neural CLI!{Colors.ENDC}

Neural is a powerful tool for building, training, and visualizing neural networks.
Here are some commands to get you started:

  {Colors.BOLD}neural version{Colors.ENDC}              - Show version information
  {Colors.BOLD}neural visualize <file>{Colors.ENDC}     - Visualize a neural network
  {Colors.BOLD}neural compile <file>{Colors.ENDC}       - Compile a Neural DSL file
  {Colors.BOLD}neural run <file>{Colors.ENDC}           - Run a compiled model
  {Colors.BOLD}neural debug <file>{Colors.ENDC}         - Debug a neural network
  {Colors.BOLD}neural no-code{Colors.ENDC}              - Launch the no-code interface
  {Colors.BOLD}neural clean{Colors.ENDC}                - Clean generated files

{Colors.CYAN}Experiment Tracking:{Colors.ENDC}
  {Colors.BOLD}neural track init{Colors.ENDC}           - Initialize experiment tracking
  {Colors.BOLD}neural track log{Colors.ENDC}            - Log metrics and artifacts
  {Colors.BOLD}neural track list{Colors.ENDC}           - List all experiments
  {Colors.BOLD}neural track show <id>{Colors.ENDC}      - Show experiment details
  {Colors.BOLD}neural track plot <id>{Colors.ENDC}      - Plot experiment metrics
  {Colors.BOLD}neural track compare <id1> <id2>{Colors.ENDC} - Compare experiments

For more information, run {Colors.BOLD}neural --help{Colors.ENDC} or {Colors.BOLD}neural <command> --help{Colors.ENDC}

{Colors.YELLOW}Happy neural network building!{Colors.ENDC}
"""

def show_welcome_message():
    """Show the welcome message if it's the first time the CLI is run.

    Returns:
        bool: True if the welcome message was shown, False otherwise.
    """
    try:
        # Get the user's home directory
        home_dir = Path.home()
        neural_dir = home_dir / ".neural"
        neural_dir.mkdir(exist_ok=True)
        welcome_file = neural_dir / "welcome_shown"

        if not welcome_file.exists():
            print_neural_logo()
            print(WELCOME_MESSAGE)
            logger.info("Displaying welcome message to new user")
            logger.info("Here's a preview of what Neural can visualize:")
            if not os.environ.get('NEURAL_NO_ANIMATIONS'):
                animate_neural_network(3)
            welcome_file.touch()

            if os.environ.get('NEURAL_SKIP_WELCOME') or not sys.stdout.isatty():
                return True

            print(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            input()
            return True
        return False
    except (PermissionError, OSError) as e:
        print_error(f"Failed to manage welcome message: {str(e)}")
        return False
