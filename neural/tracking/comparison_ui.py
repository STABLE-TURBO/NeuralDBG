"""
Standalone script to launch the experiment comparison UI.

Usage:
    python neural/tracking/comparison_ui.py
    python neural/tracking/comparison_ui.py --port 8052 --base-dir ./my_experiments
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.tracking import ExperimentManager, ExperimentComparisonUI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Neural Experiment Comparison UI"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="neural_experiments",
        help="Base directory containing experiments (default: neural_experiments)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8052,
        help="Port to run the UI on (default: 8052)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the UI on (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )

    args = parser.parse_args()

    print(f"Starting Neural Experiment Comparison UI...")
    print(f"Base directory: {args.base_dir}")
    print(f"URL: http://{args.host}:{args.port}")
    print()

    manager = ExperimentManager(base_dir=args.base_dir)
    
    experiments = manager.list_experiments()
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments[:5]:
        print(f"  - {exp['experiment_name']} ({exp['experiment_id']}) - {exp['status']}")
    if len(experiments) > 5:
        print(f"  ... and {len(experiments) - 5} more")
    print()

    ui = ExperimentComparisonUI(manager=manager, port=args.port)
    ui.run(debug=args.debug, host=args.host)


if __name__ == "__main__":
    main()
