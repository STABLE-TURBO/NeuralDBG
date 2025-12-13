#!/usr/bin/env python3
"""
Master Automation Script

Orchestrates all automation tasks for Neural DSL.
Run this script to handle releases, tests, and validation.
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.automation.release_automation import ReleaseAutomation
from scripts.automation.example_validator import ExampleValidator
from scripts.automation.test_automation import TestAutomation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main automation orchestrator."""
    parser = argparse.ArgumentParser(
        description="Neural DSL Master Automation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tests and validate examples
  python master_automation.py --test --validate
  
  # Full release (bump version, test, release)
  python master_automation.py --release --version-type patch
  
  # Daily maintenance tasks
  python master_automation.py --daily
        """
    )
    
    # Action flags
    parser.add_argument("--test", action="store_true",
                       help="Run tests")
    parser.add_argument("--validate", action="store_true",
                       help="Validate examples")
    parser.add_argument("--release", action="store_true",
                       help="Run full release process")
    parser.add_argument("--daily", action="store_true",
                       help="Run daily maintenance tasks")
    
    # Release options
    parser.add_argument("--version-type", choices=["major", "minor", "patch"],
                       default="patch", help="Version bump type (for releases)")
    parser.add_argument("--version", type=str,
                       help="Specific version")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip tests during release")
    parser.add_argument("--draft", action="store_true",
                       help="Create draft release")
    parser.add_argument("--test-pypi", action="store_true",
                       help="Publish to TestPyPI")
    
    # Test options
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    
    args = parser.parse_args()
    
    # If no specific action, show help
    if not any([args.test, args.validate, args.release, args.daily]):
        parser.print_help()
        return
    
    version = args.version
    
    print("=" * 70)
    print("Neural DSL Automation")
    print("=" * 70)
    print()
    
    # Daily tasks
    if args.daily:
        logger.info("Running daily maintenance tasks...")
        print()
        
        # Run tests
        logger.info("1. Running tests...")
        test_auto = TestAutomation()
        test_auto.run_and_report(coverage=args.coverage)
        print()
        
        # Validate examples
        logger.info("2. Validating examples...")
        validator = ExampleValidator()
        validator.generate_report()
        print()
        
        logger.info("✅ Daily tasks complete!")
        return
    
    # Full release
    if args.release:
        logger.info("Starting release process...")
        print()
        
        automation = ReleaseAutomation(version=version)
        automation.full_release(
            version_type=args.version_type,
            skip_tests=args.skip_tests,
            draft=args.draft,
            test_pypi=args.test_pypi
        )
        return
    
    # Individual tasks
    if args.test:
        logger.info("Running tests...")
        test_auto = TestAutomation()
        test_auto.run_and_report(coverage=args.coverage)
        print()
    
    if args.validate:
        logger.info("Validating examples...")
        validator = ExampleValidator()
        validator.generate_report()
        print()
    
    logger.info("✅ Automation complete!")


if __name__ == "__main__":
    main()
