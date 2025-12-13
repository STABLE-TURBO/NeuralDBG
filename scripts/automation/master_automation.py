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

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.automation.blog_generator import BlogGenerator
from scripts.automation.release_automation import ReleaseAutomation
from scripts.automation.example_validator import ExampleValidator
from scripts.automation.test_automation import TestAutomation
from scripts.automation.social_media_generator import SocialMediaGenerator

try:
    from scripts.automation.devto_publisher import DevToPublisher
    DEVTO_AVAILABLE = True
except ImportError:
    DEVTO_AVAILABLE = False

try:
    from scripts.automation.medium_publisher import MediumPublisher
    MEDIUM_AVAILABLE = True
except ImportError:
    MEDIUM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_marketing_automation(version: str = None, 
                            publish_devto: bool = False,
                            publish_medium: bool = False,
                            devto_publish_status: bool = False,
                            medium_publish_status: str = "draft"):
    """
    Run full marketing automation workflow.
    
    Args:
        version: Version number (auto-detected if None)
        publish_devto: Whether to publish to Dev.to
        publish_medium: Whether to publish to Medium
        devto_publish_status: Whether to publish Dev.to articles immediately
        medium_publish_status: Medium publish status (draft/public/unlisted)
    """
    logger.info("=" * 70)
    logger.info("Marketing Automation Workflow")
    logger.info("=" * 70)
    
    # Step 1: Generate blog posts
    logger.info("\n[1/4] Generating blog posts...")
    try:
        generator = BlogGenerator(version=version)
        blog_paths = generator.save_blog_posts()
        logger.info(f"✓ Generated blog posts for v{generator.version}")
        for platform, path in blog_paths.items():
            logger.info(f"  - {platform}: {path}")
    except Exception as e:
        logger.error(f"✗ Failed to generate blog posts: {e}")
        raise
    
    # Step 2: Generate social media posts
    logger.info("\n[2/4] Generating social media posts...")
    try:
        social_gen = SocialMediaGenerator(generator.version, generator.release_notes)
        social_paths = social_gen.save_posts()
        logger.info("✓ Generated social media posts")
        for platform, path in social_paths.items():
            logger.info(f"  - {platform}: {path}")
    except Exception as e:
        logger.error(f"✗ Failed to generate social media posts: {e}")
        raise
    
    # Step 3: Publish to Dev.to
    if publish_devto:
        logger.info("\n[3/4] Publishing to Dev.to...")
        
        if not DEVTO_AVAILABLE:
            logger.warning("✗ Dev.to publisher not available (requests library needed)")
            logger.warning("  Install with: pip install requests")
        else:
            try:
                devto_publisher = DevToPublisher()
                
                # Publish Dev.to article
                devto_file = blog_paths.get("devto")
                if devto_file:
                    result = devto_publisher.publish_from_file(
                        devto_file,
                        update_if_exists=True,
                        force_publish=devto_publish_status
                    )
                    logger.info(f"✓ Published to Dev.to: {result.get('url')}")
                    logger.info(f"  ID: {result.get('id')}")
                    logger.info(f"  Status: {'Published' if result.get('published') else 'Draft'}")
                else:
                    logger.warning("✗ Dev.to blog file not found")
                    
            except ValueError as e:
                logger.warning(f"✗ Dev.to publishing skipped: {e}")
            except Exception as e:
                logger.error(f"✗ Failed to publish to Dev.to: {e}")
                logger.exception(e)
    else:
        logger.info("\n[3/4] Skipping Dev.to publishing (use --publish-devto to enable)")
    
    # Step 4: Publish to Medium
    if publish_medium:
        logger.info("\n[4/4] Publishing to Medium...")
        
        if not MEDIUM_AVAILABLE:
            logger.warning("✗ Medium publisher not available (requests library needed)")
            logger.warning("  Install with: pip install requests")
        else:
            try:
                medium_publisher = MediumPublisher()
                
                # Get user info
                medium_publisher.get_user_info()
                
                # Publish Medium article
                medium_file = blog_paths.get("medium")
                if medium_file:
                    result = medium_publisher.publish_from_file(
                        medium_file,
                        publish_status=medium_publish_status,
                        notify_followers=(medium_publish_status == "public")
                    )
                    logger.info(f"✓ Published to Medium: {result.get('url')}")
                    logger.info(f"  ID: {result.get('id')}")
                    logger.info(f"  Status: {result.get('publishStatus')}")
                    logger.info(f"  Tags: {', '.join(result.get('tags', []))}")
                else:
                    logger.warning("✗ Medium blog file not found")
                    
            except ValueError as e:
                logger.warning(f"✗ Medium publishing skipped: {e}")
            except Exception as e:
                logger.error(f"✗ Failed to publish to Medium: {e}")
                logger.exception(e)
    else:
        logger.info("\n[4/4] Skipping Medium publishing (use --publish-medium to enable)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Marketing automation complete!")
    logger.info("=" * 70)


def main():
    """Main automation orchestrator."""
    parser = argparse.ArgumentParser(
        description="Neural DSL Master Automation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate blog posts for current version
  python master_automation.py --blog
  
  # Generate and publish to Dev.to (as draft)
  python master_automation.py --blog --publish-devto
  
  # Generate and publish to Medium (as draft)
  python master_automation.py --blog --publish-medium
  
  # Full marketing automation (generate + publish everywhere)
  python master_automation.py --marketing --publish-devto --publish-medium
  
  # Publish immediately (not as draft)
  python master_automation.py --marketing --publish-devto --devto-public --publish-medium --medium-status public
  
  # Run tests and validate examples
  python master_automation.py --test --validate
  
  # Full release (bump version, test, release, blog, social)
  python master_automation.py --release --version-type patch
  
  # Daily maintenance tasks
  python master_automation.py --daily

Environment Variables (for publishing):
  DEVTO_API_KEY: Dev.to API key
  MEDIUM_API_TOKEN: Medium API token
        """
    )
    
    # Action flags
    parser.add_argument("--blog", action="store_true",
                       help="Generate blog posts")
    parser.add_argument("--social", action="store_true",
                       help="Generate social media posts")
    parser.add_argument("--marketing", action="store_true",
                       help="Run full marketing automation (blog + social + publish)")
    parser.add_argument("--test", action="store_true",
                       help="Run tests")
    parser.add_argument("--validate", action="store_true",
                       help="Validate examples")
    parser.add_argument("--release", action="store_true",
                       help="Run full release process")
    parser.add_argument("--daily", action="store_true",
                       help="Run daily maintenance tasks")
    
    # Publishing flags
    parser.add_argument("--publish-devto", action="store_true",
                       help="Publish to Dev.to")
    parser.add_argument("--devto-public", action="store_true",
                       help="Publish Dev.to articles immediately (not draft)")
    parser.add_argument("--publish-medium", action="store_true",
                       help="Publish to Medium")
    parser.add_argument("--medium-status", type=str, 
                       choices=["draft", "public", "unlisted"],
                       default="draft",
                       help="Medium publish status (default: draft)")
    
    # Release options
    parser.add_argument("--version-type", choices=["major", "minor", "patch"],
                       default="patch", help="Version bump type (for releases)")
    parser.add_argument("--version", type=str,
                       help="Specific version (for blog/social generation)")
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
    if not any([args.blog, args.social, args.test, args.validate, 
                args.release, args.daily, args.marketing]):
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
    
    # Full marketing automation
    if args.marketing:
        logger.info("Starting marketing automation...")
        print()
        
        run_marketing_automation(
            version=version,
            publish_devto=args.publish_devto,
            publish_medium=args.publish_medium,
            devto_publish_status=args.devto_public,
            medium_publish_status=args.medium_status
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
    
    if args.blog:
        logger.info("Generating blog posts...")
        generator = BlogGenerator(version=version)
        blog_paths = generator.save_blog_posts()
        print()
        
        # Publish if requested
        if args.publish_devto:
            logger.info("Publishing to Dev.to...")
            if DEVTO_AVAILABLE:
                try:
                    devto_publisher = DevToPublisher()
                    devto_file = blog_paths.get("devto")
                    if devto_file:
                        result = devto_publisher.publish_from_file(
                            devto_file,
                            update_if_exists=True,
                            force_publish=args.devto_public
                        )
                        logger.info(f"✓ Published to Dev.to: {result.get('url')}")
                except Exception as e:
                    logger.error(f"✗ Failed to publish to Dev.to: {e}")
            else:
                logger.warning("✗ Dev.to publisher not available")
            print()
        
        if args.publish_medium:
            logger.info("Publishing to Medium...")
            if MEDIUM_AVAILABLE:
                try:
                    medium_publisher = MediumPublisher()
                    medium_publisher.get_user_info()
                    medium_file = blog_paths.get("medium")
                    if medium_file:
                        result = medium_publisher.publish_from_file(
                            medium_file,
                            publish_status=args.medium_status
                        )
                        logger.info(f"✓ Published to Medium: {result.get('url')}")
                except Exception as e:
                    logger.error(f"✗ Failed to publish to Medium: {e}")
            else:
                logger.warning("✗ Medium publisher not available")
            print()
    
    if args.social:
        logger.info("Generating social media posts...")
        generator = BlogGenerator(version=version)
        social_gen = SocialMediaGenerator(version or generator.version, generator.release_notes)
        social_gen.save_posts()
        print()
    
    logger.info("✅ Automation complete!")


if __name__ == "__main__":
    main()
