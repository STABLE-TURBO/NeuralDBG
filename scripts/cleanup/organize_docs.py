#!/usr/bin/env python3
"""
Script to organize Neural DSL documentation files.

Moves implementation summaries, release notes, and specialized docs
to organized subdirectories in docs/.
"""

from pathlib import Path
import shutil
from typing import Dict, List

# Root directory
ROOT = Path(__file__).parent.parent.parent

# Mapping of file patterns to destination directories
DOC_ORGANIZATION: Dict[str, List[str]] = {
    "docs/archive/": [
        "*_IMPLEMENTATION*.md",
        "*_SUMMARY.md",
        "RELEASE_*.md",
        "GITHUB_RELEASE_*.md",
        "V0.3.0_*.md",
        "MIGRATION_*.md",
        "CHANGES_SUMMARY.md",
        "BUG_FIXES.md",
        "IMPLEMENTATION_*.md",
        "POST_RELEASE_*.md",
    ],
    "docs/automation/": [
        "AUTOMATION_GUIDE.md",
        "QUICK_START_AUTOMATION.md",
        "POST_RELEASE_AUTOMATION_*.md",
    ],
    "docs/dependencies/": [
        "DEPENDENCY_*.md",
        "MIGRATION_GUIDE_DEPENDENCIES.md",
    ],
    "docs/distribution/": [
        "DISTRIBUTION_*.md",
        "EXTRACTED_PROJECTS.md",
        "GITHUB_PUBLISHING_GUIDE.md",
    ],
    "docs/setup/": [
        "INSTALL.md",
        "SETUP_STATUS.md",
        "ERROR_MESSAGES_GUIDE.md",
    ],
    "docs/features/": [
        "DEPLOYMENT_FEATURES.md",
        "TRANSFORMER_*.md",
        "MULTIHEADATTENTION_*.md",
        "POSITIONAL_ENCODING_*.md",
    ],
}

# Files to keep in root (essential only)
KEEP_IN_ROOT = [
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "LICENSE.md",
    "SECURITY.md",
    "AGENTS.md",
    "GETTING_STARTED.md",
    "CLEANUP_PLAN.md",  # Keep for reference during cleanup phase
]


def organize_docs(dry_run: bool = True) -> None:
    """Organize documentation files into structured directories."""
    
    print("Documentation Organization Script")
    print("=" * 50)
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL MOVE'}")
    print()
    
    # Create destination directories
    for dest_dir in DOC_ORGANIZATION.keys():
        dest_path = ROOT / dest_dir
        if not dry_run:
            dest_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensuring directory exists: {dest_dir}")
    
    print()
    
    # Process each category
    moved_count = 0
    for dest_dir, patterns in DOC_ORGANIZATION.items():
        print(f"\n{dest_dir}")
        print("-" * 50)
        
        for pattern in patterns:
            matches = list(ROOT.glob(pattern))
            for source_file in matches:
                # Skip if in subdirectory or if it should stay in root
                if source_file.parent != ROOT or source_file.name in KEEP_IN_ROOT:
                    continue
                
                dest_file = ROOT / dest_dir / source_file.name
                
                if dry_run:
                    print(f"  Would move: {source_file.name} -> {dest_dir}")
                else:
                    try:
                        shutil.move(str(source_file), str(dest_file))
                        print(f"  Moved: {source_file.name} -> {dest_dir}")
                    except Exception as e:
                        print(f"  ERROR moving {source_file.name}: {e}")
                
                moved_count += 1
    
    print()
    print("=" * 50)
    print(f"Total files {'to be moved' if dry_run else 'moved'}: {moved_count}")
    
    # Show what would remain in root
    print("\nFiles remaining in root directory:")
    print("-" * 50)
    remaining = sorted([f.name for f in ROOT.glob("*.md")])
    for filename in remaining:
        status = "✓ Essential" if filename in KEEP_IN_ROOT else "⚠ Check"
        print(f"  {status}: {filename}")
    
    if dry_run:
        print("\n" + "=" * 50)
        print("This was a DRY RUN. No files were moved.")
        print("Run with --execute to actually move files.")
        print("=" * 50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize Neural DSL documentation files"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry run)"
    )
    
    args = parser.parse_args()
    
    organize_docs(dry_run=not args.execute)


if __name__ == "__main__":
    main()
