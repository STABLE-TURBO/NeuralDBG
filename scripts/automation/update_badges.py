#!/usr/bin/env python3
"""
Update README.md badges with new version and status information.
"""

from datetime import datetime
from pathlib import Path
import re
import sys


def update_badges(version: str, readme_path: Path = None) -> bool:
    """
    Update badges in README.md with new version information.
    
    Args:
        version: New version number (e.g., "0.3.0")
        readme_path: Path to README.md (defaults to repo root)
    
    Returns:
        True if successful, False otherwise
    """
    if readme_path is None:
        repo_root = Path(__file__).parent.parent.parent
        readme_path = repo_root / "README.md"
    
    if not readme_path.exists():
        print(f"ERROR: README.md not found at {readme_path}", file=sys.stderr)
        return False
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    
    # Update PyPI version badge (if exists)
    content = re.sub(
        r'!\[PyPI version\]\(https://badge\.fury\.io/py/neural-dsl\.svg\)',
        '![PyPI version](https://badge.fury.io/py/neural-dsl.svg)',
        content
    )
    
    # Add or update version badge
    if '![Version]' not in content:
        # Add version badge after license badge
        content = re.sub(
            r'(\[!\[License: MIT\].*?\]\(LICENSE\))',
            rf'\1\n  ![Version](https://img.shields.io/badge/version-{version}-blue.svg)',
            content
        )
    else:
        # Update existing version badge
        content = re.sub(
            r'!\[Version\]\(https://img\.shields\.io/badge/version-[^-]+-blue\.svg\)',
            f'![Version](https://img.shields.io/badge/version-{version}-blue.svg)',
            content
        )
    
    # Update or add release date badge
    release_date = datetime.now().strftime('%Y-%m-%d')
    if '![Release Date]' not in content:
        # Add after version badge
        content = re.sub(
            r'(!\[Version\].*?\n)',
            rf'\1  ![Release Date](https://img.shields.io/badge/release_date-{release_date}-green.svg)\n',
            content
        )
    else:
        # Update existing release date badge
        content = re.sub(
            r'!\[Release Date\]\(https://img\.shields\.io/badge/release_date-[^-]+-green\.svg\)',
            f'![Release Date](https://img.shields.io/badge/release_date-{release_date}-green.svg)',
            content
        )
    
    # Update any hardcoded version references in installation instructions
    content = re.sub(
        r'pip install neural-dsl==\d+\.\d+\.\d+',
        f'pip install neural-dsl=={version}',
        content
    )
    
    # Update beta status warning with latest version
    content = re.sub(
        r'> ⚠️ \*\*BETA STATUS\*\*: Neural-dsl v\d+\.\d+\.\d+',
        f'> ⚠️ **BETA STATUS**: Neural-dsl v{version}',
        content
    )
    
    # Update version in note at bottom of long_description if present
    content = re.sub(
        r'See v\d+\.\d+\.\d+ release notes',
        f'See v{version} release notes',
        content
    )
    
    # Only write if content changed
    if content != original_content:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Updated README.md badges for version {version}")
        return True
    else:
        print("✓ No badge updates needed in README.md")
        return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python update_badges.py <version>", file=sys.stderr)
        sys.exit(1)
    
    version = sys.argv[1].lstrip('v')
    success = update_badges(version)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
