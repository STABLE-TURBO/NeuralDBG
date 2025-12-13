#!/usr/bin/env python3
"""
Parse CHANGELOG.md and extract release notes for a specific version.
"""

from pathlib import Path
import re
import sys


def parse_changelog(version: str, changelog_path: Path = None) -> str:
    """
    Parse CHANGELOG.md and extract release notes for a specific version.
    
    Args:
        version: Version to extract (e.g., "0.3.0")
        changelog_path: Path to CHANGELOG.md (defaults to repo root)
    
    Returns:
        Formatted release notes for the version
    """
    if changelog_path is None:
        repo_root = Path(__file__).parent.parent.parent
        changelog_path = repo_root / "CHANGELOG.md"
    
    if not changelog_path.exists():
        return f"# Release v{version}\n\nNo changelog found."
    
    with open(changelog_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Match version section (handles both [0.3.0] and [0.3.0-dev] formats)
    version_pattern = re.escape(version).replace(r"\.", r"\.").replace(r"\-", r"\-?")
    pattern = rf"## \[{version_pattern}[^\]]*\][^\n]*\n(.*?)(?=\n## \[|$)"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        # Try matching without brackets
        pattern = rf"## {version_pattern}[^\n]*\n(.*?)(?=\n## |$)"
        match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return f"# Release v{version}\n\nNo release notes found for this version."
    
    release_notes = match.group(1).strip()
    
    # Clean up the release notes
    # Remove excessive newlines
    release_notes = re.sub(r'\n{3,}', '\n\n', release_notes)
    
    # Format the release notes with proper header
    formatted_notes = f"""# Neural DSL v{version}

{release_notes}

---

## Installation

```bash
pip install neural-dsl=={version}
```

### Quick Start

```bash
# Minimal installation
pip install neural-dsl

# With all features
pip install neural-dsl[full]

# For development
pip install neural-dsl[dev]
```

## Documentation

- [Documentation](https://github.com/Lemniscate-world/Neural/blob/main/README.md)
- [Examples](https://github.com/Lemniscate-world/Neural/tree/main/examples)
- [Contributing Guide](https://github.com/Lemniscate-world/Neural/blob/main/CONTRIBUTING.md)

## Links

- [GitHub Repository](https://github.com/Lemniscate-world/Neural)
- [PyPI Package](https://pypi.org/project/neural-dsl/)
- [Discord Community](https://discord.gg/KFku4KvS)
"""
    
    return formatted_notes


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python parse_changelog.py <version>", file=sys.stderr)
        sys.exit(1)
    
    version = sys.argv[1].lstrip('v')
    release_notes = parse_changelog(version)
    print(release_notes)


if __name__ == "__main__":
    main()
