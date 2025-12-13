"""
Utility functions for the Neural Marketplace.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file.

    Parameters
    ----------
    file_path : str
        Path to file

    Returns
    -------
    str
        SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_license(license_name: str) -> bool:
    """Validate if license is a known open-source license.

    Parameters
    ----------
    license_name : str
        License name

    Returns
    -------
    bool
        True if valid
    """
    valid_licenses = {
        "MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "BSD-2-Clause",
        "LGPL-3.0", "MPL-2.0", "AGPL-3.0", "Unlicense", "CC0-1.0",
        "ISC", "EPL-2.0", "EUPL-1.2", "Custom"
    }
    return license_name in valid_licenses


def format_model_size(size_bytes: int) -> str:
    """Format model size in human-readable format.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    str
        Formatted size
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def parse_version(version: str) -> tuple:
    """Parse semantic version string.

    Parameters
    ----------
    version : str
        Version string (e.g., "1.2.3")

    Returns
    -------
    tuple
        (major, minor, patch)
    """
    try:
        parts = version.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except (ValueError, IndexError):
        return (0, 0, 0)


def compare_versions(v1: str, v2: str) -> int:
    """Compare two semantic versions.

    Parameters
    ----------
    v1 : str
        First version
    v2 : str
        Second version

    Returns
    -------
    int
        -1 if v1 < v2, 0 if equal, 1 if v1 > v2
    """
    version1 = parse_version(v1)
    version2 = parse_version(v2)

    if version1 < version2:
        return -1
    elif version1 > version2:
        return 1
    else:
        return 0


def sanitize_model_name(name: str) -> str:
    """Sanitize model name for use in filenames.

    Parameters
    ----------
    name : str
        Model name

    Returns
    -------
    str
        Sanitized name
    """
    import re
    # Remove special characters, keep alphanumeric, hyphens, and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Replace multiple underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    # Convert to lowercase
    sanitized = sanitized.lower()
    return sanitized


def extract_model_metadata(model_path: str) -> Dict[str, Any]:
    """Extract metadata from a Neural DSL model file.

    Parameters
    ----------
    model_path : str
        Path to model file

    Returns
    -------
    Dict
        Extracted metadata
    """
    metadata = {
        "layers": [],
        "input_shape": None,
        "output_shape": None,
        "has_training_config": False
    }

    try:
        with open(model_path, 'r') as f:
            content = f.read()

        # Simple parsing to extract basic info
        lines = content.split('\n')
        for line in lines:
            line = line.strip()

            # Extract input shape
            if line.startswith('Input:') and 'shape=' in line:
                import re
                match = re.search(r'shape=\(([^)]+)\)', line)
                if match:
                    shape_str = match.group(1)
                    metadata["input_shape"] = [int(x.strip()) for x in shape_str.split(',')]

            # Extract layer types
            if ':' in line and not line.startswith('Network') and not line.startswith('Input:') and not line.startswith('Output:'):
                layer_type = line.split(':')[0].strip()
                if layer_type and not layer_type.startswith('{'):
                    metadata["layers"].append(layer_type)

            # Check for training config
            if line.startswith('Output:') and ('loss=' in line or 'optimizer=' in line):
                metadata["has_training_config"] = True

    except Exception:
        pass

    return metadata


def generate_model_card(model_info: Dict[str, Any]) -> str:
    """Generate a model card in Markdown format.

    Parameters
    ----------
    model_info : Dict
        Model information

    Returns
    -------
    str
        Model card content
    """
    card = f"""# {model_info['name']}

**Author:** {model_info['author']}
**Version:** {model_info.get('version', '1.0.0')}
**License:** {model_info.get('license', 'MIT')}

## Description

{model_info.get('description', 'No description available.')}

## Tags

{', '.join(f'`{tag}`' for tag in model_info.get('tags', []))}

## Usage

To use this model with Neural DSL:

```bash
# Download the model
neural marketplace download {model_info['id']}

# Compile for your backend
neural compile {model_info['file']} --backend tensorflow

# Run the model
neural run {model_info['file'].replace('.neural', '_tensorflow.py')}
```

## Statistics

- **Downloads:** {model_info.get('downloads', 0)}
- **Uploaded:** {model_info.get('uploaded_at', 'Unknown')[:10]}
- **Last Updated:** {model_info.get('updated_at', 'Unknown')[:10]}

## Framework

Built with Neural DSL {model_info.get('framework', 'neural-dsl')}

---

*Generated by Neural Marketplace*
"""
    return card


def export_registry_summary(registry) -> Dict[str, Any]:
    """Export a summary of the registry.

    Parameters
    ----------
    registry : ModelRegistry
        Registry instance

    Returns
    -------
    Dict
        Registry summary
    """
    summary = {
        "total_models": len(registry.metadata["models"]),
        "total_authors": len(registry.metadata["authors"]),
        "total_tags": len(registry.metadata["tags"]),
        "total_downloads": sum(s.get("downloads", 0) for s in registry.stats.values()),
        "total_views": sum(s.get("views", 0) for s in registry.stats.values()),
        "popular_models": [],
        "recent_models": [],
        "trending_tags": []
    }

    # Get popular models
    popular = registry.get_popular_models(limit=10)
    summary["popular_models"] = [
        {
            "name": m["name"],
            "author": m["author"],
            "downloads": m.get("downloads", 0)
        }
        for m in popular
    ]

    # Get recent models
    recent = registry.get_recent_models(limit=10)
    summary["recent_models"] = [
        {
            "name": m["name"],
            "author": m["author"],
            "uploaded_at": m.get("uploaded_at", "")
        }
        for m in recent
    ]

    # Get trending tags
    from .search import SemanticSearch
    search = SemanticSearch(registry)
    tags = search.get_trending_tags(limit=20)
    summary["trending_tags"] = [{"tag": t[0], "count": t[1]} for t in tags]

    return summary


def validate_model_file(file_path: str) -> tuple[bool, Optional[str]]:
    """Validate a Neural DSL model file.

    Parameters
    ----------
    file_path : str
        Path to model file

    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    if not file_path.endswith(('.neural', '.nr')):
        return False, "File must have .neural or .nr extension"

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        if not content.strip():
            return False, "File is empty"

        # Basic syntax check
        if 'Network' not in content:
            return False, "File must contain a Network definition"

        return True, None

    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def create_backup(registry_dir: str) -> str:
    """Create a backup of the registry.

    Parameters
    ----------
    registry_dir : str
        Registry directory

    Returns
    -------
    str
        Path to backup file
    """
    from datetime import datetime
    import shutil

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{registry_dir}_backup_{timestamp}"

    shutil.copytree(registry_dir, backup_path)

    return backup_path


def restore_backup(backup_path: str, registry_dir: str):
    """Restore registry from backup.

    Parameters
    ----------
    backup_path : str
        Path to backup
    registry_dir : str
        Target registry directory
    """
    import shutil

    if os.path.exists(registry_dir):
        shutil.rmtree(registry_dir)

    shutil.copytree(backup_path, registry_dir)
