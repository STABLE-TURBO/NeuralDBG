#!/usr/bin/env python3
"""
Cleanup script to remove all cache directories and test artifacts.
This script removes:
- Cache directories: __pycache__, .pytest_cache, .hypothesis, .mypy_cache, .ruff_cache
- Virtual environments: .venv*, venv*
- Test artifacts: test_*.html, test_*.png
- Temporary scripts: sample_tensorflow.py, sample_pytorch.py
"""
import glob
import os
from pathlib import Path
import shutil


def remove_directory(path):
    """Remove a directory and all its contents."""
    if os.path.exists(path):
        print(f"Removing directory: {path}")
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"  ✓ Removed {path}")
            return True
        except Exception as e:
            print(f"  ✗ Error removing {path}: {e}")
            return False
    return False


def remove_file(path):
    """Remove a file."""
    if os.path.exists(path):
        print(f"Removing file: {path}")
        try:
            os.remove(path)
            print(f"  ✓ Removed {path}")
            return True
        except Exception as e:
            print(f"  ✗ Error removing {path}: {e}")
            return False
    return False


def find_and_remove_pattern(root_dir, pattern, is_dir=True):
    """Find and remove all files/dirs matching a pattern."""
    count = 0
    for path in Path(root_dir).rglob(pattern):
        if is_dir and path.is_dir():
            if remove_directory(str(path)):
                count += 1
        elif not is_dir and path.is_file():
            if remove_file(str(path)):
                count += 1
    return count


def main():
    """Main cleanup function."""
    print("=" * 70)
    print("Cache and Artifacts Cleanup Script")
    print("=" * 70)
    print()
    
    root_dir = Path(__file__).parent
    os.chdir(root_dir)
    
    # Cache directories to remove (recursive search)
    cache_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".hypothesis",
        ".mypy_cache",
        ".ruff_cache",
    ]
    
    print("Removing cache directories...")
    print("-" * 70)
    total_removed = 0
    for cache_dir in cache_dirs:
        count = find_and_remove_pattern(".", cache_dir, is_dir=True)
        total_removed += count
        if count > 0:
            print(f"  Removed {count} {cache_dir} director{'y' if count == 1 else 'ies'}")
    print(f"Total cache directories removed: {total_removed}")
    print()
    
    # Virtual environment directories (only in root)
    print("Removing virtual environment directories...")
    print("-" * 70)
    venv_patterns = [".venv", ".venv*", "venv", "venv*"]
    venv_removed = 0
    for pattern in venv_patterns:
        for venv_path in glob.glob(pattern):
            if os.path.isdir(venv_path):
                if remove_directory(venv_path):
                    venv_removed += 1
    print(f"Total virtual environments removed: {venv_removed}")
    print()
    
    # Test artifacts
    print("Removing test artifacts...")
    print("-" * 70)
    test_patterns = ["test_*.html", "test_*.png"]
    artifact_removed = 0
    for pattern in test_patterns:
        count = find_and_remove_pattern(".", pattern, is_dir=False)
        artifact_removed += count
        if count > 0:
            print(f"  Removed {count} {pattern} file(s)")
    print(f"Total test artifacts removed: {artifact_removed}")
    print()
    
    # Temporary Python scripts
    print("Removing temporary Python scripts...")
    print("-" * 70)
    temp_scripts = ["sample_tensorflow.py", "sample_pytorch.py"]
    scripts_removed = 0
    for script in temp_scripts:
        if remove_file(script):
            scripts_removed += 1
    print(f"Total temporary scripts removed: {scripts_removed}")
    print()
    
    print("=" * 70)
    print("Cleanup Complete!")
    print("=" * 70)
    print("Summary:")
    print(f"  - Cache directories: {total_removed}")
    print(f"  - Virtual environments: {venv_removed}")
    print(f"  - Test artifacts: {artifact_removed}")
    print(f"  - Temporary scripts: {scripts_removed}")
    print()


if __name__ == "__main__":
    main()
