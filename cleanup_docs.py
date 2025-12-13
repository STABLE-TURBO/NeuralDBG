#!/usr/bin/env python3
"""Complete the documentation cleanup by deleting placeholder files.

This script removes redundant documentation files that have been superseded
by the Docusaurus website or are no longer needed.
"""

import os
import shutil
from pathlib import Path


def delete_files():
    """Delete all cleaned-up files and directories."""
    
    print("Documentation Cleanup Script")
    print("=" * 60)
    print("\nThis script will delete redundant documentation files.")
    print("Files have been replaced with placeholders marking them for deletion.\n")
    
    files_to_delete = [
        # Main index.html (superseded by Docusaurus)
        "docs/index.html",
        
        # Generated blog posts (16 files)
        "docs/blog/devto_v0.2.5_release.md",
        "docs/blog/devto_v0.2.6_release.md",
        "docs/blog/devto_v0.2.7_release.md",
        "docs/blog/devto_v0.2.8_release.md",
        "docs/blog/devto_v0.2.8_release_updated.md",
        "docs/blog/devto_v0.2.9_release.md",
        "docs/blog/devto_v0.3.0_release.md",
        "docs/blog/github_v0.3.0_release.md",
        "docs/blog/index.html",
        "docs/blog/medium_v0.3.0_release.md",
        "docs/blog/website_v0.2.5_release.html",
        "docs/blog/website_v0.2.5_release.md",
        "docs/blog/website_v0.2.6_release.md",
        "docs/blog/website_v0.2.7_release.md",
        "docs/blog/website_v0.2.8_release.md",
        "docs/blog/website_v0.2.9_release.md",
        
        # Duplicate documentation (superseded by website/docs/)
        "docs/installation.md",
        "docs/cli.md",
        "docs/dsl.md",
        "docs/deployment.md",
    ]
    
    dirs_to_delete = [
        # Archive directory (internal development docs)
        "docs/archive",
    ]
    
    deleted_count = 0
    not_found_count = 0
    
    # Delete files
    print("Deleting files...")
    print("-" * 60)
    for file_path_str in files_to_delete:
        file_path = Path(file_path_str)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
        else:
            print(f"⚠️  Not found: {file_path}")
            not_found_count += 1
    
    # Delete directories
    print("\nDeleting directories...")
    print("-" * 60)
    for dir_path_str in dirs_to_delete:
        dir_path = Path(dir_path_str)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"✅ Deleted directory: {dir_path}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {dir_path}: {e}")
        else:
            print(f"⚠️  Directory not found: {dir_path}")
            not_found_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Cleanup Summary:")
    print(f"  ✅ Deleted: {deleted_count} items")
    if not_found_count > 0:
        print(f"  ⚠️  Not found: {not_found_count} items")
    print("\n✅ Documentation cleanup complete!")
    print("\nSee DOCS_CLEANUP_GUIDE.md for details on what was removed.")


def main():
    """Main entry point."""
    # Confirm we're in the right directory
    if not Path("docs").exists():
        print("❌ Error: 'docs' directory not found.")
        print("   Please run this script from the repository root.")
        return 1
    
    # Run cleanup
    try:
        delete_files()
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
