# Documentation Cleanup Guide

This guide documents the cleanup of redundant documentation in the `docs/` directory.

## Summary of Changes

The following files have been cleaned up to remove redundancy and consolidate documentation:

### 1. Removed index.html (Superseded by Docusaurus)

- **File**: `docs/index.html`
- **Reason**: This static HTML homepage has been superseded by the Docusaurus website in the `website/` directory
- **Status**: ✅ Cleared (replaced with placeholder)

### 2. Removed Generated Blog Posts (16 release-specific markdown files)

The following auto-generated blog posts in `docs/blog/` have been removed:

- `devto_v0.2.5_release.md`
- `devto_v0.2.6_release.md`
- `devto_v0.2.7_release.md`
- `devto_v0.2.8_release.md`
- `devto_v0.2.8_release_updated.md`
- `devto_v0.2.9_release.md`
- `devto_v0.3.0_release.md`
- `github_v0.3.0_release.md`
- `medium_v0.3.0_release.md`
- `website_v0.2.5_release.html`
- `website_v0.2.5_release.md`
- `website_v0.2.6_release.md`
- `website_v0.2.7_release.md`
- `website_v0.2.8_release.md`
- `website_v0.2.9_release.md`
- `index.html`

**Reason**: These are auto-generated release announcements that are no longer needed. They were created by automation scripts for posting to various platforms.

**Preserved**: `README.md` and `blog-list.json` remain for future blog infrastructure.

**Status**: ✅ Cleared (replaced with placeholders)

### 3. Removed Archive Directory

- **Directory**: `docs/archive/`
- **Contents**: Only contained `CHECKLIST.md` (internal development checklist)
- **Reason**: This was an internal development tracking document that is no longer relevant
- **Status**: ✅ Cleared (CHECKLIST.md replaced with placeholder)

### 4. Consolidated Duplicate Documentation

The following files in `docs/` have been superseded by the Docusaurus documentation in `website/docs/`:

| Legacy File | Superseded By | Status |
|-------------|---------------|--------|
| `docs/installation.md` | `website/docs/getting-started/installation.md` | ✅ Cleared with redirect |
| `docs/cli.md` | `website/docs/api/cli.md` | ✅ Cleared with redirect |
| `docs/dsl.md` | `website/docs/concepts/dsl-syntax.md` | ✅ Cleared with redirect |
| `docs/deployment.md` | `website/docs/tutorial/deployment.md` | ✅ Cleared with redirect |

**Reason**: The `website/docs/` directory contains the active, maintained documentation with Docusaurus frontmatter and proper organization. The old `docs/` files were legacy versions.

## Complete File Deletion Script

Due to security restrictions, the files have been replaced with placeholders. To complete the cleanup, run this script:

```python
#!/usr/bin/env python3
"""Complete the documentation cleanup by deleting placeholder files."""

import os
from pathlib import Path

def delete_files():
    """Delete all cleaned-up files."""
    files_to_delete = [
        # Main index.html
        "docs/index.html",
        
        # Blog posts
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
        
        # Duplicate documentation
        "docs/installation.md",
        "docs/cli.md",
        "docs/dsl.md",
        "docs/deployment.md",
    ]
    
    dirs_to_delete = [
        "docs/archive",
    ]
    
    # Delete files
    for file_path_str in files_to_delete:
        file_path = Path(file_path_str)
        if file_path.exists():
            file_path.unlink()
            print(f"✅ Deleted: {file_path}")
        else:
            print(f"⚠️  Not found: {file_path}")
    
    # Delete directories
    import shutil
    for dir_path_str in dirs_to_delete:
        dir_path = Path(dir_path_str)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✅ Deleted directory: {dir_path}")
        else:
            print(f"⚠️  Not found: {dir_path}")
    
    print("\n✅ Cleanup complete!")

if __name__ == "__main__":
    delete_files()
```

### Running the Cleanup Script

```bash
python cleanup_docs.py
```

Or manually delete the files:

```bash
# On Linux/Mac
rm docs/index.html
rm docs/blog/devto_*.md docs/blog/github_*.md docs/blog/medium_*.md docs/blog/website_*.md docs/blog/website_*.html docs/blog/index.html
rm -rf docs/archive
rm docs/installation.md docs/cli.md docs/dsl.md docs/deployment.md

# On Windows
Remove-Item docs\index.html
Remove-Item docs\blog\devto_*.md, docs\blog\github_*.md, docs\blog\medium_*.md, docs\blog\website_*.md, docs\blog\website_*.html, docs\blog\index.html
Remove-Item docs\archive -Recurse
Remove-Item docs\installation.md, docs\cli.md, docs\dsl.md, docs\deployment.md
```

## What Remains

After cleanup, the `docs/` directory structure:

```
docs/
├── _static/                 # Sphinx static assets
├── _templates/              # Sphinx templates
├── api/                     # API documentation
├── aquarium/                # Aquarium IDE docs
├── blog/
│   ├── README.md            # ✅ Kept (blog infrastructure)
│   └── blog-list.json       # ✅ Kept (blog metadata)
├── colab/                   # Colab notebooks
├── diagrams/                # Documentation diagrams
├── examples/                # Example documentation
├── features/                # Feature documentation
├── images/                  # Documentation images
├── mlops/                   # MLOps documentation
├── releases/                # Release notes
├── social/                  # Social media content
├── tutorials/               # Tutorial documentation
├── *.md                     # Various documentation files
└── conf.py                  # Sphinx configuration
```

The active, maintained documentation is in:

```
website/docs/                # 🌟 Primary documentation (Docusaurus)
├── api/
│   └── cli.md
├── concepts/
│   ├── dsl-syntax.md
│   └── shape-propagation.md
├── features/
│   └── neuraldbg.md
├── getting-started/
│   ├── installation.md
│   ├── quick-start.md
│   └── first-model.md
├── tutorial/
│   ├── basics.md
│   ├── debugging.md
│   ├── deployment.md
│   ├── layers.md
│   └── training.md
└── intro.md
```

## Benefits of This Cleanup

1. **Reduced Confusion**: Clear separation between legacy docs (`docs/`) and active docs (`website/docs/`)
2. **Removed Redundancy**: No duplicate documentation to maintain
3. **Cleaner Repository**: Removed auto-generated blog posts that cluttered the repo
4. **Better Organization**: Docusaurus provides better navigation and search
5. **Easier Maintenance**: Single source of truth for user-facing documentation

## Next Steps

Consider future cleanup:

1. **Evaluate remaining docs/ files**: Determine if other files in `docs/` should be migrated to `website/docs/` or archived
2. **Update BUILD_DOCS.md**: Update documentation build instructions to reflect the new structure
3. **Update CONTRIBUTING.md**: Guide contributors to update `website/docs/` instead of `docs/`
4. **Consider deprecating Sphinx**: If Docusaurus handles all needs, consider removing Sphinx setup

## References

- Docusaurus Documentation: https://docusaurus.io/
- Project Structure: See `website/` directory
- Active Docs: `website/docs/`
