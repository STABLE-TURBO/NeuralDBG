# Documentation Cleanup - Quick Reference

**TL;DR**: Cleaned up 21 redundant doc files. Run `python cleanup_docs.py` to complete deletion.

## What Happened

✅ Removed redundant `docs/index.html` (superseded by Docusaurus)  
✅ Removed 16 auto-generated blog posts in `docs/blog/`  
✅ Removed `docs/archive/` directory (internal dev docs)  
✅ Consolidated duplicate docs (redirected to `website/docs/`)

## Complete the Cleanup

```bash
python cleanup_docs.py
```

## New Documentation Structure

| Type | Location | Purpose |
|------|----------|---------|
| **User Docs** | `website/docs/` | 🌟 Active, maintained documentation |
| **Blog** | `website/blog/` | Docusaurus blog (recommended) |
| **Legacy** | `docs/` | Reference only, not maintained |

## For Contributors

**✅ DO**: Add docs to `website/docs/`  
**❌ DON'T**: Add docs to `docs/`

## Files Replaced

### Removed (21 files)
- `docs/index.html` → Placeholder
- `docs/blog/devto_*.md` (7 files) → Placeholders
- `docs/blog/github_*.md` → Placeholder
- `docs/blog/medium_*.md` → Placeholder
- `docs/blog/website_*.md` (6 files) → Placeholders
- `docs/blog/index.html` → Placeholder
- `docs/archive/CHECKLIST.md` → Placeholder
- `docs/installation.md` → Redirect to `website/docs/getting-started/installation.md`
- `docs/cli.md` → Redirect to `website/docs/api/cli.md`
- `docs/dsl.md` → Redirect to `website/docs/concepts/dsl-syntax.md`
- `docs/deployment.md` → Redirect to `website/docs/tutorial/deployment.md`

### Created (5 files)
- `DOCS_CLEANUP_GUIDE.md` - Detailed guide
- `DOCS_CLEANUP_SUMMARY.md` - Implementation summary
- `DOCS_CLEANUP_QUICK_REF.md` - This file
- `docs/README_CLEANUP.md` - In-directory summary
- `cleanup_docs.py` - Deletion script

### Updated (1 file)
- `docs/blog/README.md` - Updated with cleanup notes

## Duplicate Documentation Mapping

| Old Location | New Location |
|--------------|--------------|
| `docs/installation.md` | `website/docs/getting-started/installation.md` |
| `docs/cli.md` | `website/docs/api/cli.md` |
| `docs/dsl.md` | `website/docs/concepts/dsl-syntax.md` |
| `docs/deployment.md` | `website/docs/tutorial/deployment.md` |

## Benefits

- 🧹 Cleaner repository (21 files cleared)
- 📚 Single source of truth (`website/docs/`)
- ✨ Reduced confusion (no duplicates)
- 🔍 Better navigation (Docusaurus)

## More Info

- **Detailed Guide**: `DOCS_CLEANUP_GUIDE.md`
- **Implementation Summary**: `DOCS_CLEANUP_SUMMARY.md`
- **In-Docs Summary**: `docs/README_CLEANUP.md`
- **Run Cleanup**: `python cleanup_docs.py`
