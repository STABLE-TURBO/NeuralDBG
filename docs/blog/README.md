# Neural DSL Blog

**Status**: 🧹 Cleaned up - Auto-generated release posts removed

This directory previously contained auto-generated blog posts for various releases. These have been removed as part of documentation cleanup.

## What Was Removed

The following auto-generated release announcement files were removed:
- 7 Dev.to release posts (`devto_v*.md`)
- 1 GitHub release post (`github_v*.md`)
- 1 Medium release post (`medium_v*.md`)
- 6 Website release posts (`website_v*.md`)
- Blog index.html

**Reason**: These were auto-generated files that are no longer needed. Release notes are better maintained in:
- `docs/releases/` directory
- GitHub Releases page
- `CHANGELOG.md`

## Current Status

This directory is preserved for future blog infrastructure but currently contains minimal files:
- `README.md` (this file)
- `blog-list.json` (blog metadata, preserved for future use)

## Future Blog Posts

If you want to add blog posts in the future, consider these options:

### Option 1: Use Docusaurus Blog (Recommended)

The Docusaurus website (`website/`) has built-in blog functionality:

```bash
# Create a blog post in the Docusaurus blog
# Location: website/blog/
# Format: YYYY-MM-DD-post-title.md
```

Benefits:
- Integrated with the main documentation site
- Automatic blog index generation
- RSS feed support
- Better SEO and navigation

### Option 2: Manual Blog Posts Here

If you prefer to keep blog posts in this directory:

1. Create a markdown file with a descriptive name
2. Follow standard markdown formatting
3. Manually update `blog-list.json` if needed

**Note**: This approach requires manual integration with the website.

## Blog Post Guidelines (If Adding New Posts)

- Keep titles concise and descriptive
- Include a clear date in the format `Month Day, Year`
- Include code examples when relevant
- Use proper markdown formatting
- Keep images to a reasonable size (max width 900px)

## Recommended Approach

**For release announcements**: Use GitHub Releases and `CHANGELOG.md`

**For technical blog posts**: Use the Docusaurus blog in `website/blog/`

**For social media**: Use dedicated social media management tools

## See Also

- `docs/releases/` - Release notes directory
- `website/blog/` - Docusaurus blog (recommended for new posts)
- `CHANGELOG.md` - Changelog file
- `docs/README_CLEANUP.md` - Documentation cleanup summary
