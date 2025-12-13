# Neural DSL Automation Scripts

This directory contains automation scripts for:
- Blog post generation
- Blog publishing (Dev.to, Medium)
- GitHub releases
- PyPI publishing
- Example validation
- Test automation
- Social media posts

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - 5-minute setup guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture and design
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete implementation details
- **[CHECKLIST.md](CHECKLIST.md)** - Setup and publishing checklists
- **[FILES_OVERVIEW.md](FILES_OVERVIEW.md)** - Overview of all files
- **[example_usage.py](example_usage.py)** - Programmatic usage examples

## Scripts

### `blog_generator.py`
Generates blog posts from CHANGELOG.md for multiple platforms.

**Usage:**
```bash
python scripts/automation/blog_generator.py [version]
```

**Output:**
- `docs/blog/medium_v{version}_release.md`
- `docs/blog/devto_v{version}_release.md`
- `docs/blog/github_v{version}_release.md`

### `release_automation.py`
Automates the full release process: version bumping, testing, GitHub releases, PyPI publishing.

**Usage:**
```bash
# Patch release (default)
python scripts/automation/release_automation.py

# Minor release
python scripts/automation/release_automation.py --version-type minor

# Major release
python scripts/automation/release_automation.py --version-type major

# Draft release (for testing)
python scripts/automation/release_automation.py --draft

# Skip tests
python scripts/automation/release_automation.py --skip-tests

# Publish to TestPyPI first
python scripts/automation/release_automation.py --test-pypi
```

### `example_validator.py`
Validates all examples in the `examples/` directory.

**Usage:**
```bash
python scripts/automation/example_validator.py
```

**Output:**
- `examples_validation_report.md`

### `test_automation.py`
Runs tests and generates reports.

**Usage:**
```bash
# Run tests with coverage
python scripts/automation/test_automation.py

# Or import and use
from scripts.automation.test_automation import TestAutomation
automation = TestAutomation()
automation.run_and_report(coverage=True)
```

**Output:**
- `test_report.md`
- `test_results.json`
- `htmlcov/` (if coverage enabled)

### `social_media_generator.py`
Generates and publishes social media posts from release information.

**Features:**
- Twitter/X API v2 integration
- Dev.to API integration
- LinkedIn API integration
- Automatic rate limit handling
- Dry-run mode for testing

**Usage:**
```bash
# Generate posts only (save to files)
python scripts/automation/social_media_generator.py [version]

# Interactive mode (prompts for posting)
python scripts/automation/social_media_generator.py

# Programmatic posting
from scripts.automation.social_media_generator import SocialMediaGenerator
from scripts.automation.blog_generator import BlogGenerator

generator = BlogGenerator(version="0.3.0")
social_gen = SocialMediaGenerator(generator.version, generator.release_notes)

# Post to all platforms
results = social_gen.post_all()

# Post to specific platforms
results = social_gen.post_all(platforms=["twitter", "devto"])
```

**Environment Variables:**
- `TWITTER_API_KEY` - Twitter API key
- `TWITTER_API_SECRET` - Twitter API secret
- `TWITTER_ACCESS_TOKEN` - Twitter access token
- `TWITTER_ACCESS_TOKEN_SECRET` - Twitter access token secret
- `TWITTER_BEARER_TOKEN` - Twitter bearer token (optional)
- `DEV_TO_API_KEY` - Dev.to API key
- `LINKEDIN_ACCESS_TOKEN` - LinkedIn access token

**Output:**
- `docs/social/twitter_v{version}.txt`
- `docs/social/linkedin_v{version}.txt`

### `devto_publisher.py`
Automates publishing articles to Dev.to via API.

**Features:**
- Frontmatter parsing (YAML)
- Article creation and updates
- Draft/published status control
- Tag management (max 4 tags)
- Canonical URL support
- Error handling and logging

**Setup:**
1. Get API key from https://dev.to/settings/extensions
2. Set environment variable:
   ```bash
   export DEVTO_API_KEY="your_api_key_here"
   ```

**Usage:**
```bash
# Publish single article as draft
python scripts/automation/devto_publisher.py --file docs/blog/devto_v0.3.0_release.md

# Publish immediately (not as draft)
python scripts/automation/devto_publisher.py --file article.md --publish

# Publish all devto_*.md files in directory
python scripts/automation/devto_publisher.py --directory docs/blog

# Update existing articles
python scripts/automation/devto_publisher.py --directory docs/blog --update
```

**Frontmatter Format:**
```yaml
---
title: Article Title
published: false
description: Brief description
tags: python, machinelearning, deeplearning, neuralnetworks
canonical_url: https://example.com/original-article
series: Series Name
---
```

### `medium_publisher.py`
Automates publishing articles to Medium via API.

**Features:**
- Markdown content support
- Draft/public/unlisted status
- Tag management (max 5 tags)
- Canonical URL support
- Publication support
- License configuration

**Setup:**
1. Get API token from https://medium.com/me/settings/security
2. Set environment variable:
   ```bash
   export MEDIUM_API_TOKEN="your_token_here"
   ```

**Usage:**
```bash
# Publish single article as draft
python scripts/automation/medium_publisher.py --file docs/blog/medium_v0.3.0_release.md

# Publish as public post
python scripts/automation/medium_publisher.py --file article.md --status public

# Publish to a publication
python scripts/automation/medium_publisher.py --file article.md --publication-id abc123

# List your publications
python scripts/automation/medium_publisher.py --list-publications

# Publish all medium_*.md files
python scripts/automation/medium_publisher.py --directory docs/blog
```

**Frontmatter Format:**
```yaml
---
title: Article Title
status: draft
tags: python, machine-learning, deep-learning
canonical_url: https://example.com/original-article
license: all-rights-reserved
---
```

### `master_automation.py`
Orchestrates all automation tasks including marketing automation.

**Features:**
- Blog post generation
- Social media post generation
- Dev.to publishing
- Medium publishing
- Test automation
- Example validation
- Full release workflow

**Usage:**
```bash
# Generate blog posts only
python scripts/automation/master_automation.py --blog

# Generate and publish to Dev.to (as draft)
python scripts/automation/master_automation.py --blog --publish-devto

# Generate and publish to Medium (as draft)
python scripts/automation/master_automation.py --blog --publish-medium

# Full marketing automation
python scripts/automation/master_automation.py --marketing --publish-devto --publish-medium

# Publish immediately (not as draft)
python scripts/automation/master_automation.py --marketing --publish-devto --devto-public --publish-medium --medium-status public

# Run daily tasks
python scripts/automation/master_automation.py --daily

# Full release workflow
python scripts/automation/master_automation.py --release --version-type patch
```

## GitHub Actions

### Automated Release Workflow
Located at `.github/workflows/automated_release.yml`

**Triggers:**
- Manual dispatch (with options)
- Tag push (v*)

**Actions:**
1. Run tests
2. Generate blog posts
3. Validate examples
4. Bump version (if manual)
5. Create GitHub release
6. Upload artifacts

### Periodic Tasks Workflow
Located at `.github/workflows/periodic_tasks.yml`

**Schedule:** Daily at 2 AM UTC

**Actions:**
1. Run tests
2. Validate examples
3. Generate reports
4. Upload artifacts

## Setup

### Required Tools

1. **GitHub CLI** (for releases):
   ```bash
   # macOS
   brew install gh
   
   # Linux
   sudo apt install gh
   
   # Or download from https://cli.github.com/
   ```

2. **Python packages**:
   ```bash
   pip install build twine pytest pytest-json-report
   
   # For publishing to Dev.to and Medium
   pip install requests
   ```

3. **API Credentials** (for blog publishing):
   - **Dev.to**: Get API key from https://dev.to/settings/extensions
     - Set `DEVTO_API_KEY` environment variable
   - **Medium**: Get API token from https://medium.com/me/settings/security
     - Set `MEDIUM_API_TOKEN` environment variable

4. **GitHub Actions Secrets** (if publishing to PyPI):
   - `PYPI_API_TOKEN` - PyPI API token
   - `TEST_PYPI_API_TOKEN` - TestPyPI API token (optional)
   - `DEVTO_API_KEY` - Dev.to API key (optional, for automated publishing)
   - `MEDIUM_API_TOKEN` - Medium API token (optional, for automated publishing)

## Workflow

### Typical Release Process

1. **Update CHANGELOG.md** with new features/fixes
2. **Run automation**:
   ```bash
   python scripts/automation/release_automation.py --version-type patch
   ```
3. **Review generated files**:
   - Blog posts in `docs/blog/`
   - Social media posts in `docs/social/`
   - Release notes
4. **Manual steps** (if needed):
   - Review and edit blog posts
   - Post to social media
   - Update documentation

### Marketing Automation Workflow

1. **Generate content**:
   ```bash
   python scripts/automation/master_automation.py --marketing
   ```
   This generates:
   - Blog posts for Medium and Dev.to
   - Social media posts for Twitter/X and LinkedIn

2. **Review and edit** generated content in:
   - `docs/blog/` - Blog post files
   - `docs/social/` - Social media post files

3. **Publish automatically**:
   ```bash
   # Publish to Dev.to as draft
   python scripts/automation/master_automation.py --marketing --publish-devto
   
   # Publish to Medium as draft
   python scripts/automation/master_automation.py --marketing --publish-medium
   
   # Publish everywhere immediately
   python scripts/automation/master_automation.py --marketing \
     --publish-devto --devto-public \
     --publish-medium --medium-status public
   ```

4. **Or publish manually** using individual publishers:
   ```bash
   # Dev.to
   python scripts/automation/devto_publisher.py --file docs/blog/devto_v0.3.0_release.md
   
   # Medium
   python scripts/automation/medium_publisher.py --file docs/blog/medium_v0.3.0_release.md
   ```

5. **Post to social media** using content from `docs/social/`

### Automated Daily Tasks

GitHub Actions runs daily to:
- Validate all examples
- Run test suite
- Generate reports
- Upload artifacts

## Customization

### Blog Post Templates
Edit `blog_generator.py` to customize:
- Post format
- Platform-specific formatting
- Additional sections

### Social Media Posts
Edit `social_media_generator.py` to customize:
- Post length
- Hashtags
- Formatting
- API endpoints
- Rate limiting behavior

### Twitter Bot
Edit `../twitter_bot.py` to customize:
- Tweet formatting
- Changelog parsing
- Character limits
- Retry behavior

### Release Process
Edit `release_automation.py` to customize:
- Version bumping logic
- Release steps
- Notification methods

## Troubleshooting

### GitHub CLI not found
Install from https://cli.github.com/ or use manual release creation.

### PyPI upload fails
Check API tokens in GitHub Secrets or use `--test-pypi` first.

### Tests fail
Review test output and fix issues before releasing.

### Dev.to API errors
- **401 Unauthorized**: Check that `DEVTO_API_KEY` is set correctly
- **422 Unprocessable Entity**: Check article format and required fields (title, body)
- **Rate limiting**: Dev.to has rate limits; wait before retrying
- **Duplicate article**: Use `--update` flag to update existing articles

### Medium API errors
- **401 Unauthorized**: Check that `MEDIUM_API_TOKEN` is set correctly
- **Invalid token**: Regenerate token from Medium settings
- **403 Forbidden**: Check publication permissions if publishing to a publication
- **Rate limiting**: Medium has rate limits; wait before retrying

### Missing requests library
If you get `ImportError: requests library required`:
```bash
pip install requests
```

### Article not updating
- For Dev.to: Check that article title matches exactly (case-sensitive)
- For Medium: Medium API doesn't support updates; each publish creates a new post

### API authentication errors
Check that all required environment variables are set:
- For Twitter: `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET`
- For Dev.to: `DEV_TO_API_KEY`
- For LinkedIn: `LINKEDIN_ACCESS_TOKEN`

### Rate limiting
The scripts include automatic rate limit handling with exponential backoff. If you hit rate limits:
- Wait for the specified retry period
- Use dry-run mode to test without consuming API quota
- Check your API usage limits on the respective platforms

## API Setup Guide

### Twitter/X API Setup
1. Go to https://developer.twitter.com/
2. Create a new app or use an existing one
3. Generate API keys and access tokens
4. Set environment variables:
   ```bash
   export TWITTER_API_KEY="your_api_key"
   export TWITTER_API_SECRET="your_api_secret"
   export TWITTER_ACCESS_TOKEN="your_access_token"
   export TWITTER_ACCESS_TOKEN_SECRET="your_access_token_secret"
   ```

### Dev.to API Setup
1. Go to https://dev.to/settings/extensions
2. Generate an API key
3. Set environment variable:
   ```bash
   export DEV_TO_API_KEY="your_api_key"
   ```

### LinkedIn API Setup
1. Create a LinkedIn app at https://www.linkedin.com/developers/
2. Request appropriate permissions (w_member_social)
3. Obtain an access token via OAuth 2.0 flow
4. Set environment variable:
   ```bash
   export LINKEDIN_ACCESS_TOKEN="your_access_token"
   ```

## Example Usage

### Complete Release Flow

```python
from scripts.automation.blog_generator import BlogGenerator
from scripts.automation.social_media_generator import SocialMediaGenerator

# Generate blog posts
blog_gen = BlogGenerator(version="0.3.0")
blog_paths = blog_gen.save_blog_posts()

# Generate and post social media
social_gen = SocialMediaGenerator(blog_gen.version, blog_gen.release_notes)
social_paths = social_gen.save_posts()

# Post to social media (requires API credentials)
results = social_gen.post_all(platforms=["twitter", "devto"])

for platform, result in results.items():
    if result.get("success"):
        print(f"✓ {platform}: {result['message']}")
    else:
        print(f"✗ {platform}: {result['error']}")
```

### Standalone Twitter Posting

```bash
# Test with dry run
python scripts/twitter_bot.py 0.3.0 --dry-run

# Post if dry run looks good
python scripts/twitter_bot.py 0.3.0
```

## Future Enhancements

- [ ] Automated documentation generation
- [ ] Automated example generation
- [ ] Automated dependency updates
- [ ] Automated security scanning
- [ ] Automated performance benchmarking
- [ ] Automated changelog generation from commits
- [ ] Reddit API integration
- [ ] Mastodon API integration
- [ ] Discord webhook integration

