# Neural DSL Automation Guide

## Overview

Over time, we've built a comprehensive automation system for Neural DSL. It started with a simple script to generate blog posts from the changelog, and grew into a full release pipeline. Here's what we've automated so far:

- Blog post generation from CHANGELOG
- Publishing to Dev.to and Medium via their APIs
- Social media posting to Twitter/X and LinkedIn
- GitHub discussions for release announcements
- Version bumping and GitHub releases
- PyPI package publishing
- Post-release cleanup and notifications
- Example validation (because broken examples are embarrassing)
- Test automation with coverage reports
- Daily maintenance tasks

## Quick Start

### Generate Blog Posts

```bash
python scripts/automation/master_automation.py --blog
```

This reads your CHANGELOG.md and generates three versions:
- `docs/blog/medium_v{version}_release.md`
- `docs/blog/devto_v{version}_release.md`
- `docs/blog/github_v{version}_release.md`

Each version is tailored to its platform's formatting and audience.

### Run Tests and Validation

```bash
python scripts/automation/master_automation.py --test --validate
```

This runs the full test suite and validates all example files. If you're about to release, you want these passing.

### Full Release

```bash
# Patch release (0.3.0 -> 0.3.1)
python scripts/automation/master_automation.py --release --version-type patch

# Minor release (0.3.0 -> 0.4.0)
python scripts/automation/master_automation.py --release --version-type minor

# Major release (0.3.0 -> 1.0.0)
python scripts/automation/master_automation.py --release --version-type major
```

The release script handles everything: version bumping, running tests, building packages, creating GitHub releases, and publishing to PyPI.

### Daily Maintenance

```bash
python scripts/automation/master_automation.py --daily
```

Or just let GitHub Actions do it automatically - it runs every day at 2 AM UTC. We set it up this way so we'd notice if tests start failing overnight.

## Automation Scripts

### 1. Blog Generator (`blog_generator.py`)

This script parses your CHANGELOG.md and turns it into readable blog posts. It's not perfect - sometimes you'll want to edit the output - but it saves a ton of time.

**Usage:**
```bash
python scripts/automation/blog_generator.py [version]
```

**What you get:**
- A Medium-style post with proper formatting
- A Dev.to post with their specific frontmatter
- GitHub release notes in markdown

### 2. Release Automation (`release_automation.py`)

The big one. This script orchestrates the entire release process. We built it after manually releasing v0.1.0 and realizing we'd never remember all the steps.

**What it does:**
- Bumps version in setup.py and __init__.py
- Runs the test suite (and stops if tests fail)
- Generates release notes from the changelog
- Creates a GitHub release
- Builds and publishes to PyPI
- Generates blog posts

**Usage:**
```bash
python scripts/automation/release_automation.py \
  --version-type patch \
  --skip-tests false \
  --draft false \
  --test-pypi false
```

**Fair warning:** If you use `--skip-tests`, you're living dangerously. We learned this the hard way.

### 3. Example Validator (`example_validator.py`)

Validates every .neural file in the examples directory. This catches syntax errors and compilation issues before users see them.

**Usage:**
```bash
python scripts/automation/example_validator.py
```

**Output:**
- `examples_validation_report.md` with detailed results

We run this in CI because nothing is worse than shipping broken examples.

### 4. Test Automation (`test_automation.py`)

Wraps pytest and generates nice reports. The coverage report is especially useful for finding untested code paths.

**Usage:**
```bash
python scripts/automation/test_automation.py
```

**Output:**
- `test_report.md` with results summary
- `test_results.json` for programmatic access
- `htmlcov/` directory with detailed coverage

### 5. Social Media Generator (`social_media_generator.py`)

Generates posts for Twitter/X and LinkedIn from your changelog. Character limits are annoying, so this script handles the truncation and hashtag management.

**Usage:**
```bash
python scripts/automation/social_media_generator.py
```

**Output:**
- `docs/social/twitter_v{version}.txt` (stays under 280 chars)
- `docs/social/linkedin_v{version}.txt` (more verbose)

## GitHub Actions Workflows

### Marketing Automation (`.github/workflows/marketing_automation.yml`)

The newest addition to our automation suite. This workflow publishes your release across multiple platforms automatically.

**When it runs:**
- Automatically when you publish a GitHub release
- Manually via workflow dispatch

**What it does:**
1. Validates that you've set up all the API tokens (fails early if not)
2. Generates blog posts and social content
3. Publishes to Dev.to immediately (goes live right away)
4. Creates a draft on Medium (you can review before publishing)
5. Posts to Twitter/X
6. Posts to LinkedIn
7. Creates a GitHub Discussion for the release
8. Commits all generated files back to the repo

**Important notes:**
- Each platform step has `continue-on-error: true` so one failure doesn't break everything
- Medium posts are created as drafts, giving you a chance to review
- Dev.to posts go live immediately, so make sure your changelog is polished
- You need API keys for all platforms (see Setup Requirements below)

**Documentation:**
- [Full Marketing Guide](docs/MARKETING_AUTOMATION_GUIDE.md)
- [Quick Reference](docs/MARKETING_AUTOMATION_QUICK_REF.md)
- [Workflows README](.github/workflows/README.md)

### Automated Release (`.github/workflows/automated_release.yml`)

This workflow handles the actual release process in CI.

**Triggers:**
- Manual dispatch (you control the options)
- Tag push matching `v*` pattern

**What it does:**
1. Checks out your code
2. Sets up Python
3. Installs dependencies
4. Runs the test suite
5. Generates blog posts
6. Validates all examples
7. Bumps version (if triggered manually)
8. Creates GitHub release
9. Uploads build artifacts

**Usage:**
1. Go to the Actions tab on GitHub
2. Select "Automated Release"
3. Click "Run workflow"
4. Choose your options:
   - Version type (patch/minor/major)
   - Whether to skip tests (not recommended)
   - Whether to create a draft release

### Periodic Tasks (`.github/workflows/periodic_tasks.yml`)

Runs daily at 2 AM UTC. We set this up after discovering a breaking change a week after it was introduced.

**What it does:**
1. Runs the full test suite
2. Validates all examples
3. Generates reports
4. Uploads artifacts (kept for 30 days)

If tests start failing, you'll see it in GitHub's Actions tab. No manual intervention needed unless something breaks.

### Post-Release Automation (`.github/workflows/post_release_automation.yml`)

Handles all the cleanup after a release is published.

**Triggers:**
- Automatically after a release is published
- Manually via workflow dispatch

**What it does:**
1. Bumps version to next dev version (e.g., 0.3.0 → 0.3.1.dev0)
2. Creates a GitHub Discussion announcing the release
3. Updates documentation links if needed
4. Triggers Netlify/Vercel deployments for docs
5. Sends a Discord notification (if configured)
6. Creates a planning issue for the next release

**See also:** [POST_RELEASE_AUTOMATION_QUICK_REF.md](POST_RELEASE_AUTOMATION_QUICK_REF.md)

## Typical Workflow

### For a New Release

Here's what we typically do when releasing a new version:

1. **Update CHANGELOG.md** - Document all the new features, bug fixes, and changes. Be specific. Future you will appreciate it.

2. **Run automation:**
   ```bash
   python scripts/automation/master_automation.py --release --version-type patch
   ```

3. **Review generated files:**
   - Check the blog posts in `docs/blog/` - they're auto-generated but often need tweaking
   - Review social media posts in `docs/social/` - especially the Twitter one, character limits are strict
   - Skim the release notes

4. **Manual steps (if needed):**
   - Edit blog posts to add personality or clarify points
   - Post to social media (or let the marketing automation handle it)
   - Update documentation if there are major changes

### For Daily Maintenance

The periodic tasks workflow runs automatically. You don't need to do anything unless something fails.

If you want to run it manually:
```bash
python scripts/automation/master_automation.py --daily
```

## Setup Requirements

### Local Setup

If you want to run these scripts locally:

1. **Install dependencies:**
   ```bash
   pip install build twine pytest pytest-json-report
   ```

2. **Install GitHub CLI** (for creating releases):
   ```bash
   # macOS
   brew install gh
   
   # Linux
   sudo apt install gh
   ```

3. **Authenticate GitHub CLI:**
   ```bash
   gh auth login
   ```

### GitHub Actions Setup

To use the full automation in CI, you'll need to set up some secrets in your GitHub repository settings:

1. **For PyPI Publishing:**
   - `PYPI_API_TOKEN` - Your PyPI API token (get it from pypi.org/manage/account)
   - `TEST_PYPI_API_TOKEN` - TestPyPI token (optional, but useful for testing)

2. **For Marketing Automation:**
   - `DEVTO_API_KEY` - From dev.to/settings/extensions
   - `MEDIUM_API_KEY` - Medium integration token (trickier to get, needs OAuth)
   - `MEDIUM_USER_ID` - Your Medium user ID (optional, used for author attribution)
   - `TWITTER_API_KEY` - From Twitter Developer Portal
   - `TWITTER_API_SECRET` - Also from Twitter Developer Portal
   - `TWITTER_ACCESS_TOKEN` - Generated during OAuth flow
   - `TWITTER_ACCESS_TOKEN_SECRET` - Also from OAuth flow
   - `LINKEDIN_ACCESS_TOKEN` - LinkedIn API token (expires every 60 days, unfortunately)
   - `LINKEDIN_PERSON_URN` - Your LinkedIn person URN (optional)

3. **Permissions:**
   Make sure your GitHub token (automatically provided by Actions) has:
   - Contents: write (for releases and commits)
   - Actions: read (for workflows)
   - Discussions: write (for announcements)

**Note:** Getting all these API keys set up is tedious. Budget an hour or two. The LinkedIn token especially is annoying because it expires.

## Customization

### Blog Post Templates

If you want to change how blog posts are formatted:

Edit `scripts/automation/blog_generator.py` - look for the template strings. You can customize:
- Post structure and sections
- Platform-specific formatting
- Code snippet styling
- Call-to-action text

### Social Media Posts

Edit `scripts/automation/social_media_generator.py` to adjust:
- Post length (Twitter's 280 char limit is non-negotiable though)
- Hashtags (but don't go overboard)
- Emoji usage (we try to keep it minimal)
- Formatting and line breaks

### Release Process

If you need custom steps in your release process:

Edit `scripts/automation/release_automation.py` and add your steps. Common customizations:
- Custom version bumping logic
- Additional build steps
- Custom notification methods
- Pre-release validation checks

## Troubleshooting

### GitHub CLI not found

```bash
# Install from https://cli.github.com/
```

If you don't want to install it, you can create releases manually through the GitHub web UI. The automation will still generate the release notes for you.

### PyPI upload fails

Common causes:
- Incorrect API token (double-check it's saved correctly in GitHub secrets)
- Version already exists on PyPI (you can't re-upload the same version)
- Package name collision (unlikely but possible)

Try `--test-pypi` first to test on TestPyPI before going to production.

### Tests fail

When tests fail during release:
- Review the test output carefully
- Fix the failing tests
- Consider using `--skip-tests` only for emergency hotfixes (and understand the risk)

We've never had a situation where skipping tests was the right call, but the option exists.

### Blog posts not generating

Check these things:
- Is your CHANGELOG.md formatted correctly? The script looks for specific headers.
- Does the version you specified actually exist in the changelog?
- File permissions - can the script write to docs/blog/?

Look at existing blog posts to see the expected format.

## Limitations and Trade-offs

**What works well:**
- Automating repetitive tasks (version bumping, file generation)
- Catching errors early (example validation, test runs)
- Consistent release process

**What needs work:**
- Blog posts are generic - they usually need human editing for personality
- API tokens expire (especially LinkedIn's 60-day tokens)
- Social media character limits mean important details get cut
- The marketing automation assumes you want to publish everywhere at once

**Known issues:**
- Medium's API can be flaky - posts sometimes fail to create
- Twitter API rate limits can bite you if you're testing multiple releases
- The changelog parser is brittle - stick to the format

## Future Enhancements

Things we've considered adding:

- Automated documentation generation from docstrings
- Example generation from templates (for common architectures)
- Dependabot integration for dependency updates
- Security scanning automation
- Performance benchmarking on each release
- Changelog generation from commit messages
- Translation of blog posts to other languages
- Newsletter generation
- Support for Hashnode and other blog platforms

Some of these might happen, some might not. We add features as we need them.

## Support

If something breaks or you have questions:
- Open an issue on GitHub with details
- Check `scripts/automation/README.md` for more technical docs
- Review GitHub Actions logs - they're usually helpful for debugging

Remember: automation is meant to save time, not create more work. If a script isn't helping, don't use it.

---

**Last Updated:** October 18, 2025
**Version:** 1.0.0
