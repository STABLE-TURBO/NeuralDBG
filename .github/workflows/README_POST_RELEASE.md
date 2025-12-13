# Post-Release Automation Workflow

## Overview

The `post_release_automation.yml` workflow provides comprehensive automation after each release publication.

## Quick Start

### Automatic Trigger

The workflow runs automatically when a release is published on GitHub.

### Manual Trigger

```bash
gh workflow run post_release_automation.yml \
  -f version="0.3.0" \
  -f skip_version_bump=false \
  -f skip_discussion=false \
  -f skip_deployment=false \
  -f skip_notifications=false
```

## What It Does

1. **Version Management** (1-2 min)
   - Updates `setup.py` to next dev version (e.g., 0.3.0 → 0.3.1.dev0)
   - Updates `neural/__init__.py` version
   - Commits and pushes changes

2. **GitHub Discussion** (30 sec)
   - Creates announcement discussion
   - Includes release notes from CHANGELOG
   - Adds installation instructions and links

3. **Documentation** (30 sec)
   - Updates version references in README
   - Updates docs/index.md if exists

4. **Deployments** (varies)
   - Netlify: If `NETLIFY_BUILD_HOOK` is set
   - Vercel: If `VERCEL_DEPLOY_HOOK` is set
   - GitHub Pages: Automatic rebuild

5. **Notifications** (5 sec)
   - Discord: If `DISCORD_WEBHOOK_URL` is set
   - Rich embed with version, links, install command

6. **Planning** (10 sec)
   - Creates issue for next release planning
   - Includes checklist and suggested timeline

**Total Time:** 2-5 minutes depending on configuration

## Required Secrets

| Secret | Required | Purpose |
|--------|----------|---------|
| `GITHUB_TOKEN` | Yes (auto) | GitHub API access |
| `NETLIFY_BUILD_HOOK` | Optional | Trigger Netlify deploy |
| `VERCEL_DEPLOY_HOOK` | Optional | Trigger Vercel deploy |
| `DISCORD_WEBHOOK_URL` | Optional | Discord notifications |

## Setup Instructions

### 1. Enable Discussions

```
Repository Settings → General → Features → Discussions (enable)
```

### 2. Configure Secrets (Optional)

**Netlify:**
1. Go to Netlify Dashboard → Site Settings → Build & Deploy → Build Hooks
2. Create hook: "Post-Release Deploy"
3. Copy URL
4. Add to GitHub: Settings → Secrets → New repository secret
   - Name: `NETLIFY_BUILD_HOOK`
   - Value: (paste URL)

**Vercel:**
1. Go to Vercel Dashboard → Project Settings → Git → Deploy Hooks
2. Create hook: "Post-Release Deploy"
3. Copy URL
4. Add to GitHub: Settings → Secrets → New repository secret
   - Name: `VERCEL_DEPLOY_HOOK`
   - Value: (paste URL)

**Discord:**
1. Go to Discord Server → Channel Settings → Integrations → Webhooks
2. Create webhook: "Neural DSL Releases"
3. Copy Webhook URL
4. Add to GitHub: Settings → Secrets → New repository secret
   - Name: `DISCORD_WEBHOOK_URL`
   - Value: (paste URL)

### 3. Verify Permissions

```
Repository Settings → Actions → General → Workflow permissions
☑ Read and write permissions
☑ Allow GitHub Actions to create and approve pull requests
```

## Testing

### Test Without Release

```bash
# Dry run with all features
gh workflow run post_release_automation.yml \
  -f version="0.2.9" \
  -f skip_version_bump=true \
  -f skip_discussion=false \
  -f skip_deployment=false \
  -f skip_notifications=false

# Watch progress
gh run watch

# Check results
gh run view --log
```

### Test Individual Features

```bash
# Only test discussion creation
gh workflow run post_release_automation.yml \
  -f version="0.2.9" \
  -f skip_version_bump=true \
  -f skip_deployment=true \
  -f skip_notifications=true

# Only test deployments
gh workflow run post_release_automation.yml \
  -f version="0.2.9" \
  -f skip_version_bump=true \
  -f skip_discussion=true \
  -f skip_notifications=true
```

## Monitoring

### View Recent Runs

```bash
# List runs
gh run list --workflow=post_release_automation.yml

# View specific run
gh run view <run-id>

# View logs
gh run view <run-id> --log

# Download logs
gh run download <run-id>
```

### Check Actions Taken

After workflow completes, verify:

- [ ] `setup.py` version updated
- [ ] `neural/__init__.py` version updated
- [ ] New commit with message "Bump version to X.Y.Z.dev0 [skip ci]"
- [ ] Discussion created in Announcements category
- [ ] Planning issue created with `release` and `planning` labels
- [ ] Netlify deployment triggered (check dashboard)
- [ ] Vercel deployment triggered (check dashboard)
- [ ] Discord message received (check channel)

## Troubleshooting

### Workflow Failed

**Check logs:**
```bash
gh run list --workflow=post_release_automation.yml --limit 1
gh run view <run-id> --log
```

**Common issues:**
- Permission errors: Check workflow permissions in settings
- API rate limit: Wait and retry
- Webhook errors: Verify webhook URLs are valid
- Discussion errors: Ensure Discussions feature is enabled

### Version Not Updated

```bash
# Check if commit was made
git log --oneline -5

# Check for merge conflicts
git status

# Manually update if needed
sed -i 's/version="0.3.0.dev0"/version="0.3.1.dev0"/' setup.py
sed -i 's/__version__ = "0.3.0.dev0"/__version__ = "0.3.1.dev0"/' neural/__init__.py
git add setup.py neural/__init__.py
git commit -m "Bump version to 0.3.1.dev0 [skip ci]"
git push
```

### Discussion Not Created

**Possible causes:**
- Discussions not enabled
- Invalid category name
- API error

**Manual creation:**
```bash
# Use helper script
export GITHUB_TOKEN="your_token"
export GITHUB_REPOSITORY="owner/repo"
python scripts/automation/post_release_helper.py \
  --action discussion \
  --version "0.3.0"
```

### Deployment Not Triggered

**Test webhook:**
```bash
# Netlify
curl -X POST -d {} "$NETLIFY_BUILD_HOOK"

# Vercel
curl -X POST "$VERCEL_DEPLOY_HOOK"
```

**Check webhook logs:**
- Netlify: Dashboard → Site Settings → Build Hooks → Recent triggers
- Vercel: Dashboard → Deployments → Check for triggered deployment

### Discord Not Notified

**Test webhook:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"content":"Test message"}' \
  "$DISCORD_WEBHOOK_URL"
```

**Or use helper script:**
```bash
export GITHUB_TOKEN="your_token"
export GITHUB_REPOSITORY="owner/repo"
python scripts/automation/post_release_helper.py \
  --action discord \
  --version "0.3.0" \
  --webhook-url "$DISCORD_WEBHOOK_URL"
```

## Integration with Release Process

```
Developer actions:
  1. Create tag: git tag v0.3.0
  2. Push tag: git push origin v0.3.0

Automatic workflows:
  3. release.yml → Creates GitHub Release
  4. pypi.yml → Publishes to PyPI  
  5. post_release.yml → Sends Twitter post
  6. post_release_automation.yml → Full automation (this workflow)
     - Bumps version to 0.3.1.dev0
     - Creates discussion
     - Updates docs
     - Triggers deployments
     - Sends notifications
     - Creates planning issue

Development continues:
  7. Work on 0.3.1.dev0
  8. Repeat when ready for next release
```

## Customization

### Skip Specific Steps

Add to workflow inputs or environment:

```yaml
# Skip version bump
skip_version_bump: true

# Skip discussion
skip_discussion: true

# Skip deployments
skip_deployment: true

# Skip notifications
skip_notifications: true
```

### Modify Version Increment

Current: patch (0.3.0 → 0.3.1)

To change to minor (0.3.0 → 0.4.0), modify workflow:

```yaml
- name: Extract version information
  run: |
    IFS='.' read -r major minor patch <<< "$VERSION"
    minor=$((minor + 1))  # Changed from patch
    NEXT_DEV_VERSION="${major}.${minor}.0.dev0"  # Changed
```

### Add More Notifications

Add steps for:
- Slack: Use Slack webhook API
- Email: Use sendgrid or mailgun
- Twitter: Use existing post_release.yml
- Mastodon: Use Mastodon API
- Reddit: Use Reddit API

Example Slack notification:

```yaml
- name: Send Slack notification
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
  run: |
    curl -X POST \
      -H "Content-Type: application/json" \
      -d '{"text":"Neural DSL v${{ steps.version.outputs.current }} released!"}' \
      "$SLACK_WEBHOOK_URL"
```

## Related Files

- Workflow: `.github/workflows/post_release_automation.yml`
- Helper script: `scripts/automation/post_release_helper.py`
- Documentation: `docs/POST_RELEASE_AUTOMATION.md`
- Release guide: `GITHUB_PUBLISHING_GUIDE.md`
- Automation guide: `AUTOMATION_GUIDE.md`

## Support

Issues? Questions?

1. Check this README
2. Check `docs/POST_RELEASE_AUTOMATION.md`
3. Review workflow logs
4. Open issue with `automation` label

---

**Last Updated:** 2025
**Workflow Version:** 1.0
**Maintainer:** Repository maintainers
