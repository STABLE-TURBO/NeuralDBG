# Post-Release Automation - Quick Reference

## What It Does

After publishing a release (e.g., v0.3.0), this workflow automatically:

1. ✅ Updates version to next dev cycle (0.3.1.dev0)
2. ✅ Creates GitHub Discussion announcement
3. ✅ Updates documentation links
4. ✅ Triggers website deployments (Netlify/Vercel)
5. ✅ Sends Discord notification
6. ✅ Creates planning issue for next release

## Setup (One-Time)

### 1. Enable Discussions
```
Repository → Settings → Features → ☑ Discussions
```

### 2. Set Permissions
```
Repository → Settings → Actions → General → Workflow permissions
☑ Read and write permissions
```

### 3. Add Secrets (Optional)
```
Repository → Settings → Secrets → Actions → New repository secret
```

| Secret Name | Get From | Required |
|-------------|----------|----------|
| `NETLIFY_BUILD_HOOK` | Netlify → Build Hooks | Optional |
| `VERCEL_DEPLOY_HOOK` | Vercel → Deploy Hooks | Optional |
| `DISCORD_WEBHOOK_URL` | Discord → Channel → Webhooks | Optional |

## Usage

### Automatic (Recommended)
1. Create and push tag: `git tag v0.3.0 && git push origin v0.3.0`
2. Workflow runs automatically after release is published
3. Check Actions tab for progress

### Manual Trigger
```bash
gh workflow run post_release_automation.yml \
  -f version="0.3.0"
```

## Verify It Worked

After workflow completes (2-5 minutes):

- [ ] Check `setup.py` - version should be `0.3.1.dev0`
- [ ] Check `neural/__init__.py` - `__version__` should be `0.3.1.dev0`
- [ ] Check Discussions tab - new announcement should exist
- [ ] Check Issues tab - planning issue should exist (labels: `release`, `planning`)
- [ ] Check Netlify/Vercel dashboard - deployment should be triggered
- [ ] Check Discord channel - release notification should appear

## Troubleshooting

### Version Not Updated
```bash
git log --oneline -5  # Check if commit was made
```
If missing, manually update and commit.

### Discussion Not Created
```bash
# Ensure Discussions are enabled in Settings
# Try manual creation:
python scripts/automation/post_release_helper.py \
  --action discussion --version "0.3.0"
```

### Deployment Not Triggered
```bash
# Test webhook manually:
curl -X POST "$NETLIFY_BUILD_HOOK"
curl -X POST "$VERCEL_DEPLOY_HOOK"
```

### Discord Not Notified
```bash
# Test webhook:
curl -X POST -H "Content-Type: application/json" \
  -d '{"content":"Test"}' "$DISCORD_WEBHOOK_URL"
```

## Common Commands

```bash
# List recent runs
gh run list --workflow=post_release_automation.yml

# Watch current run
gh run watch

# View run logs
gh run view --log

# Manual trigger with all options
gh workflow run post_release_automation.yml \
  -f version="0.3.0" \
  -f skip_version_bump=false \
  -f skip_discussion=false \
  -f skip_deployment=false \
  -f skip_notifications=false
```

## Files

| File | Purpose |
|------|---------|
| `.github/workflows/post_release_automation.yml` | Main workflow |
| `scripts/automation/post_release_helper.py` | Helper script |
| `docs/POST_RELEASE_AUTOMATION.md` | Full documentation |
| `.github/workflows/README_POST_RELEASE.md` | Detailed guide |

## Skip Options

When manually triggering, you can skip specific steps:

```bash
# Skip version bump (useful for testing)
-f skip_version_bump=true

# Skip discussion (if already created)
-f skip_discussion=true

# Skip deployments (for testing)
-f skip_deployment=true

# Skip notifications (for testing)
-f skip_notifications=true
```

## Integration Flow

```
1. git tag v0.3.0 && git push origin v0.3.0
   ↓
2. release.yml (creates GitHub Release)
   ↓
3. pypi.yml (publishes to PyPI)
   ↓
4. post_release.yml (Twitter announcement)
   ↓
5. post_release_automation.yml (this workflow)
   - Bump to 0.3.1.dev0
   - Create discussion
   - Update docs
   - Trigger deployments
   - Send notifications
   - Create planning issue
   ↓
6. Continue development on v0.3.1.dev0
```

## Need Help?

1. Check full docs: `docs/POST_RELEASE_AUTOMATION.md`
2. Check workflow README: `.github/workflows/README_POST_RELEASE.md`
3. View logs: `gh run view --log`
4. Open issue with `automation` label

---

**Quick Links:**
- [Full Documentation](docs/POST_RELEASE_AUTOMATION.md)
- [Workflow README](.github/workflows/README_POST_RELEASE.md)
- [Helper Script](scripts/automation/post_release_helper.py)
- [Automation Guide](AUTOMATION_GUIDE.md)
