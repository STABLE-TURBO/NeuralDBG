# Post-Release Automation Guide

This document describes the automated post-release workflow that runs after each Neural DSL release is published.

## Overview

The post-release automation workflow (`.github/workflows/post_release_automation.yml`) automatically handles several important tasks after a successful release publication:

1. **Version Management** - Updates version to next development cycle
2. **Community Engagement** - Creates announcement discussions
3. **Documentation** - Updates project documentation and website
4. **Deployment** - Triggers documentation site deployments
5. **Notifications** - Sends release notifications to Discord
6. **Planning** - Creates planning issue for next release

## Trigger Conditions

The workflow triggers automatically on:

- **Release Publication**: When a new GitHub release is published
- **Manual Dispatch**: Can be triggered manually with custom parameters

## Automated Tasks

### 1. Version Bump to Next Dev Version

After releasing version `X.Y.Z`, the workflow automatically updates:

- `setup.py` - Changes version to `X.Y.(Z+1).dev0`
- `neural/__init__.py` - Updates `__version__` variable

**Example:**
- Released: `0.3.0`
- Auto-updated to: `0.3.1.dev0`

This ensures the main branch is always ready for next development cycle.

### 2. GitHub Discussion Announcement

Creates a new discussion in the repository's Discussions section with:

- Release highlights extracted from CHANGELOG.md
- Installation instructions
- Links to PyPI and GitHub release
- Call for community feedback

**Format:**
```markdown
🎉 Neural DSL vX.Y.Z Released!

[Release notes from CHANGELOG.md]

---

**Installation:**
pip install --upgrade neural-dsl

**PyPI:** https://pypi.org/project/neural-dsl/X.Y.Z/
**GitHub Release:** https://github.com/repo/releases/tag/vX.Y.Z
```

### 3. Documentation Updates

Updates version references in:

- `README.md` - Badge URLs and version numbers
- `docs/index.md` - Current version references
- Other documentation files as needed

### 4. Deployment Triggers

Triggers rebuilds of:

- **Netlify**: If `NETLIFY_BUILD_HOOK` secret is configured
- **Vercel**: If `VERCEL_DEPLOY_HOOK` secret is configured
- **GitHub Pages**: Automatically rebuilds if gh-pages branch exists

This ensures documentation sites show the latest version immediately.

### 5. Discord Notification

Sends a rich embed notification to Discord webhook with:

- Release version
- Links to GitHub release and PyPI
- Installation command
- Timestamp

**Requirements:**
- `DISCORD_WEBHOOK_URL` secret must be configured

### 6. Next Release Planning Issue

Creates a planning issue for the next release with:

- Suggested release date (5 weeks from current release)
- Pre-release checklist
- Areas to consider (bugs, features, docs, security)
- Discussion prompts

Labels: `release`, `planning`

## Configuration

### Required Secrets

The workflow uses these GitHub secrets (all optional based on needs):

| Secret | Purpose | Required |
|--------|---------|----------|
| `GITHUB_TOKEN` | GitHub API access (auto-provided) | Yes |
| `NETLIFY_BUILD_HOOK` | Trigger Netlify deployment | No |
| `VERCEL_DEPLOY_HOOK` | Trigger Vercel deployment | No |
| `DISCORD_WEBHOOK_URL` | Send Discord notifications | No |

### Setting Up Secrets

1. **Netlify Build Hook:**
   ```
   Go to Netlify Dashboard → Site Settings → Build & Deploy → Build Hooks
   Create a new build hook and copy the URL
   Add as secret: NETLIFY_BUILD_HOOK
   ```

2. **Vercel Deploy Hook:**
   ```
   Go to Vercel Dashboard → Project Settings → Git → Deploy Hooks
   Create a new deploy hook and copy the URL
   Add as secret: VERCEL_DEPLOY_HOOK
   ```

3. **Discord Webhook:**
   ```
   Go to Discord Server → Channel Settings → Integrations → Webhooks
   Create a new webhook and copy the URL
   Add as secret: DISCORD_WEBHOOK_URL
   ```

## Manual Trigger

You can manually trigger the workflow with custom options:

```bash
gh workflow run post_release_automation.yml \
  -f version="0.3.0" \
  -f skip_version_bump=false \
  -f skip_discussion=false \
  -f skip_deployment=false \
  -f skip_notifications=false
```

Or via GitHub UI:
1. Go to Actions tab
2. Select "Post-Release Automation"
3. Click "Run workflow"
4. Fill in parameters
5. Click "Run workflow"

### Manual Trigger Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `version` | string | Version that was released (e.g., 0.3.0) | required |
| `skip_version_bump` | boolean | Skip version bump to next dev version | false |
| `skip_discussion` | boolean | Skip GitHub Discussion creation | false |
| `skip_deployment` | boolean | Skip deployment triggers | false |
| `skip_notifications` | boolean | Skip notifications | false |

## Workflow Permissions

The workflow requires these permissions:

- `contents: write` - To commit version bump
- `discussions: write` - To create announcement discussions
- `issues: write` - To create planning issues
- `pull-requests: write` - For potential PR creation (future use)

## Skipping Automation

To skip the workflow for a specific release, you can:

1. **Manually control via workflow_dispatch** with skip flags
2. **Delete the workflow file temporarily** before release
3. **Disable the workflow** in GitHub Actions settings

## Troubleshooting

### Version Bump Failed

**Issue**: Version not updated after release

**Solutions:**
- Check if commit was made: `git log --oneline -5`
- Verify bot has write permissions
- Check for merge conflicts
- Review workflow logs for errors

### Discussion Not Created

**Issue**: GitHub Discussion announcement missing

**Possible Causes:**
- Discussions feature not enabled for repository
- CHANGELOG.md format issues
- API rate limiting

**Solutions:**
- Enable Discussions in repository settings
- Check CHANGELOG.md has proper version headers
- Wait and retry manually if rate limited

### Deployment Not Triggered

**Issue**: Website not updating after release

**Solutions:**
- Verify webhook secrets are set correctly
- Test webhook URLs manually with curl
- Check webhook delivery logs in Netlify/Vercel
- Ensure webhook URLs are not expired

### Discord Notification Failed

**Issue**: No Discord message received

**Solutions:**
- Verify `DISCORD_WEBHOOK_URL` secret is set
- Test webhook URL: `curl -X POST -H "Content-Type: application/json" -d '{"content":"test"}' $WEBHOOK_URL`
- Check Discord channel permissions
- Verify webhook hasn't been deleted

### Planning Issue Not Created

**Issue**: Next release planning issue missing

**Solutions:**
- Check if issue already exists for that version
- Verify bot has issues write permission
- Review workflow logs for API errors
- Create manually if needed

## Monitoring

### Check Workflow Runs

```bash
# List recent workflow runs
gh run list --workflow=post_release_automation.yml

# View specific run
gh run view <run-id>

# View logs
gh run view <run-id> --log
```

### Verify Actions Taken

After each release, verify:

1. **Version updated**: Check `setup.py` and `neural/__init__.py`
2. **Discussion created**: Check repository Discussions tab
3. **Issue created**: Check Issues with `release` and `planning` labels
4. **Deployments triggered**: Check Netlify/Vercel dashboards
5. **Discord notified**: Check Discord channel

## Best Practices

1. **Test Webhooks**: Test all webhook URLs before release
2. **Review Changelog**: Ensure CHANGELOG.md is updated before release
3. **Monitor First Run**: Watch the workflow run for first few releases
4. **Keep Secrets Updated**: Rotate webhook URLs periodically
5. **Document Changes**: Update this guide if workflow changes

## Integration with Release Process

This workflow complements the main release workflows:

```
1. Tag pushed (vX.Y.Z)
   ↓
2. release.yml → Creates GitHub Release
   ↓
3. pypi.yml → Publishes to PyPI
   ↓
4. post_release.yml → Sends Twitter announcement
   ↓
5. post_release_automation.yml → This workflow (full automation)
   ↓
6. Development continues on vX.Y.(Z+1).dev0
```

## Future Enhancements

Potential improvements for this workflow:

- [ ] Slack notifications support
- [ ] Automated blog post generation
- [ ] Docker image rebuild trigger
- [ ] Update conda-forge feedstock
- [ ] Send release email to subscribers
- [ ] Update package managers (Homebrew, Chocolatey)
- [ ] Generate release comparison report
- [ ] Automated documentation translation updates

## Related Documentation

- [Release Process](../GITHUB_PUBLISHING_GUIDE.md)
- [Automation Guide](../AUTOMATION_GUIDE.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Distribution Plan](../DISTRIBUTION_PLAN.md)

## Support

If you encounter issues with the post-release automation:

1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Open an issue with the `automation` label
4. Contact maintainers for urgent issues

---

**Note**: This workflow is designed to be robust and fail gracefully. If any step fails, other steps will still execute, and you'll receive detailed logs to help diagnose issues.
