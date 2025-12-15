# GitHub Pages Setup Summary

This document summarizes the GitHub Pages deployment configuration for the Neural DSL website.

## Files Created/Modified

### 1. GitHub Actions Workflow
**File**: `.github/workflows/deploy-docs.yml`

Automated workflow that:
- Triggers on push to `main` branch when `website/` files change
- Can be manually triggered via Actions tab
- Builds the Docusaurus site
- Deploys to GitHub Pages using latest Actions (v4)
- Uses Node.js 18 with npm caching

### 2. Static Files
**Files**: 
- `website/static/.nojekyll` - Prevents Jekyll processing
- `website/static/CNAME` - Template for custom domain (commented out)

### 3. Docusaurus Configuration
**File**: `website/docusaurus.config.js`

Key settings:
```js
url: 'https://lemniscate-world.github.io',
baseUrl: '/Neural/',
organizationName: 'Lemniscate-world',
projectName: 'Neural',
trailingSlash: false,
```

### 4. Documentation
**Files**:
- `website/DEPLOYMENT.md` - Updated with GitHub Pages section
- `website/QUICKSTART.md` - Rewritten with GitHub Pages quick start
- `website/README.md` - Updated with GitHub Pages deployment info
- `website/.github-pages-checklist.md` - Comprehensive deployment checklist
- `README.md` (root) - Added website link

## Deployment Process

### Automatic (Recommended)
1. Push changes to `main` branch affecting `website/` directory
2. Workflow automatically triggers
3. Site builds and deploys
4. Live in 2-5 minutes at https://lemniscate-world.github.io/Neural/

### Manual
1. Go to repository Actions tab
2. Select "Deploy Docusaurus to GitHub Pages"
3. Click "Run workflow"
4. Select `main` branch
5. Monitor deployment progress

## Prerequisites

### One-Time Setup
1. Enable GitHub Pages in repository Settings > Pages
2. Set source to "GitHub Actions" (not branch)
3. Ensure repository is public or has GitHub Pages enabled for private

### Local Development
```bash
cd website
npm install
npm run build
npm run serve
```

## Workflow Features

- **Build caching**: Uses npm cache for faster builds
- **Artifact upload**: Uses upload-pages-artifact@v3
- **Deploy**: Uses deploy-pages@v4 (latest)
- **Permissions**: Properly scoped for security
- **Concurrency**: Prevents conflicting deployments
- **Working directory**: Set to `website/` for cleaner commands

## URL Structure

- **Production**: https://lemniscate-world.github.io/Neural/
- **Homepage**: /Neural/
- **Docs**: /Neural/docs/intro
- **Playground**: /Neural/playground
- **Pricing**: /Neural/pricing
- **Showcase**: /Neural/showcase
- **Comparison**: /Neural/comparison

## Custom Domain (Future)

To add custom domain:
1. Uncomment domain in `website/static/CNAME`
2. Configure DNS records (A or CNAME)
3. Update `url` in docusaurus.config.js
4. Change `baseUrl` to '/'
5. Enable in GitHub Settings > Pages

## Monitoring

- **Build status**: Actions tab shows workflow runs
- **Logs**: Click workflow run to see detailed build logs
- **Deployment**: Environment page shows deployment history
- **URL**: Live site at https://lemniscate-world.github.io/Neural/

## Troubleshooting

### Build fails
- Check Actions logs for errors
- Verify Node version (18+)
- Ensure package-lock.json is committed
- Test locally: `npm run build`

### 404 errors
- Verify GitHub Pages is enabled
- Check source is "GitHub Actions"
- Wait 2-5 minutes after deployment
- Clear browser cache

### Assets don't load
- Verify `baseUrl: '/Neural/'` is correct
- Check browser console for 404s
- Ensure files are in `website/static/`

## Security

- Workflow uses scoped permissions
- Only read content, write pages
- Uses official GitHub Actions
- No secrets required for basic deployment
- HTTPS enforced automatically

## Performance

- Build time: ~2-3 minutes
- Deploy time: ~1 minute
- Total: ~3-5 minutes from push to live
- Cached builds are faster (~1-2 minutes)

## Maintenance

### Update Dependencies
```bash
cd website
npm update
npm install
git commit -am "Update dependencies"
git push
```

### Update Workflow
Edit `.github/workflows/deploy-docs.yml` as needed. The workflow uses:
- `actions/checkout@v4`
- `actions/setup-node@v4`
- `actions/upload-pages-artifact@v3`
- `actions/deploy-pages@v4`

### Rollback
If deployment breaks:
1. Go to Actions tab
2. Find last successful run
3. Re-run workflow
4. Or revert commit: `git revert HEAD && git push`

## Resources

- [DEPLOYMENT.md](./DEPLOYMENT.md) - Full deployment guide
- [QUICKSTART.md](./QUICKSTART.md) - Quick start guide  
- [.github-pages-checklist.md](./.github-pages-checklist.md) - Deployment checklist
- [README.md](./README.md) - Development guide
- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [Docusaurus Deployment](https://docusaurus.io/docs/deployment)
- [GitHub Actions](https://docs.github.com/en/actions)

## Next Steps

1. Enable GitHub Pages in repository settings
2. Push this code to `main` branch
3. Monitor deployment in Actions tab
4. Visit https://lemniscate-world.github.io/Neural/
5. Verify all pages load correctly
6. Share the link!

## Support

Questions or issues?
- Check [DEPLOYMENT.md](./DEPLOYMENT.md) troubleshooting section
- Review [.github-pages-checklist.md](./.github-pages-checklist.md)
- Ask in [Discord](https://discord.gg/KFku4KvS)
- Open a [GitHub Issue](https://github.com/Lemniscate-world/Neural/issues)
