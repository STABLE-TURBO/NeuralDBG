# Quick Start - GitHub Pages Deployment

This guide will help you quickly deploy the Neural DSL website to GitHub Pages.

## Step 1: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** (top navigation)
3. Click on **Pages** (left sidebar)
4. Under "Source", select **GitHub Actions**

That's it! No branch selection needed.

## Step 2: Trigger Deployment

### Option A: Automatic (Recommended)

Simply push any changes to the `website/` directory on the `main` branch:

```bash
cd website
# Make any changes to files...
git add .
git commit -m "Update website content"
git push origin main
```

The GitHub Actions workflow will automatically build and deploy.

### Option B: Manual Trigger

1. Go to the **Actions** tab in your repository
2. Click on "Deploy Docusaurus to GitHub Pages" workflow
3. Click **Run workflow** button (right side)
4. Select `main` branch
5. Click **Run workflow**

## Step 3: Monitor Deployment

1. Go to **Actions** tab
2. Click on the running workflow
3. Watch the build and deploy process
4. Once complete (green checkmark), your site is live!

## Step 4: Visit Your Site

Your website will be available at:

**https://lemniscate-world.github.io/Neural/**

## Local Testing (Optional but Recommended)

Before pushing changes, test locally:

```bash
cd website
npm install
npm run build
npm run serve
```

Visit http://localhost:3000 to preview your site.

## Troubleshooting

### Build Fails

**Check the workflow logs:**
1. Go to Actions tab
2. Click on the failed workflow run
3. Expand the failing step to see error details

**Common issues:**
- Missing `package-lock.json`: Run `npm install` locally and commit the file
- Node version mismatch: Workflow uses Node 18
- Syntax errors in markdown or config files

### 404 Error

If you see a 404 error:

1. Verify GitHub Pages is enabled (Settings > Pages)
2. Check that source is set to "GitHub Actions"
3. Ensure the workflow completed successfully
4. Wait a few minutes for DNS propagation

### Deployment Not Triggering

Workflow only triggers when:
- Files in `website/` directory change
- Or the workflow file itself changes
- Or manually triggered via Actions tab

**To force trigger:**
- Make a small change to any file in `website/`
- Or use manual trigger (Option B above)

## Next Steps

- **Custom Domain**: See [DEPLOYMENT.md](./DEPLOYMENT.md) for custom domain setup
- **Content Updates**: See [README.md](./README.md) for adding documentation and blog posts
- **Configuration**: Edit `docusaurus.config.js` for site settings

## Support

Need help?
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Full deployment guide
- [README.md](./README.md) - Development guide
- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [Discord Community](https://discord.gg/KFku4KvS)
