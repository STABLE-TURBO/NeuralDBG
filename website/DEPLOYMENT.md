# Deployment Guide

This guide covers deploying the Neural DSL website to GitHub Pages, Netlify, or Vercel.

## Prerequisites

- GitHub repository with website code
- Node.js 18+ installed locally
- GitHub Pages, Netlify, or Vercel account

## Option 1: Deploy to GitHub Pages (Recommended)

### Automated Deployment with GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/deploy-docs.yml`) that automatically deploys to GitHub Pages.

**Prerequisites:**
1. Enable GitHub Pages in repository settings:
   - Go to repository Settings > Pages
   - Source: GitHub Actions
   - No branch selection needed (Actions handles it)

2. The workflow triggers automatically on:
   - Push to `main` branch (when `website/` files change)
   - Manual trigger via Actions tab

**Deployment URL:**
- Site will be available at: `https://lemniscate-world.github.io/Neural/`

**Configuration:**
The `docusaurus.config.js` is already configured with:
```js
url: 'https://lemniscate-world.github.io',
baseUrl: '/Neural/',
organizationName: 'Lemniscate-world',
projectName: 'Neural',
```

**Manual Trigger:**
1. Go to repository Actions tab
2. Select "Deploy Docusaurus to GitHub Pages"
3. Click "Run workflow"
4. Select `main` branch
5. Click "Run workflow"

**Local Testing:**
```bash
cd website
npm install
npm run build
npm run serve
```

### Custom Domain (Optional)

To use a custom domain with GitHub Pages:

1. **Add Domain File:**
   - Edit `website/static/CNAME`
   - Uncomment and set your domain (e.g., `neuraldsl.com`)

2. **Configure DNS:**
   ```
   # Apex domain
   A     @    185.199.108.153
   A     @    185.199.109.153
   A     @    185.199.110.153
   A     @    185.199.111.153
   
   # Subdomain (www)
   CNAME www  lemniscate-world.github.io
   ```

3. **Update Config:**
   - Edit `website/docusaurus.config.js`
   - Change `url` to your custom domain
   - Keep `baseUrl: '/'` for custom domain

4. **Enable in GitHub:**
   - Go to Settings > Pages
   - Add custom domain
   - Check "Enforce HTTPS"

## Option 2: Deploy to Netlify

### Method A: Netlify UI

1. **Connect Repository**
   - Go to [Netlify](https://netlify.com)
   - Click "New site from Git"
   - Connect your GitHub repository
   - Select the repository

2. **Configure Build Settings**
   - Base directory: `website`
   - Build command: `npm install && npm run build`
   - Publish directory: `website/build`
   - Node version: `18`

3. **Deploy**
   - Click "Deploy site"
   - Netlify will automatically build and deploy

4. **Custom Domain (Optional)**
   - Go to Site settings > Domain management
   - Add your custom domain
   - Configure DNS records

### Method B: Netlify CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Build site
cd website
npm install
npm run build

# Deploy
netlify deploy --prod --dir=build
```

### Netlify Configuration

The site uses `netlify.toml` for configuration:

```toml
[build]
  base = "website/"
  command = "npm install && npm run build"
  publish = "build/"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

## Option 3: Deploy to Vercel

### Method A: Vercel UI

1. **Import Project**
   - Go to [Vercel](https://vercel.com)
   - Click "New Project"
   - Import from GitHub
   - Select the repository

2. **Configure Project**
   - Framework Preset: Docusaurus
   - Root Directory: `website`
   - Build Command: `npm run build`
   - Output Directory: `build`

3. **Deploy**
   - Click "Deploy"
   - Vercel will build and deploy automatically

4. **Custom Domain (Optional)**
   - Go to Project Settings > Domains
   - Add your custom domain

### Method B: Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy from website directory
cd website
vercel --prod
```

### Vercel Configuration

The site uses `vercel.json` for configuration:

```json
{
  "buildCommand": "npm install && npm run build",
  "outputDirectory": "build",
  "framework": "docusaurus"
}
```

## Environment Variables

### Required (Optional)

- `GOOGLE_ANALYTICS_ID` - Google Analytics tracking ID
- `ALGOLIA_APP_ID` - Algolia search app ID (if using Algolia)
- `ALGOLIA_API_KEY` - Algolia API key
- `ALGOLIA_INDEX_NAME` - Algolia index name

Add these in your hosting platform's dashboard:
- **Netlify**: Site settings > Environment variables
- **Vercel**: Project settings > Environment Variables

## Post-Deployment

### 1. Test Deployment

Visit your deployed URL and check:
- Homepage loads correctly
- Documentation is accessible
- Playground works
- All internal links function
- Mobile responsiveness

### 2. Configure DNS

Point your domain to the deployment:

**Netlify:**
- A record: `75.2.60.5`
- AAAA record: `2600:1f14:fff:aaaa::5`

**Vercel:**
- CNAME: `cname.vercel-dns.com`

### 3. Enable HTTPS

Both Netlify and Vercel provide automatic HTTPS:
- SSL certificate is provisioned automatically
- Force HTTPS redirect is enabled by default

### 4. Set up Monitoring

**Google Analytics:**
Update `docusaurus.config.js` with your tracking ID:
```js
gtag: {
  trackingID: 'G-XXXXXXXXXX',
},
```

**Vercel Analytics:**
Enable in project settings for performance monitoring.

## Continuous Deployment

Both platforms support automatic deployments:

### Branch Deployments
- `main` branch → Production
- `develop` branch → Staging/Preview
- Pull requests → Preview deployments

### Configure in Platform
- **Netlify**: Deploy settings > Continuous Deployment
- **Vercel**: Git integration is automatic

## Custom Domain Setup

### 1. Purchase Domain
Buy a domain from:
- Namecheap
- GoDaddy
- Google Domains
- Cloudflare

### 2. Configure DNS

Add DNS records pointing to your hosting:

```
# Netlify
A     @    75.2.60.5
CNAME www  [your-site].netlify.app

# Vercel  
CNAME @    cname.vercel-dns.com
CNAME www  cname.vercel-dns.com
```

### 3. Add Domain in Platform
- Go to domain settings
- Add your domain
- Wait for DNS propagation (up to 48 hours)

## Performance Optimization

### 1. Enable Caching

Headers are configured in `netlify.toml` / `vercel.json`:
- Static assets: 1 year cache
- HTML: No cache (always fresh)

### 2. Image Optimization

Use optimized images:
- WebP format when possible
- Compress images before upload
- Use appropriate dimensions

### 3. Bundle Optimization

Docusaurus includes:
- Code splitting
- Tree shaking
- Minification
- Compression

## Troubleshooting

### Build Fails

**Check build logs:**
- GitHub Pages: Actions tab > Workflow runs
- Netlify: Deploys > Deploy log
- Vercel: Deployments > Build logs

**Common issues:**
- Node version mismatch: Set to 18 in settings
- Missing dependencies: Ensure `package.json` is up to date
- Build command error: Verify command works locally
- GitHub Pages: Ensure Pages is enabled in repository settings with "GitHub Actions" source

### 404 Errors

**Ensure redirects are configured:**
- Check `netlify.toml` or `vercel.json`
- SPA fallback should redirect all routes to index.html

### Slow Build Times

**Optimize:**
- Use dependency caching
- Remove unused dependencies
- Parallelize builds if possible

## Rollback

### GitHub Pages
1. Go to Actions tab
2. Find previous successful workflow run
3. Re-run the workflow
4. Or revert the commit and push

### Netlify
1. Go to Deploys
2. Find previous working deploy
3. Click "Publish deploy"

### Vercel
1. Go to Deployments
2. Find previous working deployment
3. Click "Promote to Production"

## Support

For deployment issues:
- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [Netlify Docs](https://docs.netlify.com/)
- [Vercel Docs](https://vercel.com/docs)
- [Docusaurus Deployment](https://docusaurus.io/docs/deployment)
- [Our Discord](https://discord.gg/KFku4KvS)

## Security

All platforms provide:
- Automatic HTTPS
- DDoS protection
- CDN distribution
- Security headers (configured in config files)

Review security headers in:
- `netlify.toml` → `[[headers]]`
- `vercel.json` → `headers`
- GitHub Pages: Automatic HTTPS enforcement
