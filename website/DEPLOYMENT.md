# Deployment Guide

This guide covers deploying the Neural DSL website to Netlify or Vercel.

## Prerequisites

- GitHub repository with website code
- Netlify or Vercel account
- Node.js 18+ installed locally

## Option 1: Deploy to Netlify

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

## Option 2: Deploy to Vercel

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
- Netlify: Deploys > Deploy log
- Vercel: Deployments > Build logs

**Common issues:**
- Node version mismatch: Set to 18 in settings
- Missing dependencies: Ensure `package.json` is up to date
- Build command error: Verify command works locally

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
- [Netlify Docs](https://docs.netlify.com/)
- [Vercel Docs](https://vercel.com/docs)
- [Docusaurus Deployment](https://docusaurus.io/docs/deployment)
- [Our Discord](https://discord.gg/KFku4KvS)

## Security

Both platforms provide:
- Automatic HTTPS
- DDoS protection
- CDN distribution
- Security headers (configured in config files)

Review security headers in:
- `netlify.toml` → `[[headers]]`
- `vercel.json` → `headers`
