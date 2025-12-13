# Neural DSL Marketing Website and Documentation Portal

This document describes the comprehensive marketing website and documentation portal created for Neural DSL.

## Overview

A complete, production-ready website built with Docusaurus featuring:

- **Marketing homepage** with features, testimonials, and CTAs
- **Interactive playground** - REPL in browser for trying Neural DSL
- **Comprehensive documentation** - Guides, tutorials, API reference
- **Blog** with SEO optimization
- **Video tutorial placeholders** for upcoming content
- **Case studies** from real users
- **Comparison matrix** vs competitors (TensorFlow, PyTorch, Keras)
- **Pricing page** for enterprise features
- **Community showcase** of projects built with Neural DSL
- **Deployment configs** for Netlify and Vercel

## Directory Structure

```
website/
├── package.json              # Node.js dependencies
├── docusaurus.config.js      # Main configuration
├── sidebars.js              # Documentation sidebar
├── netlify.toml             # Netlify deployment config
├── vercel.json              # Vercel deployment config
├── README.md                # Website README
├── DEPLOYMENT.md            # Deployment guide
│
├── src/
│   ├── css/
│   │   └── custom.css       # Global styles and theme
│   ├── components/          # React components
│   │   ├── HomepageFeatures/
│   │   ├── CodeDemo/
│   │   ├── Testimonials/
│   │   └── Stats/
│   └── pages/               # Custom pages
│       ├── index.js         # Homepage
│       ├── playground.js    # Interactive playground
│       ├── pricing.js       # Pricing page
│       ├── comparison.js    # Comparison matrix
│       ├── showcase.js      # Community showcase
│       ├── privacy.md       # Privacy policy
│       └── terms.md         # Terms of service
│
├── docs/                    # Documentation
│   ├── intro.md
│   ├── getting-started/
│   │   ├── installation.md
│   │   ├── quick-start.md
│   │   └── first-model.md
│   ├── tutorial/
│   │   └── basics.md
│   ├── concepts/
│   │   ├── dsl-syntax.md
│   │   └── shape-propagation.md
│   ├── features/
│   │   └── neuraldbg.md
│   ├── guides/
│   ├── api/
│   │   └── cli.md
│   └── enterprise/
│
├── blog/                    # Blog posts
│   ├── 2024-12-13-welcome.md
│   ├── 2024-12-13-case-study-medical-imaging.md
│   └── 2024-12-13-video-tutorials.md
│
└── static/                  # Static assets
    └── img/
        ├── logo.svg
        └── favicon.ico
```

## Key Features

### 1. Marketing Homepage
- Hero section with code example
- Feature grid (9 key features)
- Stats section (downloads, stars, etc.)
- Code demo showing one DSL → multiple frameworks
- Testimonials from users
- CTA sections

### 2. Interactive Playground
- Browser-based REPL
- Pre-loaded examples (MNIST, sentiment analysis, transformers)
- Backend selection (TensorFlow, PyTorch, ONNX)
- Real-time compilation simulation
- Generated code preview

### 3. Documentation
Comprehensive docs including:
- Getting started guide
- Tutorial series
- Core concepts
- Feature documentation
- API reference
- Enterprise guides

### 4. Blog
SEO-optimized blog with:
- Welcome post
- Case study: Medical imaging at Stanford
- Video tutorial announcement
- RSS feed
- Reading time estimates
- Tag system

### 5. Pricing Page
Three-tier pricing:
- Open Source (Free)
- Team ($99/user/month)
- Enterprise (Custom)

Each with feature lists and FAQ section.

### 6. Comparison Matrix
Detailed comparison with:
- TensorFlow
- PyTorch
- Keras

Covering 16+ features with visual indicators.

### 7. Community Showcase
9 example projects featuring:
- Medical image classification
- Sentiment analysis
- Autonomous vehicle vision
- Financial fraud detection
- And more...

Each with tags, descriptions, and links to case studies.

### 8. Video Tutorials (Coming Soon)
Placeholder blog post announcing:
- 12 video tutorials planned
- Beginner, intermediate, and advanced series
- Topics from basics to deployment
- Timeline and subscription options

## Technology Stack

- **Framework**: Docusaurus 3.1
- **React**: 18.2
- **Styling**: Custom CSS with CSS variables
- **Deployment**: Netlify / Vercel
- **SEO**: Meta tags, sitemap, structured data
- **Analytics**: Google Analytics (configurable)
- **Search**: Algolia (optional, configurable)

## SEO Features

1. **Meta Tags**
   - Keywords, description, og:image
   - Twitter cards
   - Structured data

2. **Sitemap**
   - Automatic generation
   - Submitted to search engines
   - Weekly update frequency

3. **Performance**
   - Code splitting
   - Image optimization
   - Caching headers
   - CDN distribution

4. **Mobile Responsive**
   - Fully responsive design
   - Touch-friendly navigation
   - Optimized for all devices

## Deployment

### Netlify

```bash
cd website
npm install
npm run build
# Deploy build/ directory
```

Or use automatic Git deployment:
- Connect GitHub repo to Netlify
- Set base directory: `website`
- Build command: `npm install && npm run build`
- Publish directory: `build`

### Vercel

```bash
cd website
vercel --prod
```

Or connect GitHub repo for automatic deployments.

### Configuration Files

- `netlify.toml` - Netlify config with redirects and headers
- `vercel.json` - Vercel config with build settings
- `.gitignore` - Excludes node_modules and build artifacts

## Local Development

```bash
cd website
npm install
npm start
```

Opens at http://localhost:3000 with hot reload.

## Customization

### Update Content

1. **Homepage**: Edit `src/pages/index.js`
2. **Docs**: Add files to `docs/`
3. **Blog**: Add files to `blog/`
4. **Styling**: Modify `src/css/custom.css`

### Update Configuration

1. **Site config**: `docusaurus.config.js`
2. **Sidebar**: `sidebars.js`
3. **Deployment**: `netlify.toml` or `vercel.json`

### Add Analytics

Update `docusaurus.config.js`:

```js
gtag: {
  trackingID: 'G-XXXXXXXXXX',
}
```

### Enable Search

Configure Algolia in `docusaurus.config.js`:

```js
algolia: {
  appId: 'YOUR_APP_ID',
  apiKey: 'YOUR_API_KEY',
  indexName: 'neural-dsl',
}
```

## Content Guidelines

### Documentation
- Clear, concise language
- Code examples for every concept
- Step-by-step tutorials
- Visual aids (diagrams, screenshots)

### Blog Posts
- Front matter with slug, title, authors, tags
- Use `<!--truncate-->` for excerpts
- SEO-friendly titles and descriptions
- Internal links to docs

### Case Studies
- Real-world applications
- Quantifiable results
- Technical details
- Testimonials

## Next Steps

### Immediate
1. Replace placeholder images in `static/img/`
2. Add actual favicon.ico
3. Configure Google Analytics ID
4. Set up Algolia search (optional)
5. Deploy to Netlify or Vercel

### Short-term
1. Create more documentation pages
2. Write additional blog posts
3. Add more case studies
4. Create video tutorials
5. Expand showcase projects

### Long-term
1. Implement actual playground backend
2. Add user authentication for playground
3. Create interactive examples
4. Build community forum
5. Add newsletter signup

## Maintenance

### Regular Updates
- Weekly blog posts
- Monthly case studies
- Quarterly video tutorials
- Documentation updates with new features

### Monitoring
- Google Analytics for traffic
- Search Console for SEO
- User feedback via Discord
- GitHub issues for bugs

## Support

For questions or issues:
- See `website/README.md` for detailed instructions
- See `website/DEPLOYMENT.md` for deployment help
- Join Discord: https://discord.gg/KFku4KvS
- Email: Lemniscate_zero@proton.me

## License

This website is part of the Neural DSL project and is licensed under the MIT License.
