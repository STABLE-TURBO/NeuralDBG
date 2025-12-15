# Neural DSL Website

This is the official website and documentation portal for Neural DSL, built with [Docusaurus](https://docusaurus.io/).

## Features

- 📚 **Comprehensive Documentation** - Complete guides, tutorials, and API reference
- 🎮 **Interactive Playground** - Try Neural DSL in your browser
- 💰 **Pricing Page** - Enterprise feature pricing
- 🏆 **Community Showcase** - Projects built with Neural DSL
- 📊 **Comparison Matrix** - Compare Neural DSL with competitors
- 📝 **Blog** - Latest updates, tutorials, and case studies
- 🎥 **Video Tutorials** - Coming soon
- 🔍 **SEO Optimized** - Meta tags, sitemap, and semantic HTML

## Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd website
npm install
```

### Local Development

```bash
npm start
```

This starts a local development server at http://localhost:3000 with hot reload.

### Build

```bash
npm run build
```

This generates static content into the `build` directory.

### Deployment

#### GitHub Pages (Recommended)

The site is configured for automatic deployment to GitHub Pages via GitHub Actions:

1. **Enable GitHub Pages:**
   - Go to repository Settings > Pages
   - Source: GitHub Actions

2. **Automatic Deployment:**
   - Push to `main` branch triggers deployment
   - Or manually trigger via Actions tab

3. **Live URL:**
   - https://lemniscate-world.github.io/Neural/

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed instructions.

#### Netlify

The site is also configured for Netlify:

```bash
# Deploy to Netlify
npm run build
netlify deploy --prod
```

#### Vercel

Deploy to Vercel:

```bash
vercel --prod
```

Or connect your GitHub repository to Vercel for automatic deployments.

## Project Structure

```
website/
├── blog/                   # Blog posts
│   └── *.md
├── docs/                   # Documentation
│   ├── getting-started/
│   ├── tutorial/
│   ├── concepts/
│   ├── features/
│   ├── guides/
│   ├── api/
│   └── enterprise/
├── src/
│   ├── components/        # React components
│   │   ├── HomepageFeatures/
│   │   ├── CodeDemo/
│   │   ├── Testimonials/
│   │   └── Stats/
│   ├── css/              # Global styles
│   │   └── custom.css
│   └── pages/            # Custom pages
│       ├── index.js      # Homepage
│       ├── playground.js # Interactive playground
│       ├── pricing.js    # Pricing page
│       ├── comparison.js # Comparison matrix
│       └── showcase.js   # Community showcase
├── static/               # Static assets
│   └── img/
├── docusaurus.config.js  # Docusaurus configuration
├── sidebars.js          # Documentation sidebar
├── package.json         # Dependencies
├── netlify.toml         # Netlify configuration
└── vercel.json          # Vercel configuration
```

## Adding Content

### New Blog Post

Create a file in `blog/`:

```markdown
---
slug: my-post
title: My Post Title
authors: [neural-team]
tags: [tag1, tag2]
---

Your content here...

<!--truncate-->

More content after the fold...
```

### New Documentation Page

Create a file in the appropriate `docs/` subdirectory:

```markdown
---
sidebar_position: 1
---

# Page Title

Your content here...
```

### New Custom Page

Create a file in `src/pages/`:

```jsx
import React from 'react';
import Layout from '@theme/Layout';

export default function MyPage() {
  return (
    <Layout title="My Page">
      <div className="container margin-vert--lg">
        <h1>My Page</h1>
      </div>
    </Layout>
  );
}
```

## SEO Configuration

SEO is configured in `docusaurus.config.js`:

- Meta tags for social sharing
- Sitemap generation
- Google Analytics integration
- Structured data markup

Update the following:
- `themeConfig.metadata` - Meta tags
- `gtag.trackingID` - Google Analytics ID
- `algolia` - Search configuration (optional)

## Customization

### Styling

Global styles are in `src/css/custom.css`. Use CSS variables for theming:

```css
:root {
  --ifm-color-primary: #6366f1;
  /* More variables... */
}
```

### Components

React components are in `src/components/`. Extend or create new components as needed.

### Configuration

Main configuration is in `docusaurus.config.js`. See [Docusaurus docs](https://docusaurus.io/docs/configuration) for all options.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This website is part of the Neural DSL project and is licensed under the MIT License.

## Support

- [Documentation](https://neural-dsl.dev/docs)
- [Discord](https://discord.gg/KFku4KvS)
- [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- [Email](mailto:Lemniscate_zero@proton.me)
