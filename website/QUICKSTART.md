# Website Quick Start Guide

Get the Neural DSL website running in 5 minutes.

## Prerequisites

- Node.js 18+ ([Download](https://nodejs.org/))
- npm (comes with Node.js)
- Git

## Quick Setup

```bash
# Navigate to website directory
cd website

# Install dependencies
npm install

# Start development server
npm start
```

The website will open at http://localhost:3000 with hot reload.

## Build for Production

```bash
# Build static files
npm run build

# Test production build locally
npm run serve
```

## Deploy to Netlify (Recommended)

### Option 1: Netlify UI (Easiest)

1. Push code to GitHub
2. Go to [Netlify](https://netlify.com) and sign up
3. Click "New site from Git"
4. Select your repository
5. Configure:
   - Base directory: `website`
   - Build command: `npm install && npm run build`
   - Publish directory: `build`
6. Click "Deploy site"

### Option 2: Netlify CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Build
npm run build

# Deploy
netlify deploy --prod --dir=build
```

## Deploy to Vercel

### Option 1: Vercel UI

1. Push code to GitHub
2. Go to [Vercel](https://vercel.com) and sign up
3. Click "New Project"
4. Import your repository
5. Configure:
   - Framework: Docusaurus
   - Root directory: `website`
6. Click "Deploy"

### Option 2: Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

## Configuration

### 1. Update Site Metadata

Edit `docusaurus.config.js`:

```js
const config = {
  title: 'Neural DSL',
  tagline: 'Your tagline here',
  url: 'https://your-domain.com',
  // ...
};
```

### 2. Add Google Analytics

In `docusaurus.config.js`:

```js
gtag: {
  trackingID: 'G-XXXXXXXXXX', // Your GA ID
}
```

### 3. Configure Search (Optional)

Sign up for [Algolia DocSearch](https://docsearch.algolia.com/) and update:

```js
algolia: {
  appId: 'YOUR_APP_ID',
  apiKey: 'YOUR_API_KEY',
  indexName: 'neural-dsl',
}
```

## Adding Content

### New Documentation Page

Create file in `docs/`:

```markdown
---
sidebar_position: 1
---

# Your Title

Content here...
```

### New Blog Post

Create file in `blog/`:

```markdown
---
slug: my-post
title: My Post Title
authors: [neural-team]
tags: [tag1, tag2]
---

Content here...

<!--truncate-->

More content...
```

### New Custom Page

Create file in `src/pages/`:

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

## Styling

### Update Colors

Edit `src/css/custom.css`:

```css
:root {
  --ifm-color-primary: #6366f1;
  --ifm-color-primary-dark: #4f46e5;
  /* More colors... */
}
```

### Dark Mode

Colors for dark mode:

```css
[data-theme='dark'] {
  --ifm-color-primary: #818cf8;
  /* More colors... */
}
```

## Common Commands

```bash
# Development
npm start          # Start dev server
npm run build      # Build for production
npm run serve      # Serve production build locally
npm run clear      # Clear cache

# Deployment
netlify deploy --prod     # Deploy to Netlify
vercel --prod             # Deploy to Vercel
```

## Project Structure

```
website/
├── src/
│   ├── components/    # React components
│   ├── css/          # Styles
│   └── pages/        # Custom pages
├── docs/             # Documentation
├── blog/             # Blog posts
├── static/           # Static files
├── docusaurus.config.js
└── package.json
```

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 3000
# Linux/Mac:
kill -9 $(lsof -t -i:3000)

# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use different port:
npm start -- --port 3001
```

### Build Fails

```bash
# Clear cache and rebuild
npm run clear
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Broken Links

Check for broken links:

```bash
npm run build
# Docusaurus will warn about broken links
```

## Next Steps

1. **Replace placeholder content**
   - Add real images to `static/img/`
   - Update testimonials
   - Add more case studies

2. **Configure integrations**
   - Set up Google Analytics
   - Configure Algolia search
   - Add social media links

3. **Deploy**
   - Choose Netlify or Vercel
   - Set up custom domain
   - Configure DNS

4. **Optimize**
   - Add more documentation
   - Write blog posts
   - Create video tutorials

## Resources

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [Netlify Docs](https://docs.netlify.com/)
- [Vercel Docs](https://vercel.com/docs)
- [React Documentation](https://react.dev/)

## Getting Help

- Read `website/README.md` for details
- See `website/DEPLOYMENT.md` for deployment
- Join [Discord](https://discord.gg/KFku4KvS)
- Check [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)

## License

MIT License - see LICENSE.md
