// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@docusaurus/module-type-aliases`)

import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Neural DSL',
  tagline: 'The Modern Neural Network Programming Language',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://neural-dsl.dev',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/',

  // GitHub pages deployment config
  organizationName: 'Lemniscate-world',
  projectName: 'Neural',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/Lemniscate-world/Neural/tree/main/website/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        gtag: {
          trackingID: 'G-XXXXXXXXXX',
          anonymizeIP: true,
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          ignorePatterns: ['/tags/**'],
          filename: 'sitemap.xml',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      metadata: [
        { name: 'keywords', content: 'neural networks, deep learning, DSL, machine learning, tensorflow, pytorch, AI, debugging' },
        { name: 'description', content: 'Neural DSL: A powerful domain-specific language for defining, training, debugging, and deploying neural networks with cross-framework support.' },
        { property: 'og:image', content: 'img/neural-social-card.png' },
      ],

      announcementBar: {
        id: 'announcement-1',
        content:
          '⭐️ If you like Neural DSL, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/Lemniscate-world/Neural">GitHub</a>! ⭐️',
        backgroundColor: '#6366f1',
        textColor: '#ffffff',
        isCloseable: true,
      },

      navbar: {
        title: 'Neural DSL',
        logo: {
          alt: 'Neural DSL Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Docs',
          },
          { to: '/playground', label: 'Playground', position: 'left' },
          { to: '/showcase', label: 'Showcase', position: 'left' },
          { to: '/pricing', label: 'Pricing', position: 'left' },
          { to: '/comparison', label: 'Compare', position: 'left' },
          {
            href: 'https://github.com/Lemniscate-world/Neural',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://discord.gg/KFku4KvS',
            label: 'Discord',
            position: 'right',
          },
        ],
      },

      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/intro',
              },
              {
                label: 'Tutorial',
                to: '/docs/tutorial/basics',
              },
              {
                label: 'API Reference',
                to: '/docs/api',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Discord',
                href: 'https://discord.gg/KFku4KvS',
              },
              {
                label: 'GitHub Discussions',
                href: 'https://github.com/Lemniscate-world/Neural/discussions',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/Lemniscate-world/Neural',
              },
            ],
          },
          {
            title: 'Legal',
            items: [
              {
                label: 'Privacy Policy',
                to: '/privacy',
              },
              {
                label: 'Terms of Service',
                to: '/terms',
              },
              {
                label: 'License',
                href: 'https://github.com/Lemniscate-world/Neural/blob/main/LICENSE.md',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Neural DSL Team. Built with Docusaurus.`,
      },

      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'yaml'],
      },

      algolia: {
        appId: 'YOUR_APP_ID',
        apiKey: 'YOUR_SEARCH_API_KEY',
        indexName: 'neural-dsl',
        contextualSearch: true,
        searchPagePath: 'search',
      },

      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
    }),

  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'community',
        path: 'community',
        routeBasePath: 'community',
        sidebarPath: './sidebarsCommunity.js',
      },
    ],
  ],
};

export default config;
