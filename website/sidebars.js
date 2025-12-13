/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/first-model',
      ],
    },
    {
      type: 'category',
      label: 'Tutorial',
      items: [
        'tutorial/basics',
        'tutorial/layers',
        'tutorial/training',
        'tutorial/debugging',
        'tutorial/deployment',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/dsl-syntax',
        'concepts/shape-propagation',
        'concepts/backends',
        'concepts/hpo',
      ],
    },
    {
      type: 'category',
      label: 'Features',
      items: [
        'features/neuraldbg',
        'features/visualization',
        'features/cloud-integration',
        'features/experiment-tracking',
        'features/no-code',
        'features/ai-integration',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/mnist',
        'guides/sentiment-analysis',
        'guides/transformers',
        'guides/custom-layers',
        'guides/onnx-export',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/cli',
        'api/layers',
        'api/optimizers',
        'api/losses',
        'api/python-api',
      ],
    },
    {
      type: 'category',
      label: 'Enterprise',
      items: [
        'enterprise/features',
        'enterprise/support',
        'enterprise/security',
        'enterprise/deployment',
      ],
    },
  ],
};

export default sidebars;
