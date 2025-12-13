import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

const pricingTiers = [
  {
    name: 'Open Source',
    price: 'Free',
    period: 'forever',
    description: 'Perfect for individual developers and researchers',
    features: [
      'Full DSL functionality',
      'All backend support (TensorFlow, PyTorch, ONNX)',
      'Shape propagation & validation',
      'NeuralDbg debugger',
      'Visualization tools',
      'Community support',
      'Cloud integration',
      'Experiment tracking',
      'Documentation access',
    ],
    cta: 'Get Started',
    ctaLink: '/docs/getting-started/installation',
    featured: false,
  },
  {
    name: 'Team',
    price: '$99',
    period: 'per user/month',
    description: 'For teams building production ML systems',
    features: [
      'Everything in Open Source',
      'Priority email support',
      'Advanced HPO features',
      'Team collaboration tools',
      'Shared model registry',
      'Enterprise SSO (SAML)',
      'Audit logs',
      'SLA: 99.9% uptime',
      'Dedicated Slack channel',
    ],
    cta: 'Contact Sales',
    ctaLink: 'mailto:Lemniscate_zero@proton.me?subject=Neural DSL Team Plan',
    featured: true,
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: 'contact us',
    description: 'For organizations with advanced requirements',
    features: [
      'Everything in Team',
      '24/7 phone & email support',
      'On-premise deployment',
      'Custom integrations',
      'Dedicated account manager',
      'Training & onboarding',
      'Custom SLA',
      'Legal & compliance review',
      'Priority feature requests',
      'White-label options',
    ],
    cta: 'Contact Sales',
    ctaLink: 'mailto:Lemniscate_zero@proton.me?subject=Neural DSL Enterprise Plan',
    featured: false,
  },
];

const faq = [
  {
    question: 'Is Neural DSL really free?',
    answer: 'Yes! Neural DSL is 100% open source under the MIT license. You can use it for any purpose, including commercial projects, at no cost.',
  },
  {
    question: 'What payment methods do you accept?',
    answer: 'For Team and Enterprise plans, we accept credit cards, wire transfers, and purchase orders.',
  },
  {
    question: 'Can I upgrade or downgrade my plan?',
    answer: 'Yes, you can change your plan at any time. Upgrades take effect immediately, and downgrades apply at the end of your billing cycle.',
  },
  {
    question: 'Do you offer academic discounts?',
    answer: 'Yes! Educational institutions and students can receive up to 50% off Team plans. Contact us for details.',
  },
  {
    question: 'What kind of support is included?',
    answer: 'Open source users get community support via Discord and GitHub. Team plans include priority email support, and Enterprise includes 24/7 phone support.',
  },
  {
    question: 'Can I self-host Neural DSL?',
    answer: 'The open source version can be self-hosted on any platform. Enterprise plans include support for on-premise deployments with dedicated infrastructure.',
  },
];

function PricingCard({ tier }) {
  return (
    <div className={`pricing-card ${tier.featured ? 'pricing-card--featured' : ''}`}>
      <div className="pricing-card__name">{tier.name}</div>
      <div className="pricing-card__price">{tier.price}</div>
      <div className="pricing-card__period">{tier.period}</div>
      <p>{tier.description}</p>
      <ul className="pricing-card__features">
        {tier.features.map((feature, idx) => (
          <li key={idx}>{feature}</li>
        ))}
      </ul>
      <Link
        className={`button button--${tier.featured ? 'primary' : 'secondary'} button--block button--lg`}
        to={tier.ctaLink}>
        {tier.cta}
      </Link>
    </div>
  );
}

export default function Pricing() {
  return (
    <Layout
      title="Pricing"
      description="Choose the right Neural DSL plan for your needs">
      <div className="container margin-vert--lg">
        <div style={{textAlign: 'center', marginBottom: '3rem'}}>
          <h1>Simple, Transparent Pricing</h1>
          <p style={{fontSize: '1.25rem', color: 'var(--ifm-color-emphasis-600)'}}>
            Start free, scale when you need to
          </p>
        </div>

        <div className="pricing-cards">
          {pricingTiers.map((tier, idx) => (
            <PricingCard key={idx} tier={tier} />
          ))}
        </div>

        <div style={{marginTop: '4rem'}}>
          <h2 style={{textAlign: 'center', marginBottom: '2rem'}}>Frequently Asked Questions</h2>
          <div style={{maxWidth: '800px', margin: '0 auto'}}>
            {faq.map((item, idx) => (
              <details key={idx} style={{marginBottom: '1rem', padding: '1rem', border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: '8px'}}>
                <summary style={{fontWeight: '600', cursor: 'pointer', fontSize: '1.1rem'}}>
                  {item.question}
                </summary>
                <p style={{marginTop: '1rem', color: 'var(--ifm-color-emphasis-700)'}}>
                  {item.answer}
                </p>
              </details>
            ))}
          </div>
        </div>

        <div style={{textAlign: 'center', marginTop: '4rem', padding: '3rem', background: 'var(--ifm-color-emphasis-100)', borderRadius: '8px'}}>
          <h2>Need a Custom Solution?</h2>
          <p style={{fontSize: '1.1rem', marginBottom: '2rem'}}>
            We work with organizations to create tailored plans that fit your specific requirements.
          </p>
          <Link
            className="button button--primary button--lg"
            to="mailto:Lemniscate_zero@proton.me?subject=Neural DSL Custom Solution">
            Contact Sales
          </Link>
        </div>
      </div>
    </Layout>
  );
}
