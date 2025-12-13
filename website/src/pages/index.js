import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import CodeDemo from '@site/src/components/CodeDemo';
import Testimonials from '@site/src/components/Testimonials';
import Stats from '@site/src/components/Stats';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <p className={styles.heroDescription}>
          Simplify deep learning development with a powerful DSL, cross-framework support,
          and built-in debugging tools.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started - 5min ⏱️
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/playground">
            Try Playground
          </Link>
        </div>
        <div className={styles.heroDemo}>
          <pre className={styles.codeBlock}>
{`network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Conv2D(32, (3,3), "relu")
    MaxPooling2D((2,2))
    Flatten()
    Dense(128, "relu")
    Dropout(0.5)
    Output(10, "softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 15
    batch_size: 64
  }
}`}
          </pre>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - The Modern Neural Network Programming Language`}
      description="Define, train, debug, and deploy neural networks with a powerful DSL that supports TensorFlow, PyTorch, and ONNX.">
      <HomepageHeader />
      <main>
        <Stats />
        <HomepageFeatures />
        <CodeDemo />
        <Testimonials />
        <section className="cta-section">
          <div className="container">
            <h2>Ready to Build Better Neural Networks?</h2>
            <p>Join thousands of developers and researchers using Neural DSL</p>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg"
                to="/docs/getting-started/installation">
                Install Now
              </Link>
              <Link
                className="button button--outline button--secondary button--lg"
                to="/pricing">
                View Pricing
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
