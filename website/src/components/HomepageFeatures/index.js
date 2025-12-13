import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '🎯 Simple & Intuitive',
    description: (
      <>
        Write neural networks in a clean, YAML-like syntax. No framework boilerplate,
        just pure model definition. Get started in minutes, not hours.
      </>
    ),
  },
  {
    title: '🔄 Cross-Framework',
    description: (
      <>
        Compile to TensorFlow, PyTorch, or ONNX with a single flag. Switch frameworks
        without rewriting code. True framework independence.
      </>
    ),
  },
  {
    title: '✅ Shape Validation',
    description: (
      <>
        Catch shape mismatches before runtime. Automatic shape propagation ensures
        your model architecture is correct before training.
      </>
    ),
  },
  {
    title: '🐛 Built-in Debugger',
    description: (
      <>
        NeuralDbg provides real-time execution tracing, gradient analysis, dead neuron
        detection, and anomaly detection. Debug with confidence.
      </>
    ),
  },
  {
    title: '📊 Visualization Tools',
    description: (
      <>
        Generate interactive architecture diagrams, shape flow visualizations, and
        performance charts automatically. See your model clearly.
      </>
    ),
  },
  {
    title: '🚀 Production Ready',
    description: (
      <>
        Export to ONNX, TFLite, TorchScript with optimization. Deploy configs for
        TensorFlow Serving and TorchServe included.
      </>
    ),
  },
  {
    title: '🔬 HPO Integration',
    description: (
      <>
        Built-in hyperparameter optimization with Optuna. Cross-framework HPO
        support for TensorFlow and PyTorch.
      </>
    ),
  },
  {
    title: '☁️ Cloud Native',
    description: (
      <>
        Run on Kaggle, Google Colab, AWS SageMaker. Built-in cloud integration
        with remote debugging support via ngrok tunnels.
      </>
    ),
  },
  {
    title: '📝 No-Code Interface',
    description: (
      <>
        Visual model builder for beginners and rapid prototyping. Build models
        with drag-and-drop, export to DSL or code.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="feature text--center padding-horiz--md">
        <h3 className="feature__title">{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className="features">
      <div className="container">
        <h2 className="text--center margin-bottom--lg">
          Everything You Need for Deep Learning
        </h2>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
