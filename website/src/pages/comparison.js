import React from 'react';
import Layout from '@theme/Layout';

const comparisonData = [
  {
    feature: 'Learning Curve',
    neural: '⭐⭐⭐⭐⭐',
    tensorflow: '⭐⭐',
    pytorch: '⭐⭐⭐',
    keras: '⭐⭐⭐⭐',
  },
  {
    feature: 'Shape Validation',
    neural: '✅ Automatic pre-runtime',
    tensorflow: '❌ Runtime only',
    pytorch: '❌ Runtime only',
    keras: '✅ Partial',
  },
  {
    feature: 'Framework Switching',
    neural: '✅ Single flag',
    tensorflow: '❌ N/A',
    pytorch: '❌ N/A',
    keras: '❌ N/A',
  },
  {
    feature: 'Built-in Debugger',
    neural: '✅ NeuralDbg',
    tensorflow: '⚠️ TensorBoard',
    pytorch: '⚠️ TensorBoard',
    keras: '⚠️ TensorBoard',
  },
  {
    feature: 'Real-time Tracing',
    neural: '✅ Yes',
    tensorflow: '❌ No',
    pytorch: '❌ No',
    keras: '❌ No',
  },
  {
    feature: 'Gradient Analysis',
    neural: '✅ Built-in',
    tensorflow: '⚠️ Manual',
    pytorch: '⚠️ Manual',
    keras: '⚠️ Manual',
  },
  {
    feature: 'Dead Neuron Detection',
    neural: '✅ Automatic',
    tensorflow: '❌ No',
    pytorch: '❌ No',
    keras: '❌ No',
  },
  {
    feature: 'Code Generation',
    neural: '✅ TF/PyTorch/ONNX',
    tensorflow: '❌ N/A',
    pytorch: '❌ N/A',
    keras: '⚠️ To TF only',
  },
  {
    feature: 'Architecture Visualization',
    neural: '✅ Interactive 3D',
    tensorflow: '⚠️ Basic',
    pytorch: '⚠️ Third-party',
    keras: '⚠️ Basic',
  },
  {
    feature: 'HPO Integration',
    neural: '✅ Cross-framework',
    tensorflow: '⚠️ Manual',
    pytorch: '⚠️ Manual',
    keras: '⚠️ Manual',
  },
  {
    feature: 'Deployment Export',
    neural: '✅ ONNX/TFLite/TorchScript',
    tensorflow: '✅ TFLite/SavedModel',
    pytorch: '✅ TorchScript',
    keras: '✅ TFLite',
  },
  {
    feature: 'Cloud Integration',
    neural: '✅ Kaggle/Colab/SageMaker',
    tensorflow: '⚠️ Manual',
    pytorch: '⚠️ Manual',
    keras: '⚠️ Manual',
  },
  {
    feature: 'No-Code Interface',
    neural: '✅ Yes',
    tensorflow: '❌ No',
    pytorch: '❌ No',
    keras: '❌ No',
  },
  {
    feature: 'Experiment Tracking',
    neural: '✅ Automatic',
    tensorflow: '⚠️ Manual',
    pytorch: '⚠️ Manual',
    keras: '⚠️ Manual',
  },
  {
    feature: 'Documentation',
    neural: '✅ Comprehensive',
    tensorflow: '✅ Extensive',
    pytorch: '✅ Extensive',
    keras: '✅ Good',
  },
  {
    feature: 'Community Size',
    neural: '⭐⭐ Growing',
    tensorflow: '⭐⭐⭐⭐⭐',
    pytorch: '⭐⭐⭐⭐⭐',
    keras: '⭐⭐⭐⭐',
  },
];

const useCases = [
  {
    title: 'Research & Prototyping',
    neural: '✅ Excellent - Fast iteration, easy experimentation',
    competitors: '⚠️ Good - More boilerplate code required',
  },
  {
    title: 'Production Deployment',
    neural: '✅ Excellent - Multi-format export, deployment configs',
    competitors: '✅ Excellent - Mature deployment tools',
  },
  {
    title: 'Learning & Education',
    neural: '✅ Excellent - Clear syntax, comprehensive tutorials',
    competitors: '⚠️ Good - Steeper learning curve',
  },
  {
    title: 'Multi-Framework Projects',
    neural: '✅ Perfect - Built for cross-framework work',
    competitors: '❌ Poor - Requires complete rewrites',
  },
  {
    title: 'Debugging Complex Models',
    neural: '✅ Excellent - NeuralDbg with real-time analysis',
    competitors: '⚠️ Fair - Limited debugging tools',
  },
];

export default function Comparison() {
  return (
    <Layout
      title="Comparison"
      description="See how Neural DSL compares to TensorFlow, PyTorch, and Keras">
      <div className="container margin-vert--lg">
        <div style={{textAlign: 'center', marginBottom: '3rem'}}>
          <h1>How Neural DSL Compares</h1>
          <p style={{fontSize: '1.25rem', color: 'var(--ifm-color-emphasis-600)'}}>
            See how we stack up against popular deep learning frameworks
          </p>
        </div>

        <div style={{marginBottom: '4rem'}}>
          <h2>Feature Comparison</h2>
          <div style={{overflowX: 'auto'}}>
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Neural DSL</th>
                  <th>TensorFlow</th>
                  <th>PyTorch</th>
                  <th>Keras</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.map((row, idx) => (
                  <tr key={idx}>
                    <td><strong>{row.feature}</strong></td>
                    <td>{row.neural}</td>
                    <td>{row.tensorflow}</td>
                    <td>{row.pytorch}</td>
                    <td>{row.keras}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{marginBottom: '4rem'}}>
          <h2>Use Case Recommendations</h2>
          <div style={{overflowX: 'auto'}}>
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Use Case</th>
                  <th>Neural DSL</th>
                  <th>TensorFlow/PyTorch/Keras</th>
                </tr>
              </thead>
              <tbody>
                {useCases.map((useCase, idx) => (
                  <tr key={idx}>
                    <td><strong>{useCase.title}</strong></td>
                    <td>{useCase.neural}</td>
                    <td>{useCase.competitors}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{marginBottom: '4rem', padding: '2rem', background: 'var(--ifm-color-emphasis-100)', borderRadius: '8px'}}>
          <h2>When to Use Neural DSL</h2>
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem'}}>
            <div>
              <h3>✅ Great For:</h3>
              <ul>
                <li>Rapid prototyping and experimentation</li>
                <li>Cross-framework development</li>
                <li>Learning deep learning concepts</li>
                <li>Projects requiring advanced debugging</li>
                <li>Teams with mixed framework preferences</li>
                <li>Shape-sensitive architectures</li>
                <li>Research with frequent iterations</li>
              </ul>
            </div>
            <div>
              <h3>⚠️ Consider Alternatives If:</h3>
              <ul>
                <li>You need cutting-edge research features (use PyTorch)</li>
                <li>You have existing large TensorFlow codebase</li>
                <li>You require specific framework optimizations</li>
                <li>Your team is already expert in one framework</li>
                <li>You need maximum performance tuning</li>
                <li>You're working on very specialized architectures</li>
              </ul>
            </div>
          </div>
        </div>

        <div style={{marginBottom: '4rem'}}>
          <h2>Code Comparison</h2>
          <p>See how the same MNIST model looks in different frameworks:</p>
          
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem'}}>
            <div>
              <h3>Neural DSL</h3>
              <pre style={{background: 'var(--ifm-color-emphasis-100)', padding: '1rem', borderRadius: '8px', fontSize: '0.9rem'}}>
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
              <p style={{marginTop: '1rem', color: 'var(--ifm-color-success)'}}>✅ ~15 lines of code</p>
            </div>
            
            <div>
              <h3>TensorFlow/Keras</h3>
              <pre style={{background: 'var(--ifm-color-emphasis-100)', padding: '1rem', borderRadius: '8px', fontSize: '0.9rem'}}>
{`import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), 
        activation='relu',
        input_shape=(28, 28, 1)
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2
)`}
              </pre>
              <p style={{marginTop: '1rem', color: 'var(--ifm-color-warning)'}}>⚠️ ~25 lines of code</p>
            </div>
          </div>
        </div>

        <div style={{textAlign: 'center', padding: '3rem', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', borderRadius: '8px', color: 'white'}}>
          <h2 style={{color: 'white'}}>Ready to Experience the Difference?</h2>
          <p style={{fontSize: '1.1rem', marginBottom: '2rem'}}>
            Try Neural DSL and see how it can accelerate your deep learning workflow
          </p>
          <a
            className="button button--secondary button--lg"
            href="/docs/getting-started/installation"
            style={{marginRight: '1rem'}}>
            Get Started
          </a>
          <a
            className="button button--outline button--secondary button--lg"
            href="/playground">
            Try Playground
          </a>
        </div>
      </div>
    </Layout>
  );
}
