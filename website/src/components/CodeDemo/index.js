import React from 'react';
import styles from './styles.module.css';

export default function CodeDemo() {
  return (
    <section className="code-demo">
      <div className="container">
        <h2 className="text--center margin-bottom--lg">
          One Language, Multiple Frameworks
        </h2>
        <div className="code-demo__container">
          <div>
            <h3>Write Once</h3>
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
  
  loss: "categorical_crossentropy"
  optimizer: Adam(lr=0.001)
}`}
            </pre>
          </div>
          <div>
            <h3>Deploy Everywhere</h3>
            <div className={styles.deployOptions}>
              <div className={styles.deployOption}>
                <h4>🔷 TensorFlow</h4>
                <code>neural compile model.neural --backend tensorflow</code>
              </div>
              <div className={styles.deployOption}>
                <h4>🔥 PyTorch</h4>
                <code>neural compile model.neural --backend pytorch</code>
              </div>
              <div className={styles.deployOption}>
                <h4>⚡ ONNX</h4>
                <code>neural export model.neural --format onnx</code>
              </div>
              <div className={styles.deployOption}>
                <h4>📱 TensorFlow Lite</h4>
                <code>neural export model.neural --format tflite</code>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
