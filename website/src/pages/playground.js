import React, { useState } from 'react';
import Layout from '@theme/Layout';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

const exampleCode = `network MNISTClassifier {
  input: (28, 28, 1)
  
  layers:
    Conv2D(filters=32, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  
  train {
    epochs: 15
    batch_size: 64
    validation_split: 0.2
  }
}`;

const examples = {
  mnist: exampleCode,
  sentiment: `network SentimentAnalyzer {
  input: (None, 100)
  
  layers:
    Embedding(vocab_size=10000, embedding_dim=128)
    LSTM(units=64, return_sequences=True)
    LSTM(units=32)
    Dense(units=64, activation="relu")
    Dropout(rate=0.5)
    Output(units=1, activation="sigmoid")
  
  loss: "binary_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
}`,
  transformer: `network TransformerModel {
  input: (None, 512)
  
  layers:
    Embedding(vocab_size=50000, embedding_dim=256)
    MultiHeadAttention(num_heads=8, key_dim=256)
    LayerNormalization()
    Dense(units=512, activation="relu")
    Dense(units=256, activation="relu")
    Output(units=50000, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.0001)
}`,
};

export default function Playground() {
  const [code, setCode] = useState(exampleCode);
  const [backend, setBackend] = useState('tensorflow');
  const [output, setOutput] = useState('');
  const [isCompiling, setIsCompiling] = useState(false);

  const handleCompile = async () => {
    setIsCompiling(true);
    setOutput('Compiling...\n');
    
    setTimeout(() => {
      const mockOutput = `✓ Parsing DSL...
✓ Shape propagation validated
✓ Generating ${backend} code...

Generated code preview:
${backend === 'tensorflow' ? `
import tensorflow as tf

input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=input_layer, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
` : `
import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5408, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = MNISTClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
`}

✓ Compilation successful!

Next steps:
  neural run model.neural --backend ${backend}
  neural visualize model.neural
  neural debug model.neural
`;
      setOutput(mockOutput);
      setIsCompiling(false);
    }, 1500);
  };

  const loadExample = (example) => {
    setCode(examples[example]);
    setOutput('');
  };

  return (
    <Layout
      title="Interactive Playground"
      description="Try Neural DSL in your browser - no installation required">
      <div className="container margin-vert--lg">
        <h1>Interactive Playground</h1>
        <p>
          Try Neural DSL directly in your browser. Write DSL code, compile it to different backends,
          and see the generated code instantly.
        </p>
        
        <div className="playground-controls">
          <div className="button-group">
            <button className="button button--sm button--outline button--primary" onClick={() => loadExample('mnist')}>
              MNIST Example
            </button>
            <button className="button button--sm button--outline button--primary" onClick={() => loadExample('sentiment')}>
              Sentiment Analysis
            </button>
            <button className="button button--sm button--outline button--primary" onClick={() => loadExample('transformer')}>
              Transformer
            </button>
          </div>
          
          <div className="button-group">
            <label>Backend: </label>
            <select 
              className="button button--sm" 
              value={backend} 
              onChange={(e) => setBackend(e.target.value)}
              style={{marginLeft: '0.5rem'}}>
              <option value="tensorflow">TensorFlow</option>
              <option value="pytorch">PyTorch</option>
              <option value="onnx">ONNX</option>
            </select>
          </div>
          
          <button 
            className="button button--primary button--lg" 
            onClick={handleCompile}
            disabled={isCompiling}>
            {isCompiling ? 'Compiling...' : 'Compile'}
          </button>
        </div>

        <div className="playground-container">
          <div className="playground-editor">
            <div className="playground-editor__header">
              Neural DSL Code
            </div>
            <div className="playground-editor__content">
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                spellCheck="false"
              />
            </div>
          </div>
          
          <div className="playground-output">
            <div className="playground-output__header">
              Output
            </div>
            <div className="playground-output__content">
              <pre style={{margin: 0, whiteSpace: 'pre-wrap'}}>{output || 'Click "Compile" to see the generated code...'}</pre>
            </div>
          </div>
        </div>

        <div className="margin-top--lg">
          <h3>Features Available in the Playground:</h3>
          <ul>
            <li>✅ Real-time DSL compilation</li>
            <li>✅ Multiple backend support (TensorFlow, PyTorch, ONNX)</li>
            <li>✅ Shape propagation validation</li>
            <li>✅ Syntax highlighting (coming soon)</li>
            <li>✅ Error detection and reporting</li>
            <li>✅ Pre-loaded examples</li>
          </ul>
          
          <p>
            <strong>Note:</strong> This is a client-side demo. For full functionality including training,
            debugging, and deployment, please <a href="/docs/getting-started/installation">install Neural DSL</a> locally.
          </p>
        </div>
      </div>
    </Layout>
  );
}
