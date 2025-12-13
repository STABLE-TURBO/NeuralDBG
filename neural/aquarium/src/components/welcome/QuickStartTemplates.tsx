import React from 'react';
import './QuickStartTemplates.css';

interface Template {
  id: string;
  title: string;
  description: string;
  category: string;
  icon: string;
  dslCode: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

interface QuickStartTemplatesProps {
  onLoadTemplate: (template: string) => void;
}

const templates: Template[] = [
  {
    id: 'image-classification',
    title: 'Image Classification',
    description: 'CNN for classifying images into categories (MNIST, CIFAR-10)',
    category: 'Computer Vision',
    icon: '🖼️',
    difficulty: 'beginner',
    dslCode: `network ImageClassifier {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        MaxPooling2D(pool_size=(2, 2))
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu")
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation="relu")
        Dropout(rate=0.5)
        Dense(units=10, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "Adam"
}`,
  },
  {
    id: 'text-classification',
    title: 'Text Classification',
    description: 'LSTM for sentiment analysis and text categorization',
    category: 'NLP',
    icon: '📝',
    difficulty: 'beginner',
    dslCode: `network TextClassifier {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=128)
        LSTM(units=64, return_sequences=true)
        LSTM(units=64)
        Dense(units=64, activation="relu")
        Dropout(rate=0.5)
        Dense(units=1, activation="sigmoid")
    loss: "binary_crossentropy"
    optimizer: "Adam"
}`,
  },
  {
    id: 'time-series',
    title: 'Time Series Forecasting',
    description: 'LSTM network for predicting future values in time series data',
    category: 'Time Series',
    icon: '📈',
    difficulty: 'intermediate',
    dslCode: `network TimeSeriesForecaster {
    input: (None, 50, 1)
    layers:
        LSTM(units=128, return_sequences=true)
        Dropout(rate=0.2)
        LSTM(units=64, return_sequences=true)
        Dropout(rate=0.2)
        LSTM(units=32)
        Dense(units=16, activation="relu")
        Dense(units=1)
    loss: "mse"
    optimizer: "Adam"
}`,
  },
  {
    id: 'autoencoder',
    title: 'Autoencoder',
    description: 'Deep autoencoder for dimensionality reduction and feature learning',
    category: 'Unsupervised',
    icon: '🔄',
    difficulty: 'intermediate',
    dslCode: `network Autoencoder {
    input: (None, 784)
    layers:
        Dense(units=256, activation="relu")
        Dense(units=128, activation="relu")
        Dense(units=64, activation="relu")
        Dense(units=128, activation="relu")
        Dense(units=256, activation="relu")
        Dense(units=784, activation="sigmoid")
    loss: "mse"
    optimizer: "Adam"
}`,
  },
  {
    id: 'seq2seq',
    title: 'Sequence-to-Sequence',
    description: 'Encoder-decoder architecture for machine translation',
    category: 'NLP',
    icon: '🔀',
    difficulty: 'advanced',
    dslCode: `network Seq2Seq {
    input: (None, 100)
    layers:
        Embedding(input_dim=10000, output_dim=256)
        LSTM(units=256, return_sequences=true)
        LSTM(units=256)
        RepeatVector(n=100)
        LSTM(units=256, return_sequences=true)
        LSTM(units=256, return_sequences=true)
        TimeDistributed(Dense(units=10000, activation="softmax"))
    loss: "categorical_crossentropy"
    optimizer: "Adam"
}`,
  },
  {
    id: 'gan',
    title: 'Generative Adversarial Network',
    description: 'Generator network for creating synthetic data',
    category: 'Generative',
    icon: '🎨',
    difficulty: 'advanced',
    dslCode: `network Generator {
    input: (None, 100)
    layers:
        Dense(units=256, activation="relu")
        BatchNormalization()
        Dense(units=512, activation="relu")
        BatchNormalization()
        Dense(units=1024, activation="relu")
        BatchNormalization()
        Dense(units=784, activation="tanh")
    loss: "binary_crossentropy"
    optimizer: "Adam"
}`,
  },
];

const QuickStartTemplates: React.FC<QuickStartTemplatesProps> = ({ onLoadTemplate }) => {
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner':
        return '#4caf50';
      case 'intermediate':
        return '#ff9800';
      case 'advanced':
        return '#f44336';
      default:
        return '#999';
    }
  };

  return (
    <div className="quickstart-templates">
      <div className="templates-header">
        <h2>Quick Start Templates</h2>
        <p>Choose a template to get started quickly with common neural network architectures</p>
      </div>

      <div className="templates-grid">
        {templates.map((template) => (
          <div key={template.id} className="template-card">
            <div className="template-icon">{template.icon}</div>
            <div className="template-content">
              <div className="template-header-section">
                <h3>{template.title}</h3>
                <span
                  className="difficulty-badge"
                  style={{ backgroundColor: getDifficultyColor(template.difficulty) }}
                >
                  {template.difficulty}
                </span>
              </div>
              <p className="template-category">{template.category}</p>
              <p className="template-description">{template.description}</p>
            </div>
            <div className="template-actions">
              <button
                className="load-template-btn"
                onClick={() => onLoadTemplate(template.dslCode)}
              >
                Load Template
              </button>
              <button
                className="preview-template-btn"
                onClick={() => {
                  const codePreview = prompt('Template Code:', template.dslCode);
                }}
              >
                Preview
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default QuickStartTemplates;
