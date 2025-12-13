import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './DocumentationBrowser.css';

interface DocSection {
  id: string;
  title: string;
  path: string;
  category: string;
}

const DocumentationBrowser: React.FC = () => {
  const [docSections] = useState<DocSection[]>([
    {
      id: 'quickstart',
      title: 'Quick Start Guide',
      path: '/docs/QUICKSTART.md',
      category: 'Getting Started',
    },
    {
      id: 'dsl-syntax',
      title: 'DSL Syntax Reference',
      path: '/docs/DSL_SYNTAX.md',
      category: 'Language',
    },
    {
      id: 'layers',
      title: 'Layer Types',
      path: '/docs/LAYERS.md',
      category: 'Language',
    },
    {
      id: 'debugger',
      title: 'Debugger Features',
      path: '/docs/DEBUGGER_FEATURES.md',
      category: 'Tools',
    },
    {
      id: 'deployment',
      title: 'Deployment Guide',
      path: '/docs/DEPLOYMENT.md',
      category: 'Advanced',
    },
    {
      id: 'integrations',
      title: 'Platform Integrations',
      path: '/docs/INTEGRATION.md',
      category: 'Advanced',
    },
  ]);

  const [selectedDoc, setSelectedDoc] = useState<string>('quickstart');
  const [docContent, setDocContent] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]);

  useEffect(() => {
    loadDocumentation(selectedDoc);
  }, [selectedDoc]);

  const loadDocumentation = async (docId: string) => {
    const doc = docSections.find((d) => d.id === docId);
    if (!doc) return;

    setLoading(true);
    try {
      const response = await axios.get(`/api/docs${doc.path}`);
      setDocContent(response.data);
    } catch (err) {
      console.error('Error loading documentation:', err);
      setDocContent(getPlaceholderContent(docId));
    } finally {
      setLoading(false);
    }
  };

  const getPlaceholderContent = (docId: string): string => {
    const placeholders: { [key: string]: string } = {
      quickstart: `# Quick Start Guide

## Welcome to Neural DSL

Neural DSL is a domain-specific language for defining neural networks with a clean, intuitive syntax.

### Basic Example

\`\`\`neural
network MyFirstNetwork {
    input: (None, 784)
    layers:
        Dense(units=128, activation="relu")
        Dropout(rate=0.5)
        Dense(units=10, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "Adam"
}
\`\`\`

### Key Features

- **Visual Editor**: Design networks with drag-and-drop
- **AI Assistant**: Generate models from natural language
- **Real-time Debugging**: Monitor training with live visualizations
- **Multi-backend**: Deploy to TensorFlow, PyTorch, or ONNX

### Next Steps

1. Try the Quick Start templates
2. Explore the Example Gallery
3. Read the DSL Syntax Reference
4. Watch the Video Tutorials`,

      'dsl-syntax': `# DSL Syntax Reference

## Network Definition

\`\`\`neural
network NetworkName {
    input: (batch_size, ...dimensions)
    layers:
        LayerType(parameters)
        ...
    loss: "loss_function"
    optimizer: "optimizer_name"
}
\`\`\`

## Layer Types

### Dense Layers
- **Dense**: Fully connected layer
- **Dropout**: Regularization layer

### Convolutional Layers
- **Conv2D**: 2D convolution
- **MaxPooling2D**: Max pooling
- **AveragePooling2D**: Average pooling

### Recurrent Layers
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit
- **SimpleRNN**: Basic RNN

### Other Layers
- **Embedding**: Word embeddings
- **BatchNormalization**: Batch normalization
- **Flatten**: Flatten layer`,

      layers: `# Layer Types Reference

## Dense Layers

### Dense
Fully connected neural network layer.

**Parameters:**
- units: Number of neurons
- activation: Activation function

### Dropout
Applies dropout regularization.

**Parameters:**
- rate: Dropout rate (0-1)

## Convolutional Layers

### Conv2D
2D convolutional layer for images.

**Parameters:**
- filters: Number of filters
- kernel_size: Size of convolution kernel
- activation: Activation function
- padding: 'valid' or 'same'

### MaxPooling2D
Max pooling operation.

**Parameters:**
- pool_size: Size of pooling window`,

      debugger: `# Debugger Features

## Real-time Training Monitoring

- Live loss and accuracy plots
- Layer activation visualizations
- Gradient flow analysis
- Resource usage tracking

## Breakpoints

Set breakpoints at specific:
- Epochs
- Batch numbers
- Loss thresholds

## Inspection Tools

- Weight visualization
- Activation maps
- Gradient histograms
- Learning rate schedules`,

      deployment: `# Deployment Guide

## Export Options

### TensorFlow
\`\`\`bash
neural compile model.neural --backend tensorflow --output model.py
\`\`\`

### PyTorch
\`\`\`bash
neural compile model.neural --backend pytorch --output model.py
\`\`\`

### ONNX
\`\`\`bash
neural compile model.neural --backend onnx --output model.onnx
\`\`\`

## Cloud Deployment

Supports deployment to:
- AWS SageMaker
- Google Vertex AI
- Azure ML
- Databricks`,

      integrations: `# Platform Integrations

## Supported Platforms

### AWS SageMaker
Deploy and train on AWS infrastructure.

### Google Vertex AI
Use Google Cloud's ML platform.

### Azure ML
Integrate with Microsoft Azure.

### Databricks
Run on Databricks clusters.

### Paperspace
GPU cloud computing.

### Run:AI
Kubernetes-based ML operations.`,
    };

    return placeholders[docId] || '# Documentation\n\nContent not available.';
  };

  const handleSearch = () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    const results = docSections.filter(
      (doc) =>
        doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        doc.category.toLowerCase().includes(searchQuery.toLowerCase())
    );

    setSearchResults(results.map((r) => r.id));
  };

  const categories = [...new Set(docSections.map((doc) => doc.category))];

  return (
    <div className="documentation-browser">
      <div className="docs-header">
        <h2>Documentation</h2>
        <p>Learn how to use Neural DSL and the Aquarium IDE</p>
      </div>

      <div className="docs-search">
        <input
          type="text"
          placeholder="Search documentation..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          className="docs-search-input"
        />
        <button className="search-btn" onClick={handleSearch}>
          🔍
        </button>
      </div>

      <div className="docs-content-wrapper">
        <div className="docs-sidebar">
          {categories.map((category) => (
            <div key={category} className="docs-category">
              <h4>{category}</h4>
              <ul>
                {docSections
                  .filter((doc) => doc.category === category)
                  .map((doc) => (
                    <li
                      key={doc.id}
                      className={`doc-item ${selectedDoc === doc.id ? 'active' : ''} ${
                        searchResults.length > 0 && !searchResults.includes(doc.id)
                          ? 'dimmed'
                          : ''
                      }`}
                      onClick={() => setSelectedDoc(doc.id)}
                    >
                      {doc.title}
                    </li>
                  ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="docs-main-content">
          {loading ? (
            <div className="docs-loading">Loading documentation...</div>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown>{docContent}</ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentationBrowser;
