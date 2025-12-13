import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ExampleGallery.css';

interface Example {
  name: string;
  path: string;
  description: string;
  category: string;
  tags: string[];
  complexity: string;
}

interface ExampleGalleryProps {
  onLoadExample: (code: string) => void;
}

const ExampleGallery: React.FC<ExampleGalleryProps> = ({ onLoadExample }) => {
  const [examples, setExamples] = useState<Example[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadExamples();
  }, []);

  const loadExamples = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/examples/list');
      setExamples(response.data.examples || []);
      setError(null);
    } catch (err) {
      console.error('Error loading examples:', err);
      setError('Failed to load examples. Using built-in examples.');
      setExamples(getBuiltInExamples());
    } finally {
      setLoading(false);
    }
  };

  const getBuiltInExamples = (): Example[] => {
    return [
      {
        name: 'MNIST CNN',
        path: 'examples/mnist_cnn.neural',
        description: 'Convolutional Neural Network for MNIST digit classification',
        category: 'Computer Vision',
        tags: ['cnn', 'classification', 'mnist'],
        complexity: 'Beginner',
      },
      {
        name: 'LSTM Text Classifier',
        path: 'examples/lstm_text.neural',
        description: 'LSTM network for text classification and sentiment analysis',
        category: 'NLP',
        tags: ['lstm', 'nlp', 'text', 'classification'],
        complexity: 'Beginner',
      },
      {
        name: 'ResNet Image Classifier',
        path: 'examples/resnet.neural',
        description: 'Deep residual network for advanced image classification',
        category: 'Computer Vision',
        tags: ['resnet', 'cnn', 'deep-learning'],
        complexity: 'Advanced',
      },
      {
        name: 'Transformer Model',
        path: 'examples/transformer.neural',
        description: 'Attention-based transformer for sequence modeling',
        category: 'NLP',
        tags: ['transformer', 'attention', 'nlp'],
        complexity: 'Advanced',
      },
      {
        name: 'VAE',
        path: 'examples/vae.neural',
        description: 'Variational Autoencoder for generative modeling',
        category: 'Generative',
        tags: ['vae', 'autoencoder', 'generative'],
        complexity: 'Intermediate',
      },
    ];
  };

  const loadExampleCode = async (path: string) => {
    try {
      const response = await axios.get(`/api/examples/load?path=${encodeURIComponent(path)}`);
      onLoadExample(response.data.code);
    } catch (err) {
      console.error('Error loading example code:', err);
      alert('Failed to load example code. Please try again.');
    }
  };

  const categories = ['all', ...new Set(examples.map((ex) => ex.category))];

  const filteredExamples = examples.filter((example) => {
    const matchesCategory = selectedCategory === 'all' || example.category === selectedCategory;
    const matchesSearch =
      searchQuery === '' ||
      example.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      example.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      example.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  if (loading) {
    return (
      <div className="example-gallery loading">
        <div className="loader">Loading examples...</div>
      </div>
    );
  }

  return (
    <div className="example-gallery">
      <div className="gallery-header">
        <h2>Example Gallery</h2>
        <p>Browse and load example neural network models from the repository</p>
      </div>

      <div className="gallery-controls">
        <div className="search-box">
          <input
            type="text"
            placeholder="Search examples..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>

        <div className="category-filters">
          {categories.map((category) => (
            <button
              key={category}
              className={`category-filter ${selectedCategory === category ? 'active' : ''}`}
              onClick={() => setSelectedCategory(category)}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      <div className="examples-grid">
        {filteredExamples.map((example) => (
          <div key={example.path} className="example-card">
            <div className="example-header">
              <h3>{example.name}</h3>
              <span className="complexity-badge">{example.complexity}</span>
            </div>
            <p className="example-category">{example.category}</p>
            <p className="example-description">{example.description}</p>
            <div className="example-tags">
              {example.tags.map((tag) => (
                <span key={tag} className="tag">
                  {tag}
                </span>
              ))}
            </div>
            <button
              className="load-example-btn"
              onClick={() => loadExampleCode(example.path)}
            >
              Load Example
            </button>
          </div>
        ))}
      </div>

      {filteredExamples.length === 0 && (
        <div className="no-results">
          <p>No examples found matching your criteria.</p>
        </div>
      )}
    </div>
  );
};

export default ExampleGallery;
