import { LayerCategory } from '../types';

export interface LayerDefinition {
  type: string;
  category: LayerCategory;
  defaultParams: Record<string, any>;
  color: string;
  icon: string;
  description: string;
}

export const LAYER_DEFINITIONS: Record<string, LayerDefinition> = {
  Conv1D: {
    type: 'Conv1D',
    category: 'Convolutional',
    defaultParams: { filters: 32, kernel_size: 3, activation: 'relu', padding: 'valid' },
    color: '#FF6B6B',
    icon: '⊞',
    description: '1D Convolution layer'
  },
  Conv2D: {
    type: 'Conv2D',
    category: 'Convolutional',
    defaultParams: { filters: 32, kernel_size: [3, 3], activation: 'relu', padding: 'valid' },
    color: '#FF6B6B',
    icon: '⊞',
    description: '2D Convolution layer'
  },
  Conv3D: {
    type: 'Conv3D',
    category: 'Convolutional',
    defaultParams: { filters: 32, kernel_size: [3, 3, 3], activation: 'relu', padding: 'valid' },
    color: '#FF6B6B',
    icon: '⊞',
    description: '3D Convolution layer'
  },
  SeparableConv2D: {
    type: 'SeparableConv2D',
    category: 'Convolutional',
    defaultParams: { filters: 32, kernel_size: [3, 3], activation: 'relu' },
    color: '#FF6B6B',
    icon: '⊞',
    description: 'Separable 2D Convolution'
  },
  MaxPooling2D: {
    type: 'MaxPooling2D',
    category: 'Pooling',
    defaultParams: { pool_size: [2, 2], strides: null, padding: 'valid' },
    color: '#4ECDC4',
    icon: '▽',
    description: '2D Max Pooling'
  },
  AveragePooling2D: {
    type: 'AveragePooling2D',
    category: 'Pooling',
    defaultParams: { pool_size: [2, 2], strides: null, padding: 'valid' },
    color: '#4ECDC4',
    icon: '▽',
    description: '2D Average Pooling'
  },
  GlobalMaxPooling2D: {
    type: 'GlobalMaxPooling2D',
    category: 'Pooling',
    defaultParams: {},
    color: '#4ECDC4',
    icon: '▽',
    description: 'Global 2D Max Pooling'
  },
  GlobalAveragePooling2D: {
    type: 'GlobalAveragePooling2D',
    category: 'Pooling',
    defaultParams: {},
    color: '#4ECDC4',
    icon: '▽',
    description: 'Global 2D Average Pooling'
  },
  Dense: {
    type: 'Dense',
    category: 'Core',
    defaultParams: { units: 128, activation: 'relu' },
    color: '#95E1D3',
    icon: '●',
    description: 'Fully connected layer'
  },
  Flatten: {
    type: 'Flatten',
    category: 'Core',
    defaultParams: {},
    color: '#95E1D3',
    icon: '▭',
    description: 'Flatten input'
  },
  Dropout: {
    type: 'Dropout',
    category: 'Regularization',
    defaultParams: { rate: 0.5 },
    color: '#AA96DA',
    icon: '⊗',
    description: 'Dropout layer'
  },
  BatchNormalization: {
    type: 'BatchNormalization',
    category: 'Normalization',
    defaultParams: { axis: -1, momentum: 0.99, epsilon: 0.001 },
    color: '#F38181',
    icon: 'N',
    description: 'Batch Normalization'
  },
  LayerNormalization: {
    type: 'LayerNormalization',
    category: 'Normalization',
    defaultParams: { axis: -1, epsilon: 0.001 },
    color: '#F38181',
    icon: 'N',
    description: 'Layer Normalization'
  },
  LSTM: {
    type: 'LSTM',
    category: 'Recurrent',
    defaultParams: { units: 128, activation: 'tanh', recurrent_activation: 'sigmoid', return_sequences: false },
    color: '#FCBAD3',
    icon: '↻',
    description: 'Long Short-Term Memory'
  },
  GRU: {
    type: 'GRU',
    category: 'Recurrent',
    defaultParams: { units: 128, activation: 'tanh', recurrent_activation: 'sigmoid', return_sequences: false },
    color: '#FCBAD3',
    icon: '↻',
    description: 'Gated Recurrent Unit'
  },
  SimpleRNN: {
    type: 'SimpleRNN',
    category: 'Recurrent',
    defaultParams: { units: 128, activation: 'tanh', return_sequences: false },
    color: '#FCBAD3',
    icon: '↻',
    description: 'Simple RNN'
  },
  MultiHeadAttention: {
    type: 'MultiHeadAttention',
    category: 'Attention',
    defaultParams: { num_heads: 8, key_dim: 64 },
    color: '#FFD93D',
    icon: '👁',
    description: 'Multi-Head Attention'
  },
  Attention: {
    type: 'Attention',
    category: 'Attention',
    defaultParams: {},
    color: '#FFD93D',
    icon: '👁',
    description: 'Attention mechanism'
  },
  Embedding: {
    type: 'Embedding',
    category: 'Embedding',
    defaultParams: { input_dim: 1000, output_dim: 64 },
    color: '#6BCB77',
    icon: '≡',
    description: 'Embedding layer'
  },
  ReLU: {
    type: 'ReLU',
    category: 'Activation',
    defaultParams: { max_value: null, negative_slope: 0.0, threshold: 0.0 },
    color: '#A8D8EA',
    icon: '∿',
    description: 'ReLU activation'
  },
  Softmax: {
    type: 'Softmax',
    category: 'Activation',
    defaultParams: { axis: -1 },
    color: '#A8D8EA',
    icon: '∿',
    description: 'Softmax activation'
  }
};

export const LAYER_CATEGORIES: LayerCategory[] = [
  'Convolutional',
  'Pooling',
  'Core',
  'Recurrent',
  'Attention',
  'Normalization',
  'Regularization',
  'Activation',
  'Embedding'
];

export function getLayersByCategory(category: LayerCategory): LayerDefinition[] {
  return Object.values(LAYER_DEFINITIONS).filter(layer => layer.category === category);
}

export function getLayerDefinition(type: string): LayerDefinition | undefined {
  return LAYER_DEFINITIONS[type];
}
