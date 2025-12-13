export interface LayerParameter {
  name: string;
  value: string | number | boolean | null;
  type: 'number' | 'string' | 'boolean' | 'tuple' | 'enum';
  options?: string[];
}

export interface LayerConfig {
  type: string;
  category: LayerCategory;
  parameters: Record<string, LayerParameter>;
  icon?: string;
  color?: string;
}

export type LayerCategory = 
  | 'Convolutional' 
  | 'Pooling' 
  | 'Core' 
  | 'Recurrent' 
  | 'Attention' 
  | 'Normalization'
  | 'Regularization'
  | 'Activation'
  | 'Embedding';

export interface LayerNodeData {
  label: string;
  layerType: string;
  category: LayerCategory;
  parameters: Record<string, any>;
  outputShape?: string;
  parameterCount?: number;
}

export interface NetworkModel {
  name: string;
  inputShape: string;
  layers: LayerConfig[];
  loss: string;
  optimizer: {
    type: string;
    params: Record<string, any>;
  };
}

export interface ConnectionValidationResult {
  valid: boolean;
  message?: string;
}
