export const formatShape = (shape) => {
  if (!shape) return 'N/A';
  
  if (Array.isArray(shape)) {
    return `[${shape.map(s => s === null || s === undefined ? 'None' : s).join(', ')}]`;
  }
  
  if (typeof shape === 'string') {
    return shape;
  }
  
  return String(shape);
};

export const formatNumber = (num) => {
  if (!num || num === 0) return '0';
  
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
  
  return num.toFixed(0);
};

export const formatMemory = (bytes) => {
  if (!bytes || bytes === 0) return '0 B';
  
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  }
  
  return `${bytes.toFixed(0)} B`;
};

export const parseShape = (shapeStr) => {
  if (Array.isArray(shapeStr)) return shapeStr;
  
  if (typeof shapeStr === 'string') {
    const cleaned = shapeStr.replace(/[\[\]()]/g, '');
    return cleaned.split(',').map(s => {
      const trimmed = s.trim();
      if (trimmed === 'None' || trimmed === 'null') return null;
      return parseInt(trimmed, 10);
    });
  }
  
  return [shapeStr];
};

export const checkShapeMismatch = (outputShape, inputShape) => {
  if (!outputShape || !inputShape) return false;
  
  const out = Array.isArray(outputShape) ? outputShape : parseShape(outputShape);
  const inp = Array.isArray(inputShape) ? inputShape : parseShape(inputShape);
  
  if (out.length !== inp.length) return true;
  
  for (let i = 1; i < out.length; i++) {
    if (out[i] !== inp[i] && out[i] !== null && inp[i] !== null) {
      return true;
    }
  }
  
  return false;
};

export const calculateTensorSize = (shape) => {
  if (!shape) return 0;
  
  const arr = Array.isArray(shape) ? shape : parseShape(shape);
  
  return arr.reduce((acc, val) => {
    if (val === null || val === undefined) return acc;
    return acc * val;
  }, 1);
};

export const getLayerColor = (layerType) => {
  const colorMap = {
    'Conv': '#e74c3c',
    'Dense': '#9b59b6',
    'Pool': '#2ecc71',
    'MaxPool': '#2ecc71',
    'AvgPool': '#27ae60',
    'Flatten': '#f39c12',
    'Dropout': '#e67e22',
    'BatchNorm': '#3498db',
    'Activation': '#1abc9c',
    'Input': '#34495e',
    'Output': '#2c3e50',
    'LSTM': '#8e44ad',
    'GRU': '#9b59b6',
    'Embedding': '#16a085',
    'Reshape': '#d35400',
  };
  
  for (const [key, color] of Object.entries(colorMap)) {
    if (layerType.includes(key)) return color;
  }
  
  return '#95a5a6';
};

export const exportToJson = (shapeHistory) => {
  const data = {
    timestamp: new Date().toISOString(),
    layers: shapeHistory.map((item, index) => ({
      index,
      name: item.layer_name || item.layer,
      inputShape: item.input_shape,
      outputShape: item.output_shape,
      parameters: item.parameters || 0,
      memory: item.memory || 0,
      flops: item.flops || 0,
      error: item.error || false,
      errorMessage: item.error_message || null,
    })),
  };
  
  return JSON.stringify(data, null, 2);
};

export const downloadFile = (content, filename, contentType = 'application/json') => {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
