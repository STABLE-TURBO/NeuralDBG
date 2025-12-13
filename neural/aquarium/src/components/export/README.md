# Model Export and Deployment Panel

The Export Panel provides a comprehensive interface for exporting Neural DSL models to various deployment formats and deploying them to different serving platforms.

## Features

### Export Formats

- **ONNX**: Universal cross-platform format for maximum compatibility
- **TensorFlow Lite**: Optimized for mobile and edge devices
- **TorchScript**: Production-ready PyTorch format
- **SavedModel**: TensorFlow Serving compatible format

### Optimization Options

#### General Optimizations
- Constant folding
- Layer fusion
- Dead code elimination
- Graph optimization passes

#### Quantization
- **INT8**: Full integer quantization for maximum compression
- **Float16**: Half-precision for GPU acceleration
- **Dynamic Range**: Weight quantization without dataset

#### Pruning
- Adjustable sparsity levels (0-90%)
- Remove less important weights
- Reduce model size with minimal accuracy loss

### Deployment Targets

- **Cloud**: AWS, GCP, Azure deployments
- **Edge**: IoT and embedded systems
- **Mobile**: iOS and Android devices
- **Server**: On-premise server deployments

### Serving Platforms

- **TorchServe**: PyTorch model serving
- **TensorFlow Serving**: TensorFlow model serving
- **ONNX Runtime**: Cross-platform ONNX inference
- **NVIDIA Triton**: Multi-framework inference server

## Usage

### Basic Export

```typescript
import { ExportPanel } from './components/export';

<ExportPanel
  modelData={modelData}
  backend="tensorflow"
  onExportComplete={(result) => {
    console.log('Export complete:', result);
  }}
/>
```

### With Deployment Callback

```typescript
<ExportPanel
  modelData={modelData}
  backend="pytorch"
  onExportComplete={(result) => {
    console.log('Model exported to:', result.exportPath);
  }}
  onDeploymentComplete={(result) => {
    console.log('Deployment endpoint:', result.endpoint);
  }}
/>
```

## API Integration

The panel integrates with the backend API through the `ExportService`:

### Export Endpoint
```
POST /api/export/model
```

### Deploy Endpoint
```
POST /api/deployment/deploy
```

### Serving Config Generation
```
POST /api/deployment/serving-config
```

## Components

### ExportPanel
Main container component that orchestrates the export and deployment workflow.

### ExportFormatSelector
Radio button selector for choosing the export format based on backend compatibility.

### OptimizationOptions
Collapsible sections for configuring optimization, quantization, and pruning options.

### DeploymentTargetSelector
Target and platform selection with visual cards and compatibility indicators.

### ServingConfigGenerator
Generates serving configurations and deployment scripts for selected platform.

### ExportProgress
Shows export progress, success/error states, and exported model details.

## Backend Integration

The panel connects to:

- `neural.code_generation.export.ModelExporter`: Core export functionality
- `neural.mlops.deployment.DeploymentManager`: Deployment management
- Existing deployment features from `neural/mlops/`

## Configuration

### Export Options
```typescript
{
  format: 'onnx' | 'tflite' | 'torchscript' | 'savedmodel',
  outputPath: string,
  backend: 'tensorflow' | 'pytorch',
  optimize: boolean,
  quantization: {
    enabled: boolean,
    type: 'none' | 'int8' | 'float16' | 'dynamic'
  },
  pruning: {
    enabled: boolean,
    sparsity: number  // 0.0 to 1.0
  }
}
```

### Deployment Configuration
```typescript
{
  target: 'cloud' | 'edge' | 'mobile' | 'server',
  servingPlatform: 'torchserve' | 'tfserving' | 'onnxruntime' | 'triton',
  modelName: string,
  version: string,
  resources: {
    gpuEnabled: boolean,
    replicas: number,
    batchSize: number,
    maxBatchDelay: number
  },
  networking: {
    port: number,
    enableMetrics: boolean,
    enableHealthCheck: boolean
  }
}
```

## Styling

All components follow the Aquarium dark theme with VS Code-inspired styling:

- Background: `#1e1e1e`
- Panels: `#252525`
- Borders: `#333`
- Accent: `#007acc`
- Success: `#4caf50`
- Error: `#d32f2f`
- Warning: `#ff9800`

## Testing

The panel validates:
- Format compatibility with backend
- Required output paths
- Quantization settings
- Pruning parameters
- Export success before deployment

## Future Enhancements

- [ ] Cloud provider direct integration (AWS SageMaker, GCP Vertex AI, Azure ML)
- [ ] A/B testing deployment configuration
- [ ] Model versioning and rollback
- [ ] Performance benchmarking
- [ ] Automated optimization tuning
- [ ] Multi-target batch export
- [ ] Deployment status monitoring
- [ ] Cost estimation for cloud deployments
