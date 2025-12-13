# Export and Deployment Panel Integration Guide

This document describes how the Export and Deployment Panel integrates with Neural DSL's existing deployment infrastructure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Aquarium Export Panel                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ExportPanel.tsx - Main orchestration component      │   │
│  │  ├─ ExportFormatSelector - Format selection UI      │   │
│  │  ├─ OptimizationOptions - Optimization config UI    │   │
│  │  ├─ DeploymentTargetSelector - Target selection UI  │   │
│  │  ├─ ServingConfigGenerator - Config generation UI   │   │
│  │  └─ ExportProgress - Status and results display     │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │ TypeScript/React
                        │
                    ┌───▼────┐
                    │ Axios  │
                    │  HTTP  │
                    └───┬────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Flask API (neural/aquarium/api)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  export_api.py - Export/deployment endpoints         │   │
│  │  POST /api/export/model                              │   │
│  │  POST /api/deployment/deploy                         │   │
│  │  POST /api/deployment/serving-config                 │   │
│  │  GET  /api/deployment/<id>/status                    │   │
│  │  GET  /api/deployment/list                           │   │
│  └──────────────────┬───────────────────────────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                             │
┌───────▼────────────┐    ┌──────────▼──────────────┐
│ neural/code_gen/   │    │  neural/mlops/          │
│     export.py      │    │    deployment.py        │
│                    │    │                         │
│ ModelExporter      │    │  DeploymentManager      │
│ ├─ export_onnx     │    │  ├─ create_deployment  │
│ ├─ export_tflite   │    │  ├─ start_deployment   │
│ ├─ export_torch... │    │  ├─ shadow_deploy      │
│ ├─ export_saved... │    │  ├─ rollback_...       │
│ └─ create_*_config │    │  └─ check_health       │
└────────────────────┘    └─────────────────────────┘
```

## Component Responsibilities

### Frontend (React/TypeScript)

**ExportPanel.tsx**
- Main state management for export and deployment workflows
- Coordinates sub-components
- Handles validation and API communication
- Manages export → deployment workflow

**ExportService.ts**
- HTTP client wrapper for export/deployment APIs
- Handles request/response transformation
- Provides validation for export options
- Error handling and retry logic

**Type Definitions (types/export.ts)**
- TypeScript interfaces for all export/deployment configs
- Ensures type safety across components
- Shared types between frontend and API contracts

### Backend (Python/Flask)

**export_api.py**
- REST API endpoints for export and deployment
- Request validation and error handling
- Integration with ModelExporter and DeploymentManager
- Response formatting

**Integration Points:**

1. **neural.code_generation.export.ModelExporter**
   - Core export functionality
   - Format conversion (ONNX, TFLite, TorchScript, SavedModel)
   - Optimization passes
   - Serving config generation

2. **neural.mlops.deployment.DeploymentManager**
   - Deployment lifecycle management
   - Strategy selection (direct, blue-green, canary, shadow, rolling)
   - Health monitoring
   - Rollback capabilities
   - Deployment tracking

## API Endpoints

### Export Model
```http
POST /api/export/model
Content-Type: application/json

{
  "model_data": { ... },
  "options": {
    "format": "onnx",
    "outputPath": "./model",
    "backend": "tensorflow",
    "optimize": true,
    "quantization": {
      "enabled": true,
      "type": "int8"
    },
    "pruning": {
      "enabled": false,
      "sparsity": 0
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "exportPath": "./model.onnx",
  "format": "onnx",
  "size": 12345678,
  "message": "Model successfully exported to ONNX format"
}
```

### Deploy Model
```http
POST /api/deployment/deploy
Content-Type: application/json

{
  "export_path": "./model.onnx",
  "config": {
    "target": "cloud",
    "servingPlatform": "torchserve",
    "modelName": "neural_model",
    "version": "1.0",
    "resources": {
      "gpuEnabled": false,
      "replicas": 1,
      "batchSize": 1,
      "maxBatchDelay": 100
    },
    "networking": {
      "port": 8080,
      "enableMetrics": true,
      "enableHealthCheck": true
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "deploymentId": "deploy_20231213_142530_neural_model_1.0",
  "endpoint": "http://localhost:8080/predictions/neural_model",
  "message": "Model deployed successfully with direct strategy"
}
```

### Generate Serving Config
```http
POST /api/deployment/serving-config
Content-Type: application/json

{
  "model_path": "./model.onnx",
  "platform": "torchserve",
  "model_name": "neural_model",
  "config": { ... }
}
```

**Response:**
```json
{
  "configPath": "./serving_config/config.properties",
  "modelStorePath": "./serving_config/model-store",
  "scripts": [
    "./serving_config/start_torchserve.sh",
    "./serving_config/stop_torchserve.sh",
    "./serving_config/test_inference.py"
  ],
  "instructions": [
    "Install TorchServe: pip install torchserve torch-model-archiver",
    "Archive your model and place it in ./serving_config/model-store",
    "Run ./start_torchserve.sh to start the server",
    "Test inference with python test_inference.py",
    "Monitor logs at logs/model_metrics.log"
  ]
}
```

## Integration with Existing Features

### 1. MLOps Deployment Module
The panel uses `neural.mlops.deployment.DeploymentManager` for:
- Creating deployments with different strategies
- Tracking deployment lifecycle
- Health monitoring and rollback
- Shadow deployments for testing

### 2. Code Generation Export
The panel uses `neural.code_generation.export.ModelExporter` for:
- Converting models to various formats
- Applying optimizations
- Quantization and pruning
- Generating serving configurations

### 3. Integration Connectors
Future integration with `neural.integrations.*`:
- AWS SageMaker deployment
- GCP Vertex AI deployment
- Azure ML deployment
- Databricks deployment
- Other cloud platforms

## Deployment Strategies

The panel supports all strategies from `DeploymentManager`:

1. **Direct**: Immediate replacement
2. **Blue-Green**: Zero-downtime deployment with instant rollback
3. **Canary**: Gradual rollout with traffic splitting
4. **Shadow**: Risk-free testing with traffic mirroring
5. **Rolling**: Progressive replica replacement

## Optimization Features

### Quantization
- **INT8**: Maximum compression, requires calibration dataset
- **Float16**: GPU-optimized half precision
- **Dynamic**: Automatic weight quantization

### Pruning
- Magnitude-based pruning
- Structured and unstructured pruning
- Configurable sparsity levels (0-90%)

### General Optimizations
- Constant folding
- Layer fusion
- Dead code elimination
- Graph optimization

## Serving Platforms

### TorchServe
- PyTorch model serving
- Multi-model serving
- A/B testing support
- Metrics and logging

### TensorFlow Serving
- REST and gRPC APIs
- Model versioning
- Batching optimization
- GPU support

### ONNX Runtime
- Cross-platform inference
- Multiple backends (CPU, CUDA, TensorRT)
- Mobile and edge deployment
- WebAssembly support

### NVIDIA Triton
- Multi-framework support
- Dynamic batching
- Model ensemble
- GPU optimization

## Usage Example

```typescript
import { ExportPanel } from './components/export';

function MyApp() {
  const handleExportComplete = (result) => {
    console.log('Export successful:', result.exportPath);
  };

  const handleDeploymentComplete = (result) => {
    console.log('Deployed at:', result.endpoint);
  };

  return (
    <ExportPanel
      modelData={myModelData}
      backend="tensorflow"
      onExportComplete={handleExportComplete}
      onDeploymentComplete={handleDeploymentComplete}
    />
  );
}
```

## Testing

### Frontend Testing
```bash
cd neural/aquarium
npm test -- --testPathPattern=export
```

### Backend Testing
```bash
pytest tests/test_export_api.py -v
```

### Integration Testing
```bash
pytest tests/integration/test_export_deployment.py -v
```

## Future Enhancements

1. **Cloud Provider Integration**
   - Direct deployment to AWS, GCP, Azure
   - Cost estimation
   - Resource provisioning

2. **Advanced Features**
   - A/B testing configuration
   - Model versioning UI
   - Performance benchmarking
   - Deployment monitoring dashboard

3. **Optimization Tuning**
   - Automated hyperparameter search
   - Accuracy vs. size trade-off analysis
   - Platform-specific optimizations

4. **Multi-Model Support**
   - Batch export/deployment
   - Model ensemble configuration
   - Pipeline deployment

## Troubleshooting

### Export Failures
- Check backend compatibility with format
- Verify output path permissions
- Ensure required dependencies installed

### Deployment Failures
- Verify exported model exists
- Check serving platform installation
- Validate port availability
- Review deployment logs

### API Connection Issues
- Ensure Flask API is running
- Check CORS configuration
- Verify API URL in environment variables

## Support

For issues or questions:
- Check component README files
- Review API documentation
- Consult Neural DSL main documentation
- Open GitHub issue with reproduction steps
