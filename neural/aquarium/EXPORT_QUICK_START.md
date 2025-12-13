# Export Panel Quick Start Guide

Quick reference for using the Neural Aquarium Export and Deployment Panel.

## Starting the Services

### 1. Start Backend API
```bash
cd neural/aquarium/api
python shape_api.py
```
This starts the Flask API on `http://localhost:5002` with export endpoints included.

### 2. Start Aquarium UI
```bash
cd neural/aquarium
npm start
```
This starts the React development server on `http://localhost:3000`.

## Basic Export Workflow

### Step 1: Select Export Format
- **ONNX**: Works with both TensorFlow and PyTorch
- **TFLite**: TensorFlow only, best for mobile
- **TorchScript**: PyTorch only, for production
- **SavedModel**: TensorFlow only, for TF Serving

### Step 2: Configure Optimizations
- Toggle general optimizations ON for better performance
- Enable quantization for smaller models (INT8, Float16, or Dynamic)
- Enable pruning to remove unnecessary weights (adjust sparsity slider)

### Step 3: Set Output Path
```
./exported_models/my_model
```
File extension added automatically based on format.

### Step 4: Export Model
Click "Export Model" button. Success shows:
- Export path
- Format
- File size
- Success message

## Deployment Workflow

### Step 1: Export Model First
Complete the export workflow above before deploying.

### Step 2: Switch to Deploy Tab
Click the "Deploy" tab in the panel header.

### Step 3: Configure Deployment
- **Model Name**: Unique identifier for your model
- **Version**: Semantic version (e.g., 1.0, 2.0)

### Step 4: Select Target and Platform
- **Target**: Cloud, Edge, Mobile, or Server
- **Platform**: TorchServe, TensorFlow Serving, ONNX Runtime, or Triton

### Step 5: Configure Resources
- GPU Enabled: Check if using GPU
- Replicas: Number of model instances (1-10)
- Batch Size: Requests per batch (1-32)
- Max Batch Delay: Wait time in ms (0-1000)

### Step 6: Configure Networking
- Port: Service port (default 8080)
- Enable Metrics: Check for monitoring
- Enable Health Check: Check for liveness probes

### Step 7: Generate Serving Config
Click "Generate Serving Config" to create:
- Configuration files
- Deployment scripts
- Setup instructions

### Step 8: Deploy
Click "Deploy Model" button. Success shows:
- Deployment ID
- Endpoint URL
- Success message

## Quick Format Selection Guide

| Use Case | Format | Backend | Quantization |
|----------|--------|---------|--------------|
| Production API | ONNX | Any | Float16 |
| Mobile App | TFLite | TensorFlow | INT8 |
| Edge Device | TFLite | TensorFlow | INT8 |
| PyTorch Serving | TorchScript | PyTorch | None |
| TF Serving | SavedModel | TensorFlow | None |
| Cross-Platform | ONNX | Any | Dynamic |

## Platform Selection Guide

| Platform | Best For | Formats | GPU |
|----------|----------|---------|-----|
| TorchServe | PyTorch models | TorchScript, ONNX | ✓ |
| TF Serving | TensorFlow models | SavedModel, TFLite | ✓ |
| ONNX Runtime | Any framework | ONNX | ✓ |
| Triton | Multi-framework | All | ✓ |

## Optimization Recommendations

### For Cloud Deployment
```
- Optimize: ON
- Quantization: Float16 (if GPU) or Dynamic (if CPU)
- Pruning: OFF or Low (10-20%)
```

### For Edge/Mobile Deployment
```
- Optimize: ON
- Quantization: INT8
- Pruning: Medium to High (30-50%)
```

### For Maximum Performance
```
- Optimize: ON
- Quantization: Float16 (GPU) or Dynamic (CPU)
- Pruning: OFF
```

### For Minimum Size
```
- Optimize: ON
- Quantization: INT8
- Pruning: High (50-70%)
- Note: Test accuracy carefully!
```

## Testing Deployments

### TorchServe
```bash
cd serving_config
./start_torchserve.sh
python test_inference.py
```

### TensorFlow Serving
```bash
cd serving_config
./start_tfserving.sh
python test_inference.py
```

### ONNX Runtime
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
result = session.run(None, {"input": input_data})
```

## Common Issues

### "Export failed: Backend mismatch"
**Solution**: TFLite/SavedModel require TensorFlow backend, TorchScript requires PyTorch.

### "Deployment failed: Export first"
**Solution**: Complete the Export workflow before switching to Deploy tab.

### "No serving platforms available"
**Solution**: Select a compatible export format for your desired platform.

### "Port already in use"
**Solution**: Change the port number or stop the conflicting service.

### "Model size too large"
**Solution**: Enable quantization (INT8 or Float16) and/or pruning.

## Environment Variables

Set in `.env` or environment:

```bash
# API endpoint (default: http://localhost:5000)
REACT_APP_API_URL=http://localhost:5000

# Export directory (default: ./exported_models)
EXPORT_DIR=./exported_models

# Deployment directory (default: ./deployments)
DEPLOYMENT_DIR=./deployments
```

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Switch to Export tab | Alt+1 |
| Switch to Deploy tab | Alt+2 |
| Export model | Ctrl+E |
| Deploy model | Ctrl+D |
| Generate config | Ctrl+G |

## API Endpoints Reference

```bash
# Export model
curl -X POST http://localhost:5000/api/export/model \
  -H "Content-Type: application/json" \
  -d '{"model_data": {...}, "options": {...}}'

# Deploy model
curl -X POST http://localhost:5000/api/deployment/deploy \
  -H "Content-Type: application/json" \
  -d '{"export_path": "...", "config": {...}}'

# Generate serving config
curl -X POST http://localhost:5000/api/deployment/serving-config \
  -H "Content-Type: application/json" \
  -d '{"model_path": "...", "platform": "torchserve", ...}'

# Get deployment status
curl http://localhost:5000/api/deployment/<deployment-id>/status

# List all deployments
curl http://localhost:5000/api/deployment/list
```

## Component Import Reference

```typescript
// Import entire panel
import { ExportPanel } from './components/export';

// Import individual components
import {
  ExportFormatSelector,
  OptimizationOptions,
  DeploymentTargetSelector,
  ServingConfigGenerator,
  ExportProgress
} from './components/export';

// Import service
import { ExportService } from './services';

// Import types
import {
  ExportOptions,
  ExportResult,
  DeploymentConfig,
  DeploymentResult,
  ExportFormat,
  QuantizationType,
  DeploymentTarget,
  ServingPlatform
} from './types';
```

## Next Steps

1. Explore advanced optimization settings
2. Configure cloud provider integration
3. Set up CI/CD pipelines
4. Monitor deployed models
5. Implement A/B testing
6. Configure auto-scaling

## Resources

- Full Documentation: `./EXPORT_INTEGRATION.md`
- Component README: `./src/components/export/README.md`
- API Documentation: `./api/export_api.py`
- Main Neural DSL Docs: `../../README.md`
