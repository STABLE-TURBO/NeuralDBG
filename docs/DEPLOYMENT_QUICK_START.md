# Deployment Quick Start

Quick reference guide for exporting and deploying Neural DSL models.

## Export Formats

### ONNX (Cross-Framework)

```bash
# Basic export
neural export model.neural --format onnx

# Optimized export
neural export model.neural --format onnx --optimize

# Specify backend
neural export model.neural --backend tensorflow --format onnx
neural export model.neural --backend pytorch --format onnx
```

**Use when:** You need cross-framework compatibility or want to use ONNX Runtime.

### TensorFlow Lite (Mobile/Edge)

```bash
# Basic export
neural export model.neural --backend tensorflow --format tflite

# With quantization
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type int8
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type float16
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type dynamic
```

**Use when:** Deploying to mobile devices (Android/iOS) or edge devices with limited resources.

### TorchScript (PyTorch Production)

```bash
# Basic export
neural export model.neural --backend pytorch --format torchscript
```

**Use when:** Deploying PyTorch models to production or using TorchServe.

### SavedModel (TensorFlow Serving)

```bash
# Basic export
neural export model.neural --backend tensorflow --format savedmodel
```

**Use when:** Using TensorFlow Serving for production deployment.

## Deployment Platforms

### TensorFlow Serving

```bash
# Export with deployment config
neural export model.neural \
  --backend tensorflow \
  --format savedmodel \
  --deployment tfserving \
  --model-name my_model

# Navigate to deployment directory
cd model_deployment

# Start server (Docker required)
./start_tfserving.sh

# Test inference
python test_inference.py

# Stop server
./stop_tfserving.sh
```

**Endpoints:**
- REST API: `http://localhost:8501/v1/models/my_model:predict`
- gRPC API: `localhost:8500`

### TorchServe

```bash
# Export with deployment config
neural export model.neural \
  --backend pytorch \
  --format torchscript \
  --deployment torchserve \
  --model-name my_model

# Navigate to deployment directory
cd model_deployment

# Create model archive (requires torch-model-archiver)
torch-model-archiver \
  --model-name my_model \
  --version 1.0 \
  --serialized-file ../model.pt \
  --handler image_classifier \
  --export-path model-store

# Start server
./start_torchserve.sh

# Test inference
python test_inference.py

# Stop server
./stop_torchserve.sh
```

**Endpoints:**
- Inference API: `http://localhost:8080/predictions/my_model`
- Management API: `http://localhost:8081/models`
- Metrics API: `http://localhost:8082/metrics`

## Python API

### Basic Export

```python
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.export import ModelExporter

# Parse DSL
parser = create_parser('network')
with open('model.neural', 'r') as f:
    content = f.read()
tree = parser.parse(content)
model_data = ModelTransformer().transform(tree)

# Export
exporter = ModelExporter(model_data, backend='tensorflow')
exporter.export_onnx('model.onnx', optimize=True)
```

### Quantized TFLite

```python
import numpy as np

exporter = ModelExporter(model_data, backend='tensorflow')

def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 28, 28, 1).astype(np.float32)]

exporter.export_tflite(
    'model_int8.tflite',
    quantize=True,
    quantization_type='int8',
    representative_dataset=representative_dataset
)
```

### TensorFlow Serving Setup

```python
exporter = ModelExporter(model_data, backend='tensorflow')

# Export SavedModel
model_path = exporter.export_savedmodel('saved_model')

# Create serving config
config_path = exporter.create_tfserving_config(
    model_path=model_path,
    model_name='my_model',
    output_dir='deployment',
    version=1
)

# Generate scripts
scripts = exporter.generate_deployment_scripts(
    output_dir='deployment',
    deployment_type='tfserving'
)
```

### TorchServe Setup

```python
exporter = ModelExporter(model_data, backend='pytorch')

# Export TorchScript
model_path = exporter.export_torchscript('model.pt')

# Create serving config
config_path, model_store = exporter.create_torchserve_config(
    model_path=model_path,
    model_name='my_model',
    output_dir='deployment',
    handler='image_classifier',
    batch_size=8
)

# Generate scripts
scripts = exporter.generate_deployment_scripts(
    output_dir='deployment',
    deployment_type='torchserve'
)
```

## Inference Examples

### ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 28, 28, 1).astype(np.float32)

outputs = session.run(None, {input_name: input_data})
predictions = outputs[0]
```

### TensorFlow Lite

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.random.randn(1, 28, 28, 1).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```

### TorchScript

```python
import torch

model = torch.jit.load('model.pt')
model.eval()

with torch.no_grad():
    input_data = torch.randn(1, 1, 28, 28)
    output = model(input_data)
```

### TensorFlow Serving (REST)

```python
import requests
import json
import numpy as np

url = "http://localhost:8501/v1/models/my_model:predict"
data = np.random.randn(1, 28, 28, 1).tolist()
payload = {"instances": data}

response = requests.post(url, json=payload)
predictions = response.json()['predictions']
```

### TorchServe (REST)

```python
import requests
import json
import numpy as np

url = "http://localhost:8080/predictions/my_model"
data = np.random.randn(1, 28, 28, 1).tolist()
payload = {"data": data}

response = requests.post(url, json=payload)
predictions = response.json()
```

## Common Workflows

### Mobile App Deployment

```bash
# 1. Export to TFLite with quantization
neural export model.neural \
  --backend tensorflow \
  --format tflite \
  --quantize \
  --quantization-type int8 \
  --output mobile_model.tflite

# 2. Test locally
python test_tflite.py

# 3. Integrate with mobile app
# - Android: Use TFLite Java/Kotlin API
# - iOS: Use TFLite Swift API or Core ML
```

### Cloud API Deployment

```bash
# 1. Export with serving config
neural export model.neural \
  --backend tensorflow \
  --format savedmodel \
  --deployment tfserving \
  --model-name production_model

# 2. Deploy to cloud
cd model_deployment
docker-compose up -d

# 3. Test endpoint
curl -X POST http://localhost:8501/v1/models/production_model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[[[0.5]]]]}'

# 4. Set up monitoring and scaling
```

### Multi-Platform Deployment

```bash
# Export to multiple formats
neural export model.neural --backend tensorflow --format onnx --output model.onnx
neural export model.neural --backend tensorflow --format tflite --output model.tflite
neural export model.neural --backend pytorch --format torchscript --output model.pt

# Deploy based on platform
# - Web: ONNX with ONNX.js
# - Mobile: TFLite
# - Server: TensorFlow Serving or TorchServe
# - Edge: TFLite with quantization
```

## Optimization Tips

### Model Size Reduction

```bash
# Quantization (4x reduction)
neural export model.neural --format tflite --quantize --quantization-type int8

# ONNX optimization
neural export model.neural --format onnx --optimize
```

### Inference Speed

1. **Use batch inference** - Process multiple inputs together
2. **Enable GPU** - Use GPU-enabled serving platforms
3. **Optimize model** - Apply quantization and pruning
4. **Use appropriate format** - TFLite for mobile, ONNX for cross-platform

### Memory Usage

1. **Quantization** - Reduce precision (int8, float16)
2. **Model pruning** - Remove unnecessary weights
3. **Batch size tuning** - Balance throughput and memory

## Troubleshooting

### Export Fails

```bash
# Enable verbose logging
neural export model.neural --format onnx --verbose

# Check dependencies
pip install onnx tf2onnx torch tensorflow
```

### Serving Issues

```bash
# Check server logs
docker logs <container_id>

# Test with simple input
curl -X POST http://localhost:8501/v1/models/my_model:predict \
  -d '{"instances": [0]}'

# Verify model path
ls -la /path/to/model
```

### Performance Issues

1. **Enable GPU** - Set `--gpus all` in Docker
2. **Increase workers** - Scale serving instances
3. **Optimize model** - Apply quantization
4. **Use batching** - Enable batch prediction

## Next Steps

- Read full [Deployment Guide](deployment.md)
- Check [Examples](../examples/deployment_example.py)
- Review [TensorFlow Serving Docs](https://www.tensorflow.org/tfx/guide/serving)
- Review [TorchServe Docs](https://pytorch.org/serve/)
