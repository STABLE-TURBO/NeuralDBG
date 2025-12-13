---
sidebar_position: 5
---

# Model Deployment

Deploy Neural DSL models to production.

## Export Formats

### ONNX (Recommended)

```bash
# Basic export
neural export model.neural --format onnx

# With optimization
neural export model.neural --format onnx --optimize

# Specify output
neural export model.neural --format onnx --output model.onnx
```

### TensorFlow Lite

```bash
# For mobile/edge devices
neural export model.neural --format tflite

# With quantization
neural export model.neural --format tflite --quantize --quantization-type int8
```

### TorchScript

```bash
# For PyTorch production
neural export model.neural --backend pytorch --format torchscript
```

### SavedModel

```bash
# TensorFlow SavedModel format
neural export model.neural --format savedmodel
```

## Deployment Platforms

### TensorFlow Serving

```bash
# Generate deployment config
neural export model.neural --format savedmodel --deployment tfserving --model-name my_model

# This creates:
# - my_model_saved_model/
# - models.config
# - deployment instructions
```

### TorchServe

```bash
# Generate TorchServe config
neural export model.neural --backend pytorch --deployment torchserve --model-name my_model

# Creates:
# - model archive (.mar file)
# - config.properties
# - deployment instructions
```

### ONNX Runtime

```python
import onnxruntime as ort

# Load model
session = ort.InferenceSession("model.onnx")

# Run inference
outputs = session.run(None, {"input": input_data})
```

## Cloud Deployment

### AWS SageMaker

```bash
# Export for SageMaker
neural export model.neural --format savedmodel --deployment sagemaker
```

### Google Cloud AI Platform

```bash
# Export for GCP
neural export model.neural --format savedmodel --deployment gcp
```

### Azure ML

```bash
# Export for Azure
neural export model.neural --format onnx --deployment azure
```

## Optimization

### Quantization

```bash
# INT8 quantization
neural export model.neural --format tflite --quantize --quantization-type int8

# Float16 quantization
neural export model.neural --format tflite --quantize --quantization-type float16
```

### Pruning

```bash
# Model pruning
neural export model.neural --format onnx --optimize --prune
```

### Graph Optimization

```bash
# ONNX graph optimization
neural export model.neural --format onnx --optimize --optimization-level 3
```

## Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install onnxruntime

# Copy model
COPY model.onnx /app/

# Copy inference script
COPY serve.py /app/

WORKDIR /app
CMD ["python", "serve.py"]
```

## REST API Example

```python
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np

app = Flask(__name__)
session = ort.InferenceSession("model.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    input_array = np.array(data, dtype=np.float32)
    outputs = session.run(None, {"input": input_array})
    return jsonify({"prediction": outputs[0].tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Performance Monitoring

```python
import time

# Measure inference time
start = time.time()
outputs = session.run(None, {"input": input_data})
inference_time = time.time() - start

print(f"Inference time: {inference_time*1000:.2f}ms")
```

## Best Practices

1. **Use ONNX** for cross-platform deployment
2. **Quantize** for mobile/edge devices
3. **Optimize graphs** before deployment
4. **Monitor performance** in production
5. **Version models** for rollback capability

## Next Steps

- [API Reference](/docs/api/cli)
- [Enterprise Features](/docs/enterprise/deployment)
