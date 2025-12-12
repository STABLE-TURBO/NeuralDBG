# Model Deployment Guide

This guide covers exporting and deploying Neural DSL models to production environments using various formats and serving platforms.

## Table of Contents

- [Export Formats](#export-formats)
  - [ONNX](#onnx)
  - [TensorFlow Lite](#tensorflow-lite)
  - [TorchScript](#torchscript)
  - [SavedModel](#savedmodel)
- [Model Serving](#model-serving)
  - [TensorFlow Serving](#tensorflow-serving)
  - [TorchServe](#torchserve)
- [Deployment Strategies](#deployment-strategies)
- [Best Practices](#best-practices)

## Export Formats

### ONNX

ONNX (Open Neural Network Exchange) is an open format for representing machine learning models, enabling interoperability across different frameworks.

#### Basic Export

```bash
# Export TensorFlow model to ONNX
neural export model.neural --backend tensorflow --format onnx --output model.onnx

# Export PyTorch model to ONNX
neural export model.neural --backend pytorch --format onnx --output model.onnx
```

#### Optimized Export

Apply optimization passes to reduce model size and improve inference speed:

```bash
neural export model.neural --backend tensorflow --format onnx --optimize --output model_optimized.onnx
```

Optimization passes include:
- Identity elimination
- Constant folding
- BatchNorm fusion into Conv layers
- Transpose optimization
- MatMul/Add fusion into GEMM

#### Python API

```python
from neural.code_generation.export import ModelExporter
from neural.parser.parser import create_parser, ModelTransformer

# Parse model
parser = create_parser('network')
with open('model.neural', 'r') as f:
    content = f.read()
tree = parser.parse(content)
model_data = ModelTransformer().transform(tree)

# Export to ONNX
exporter = ModelExporter(model_data, backend='tensorflow')
exporter.export_onnx(
    output_path='model.onnx',
    opset_version=13,
    optimize=True
)
```

#### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('model.onnx')

# Prepare input
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 28, 28, 1).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: input_data})
predictions = outputs[0]
```

### TensorFlow Lite

TensorFlow Lite is optimized for mobile and edge devices with limited compute resources.

#### Basic Export

```bash
neural export model.neural --backend tensorflow --format tflite --output model.tflite
```

#### Quantized Export

Reduce model size and improve inference speed with quantization:

```bash
# Dynamic range quantization
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type dynamic --output model_dynamic.tflite

# Float16 quantization
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type float16 --output model_fp16.tflite

# Full integer quantization (requires representative dataset)
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type int8 --output model_int8.tflite
```

Quantization benefits:
- **Dynamic**: 4x smaller model, minimal accuracy loss
- **Float16**: 2x smaller, GPU acceleration on mobile
- **Int8**: 4x smaller, fastest inference, requires calibration

#### Python API

```python
from neural.code_generation.export import ModelExporter

exporter = ModelExporter(model_data, backend='tensorflow')

# With quantization
def representative_dataset():
    """Generator for calibration data."""
    for _ in range(100):
        yield [np.random.randn(1, 28, 28, 1).astype(np.float32)]

exporter.export_tflite(
    output_path='model.tflite',
    quantize=True,
    quantization_type='int8',
    representative_dataset=representative_dataset
)
```

#### TFLite Interpreter

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
input_data = np.random.randn(1, 28, 28, 1).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
```

### TorchScript

TorchScript enables PyTorch models to be serialized and optimized for production.

#### Basic Export

```bash
neural export model.neural --backend pytorch --format torchscript --output model.pt
```

#### Python API

```python
from neural.code_generation.export import ModelExporter

exporter = ModelExporter(model_data, backend='pytorch')

# Trace-based export (recommended for most models)
exporter.export_torchscript(
    output_path='model.pt',
    method='trace'
)

# Script-based export (for models with control flow)
exporter.export_torchscript(
    output_path='model.pt',
    method='script'
)
```

#### TorchScript Inference

```python
import torch

# Load model
model = torch.jit.load('model.pt')
model.eval()

# Run inference
with torch.no_grad():
    input_data = torch.randn(1, 1, 28, 28)
    output = model(input_data)
```

### SavedModel

SavedModel is TensorFlow's recommended format for deployment, especially with TensorFlow Serving.

#### Basic Export

```bash
neural export model.neural --backend tensorflow --format savedmodel --output saved_model
```

#### Python API

```python
from neural.code_generation.export import ModelExporter

exporter = ModelExporter(model_data, backend='tensorflow')
exporter.export_savedmodel(output_path='saved_model')
```

#### Loading SavedModel

```python
import tensorflow as tf

# Load model
model = tf.saved_model.load('saved_model')

# Run inference
input_data = tf.random.normal([1, 28, 28, 1])
output = model(input_data)
```

## Model Serving

### TensorFlow Serving

TensorFlow Serving is a flexible, high-performance serving system for machine learning models.

#### Quick Start

```bash
# Export model with TF Serving config
neural export model.neural \
  --backend tensorflow \
  --format savedmodel \
  --deployment tfserving \
  --model-name my_model \
  --output saved_model
```

This generates:
- `model_deployment/` - Deployment directory
- `model_deployment/models.config` - Model configuration
- `model_deployment/docker-compose.yml` - Docker deployment config
- `model_deployment/start_tfserving.sh` - Start script
- `model_deployment/stop_tfserving.sh` - Stop script
- `model_deployment/test_inference.py` - Test client

#### Docker Deployment

```bash
cd model_deployment
./start_tfserving.sh
```

The server will be available at:
- REST API: `http://localhost:8501`
- gRPC API: `localhost:8500`

#### Manual Docker Setup

```bash
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/saved_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving
```

#### REST API Usage

```bash
# Health check
curl http://localhost:8501/v1/models/my_model

# Predict
curl -X POST http://localhost:8501/v1/models/my_model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[[[0.5]]]]}'
```

#### Python Client

```python
import requests
import json
import numpy as np

url = "http://localhost:8501/v1/models/my_model:predict"

# Prepare data
data = np.random.randn(1, 28, 28, 1).tolist()
payload = {"instances": data}

# Make prediction
response = requests.post(url, json=payload)
predictions = response.json()['predictions']
```

#### gRPC Client

```python
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.model_spec.signature_name = 'serving_default'

# Prepare input
input_data = tf.make_tensor_proto(
    np.random.randn(1, 28, 28, 1),
    dtype=tf.float32
)
request.inputs['input'].CopyFrom(input_data)

# Get prediction
result = stub.Predict(request, 10.0)
```

#### Configuration

`models.config`:
```json
{
  "model_config_list": [
    {
      "name": "my_model",
      "base_path": "/models/my_model",
      "model_platform": "tensorflow",
      "model_version_policy": {
        "specific": {
          "versions": [1, 2]
        }
      }
    }
  ]
}
```

### TorchServe

TorchServe is a production-ready serving framework for PyTorch models.

#### Quick Start

```bash
# Export model with TorchServe config
neural export model.neural \
  --backend pytorch \
  --format torchscript \
  --deployment torchserve \
  --model-name my_model \
  --output model.pt
```

This generates:
- `model_deployment/` - Deployment directory
- `model_deployment/config.properties` - Server configuration
- `model_deployment/model-store/` - Model archive storage
- `model_deployment/start_torchserve.sh` - Start script
- `model_deployment/stop_torchserve.sh` - Stop script
- `model_deployment/test_inference.py` - Test client

#### Create Model Archive

```bash
# Install torch-model-archiver
pip install torch-model-archiver

# Create MAR file
torch-model-archiver \
  --model-name my_model \
  --version 1.0 \
  --serialized-file model.pt \
  --handler image_classifier \
  --export-path model-store
```

Available handlers:
- `image_classifier` - Image classification
- `text_classifier` - Text classification
- `object_detector` - Object detection
- `image_segmenter` - Image segmentation

#### Start TorchServe

```bash
cd model_deployment
./start_torchserve.sh
```

Or manually:
```bash
torchserve --start \
  --model-store model-store \
  --models my_model=my_model.mar \
  --ts-config config.properties
```

The server will be available at:
- Inference API: `http://localhost:8080`
- Management API: `http://localhost:8081`
- Metrics API: `http://localhost:8082`

#### REST API Usage

```bash
# Health check
curl http://localhost:8080/ping

# List models
curl http://localhost:8081/models

# Predict
curl -X POST http://localhost:8080/predictions/my_model \
  -H "Content-Type: application/json" \
  -d '{"data": [[[[0.5]]]]}'
```

#### Python Client

```python
import requests
import json
import numpy as np

url = "http://localhost:8080/predictions/my_model"

# Prepare data
data = np.random.randn(1, 28, 28, 1).tolist()
payload = {"data": data}

# Make prediction
response = requests.post(url, json=payload)
predictions = response.json()
```

#### Management Operations

```bash
# Register model
curl -X POST "http://localhost:8081/models?url=my_model.mar"

# Scale workers
curl -X PUT "http://localhost:8081/models/my_model?min_worker=2&max_worker=4"

# Unregister model
curl -X DELETE "http://localhost:8081/models/my_model"

# Get model details
curl http://localhost:8081/models/my_model
```

#### Custom Handler

Create `custom_handler.py`:

```python
import torch
from ts.torch_handler.base_handler import BaseHandler

class CustomHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        # Custom initialization
    
    def preprocess(self, data):
        # Custom preprocessing
        return processed_data
    
    def inference(self, data):
        # Run model inference
        with torch.no_grad():
            predictions = self.model(data)
        return predictions
    
    def postprocess(self, data):
        # Custom postprocessing
        return processed_output
```

Use custom handler:
```bash
torch-model-archiver \
  --model-name my_model \
  --version 1.0 \
  --serialized-file model.pt \
  --handler custom_handler.py \
  --extra-files index_to_name.json \
  --export-path model-store
```

## Deployment Strategies

### Edge Deployment

For mobile and edge devices:

1. **Use TensorFlow Lite**
   ```bash
   neural export model.neural \
     --backend tensorflow \
     --format tflite \
     --quantize \
     --quantization-type int8
   ```

2. **Optimize for size**
   - Use quantization
   - Remove unnecessary operations
   - Consider model pruning

3. **Mobile Integration**
   - Android: Use TFLite Java API
   - iOS: Use TFLite Swift API or Core ML

### Cloud Deployment

For cloud platforms:

#### AWS SageMaker

```python
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

# Package model
model = TensorFlowModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework_version='2.12'
)

# Deploy
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Predict
predictions = predictor.predict(data)
```

#### Google Cloud AI Platform

```bash
# Deploy model
gcloud ai-platform models create my_model

gcloud ai-platform versions create v1 \
  --model my_model \
  --origin gs://bucket/saved_model \
  --runtime-version 2.12 \
  --python-version 3.9
```

#### Azure Machine Learning

```python
from azureml.core import Model
from azureml.core.webservice import AciWebservice, Webservice

# Register model
model = Model.register(
    workspace=ws,
    model_path='model',
    model_name='my_model'
)

# Deploy
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1
)

service = Model.deploy(
    workspace=ws,
    name='my-model-service',
    models=[model],
    deployment_config=aci_config
)
```

### Kubernetes Deployment

#### TensorFlow Serving on Kubernetes

`deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tfserving-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tfserving
  template:
    metadata:
      labels:
        app: tfserving
    spec:
      containers:
      - name: tfserving
        image: tensorflow/serving:latest
        ports:
        - containerPort: 8501
          name: rest-api
        - containerPort: 8500
          name: grpc-api
        env:
        - name: MODEL_NAME
          value: "my_model"
        volumeMounts:
        - name: model-storage
          mountPath: /models/my_model
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

#### TorchServe on Kubernetes

`deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: torchserve
  template:
    metadata:
      labels:
        app: torchserve
    spec:
      containers:
      - name: torchserve
        image: pytorch/torchserve:latest
        ports:
        - containerPort: 8080
          name: inference
        - containerPort: 8081
          name: management
        volumeMounts:
        - name: model-store
          mountPath: /home/model-server/model-store
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: model-pvc
```

## Best Practices

### Model Optimization

1. **Quantization**: Reduce model size and improve inference speed
   - Use dynamic quantization for quick wins
   - Use full quantization for maximum optimization

2. **Pruning**: Remove unnecessary weights
   ```python
   import tensorflow_model_optimization as tfmot
   
   pruning_params = {
       'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
           initial_sparsity=0.0,
           final_sparsity=0.5,
           begin_step=0,
           end_step=1000
       )
   }
   
   model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
   ```

3. **Distillation**: Train smaller models from larger ones

### Performance Monitoring

1. **Latency Tracking**
   ```python
   import time
   
   start = time.time()
   predictions = model.predict(data)
   latency = time.time() - start
   ```

2. **Throughput Measurement**
   ```python
   import concurrent.futures
   
   def benchmark(model, data, num_requests):
       with concurrent.futures.ThreadPoolExecutor() as executor:
           futures = [executor.submit(model.predict, data) 
                     for _ in range(num_requests)]
           results = [f.result() for f in futures]
   ```

3. **Resource Monitoring**
   - CPU/GPU utilization
   - Memory usage
   - Request queue length

### Security

1. **Authentication**
   - Use API keys or OAuth tokens
   - Implement rate limiting

2. **Input Validation**
   ```python
   def validate_input(data):
       if not isinstance(data, np.ndarray):
           raise ValueError("Invalid input type")
       if data.shape != expected_shape:
           raise ValueError("Invalid input shape")
       if data.dtype != np.float32:
           raise ValueError("Invalid input dtype")
   ```

3. **Model Encryption**
   - Encrypt model files at rest
   - Use secure channels (HTTPS/TLS)

### Versioning

1. **Model Versioning**
   ```
   models/
   ├── my_model/
   │   ├── 1/
   │   │   └── saved_model.pb
   │   ├── 2/
   │   │   └── saved_model.pb
   │   └── 3/
   │       └── saved_model.pb
   ```

2. **A/B Testing**
   ```python
   def route_request(request, model_v1, model_v2, traffic_split=0.5):
       if random.random() < traffic_split:
           return model_v1.predict(request)
       else:
           return model_v2.predict(request)
   ```

3. **Rollback Strategy**
   - Keep previous versions available
   - Monitor metrics after deployment
   - Automated rollback on errors

### Scaling

1. **Horizontal Scaling**: Add more instances
   - Use load balancers
   - Distribute traffic evenly

2. **Vertical Scaling**: Use more powerful hardware
   - GPU acceleration
   - Larger instance types

3. **Auto-scaling**
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: tfserving-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: tfserving-deployment
     minReplicas: 2
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use quantization
   - Enable model parallelism

2. **Slow Inference**
   - Check for bottlenecks (I/O, preprocessing)
   - Use batch inference
   - Enable GPU acceleration

3. **Model Compatibility**
   - Verify ONNX opset version
   - Check framework versions
   - Test with sample data

### Debugging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    predictions = model.predict(data)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    logger.debug(f"Input shape: {data.shape}")
    logger.debug(f"Input dtype: {data.dtype}")
    raise
```

## Resources

- [ONNX Documentation](https://onnx.ai/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [TensorFlow Serving Documentation](https://www.tensorflow.org/tfx/guide/serving)
- [TorchServe Documentation](https://pytorch.org/serve/)
- [Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
