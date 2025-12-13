# Model Deployment Guide

So you've built a model and now you want to put it in production. This guide is written by someone who's had models crash at 3 AM on Black Friday. Learn from my scars.

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
- [Hard-Won Lessons](#hard-won-lessons)

## Export Formats

### ONNX

ONNX is like the Esperanto of ML models—everyone supports it, but nobody uses it as their first language. It's brilliant when you need to move between frameworks or optimize for edge devices.

**When to use ONNX:**
- You're switching frameworks (TensorFlow to PyTorch, etc.)
- You need framework-agnostic deployment
- You're deploying to edge hardware with ONNX Runtime
- You want to use vendor-specific accelerators (NVIDIA TensorRT, Intel OpenVINO)

**When NOT to use ONNX:**
- Your model uses cutting-edge ops that ONNX doesn't support yet (check opset compatibility first)
- You have complex control flow (if/while loops)—ONNX will try but might butcher it
- You're staying within one framework ecosystem (use native formats)

#### Basic Export

```bash
# Export TensorFlow model to ONNX
neural export model.neural --backend tensorflow --format onnx --output model.onnx

# Export PyTorch model to ONNX
neural export model.neural --backend pytorch --format onnx --output model.onnx
```

**Battle-tested gotcha:** Always test your exported ONNX model with real data. I once deployed an ONNX model that worked perfectly in testing but gave different results in production because of implicit type conversions. The numerical differences were small (0.1%), but in a ranking system, that's catastrophic.

#### Optimized Export

```bash
neural export model.neural --backend tensorflow --format onnx --optimize --output model_optimized.onnx
```

Optimization passes include:
- Identity elimination (removes no-op layers—you'd be surprised how many you have)
- Constant folding (precomputes static values—free speedup)
- BatchNorm fusion into Conv layers (classic optimization, 10-15% faster)
- Transpose optimization (especially important for NHWC → NCHW conversions)
- MatMul/Add fusion into GEMM (matrix ops are faster as one kernel)

**Real-world experience:** The optimization flag once saved our bacon. We had a model that was too slow for real-time inference (85ms). After optimization, it dropped to 62ms. The difference? BatchNorm fusion and constant folding. Always optimize, but always benchmark before and after—sometimes optimizations can introduce subtle numerical differences.

#### Python API

```python
from neural.code_generation.export import ModelExporter
from neural.parser.parser import create_parser, ModelTransformer

parser = create_parser('network')
with open('model.neural', 'r') as f:
    content = f.read()
tree = parser.parse(content)
model_data = ModelTransformer().transform(tree)

exporter = ModelExporter(model_data, backend='tensorflow')
exporter.export_onnx(
    output_path='model.onnx',
    opset_version=13,  # Don't use the latest opset unless you need specific ops
    optimize=True
)
```

**Opset version wisdom:** Stick with opset 13 or 14 unless you need newer ops. Newer opsets have fewer runtime implementations and might not work on older ONNX Runtime versions. I learned this when our edge devices couldn't run opset 16 models—we had to re-export everything at 2 AM.

#### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 28, 28, 1).astype(np.float32)

outputs = session.run(None, {input_name: input_data})
predictions = outputs[0]
```

**Performance trap:** ONNX Runtime will use CPU by default. If you have a GPU, explicitly set the execution provider:

```python
session = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
```

I once spent a week debugging "slow inference" before realizing the model was running on CPU despite having GPUs available. Check your providers.

### TensorFlow Lite

TFLite is your friend when deploying to mobile/edge. It's optimized to run on devices with less RAM than your morning coffee costs.

**When to use TFLite:**
- Mobile apps (Android/iOS)
- Raspberry Pi, Jetson Nano, Coral Edge TPU
- Battery-powered devices where every watt matters
- You need <100ms inference on a phone

**When NOT to use TFLite:**
- Your model has ops TFLite doesn't support (check the compatibility list—it's long but not infinite)
- You need dynamic shapes (TFLite loves fixed shapes)
- Sub-millisecond latency requirements (use native framework with TensorRT/TVM)

#### Basic Export

```bash
neural export model.neural --backend tensorflow --format tflite --output model.tflite
```

#### Quantized Export

Quantization is like compressing a FLAC to MP3—you lose a bit of quality but gain a ton of efficiency.

```bash
# Dynamic range quantization (easiest, safest)
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type dynamic --output model_dynamic.tflite

# Float16 quantization (best for GPU acceleration)
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type float16 --output model_fp16.tflite

# Full integer quantization (maximum speed, requires calibration)
neural export model.neural --backend tensorflow --format tflite --quantize --quantization-type int8 --output model_int8.tflite
```

**The quantization decision tree:**
- **Dynamic**: Start here. 4x smaller, minimal accuracy loss (<1% typically). I've deployed dozens of models with dynamic quantization and users couldn't tell the difference.
- **Float16**: Use if you have GPU-capable mobile devices (most modern phones). 2x smaller, maintains accuracy well, GPU acceleration makes it faster than full precision on mobile.
- **Int8**: The nuclear option. 4x smaller, fastest inference, but you NEED a representative dataset for calibration. I've seen int8 models lose 5-10% accuracy when calibrated poorly.

**Horror story:** We deployed an int8-quantized model that worked beautifully in testing. In production, it started predicting garbage because our calibration dataset didn't include night-time images. The model had never seen low-light data during quantization. Always calibrate with production-representative data.

#### Python API

```python
from neural.code_generation.export import ModelExporter

exporter = ModelExporter(model_data, backend='tensorflow')

def representative_dataset():
    """
    This function is CRITICAL for int8 quantization.
    Use actual production data, not random noise.
    """
    for _ in range(100):  # 100-1000 samples is usually enough
        yield [np.random.randn(1, 28, 28, 1).astype(np.float32)]

exporter.export_tflite(
    output_path='model.tflite',
    quantize=True,
    quantization_type='int8',
    representative_dataset=representative_dataset
)
```

**Calibration dataset wisdom:** The representative_dataset function is your model's future. Use 100-1000 samples that represent actual production data distribution. More isn't always better—I've seen diminishing returns after 500 samples.

#### TFLite Interpreter

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

**Memory gotcha:** `allocate_tensors()` is not optional. Call it once after loading. If you forget, you'll get cryptic errors about uninitialized tensors. Ask me how I know.

### TorchScript

TorchScript is PyTorch's way of making your model runnable without Python. It's like taking a Python script and turning it into a binary—fast, portable, but occasionally temperamental.

**When to use TorchScript:**
- You're deploying PyTorch models to production
- You want C++ deployment without Python overhead
- You need mobile deployment (PyTorch Mobile uses TorchScript)
- You want to optimize with TorchScript's JIT compiler

**When NOT to use TorchScript:**
- Your model uses dynamic control flow that changes based on input data (TorchScript might trace it wrong)
- You're using exotic third-party ops
- You need cross-framework compatibility (use ONNX)

#### Basic Export

```bash
neural export model.neural --backend pytorch --format torchscript --output model.pt
```

#### Python API

```python
from neural.code_generation.export import ModelExporter

exporter = ModelExporter(model_data, backend='pytorch')

# Trace-based export (use this 90% of the time)
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

**Trace vs. Script—the eternal debate:**
- **Trace**: Records what the model actually does with example input. Fast, works 90% of the time. But if your model has if/else statements that depend on input values, trace will only capture one path.
- **Script**: Analyzes the source code and converts it. Handles control flow correctly but can fail on complex Python code.

**Real example:** I once traced a model that had `if x.shape[0] > 10` to switch between batch processing and single-sample mode. The traced model always used the path I traced (batch mode), so single samples crashed. Use script for control flow, trace for everything else.

#### TorchScript Inference

```python
import torch

model = torch.jit.load('model.pt')
model.eval()

with torch.no_grad():
    input_data = torch.randn(1, 1, 28, 28)
    output = model(input_data)
```

**Performance tip:** Set `torch.set_num_threads(4)` before loading if you're on CPU. Default thread count can be suboptimal. I've seen 2x speedups just by setting this correctly.

### SavedModel

SavedModel is TensorFlow's native format. If you're staying in TensorFlow land, this is the gold standard.

**When to use SavedModel:**
- TensorFlow Serving deployment
- Staying in TensorFlow ecosystem
- You need serving signatures with multiple input/output configurations
- Cloud deployment (SageMaker, Vertex AI, Azure ML all support it)

**When NOT to use SavedModel:**
- You need the smallest possible model (use TFLite)
- Cross-framework deployment (use ONNX)
- You're on PyTorch (obviously)

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

**Directory structure matters:** SavedModel creates a directory with `saved_model.pb` and `variables/`. Don't zip it, don't flatten it. Keep the structure intact. TensorFlow Serving expects this exact layout.

#### Loading SavedModel

```python
import tensorflow as tf

model = tf.saved_model.load('saved_model')
input_data = tf.random.normal([1, 28, 28, 1])
output = model(input_data)
```

**Signature gotcha:** SavedModel can have multiple serving signatures. If `model(input)` doesn't work, try:

```python
infer = model.signatures['serving_default']
output = infer(tf.constant(input_data))['output_0']
```

I spent hours debugging this once—the model loaded fine but calling it failed because I wasn't using the signature correctly.

## Model Serving

### TensorFlow Serving

TensorFlow Serving is industrial-strength model serving. It's what Google uses internally, and it shows—robust, fast, but with a learning curve.

**When to use TF Serving:**
- High-throughput production serving (1000+ RPS)
- You need model versioning and A/B testing
- gRPC performance matters
- You're already in TensorFlow ecosystem

**When NOT to use TF Serving:**
- Simple, low-traffic applications (overkill—just use Flask)
- You're on PyTorch (use TorchServe)
- You need custom preprocessing that's not in TensorFlow

#### Quick Start

```bash
neural export model.neural \
  --backend tensorflow \
  --format savedmodel \
  --deployment tfserving \
  --model-name my_model \
  --output saved_model
```

This generates a complete deployment setup including Docker configs and test scripts. I wish I had this three years ago.

#### Docker Deployment

```bash
cd model_deployment
./start_tfserving.sh
```

The server will be available at:
- REST API: `http://localhost:8501` (use for development)
- gRPC API: `localhost:8500` (use for production—3-5x faster)

**Docker gotcha:** If you're on Mac with Apple Silicon, TensorFlow Serving Docker images might not work. Use `docker pull --platform linux/amd64` or run on Linux/Intel.

#### REST API Usage

```bash
# Health check (always verify before deploying)
curl http://localhost:8501/v1/models/my_model

# Predict
curl -X POST http://localhost:8501/v1/models/my_model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[[[0.5]]]]}'
```

**REST API performance reality:** REST API is convenient for testing but 2-3x slower than gRPC. For high-throughput production, bite the bullet and implement gRPC. I resisted this for months and regretted it when our traffic spiked.

#### Python Client

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

**Production tip:** Add timeouts and retries:

```python
response = requests.post(url, json=payload, timeout=5)
```

I've seen models occasionally take 30+ seconds due to resource contention. Without timeouts, your app will hang.

#### gRPC Client

```python
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.model_spec.signature_name = 'serving_default'

input_data = tf.make_tensor_proto(
    np.random.randn(1, 28, 28, 1),
    dtype=tf.float32
)
request.inputs['input'].CopyFrom(input_data)

result = stub.Predict(request, 10.0)  # 10s timeout
```

**gRPC is faster but less forgiving:** REST API will give you nice JSON error messages. gRPC will give you cryptic protocol buffer errors. Keep the REST endpoint available for debugging.

### TorchServe

TorchServe is PyTorch's answer to TensorFlow Serving. It's younger, but it's matured nicely.

**When to use TorchServe:**
- PyTorch models (obviously)
- You need custom preprocessing/postprocessing
- Built-in handlers match your use case
- You want metrics/monitoring out of the box

**When NOT to use TorchServe:**
- You need absolute maximum performance (raw PyTorch + FastAPI can be faster for simple cases)
- You're on TensorFlow (use TF Serving)
- Your preprocessing is complex and needs non-Python libraries

#### Quick Start

```bash
neural export model.neural \
  --backend pytorch \
  --format torchscript \
  --deployment torchserve \
  --model-name my_model \
  --output model.pt
```

#### Create Model Archive

```bash
pip install torch-model-archiver

torch-model-archiver \
  --model-name my_model \
  --version 1.0 \
  --serialized-file model.pt \
  --handler image_classifier \
  --export-path model-store
```

**Handler selection matters:**
- `image_classifier`: Use for image classification (duh)
- `text_classifier`: Use for text classification
- `object_detector`: Use for object detection
- `image_segmenter`: Use for segmentation
- Custom handler: When you need preprocessing that built-ins don't provide

**Real experience:** I tried using `image_classifier` for a model that needed custom normalization. It failed silently—predictions were garbage. Always verify the handler does what you expect, or write a custom one.

#### Start TorchServe

```bash
cd model_deployment
./start_torchserve.sh
```

The server will be available at:
- Inference API: `http://localhost:8080`
- Management API: `http://localhost:8081` (for model management)
- Metrics API: `http://localhost:8082` (for Prometheus)

**Port conflicts:** If 8080/8081/8082 are taken, TorchServe will fail silently or use random ports. Check the logs and configure ports explicitly in `config.properties`.

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

#### Management Operations

```bash
# Register model (useful for hot-swapping models)
curl -X POST "http://localhost:8081/models?url=my_model.mar"

# Scale workers (do this based on CPU/GPU availability)
curl -X PUT "http://localhost:8081/models/my_model?min_worker=2&max_worker=4"

# Unregister model (for rolling updates)
curl -X DELETE "http://localhost:8081/models/my_model"
```

**Worker scaling wisdom:** Start with `min_worker=1, max_worker=4` and monitor. Each worker holds the model in memory. If you have 4GB GPU and 2GB model, you can fit 2 workers max. I once set max_workers=10 on a 8GB GPU and brought the server down.

#### Custom Handler

Create `custom_handler.py`:

```python
import torch
from ts.torch_handler.base_handler import BaseHandler

class CustomHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        # Load custom preprocessing artifacts, normalization stats, etc.
    
    def preprocess(self, data):
        # Your custom preprocessing
        # This is where normalization, resizing, etc. happens
        return processed_data
    
    def inference(self, data):
        with torch.no_grad():
            predictions = self.model(data)
        return predictions
    
    def postprocess(self, data):
        # Convert tensors to JSON-serializable format
        # Apply softmax, format output, etc.
        return processed_output
```

**Custom handler pro tip:** Put all your preprocessing in the handler, not in the model. This keeps your model clean and makes preprocessing changes easier. I've seen teams bake preprocessing into the model and regret it when they need to change image normalization.

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

Edge deployment is where models go to be humbled. Your beautiful model that runs at 120 FPS on your RTX 4090? It'll run at 2 FPS on a Raspberry Pi.

**The edge reality check:**
- Edge devices are 10-100x slower than your dev machine
- Memory is measured in MB, not GB
- Battery life matters (inference consumes power)
- You can't just "add more servers"

#### Edge Best Practices

1. **Use TensorFlow Lite or ONNX Runtime**
   ```bash
   neural export model.neural \
     --backend tensorflow \
     --format tflite \
     --quantize \
     --quantization-type int8
   ```

2. **Test on actual hardware**
   
   Don't deploy based on laptop benchmarks. I deployed a model that ran at 50ms on my laptop and 800ms on the target hardware. Test on the actual device.

3. **Optimize ruthlessly**
   - Quantize (int8 if possible)
   - Prune unnecessary layers
   - Use smaller architectures (MobileNet, EfficientNet, not ResNet-152)
   - Reduce input resolution if you can

4. **Mobile Integration Tips**
   - **Android**: Use TFLite Java API or ONNX Runtime Mobile
   - **iOS**: TFLite Swift API works, but Core ML is often faster on recent iPhones
   - Cache model in memory—loading from disk on each inference is slow
   - Use background threads for inference

**Horror story:** We deployed a model to Android phones that worked perfectly in testing. In production, phones were overheating and users were complaining. Turns out, we were running inference on the UI thread and blocking for 200ms per frame. Always run inference on background threads.

### Cloud Deployment

Cloud deployment gives you infinite scale (and infinite bills if you're not careful).

#### AWS SageMaker

```python
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

model = TensorFlowModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework_version='2.12'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'  # Start small, scale up
)

predictions = predictor.predict(data)
```

**SageMaker cost wisdom:** 
- `ml.m5.xlarge` costs ~$0.23/hour. Running 24/7 = $165/month for ONE instance.
- Use `ml.m5.large` for development ($0.115/hour)
- Enable auto-scaling for production
- Use Spot instances for 70% savings (if you can tolerate occasional interruptions)

**Real cost disaster:** A teammate forgot to tear down a SageMaker endpoint with 10 `ml.p3.2xlarge` instances ($30/hour each). The next morning, we had a $7,200 AWS bill. Always set budget alerts.

#### Google Cloud AI Platform

```bash
gcloud ai-platform models create my_model

gcloud ai-platform versions create v1 \
  --model my_model \
  --origin gs://bucket/saved_model \
  --runtime-version 2.12 \
  --python-version 3.9
```

**GCP gotcha:** AI Platform has cold start issues. The first request after idle can take 10-30 seconds. Keep the endpoint warm with periodic health checks if you need low latency.

#### Azure Machine Learning

```python
from azureml.core import Model
from azureml.core.webservice import AciWebservice, Webservice

model = Model.register(
    workspace=ws,
    model_path='model',
    model_name='my_model'
)

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

**Azure tip:** ACI (Container Instances) is easy but expensive for high traffic. Use AKS (Kubernetes Service) for production workloads.

### Kubernetes Deployment

Kubernetes is like the Swiss Army chainsaw of deployment—powerful, flexible, and easy to hurt yourself with.

**When to use Kubernetes:**
- You need auto-scaling based on traffic
- You're running multiple models and need orchestration
- You need zero-downtime rolling updates
- You're already using Kubernetes (don't adopt it just for ML)

**When NOT to use Kubernetes:**
- Simple, single-model deployments (use cloud provider's managed serving)
- You don't have K8s expertise in-house (operational complexity is real)
- Your traffic is predictable and low (K8s overhead isn't worth it)

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
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: model-storage
          mountPath: /models/my_model
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

**Resource limits are not optional:** Set both requests and limits. Without limits, one pod can OOM and kill your entire node. I learned this when a memory leak in preprocessing took down 3 nodes in a cascade failure.

#### Horizontal Pod Autoscaler

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

**HPA wisdom:**
- Don't scale on memory (it doesn't scale down cleanly)
- Target 70% CPU, not 90% (leave headroom for traffic spikes)
- Set minReplicas >= 2 for high availability
- Scale up is fast, scale down is slow (by design)

**Traffic spike story:** Black Friday, traffic 10xed in 5 minutes. HPA scaled from 2 to 20 pods in 3 minutes. Pods were ready but cold—first requests took 5s each. Solution: Pre-warm pods by sending dummy traffic after they start.

## Hard-Won Lessons

### Lesson 1: Always Version Your Models

Version your models like your life depends on it, because your uptime does.

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

**Why this matters:** I deployed v2 of a model that had higher accuracy but worse tail latency (p99 went from 50ms to 300ms). We needed to rollback to v1 immediately. Having versioned models saved us—we switched the symlink and recovered in 2 minutes.

### Lesson 2: Monitor Everything

Monitoring isn't optional. Track these metrics or suffer:

1. **Latency (p50, p95, p99)**: Average latency lies. p99 latency is what users complain about.
2. **Throughput**: Requests per second. Watch for sudden drops.
3. **Error rate**: Track 4xx and 5xx separately.
4. **Model predictions distribution**: Sudden shifts indicate data drift or bugs.
5. **Resource usage**: CPU, memory, GPU utilization.

```python
import time
import logging

def predict_with_monitoring(model, data):
    start = time.time()
    try:
        result = model.predict(data)
        latency = time.time() - start
        
        # Log metrics
        logging.info(f"Prediction latency: {latency:.3f}s")
        
        return result
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise
```

**Real disaster:** Our model started predicting class 0 for 95% of inputs. We didn't notice for 3 hours because we only monitored latency and error rate, not prediction distribution. Lost 3 hours of training data quality.

### Lesson 3: Test at Production Scale

Load testing isn't optional. Your model might work fine at 10 RPS and fall apart at 100 RPS.

```python
import concurrent.futures
import time

def load_test(url, data, num_requests=1000, concurrency=50):
    """
    Simulate production load.
    """
    def single_request():
        start = time.time()
        response = requests.post(url, json=data, timeout=10)
        return time.time() - start, response.status_code
    
    latencies = []
    errors = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                latency, status = future.result()
                latencies.append(latency)
                if status != 200:
                    errors += 1
            except Exception as e:
                errors += 1
    
    latencies.sort()
    print(f"p50: {latencies[len(latencies)//2]:.3f}s")
    print(f"p95: {latencies[int(len(latencies)*0.95)]:.3f}s")
    print(f"p99: {latencies[int(len(latencies)*0.99)]:.3f}s")
    print(f"Error rate: {errors/num_requests*100:.1f}%")
```

**Production surprise:** Our model handled 50 concurrent requests fine in testing. At 100 concurrent requests, latency went from 50ms to 2000ms. Turns out, TensorFlow Serving was queuing requests because we didn't set `--max_num_load_retries`. Load test at 2x your expected peak traffic.

### Lesson 4: Input Validation is Critical

Never trust user input. Validate everything.

```python
def validate_input(data, expected_shape, expected_dtype):
    """
    Validate input before inference.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(data)}")
    
    if data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")
    
    if data.dtype != expected_dtype:
        raise ValueError(f"Expected dtype {expected_dtype}, got {data.dtype}")
    
    # Check for NaN/Inf
    if not np.isfinite(data).all():
        raise ValueError("Input contains NaN or Inf values")
    
    # Check value ranges
    if data.min() < -1000 or data.max() > 1000:
        raise ValueError(f"Suspicious input values: min={data.min()}, max={data.max()}")
```

**Attack story:** Someone sent a 10000x10000 image to our API. No validation. The preprocessing step tried to load it into memory, OOM'd, and crashed the pod. Restart loop ensued. Validate input size/shape BEFORE doing anything expensive.

### Lesson 5: Quantization Requires Testing

Quantization can break your model in subtle ways. Always test quantized models against full-precision models with real data.

```python
def validate_quantized_model(original_model, quantized_model, test_data):
    """
    Compare quantized model predictions against original.
    """
    diffs = []
    
    for x, y_true in test_data:
        y_orig = original_model.predict(x)
        y_quant = quantized_model.predict(x)
        
        diff = np.abs(y_orig - y_quant).mean()
        diffs.append(diff)
    
    mean_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Max difference: {max_diff:.6f}")
    
    if mean_diff > 0.01:  # 1% threshold
        print("WARNING: Large quantization error detected!")
```

**Quantization gotcha:** We quantized a model to int8. Accuracy dropped from 94% to 91%. Acceptable? No—it was predicting wrong on edge cases that mattered most to users. Always test on your actual metrics, not just accuracy.

### Lesson 6: Cold Starts are Real

The first request after deploying/restarting is always slow. Plan for it.

**Cold start mitigation strategies:**
1. **Keep models loaded**: Don't lazy-load models on first request.
2. **Warm-up requests**: Send dummy traffic after deployment.
3. **Use connection pooling**: Don't create new connections per request.
4. **Pre-allocated memory**: TensorFlow and PyTorch can preallocate GPU memory.

```python
# Warm up model after loading
def warm_up_model(model, input_shape):
    """
    Run a few dummy inferences to warm up the model.
    """
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    for _ in range(10):
        _ = model.predict(dummy_input)
```

**Blue-green deployment gotcha:** We did blue-green deployments where green environment started cold. First requests to green took 5-10 seconds (loading model + JIT compilation). Solution: Send warm-up traffic to green before switching.

### Lesson 7: Batch When You Can

If you can batch requests, do it. Batching 10 requests together can be 5-10x faster than processing them individually.

```python
import asyncio
from collections import deque
import time

class BatchingPredictor:
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.processing = False
    
    async def predict(self, data):
        """
        Queue request and wait for batched inference.
        """
        future = asyncio.Future()
        self.queue.append((data, future))
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        self.processing = True
        await asyncio.sleep(self.max_wait_ms / 1000)
        
        batch = []
        futures = []
        
        while self.queue and len(batch) < self.max_batch_size:
            data, future = self.queue.popleft()
            batch.append(data)
            futures.append(future)
        
        if batch:
            batch_data = np.array(batch)
            predictions = self.model.predict(batch_data)
            
            for i, future in enumerate(futures):
                future.set_result(predictions[i])
        
        self.processing = False
```

**Batching tradeoff:** Batching adds latency (you wait for batch to fill) but increases throughput (more efficient GPU usage). For real-time APIs, keep `max_wait_ms` under 10ms.

### Lesson 8: Security Matters

ML models are software, and software has vulnerabilities.

**Security checklist:**
1. **Rate limiting**: Prevent abuse and DDoS.
2. **Authentication**: API keys, OAuth, or at minimum, API gateway auth.
3. **Input sanitization**: Validate and sanitize all inputs.
4. **Model encryption**: Encrypt models at rest (especially on edge devices).
5. **HTTPS/TLS**: Never deploy without encryption in transit.
6. **Don't log sensitive data**: Never log PII or sensitive predictions.

```python
from functools import wraps
import hashlib

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # Your prediction logic
    pass
```

**Security incident:** A competitor scraped our API for months, building a dataset from our predictions. We had no rate limiting. By the time we noticed, they had millions of predictions. Add rate limiting and monitoring from day one.

### Lesson 9: Plan for Model Updates

You'll update your model. Plan for zero-downtime updates.

**Update strategies:**
1. **Blue-green**: Run two environments, switch traffic. Safe but doubles resources.
2. **Rolling updates**: Update pods gradually. Less resource overhead, some traffic sees old model during rollout.
3. **Canary**: Route 5% traffic to new model, monitor, then route 100%. Best for risky updates.

```yaml
# Rolling update strategy
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Add 1 new pod before terminating old
      maxUnavailable: 0  # Never have less than desired replicas
```

**Update horror story:** We did a rolling update during peak traffic. Old pods terminated before new pods were ready. 30 seconds of 503 errors. Fix: Set `maxUnavailable: 0` and `readinessProbe` with appropriate delays.

### Lesson 10: Document Your Deployment

Future you (and your teammates) will thank you.

**Document these:**
1. Model version and training date
2. Expected input/output format and shapes
3. Preprocessing requirements (normalization, etc.)
4. Expected latency/throughput
5. Resource requirements (CPU, memory, GPU)
6. Deployment commands and configuration
7. Rollback procedure
8. Known issues and limitations

```yaml
# Include metadata in your deployment
metadata:
  labels:
    model-version: "v3.2.1"
    trained-date: "2024-01-15"
    framework: "tensorflow-2.12"
    accuracy: "94.2"
    p95-latency: "50ms"
```

## When Things Go Wrong

### Debugging Production Issues

**Issue: High latency**
- Check if model is running on correct device (GPU vs. CPU)
- Monitor resource usage (CPU, memory, GPU)
- Look for preprocessing bottlenecks (image decoding, etc.)
- Check for I/O issues (slow disk, network latency)
- Profile your code (cProfile for Python, TensorFlow profiler, PyTorch profiler)

**Issue: OOM (Out of Memory)**
- Reduce batch size
- Use quantization
- Enable model parallelism (multi-GPU)
- Check for memory leaks (especially in preprocessing)
- Monitor memory over time, not just at startup

**Issue: Wrong predictions**
- Verify preprocessing (normalization, resizing, color space)
- Check quantization errors (compare vs. full-precision model)
- Validate input data format (NHWC vs. NCHW, RGB vs. BGR)
- Test with known good examples
- Check for model/weight loading errors

**Issue: Inconsistent predictions**
- Check for randomness in model (dropout in inference mode?)
- Verify batch normalization is in inference mode
- Look for race conditions in preprocessing
- Check for numerical instability (especially with quantization)

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_prediction(model, data):
    """
    Debug helper for production issues.
    """
    logger.info(f"Input shape: {data.shape}")
    logger.info(f"Input dtype: {data.dtype}")
    logger.info(f"Input range: [{data.min():.3f}, {data.max():.3f}]")
    logger.info(f"Input mean: {data.mean():.3f}, std: {data.std():.3f}")
    
    try:
        start = time.time()
        predictions = model.predict(data)
        latency = time.time() - start
        
        logger.info(f"Prediction latency: {latency:.3f}s")
        logger.info(f"Output shape: {predictions.shape}")
        logger.info(f"Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise
```

## Final Wisdom

Deployment is where theory meets reality, and reality always wins. Start simple, monitor everything, and scale complexity only when needed. Your model doesn't need Kubernetes if you're serving 10 requests per hour.

Test at production scale before deploying. Have a rollback plan. Monitor prediction distribution, not just error rates. And for the love of all that's holy, set resource limits.

Good luck. You'll need it. But mostly, you'll need good monitoring.

## Resources

- [ONNX Documentation](https://onnx.ai/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [TensorFlow Serving Documentation](https://www.tensorflow.org/tfx/guide/serving)
- [TorchServe Documentation](https://pytorch.org/serve/)
- [Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/)
