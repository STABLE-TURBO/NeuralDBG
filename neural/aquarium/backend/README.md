# Neural DSL Backend Bridge

A FastAPI-based backend bridge for Neural DSL that provides REST API endpoints and WebSocket support for DSL parsing, shape propagation, code generation, compilation, and training job management.

## Features

- **DSL Parsing**: Parse Neural DSL code into structured model data
- **Shape Propagation**: Analyze and propagate tensor shapes through the model
- **Code Generation**: Generate backend-specific code (TensorFlow, PyTorch, ONNX)
- **Compilation**: Complete pipeline from DSL to executable code
- **Training Job Management**: Run and manage training jobs in separate processes
- **WebSocket Support**: Real-time updates and job monitoring
- **CORS Enabled**: Ready for frontend integration

## Installation

The backend bridge requires FastAPI and uvicorn. Install with the API extras:

```bash
pip install -e ".[api]"
```

Or install all dependencies:

```bash
pip install -e ".[full]"
```

## Running the Server

### Using Python

```bash
python -m neural.aquarium.backend.run
```

### Using the CLI

```bash
python -m neural.aquarium.backend.cli serve
```

With custom options:

```bash
python -m neural.aquarium.backend.cli serve --host 0.0.0.0 --port 8080 --reload
```

### Using uvicorn directly

```bash
uvicorn neural.aquarium.backend.server:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000` by default.

## API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Health & Info

- `GET /` - Root endpoint, returns service info
- `GET /health` - Health check endpoint

### DSL Operations

- `POST /api/parse` - Parse DSL code
- `POST /api/shape-propagation` - Propagate shapes through model
- `POST /api/generate-code` - Generate backend-specific code
- `POST /api/compile` - Complete compilation pipeline

### Training Jobs

- `POST /api/jobs/start` - Start a new training job
- `GET /api/jobs/{job_id}/status` - Get job status and output
- `POST /api/jobs/{job_id}/stop` - Stop a running job
- `GET /api/jobs` - List all jobs

### WebSocket

- `WS /ws` - General WebSocket endpoint
- `WS /ws/jobs/{job_id}` - Job-specific updates

## Usage Examples

### Parse DSL Code

```python
import requests

dsl_code = """
network SimpleModel {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=3, activation="relu")
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(128, activation="relu")
        Output(10, activation="softmax")
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}
"""

response = requests.post(
    "http://localhost:8000/api/parse",
    json={"dsl_code": dsl_code}
)

model_data = response.json()["model_data"]
```

### Generate Code

```python
response = requests.post(
    "http://localhost:8000/api/generate-code",
    json={
        "model_data": model_data,
        "backend": "tensorflow"
    }
)

code = response.json()["code"]
print(code)
```

### Complete Compilation

```python
response = requests.post(
    "http://localhost:8000/api/compile",
    json={
        "dsl_code": dsl_code,
        "backend": "pytorch"
    }
)

result = response.json()
code = result["code"]
shape_history = result["shape_history"]
```

### Start a Training Job

```python
training_code = """
import torch
print("Training started...")
# Your training code here
"""

response = requests.post(
    "http://localhost:8000/api/jobs/start",
    json={
        "code": training_code,
        "job_name": "my_training_job"
    }
)

job_id = response.json()["job_id"]
```

### Monitor Job Status

```python
response = requests.get(f"http://localhost:8000/api/jobs/{job_id}/status")
status = response.json()
print(f"Status: {status['status']}")
print(f"Output: {status['output']}")
```

### WebSocket Example

```python
import asyncio
import websockets
import json

async def monitor_job(job_id):
    uri = f"ws://localhost:8000/ws/jobs/{job_id}"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Status: {data['status']}")
            if data['status'] in ['completed', 'failed']:
                break

asyncio.run(monitor_job(job_id))
```

## Configuration

Configuration can be set via environment variables or a `.env` file:

```env
NEURAL_HOST=0.0.0.0
NEURAL_PORT=8000
NEURAL_LOG_LEVEL=INFO
NEURAL_MAX_JOB_OUTPUT_LINES=1000
NEURAL_API_KEY=your_secret_key
```

## Architecture

- **server.py**: FastAPI application with all endpoints
- **process_manager.py**: Manages training job processes
- **websocket_manager.py**: Handles WebSocket connections
- **config.py**: Application settings
- **middleware.py**: Request logging and API key validation
- **utils.py**: Helper functions
- **cli.py**: Command-line interface
- **run.py**: Simple server runner

## Integration

The backend integrates with:

- **neural.parser.parser**: DSL parsing using `create_parser()` and `ModelTransformer`
- **neural.code_generation.code_generator**: Code generation using `generate_code()`
- **neural.shape_propagation.shape_propagator**: Shape analysis using `ShapePropagator`

## Error Handling

All endpoints return structured error responses:

```json
{
    "success": false,
    "error": "Error message details"
}
```

## Security

- CORS is enabled by default for all origins (configure in production)
- Optional API key authentication via `X-API-Key` header
- Job isolation via separate processes
- Temporary files are cleaned up automatically

## Development

Run with auto-reload:

```bash
python -m neural.aquarium.backend.cli serve --reload
```

## Production

For production deployment, use a production ASGI server configuration:

```bash
uvicorn neural.aquarium.backend.server:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use with gunicorn:

```bash
gunicorn neural.aquarium.backend.server:app -w 4 -k uvicorn.workers.UvicornWorker
```
