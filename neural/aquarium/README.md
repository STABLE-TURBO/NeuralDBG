# Neural Aquarium

Aquarium is the backend bridge for Neural DSL, providing a comprehensive API for DSL parsing, shape propagation, code generation, and training job management.

## Overview

The Aquarium backend bridge exposes the core Neural DSL functionality through a modern REST API with WebSocket support, enabling:

- **DSL Parsing**: Convert Neural DSL code to structured model data
- **Shape Analysis**: Propagate and validate tensor shapes through models
- **Code Generation**: Generate TensorFlow, PyTorch, or ONNX code
- **Training Management**: Run and monitor training jobs in isolated processes
- **Real-time Updates**: WebSocket support for live job monitoring

## Quick Start

### Installation

Install with API dependencies:

```bash
pip install -e ".[api]"
```

### Start the Server

```bash
python -m neural.aquarium.backend.run
```

Or with custom configuration:

```bash
python -m neural.aquarium.backend.cli serve --host 0.0.0.0 --port 8080
```

### Use the Client Library

```python
from neural.aquarium.backend.client import create_client

# Create client
client = create_client("http://localhost:8000")

# Parse DSL
dsl_code = """
network MyModel {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=3)
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(64)
        Output(10)
    optimizer: Adam(learning_rate=0.001)
    loss: categorical_crossentropy
}
"""

result = client.compile(dsl_code, backend="tensorflow")
print(result["code"])
```

## Architecture

```
neural/aquarium/backend/
├── server.py              # FastAPI application
├── process_manager.py     # Training job management
├── websocket_manager.py   # WebSocket connections
├── client.py             # Python client library
├── config.py             # Configuration settings
├── middleware.py         # Request logging, auth
├── utils.py              # Helper functions
├── models.py             # Data models
├── cli.py                # Command-line interface
├── run.py                # Server runner
├── examples.py           # Usage examples
├── test_server.py        # Tests
├── Dockerfile            # Docker configuration
└── README.md             # Documentation
```

## API Reference

See [backend/README.md](backend/README.md) for complete API documentation.

## Features

### DSL Compilation Pipeline

1. **Parse**: Convert DSL to structured data
2. **Validate**: Check model structure and parameters
3. **Propagate**: Analyze tensor shapes through layers
4. **Generate**: Create backend-specific code
5. **Execute**: Run training in isolated process

### Process Management

- Start/stop training jobs
- Monitor job status and output
- Real-time logging via WebSocket
- Automatic cleanup on completion

### Shape Analysis

- Tensor shape propagation
- Performance estimation (FLOPs, memory)
- Issue detection
- Optimization suggestions

## Usage Examples

### Complete Workflow

```python
from neural.aquarium.backend.client import create_client

client = create_client()

# Compile DSL to code
result = client.compile(dsl_code, backend="pytorch")

# Start training job
job = client.start_job(result["code"], job_name="training")

# Wait for completion
final_status = client.wait_for_job(job["job_id"])
print(f"Training {final_status['status']}")
```

### Shape Analysis

```python
# Parse model
parse_result = client.parse(dsl_code)
model_data = parse_result["model_data"]

# Analyze shapes
shapes = client.propagate_shapes(model_data, framework="tensorflow")

print("Shape History:")
for entry in shapes["shape_history"]:
    print(f"  {entry['layer']}: {entry['output_shape']}")

if shapes["issues"]:
    print(f"\nIssues: {shapes['issues']}")

if shapes["optimizations"]:
    print(f"\nOptimizations: {shapes['optimizations']}")
```

### WebSocket Monitoring

```python
import asyncio

async def monitor():
    # Start job
    result = client.compile_and_run(dsl_code, wait=False)
    job_id = result["job_id"]
    
    # Watch via WebSocket
    def callback(data):
        print(f"Status: {data['status']}")
        if data.get('output'):
            print(f"Output: {data['output']}")
    
    await client.watch_job(job_id, callback=callback)

asyncio.run(monitor())
```

## Configuration

Environment variables (prefix with `NEURAL_`):

- `NEURAL_HOST`: Server host (default: 0.0.0.0)
- `NEURAL_PORT`: Server port (default: 8000)
- `NEURAL_LOG_LEVEL`: Logging level (default: INFO)
- `NEURAL_API_KEY`: Optional API key for authentication
- `NEURAL_MAX_JOB_OUTPUT_LINES`: Max output lines per job (default: 1000)

## Docker Deployment

Build and run with Docker:

```bash
cd neural/aquarium/backend
docker build -t neural-backend .
docker run -p 8000:8000 neural-backend
```

Or use Docker Compose:

```bash
docker-compose up
```

## Testing

Run tests:

```bash
pytest neural/aquarium/backend/test_server.py -v
```

## Examples

See complete examples:

```bash
python neural/aquarium/backend/examples.py
```

## Integration

The backend integrates with:

- `neural.parser.parser`: DSL parsing
- `neural.code_generation.code_generator`: Code generation
- `neural.shape_propagation.shape_propagator`: Shape analysis

## Security

- CORS enabled (configure for production)
- Optional API key authentication
- Process isolation for training jobs
- Automatic resource cleanup

## Performance

- Async job execution
- Non-blocking WebSocket updates
- Efficient process management
- Configurable output buffering

## License

MIT License - see project root for details.
