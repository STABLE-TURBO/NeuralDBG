# Neural Aquarium

A comprehensive suite for Neural DSL, providing both a visual network designer and backend API bridge for DSL parsing, shape propagation, code generation, and training job management.

## Visual Network Designer

A modern, interactive visual network designer with drag-and-drop layer palette, real-time connection validation, and bi-directional sync with DSL code.

### Features

#### 🎨 Drag-and-Drop Layer Palette
- Layers organized by category (Convolutional, Pooling, Core, Recurrent, Attention, etc.)
- Search functionality to quickly find layers
- Color-coded layers by category
- Visual icons for each layer type

#### 🖼️ Interactive Canvas
- React Flow-based visual editor
- Drag layers from palette onto canvas
- Connect layers by dragging between handles
- Pan and zoom controls
- Mini-map for navigation
- Auto-layout feature

#### 📊 Layer Node Components
- Display layer type and category
- Show layer parameters (up to 3 visible, expandable)
- Real-time output shape propagation
- Color-coded borders by category
- Hover effects and selection states

#### ✅ Connection Validation
- Prevents incompatible layer connections
- Cycle detection
- Shape compatibility checking
- Visual feedback on invalid connections
- Prevents duplicate connections

#### 🔄 Bi-directional DSL Sync
- Real-time DSL code generation from visual design
- Parse DSL code to visual representation
- Monaco editor for code editing
- Instant synchronization between views

#### ⚙️ Properties Panel
- Edit selected layer parameters
- Specialized inputs for different parameter types
- Category and output shape display
- Layer description and documentation

### Installation

```bash
cd neural/aquarium
npm install
```

### Development

```bash
npm run dev
```

Opens on http://localhost:3000

### Build

```bash
npm run build
```

### Usage

1. **Add Layers**: Drag layers from the left palette onto the canvas, or click to add at random position
2. **Connect Layers**: Drag from a layer's bottom handle to another layer's top handle
3. **Edit Properties**: Click a layer to select it and edit parameters in the right panel
4. **View Code**: Click "Show Code" to see the generated Neural DSL code
5. **Edit Code**: Make changes in the code editor and see them reflected in the visual design
6. **Auto Layout**: Click "Auto Layout" to organize layers vertically
7. **Clear Canvas**: Click "Clear" to start fresh

## Backend API Bridge

The Aquarium backend bridge exposes the core Neural DSL functionality through a modern REST API with WebSocket support, enabling:

- **DSL Parsing**: Convert Neural DSL code to structured model data
- **Shape Analysis**: Propagate and validate tensor shapes through models
- **Code Generation**: Generate TensorFlow, PyTorch, or ONNX code
- **Training Management**: Run and monitor training jobs in isolated processes
- **Real-time Updates**: WebSocket support for live job monitoring

### Quick Start

#### Installation

Install with API dependencies:

```bash
pip install -e ".[api]"
```

#### Start the Server

```bash
python -m neural.aquarium.backend.run
```

Or with custom configuration:

```bash
python -m neural.aquarium.backend.cli serve --host 0.0.0.0 --port 8080
```

#### Use the Client Library

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

### Frontend Architecture
```
src/
├── components/
│   └── designer/
│       ├── NetworkDesigner.tsx     # Main designer component
│       ├── LayerNode.tsx           # Custom node component
│       ├── LayerPalette.tsx        # Layer selection sidebar
│       ├── PropertiesPanel.tsx     # Layer properties editor
│       └── CodeEditor.tsx          # DSL code editor
├── data/
│   └── layerDefinitions.ts         # Layer metadata and defaults
├── types/
│   └── index.ts                    # TypeScript type definitions
└── utils/
    ├── dslParser.ts                # DSL <-> Node conversion
    └── connectionValidator.ts      # Connection validation logic
```

### Backend Architecture
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

## Technologies

### Frontend
- **React 18** - UI framework
- **ReactFlow 11** - Visual graph editor
- **Monaco Editor** - Code editor (VSCode engine)
- **TypeScript** - Type safety
- **Vite** - Build tool

### Backend
- **FastAPI** - Modern web framework
- **WebSockets** - Real-time communication
- **Python 3.8+** - Backend language

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

## Layer Categories

- **Convolutional**: Conv1D, Conv2D, Conv3D, SeparableConv2D, etc.
- **Pooling**: MaxPooling, AveragePooling, GlobalPooling variants
- **Core**: Dense, Flatten, Reshape, Permute, Lambda
- **Recurrent**: LSTM, GRU, SimpleRNN, Bidirectional
- **Attention**: MultiHeadAttention, Attention
- **Normalization**: BatchNormalization, LayerNormalization, GroupNormalization
- **Regularization**: Dropout, SpatialDropout, GaussianNoise
- **Activation**: ReLU, LeakyReLU, Softmax, Sigmoid, Tanh
- **Embedding**: Embedding

## Connection Rules

The designer enforces several validation rules:
- No cycles allowed
- Each layer (except merge layers) can have only one input
- Flattening layers cannot connect to 2D layers
- Recurrent layers require compatible input shapes
- Shape compatibility is validated in real-time

## Shape Propagation

Output shapes are automatically calculated and displayed:
- Input shape defined in Input node
- Shapes propagate through the network
- Visible in each layer node
- Used for connection validation

## Code Synchronization

Changes in either view are immediately reflected:
- **Visual → Code**: Adding/removing/editing layers updates DSL
- **Code → Visual**: Parsing DSL creates/updates visual nodes
- Topological sorting ensures correct layer order
- Parameter values synchronized automatically

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
