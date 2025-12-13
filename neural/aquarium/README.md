# Neural Aquarium

A comprehensive suite for Neural DSL, providing both a visual network designer and real-time shape propagation visualization, plus a backend API bridge for DSL parsing, code generation, and training job management.

## Features

### Visual Network Designer

A modern, interactive visual network designer with drag-and-drop layer palette, real-time connection validation, and bi-directional sync with DSL code.

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

### Real-Time Shape Propagation Panel

Live visualization showing layer-by-layer tensor shape changes, integrating with `neural/shape_propagation/shape_propagator.py`.

#### Key Features

- **Real-time Updates**: Live shape propagation with configurable auto-refresh (100-5000ms)
- **Interactive D3.js Visualization**: Click and explore shape flow diagrams with smooth animations
- **Plotly Charts**: Alternative visualization using Plotly for memory, parameters, and tensor size evolution
- **Error Highlighting**: Automatic detection and highlighting of shape mismatches (red nodes, dashed lines)
- **Detailed Tooltips**: Hover over layers to see:
  - Input/Output shapes
  - Parameters count
  - FLOPs
  - Memory usage
- **Layer Details Panel**: Click any layer to see comprehensive transformation details
- **Error Messages**: Clear error messages with expected vs actual shapes
- **Table View**: Tabular representation of all layers with sortable columns

### Backend API Bridge

The Aquarium backend bridge exposes the core Neural DSL functionality through a modern REST API with WebSocket support, enabling:

- **DSL Parsing**: Convert Neural DSL code to structured model data
- **Shape Analysis**: Propagate and validate tensor shapes through models
- **Code Generation**: Generate TensorFlow, PyTorch, or ONNX code
- **Training Management**: Run and monitor training jobs in isolated processes
- **Real-time Updates**: WebSocket support for live job monitoring

## Installation

### Frontend Setup

```bash
cd neural/aquarium
npm install
```

### Backend Setup

Install with API dependencies:

```bash
pip install -e ".[api]"
```

Or ensure you have the required packages:

```bash
pip install flask flask-cors
```

## Running

### Start Backend API

```bash
cd neural/aquarium
python api/shape_api.py
```

The API runs on `http://localhost:5002`

### Start Frontend (Development)

```bash
cd neural/aquarium
npm run dev
```

Opens on `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## Usage

### Network Designer

1. **Add Layers**: Drag layers from the left palette onto the canvas, or click to add at random position
2. **Connect Layers**: Drag from a layer's bottom handle to another layer's top handle
3. **Edit Properties**: Click a layer to select it and edit parameters in the right panel
4. **View Code**: Click "Show Code" to see the generated Neural DSL code
5. **Edit Code**: Make changes in the code editor and see them reflected in the visual design
6. **Auto Layout**: Click "Auto Layout" to organize layers vertically
7. **Clear Canvas**: Click "Clear" to start fresh

### Shape Propagation Panel

#### Automatic Mode

The panel automatically fetches shape propagation data at regular intervals (default: 1000ms).

#### Manual Mode

1. Uncheck "Auto-refresh" to stop automatic updates
2. Click "Refresh Now" to fetch data on demand

#### Interacting with Visualization

**D3.js View:**
- Hover over nodes to see detailed tooltips
- Click nodes to see full layer details in the side panel
- Red nodes indicate errors
- Dashed red lines show shape mismatches between layers

**Plotly View:**
- Use built-in Plotly controls to zoom, pan, and export
- Hover over data points for detailed information
- Three synchronized charts show different aspects of shape propagation

#### API Usage

**Propagate a Model:**

```bash
curl -X POST http://localhost:5002/api/shape-propagation/propagate \
  -H "Content-Type: application/json" \
  -d '{
    "input_shape": [null, 28, 28, 1],
    "framework": "tensorflow",
    "layers": [
      {
        "type": "Conv2D",
        "params": {
          "filters": 32,
          "kernel_size": [3, 3],
          "padding": "same"
        }
      },
      {
        "type": "MaxPooling2D",
        "params": {
          "pool_size": [2, 2]
        }
      }
    ]
  }'
```

**Get Shape History:**

```bash
curl http://localhost:5002/api/shape-propagation
```

## Architecture

### Frontend Architecture
```
src/
├── components/
│   ├── designer/               # Visual network designer
│   │   ├── NetworkDesigner.tsx
│   │   ├── LayerNode.tsx
│   │   ├── LayerPalette.tsx
│   │   ├── PropertiesPanel.tsx
│   │   └── CodeEditor.tsx
│   └── shapes/                 # Shape propagation panel
│       ├── ShapePropagationPanel.jsx
│       ├── ShapePropagationPlotly.jsx
│       └── index.js
├── data/
│   └── layerDefinitions.ts
├── types/
│   └── index.ts
└── utils/
    ├── dslParser.ts
    ├── connectionValidator.ts
    ├── shapeUtils.js
    └── api.ts
```

### Backend Architecture
```
neural/aquarium/
├── api/
│   ├── __init__.py
│   └── shape_api.py            # Flask REST API
├── backend/
│   ├── server.py              # FastAPI application
│   ├── process_manager.py     # Training job management
│   └── websocket_manager.py   # WebSocket connections
└── examples/
    └── example_usage.py
```

## Technologies

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **ReactFlow 11** - Visual graph editor
- **Monaco Editor** - Code editor (VSCode engine)
- **D3.js 7.8+** - SVG visualization
- **Plotly.js 2.26+** - Interactive charts
- **Vite** - Build tool

### Backend
- **Flask** - REST API framework
- **FastAPI** - Modern web framework
- **flask-cors** - CORS support
- **ShapePropagator** - Shape propagation engine

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

## API Reference

See [backend/README.md](backend/README.md) for complete API documentation.

## Testing

```bash
# Frontend tests
npm test

# Backend tests
python -m pytest tests/
```

## Configuration

Environment variables (prefix with `NEURAL_`):

- `NEURAL_HOST`: Server host (default: 0.0.0.0)
- `NEURAL_PORT`: Server port (default: 8000)
- `NEURAL_LOG_LEVEL`: Logging level (default: INFO)
- `NEURAL_API_KEY`: Optional API key for authentication

## Docker Deployment

Build and run with Docker:

```bash
cd neural/aquarium/backend
docker build -t neural-backend .
docker run -p 8000:8000 neural-backend
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Security

- CORS enabled (configure for production)
- Optional API key authentication
- Process isolation for training jobs
- Automatic resource cleanup

## License

MIT License - see project root for details.
