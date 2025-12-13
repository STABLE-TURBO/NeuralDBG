# Integration Guide - Neural Aquarium with Neural DSL Backend

This guide explains how to integrate the Neural Aquarium visual designer with the Neural DSL Python backend.

## Architecture

```
┌─────────────────────────────────────┐
│   Neural Aquarium (React/TS)       │
│   Port: 3000                        │
│   - Visual Designer                 │
│   - DSL Code Editor                 │
│   - Layer Palette                   │
└─────────────┬───────────────────────┘
              │ HTTP/REST API
              │
┌─────────────▼───────────────────────┐
│   Neural DSL Backend (Python)       │
│   Port: 8051                        │
│   - DSL Parser                      │
│   - Code Generator                  │
│   - Shape Propagation               │
│   - Model Compilation               │
└─────────────────────────────────────┘
```

## Setup

### 1. Frontend (Neural Aquarium)

```bash
cd neural/aquarium
npm install
npm run dev
```

Runs on http://localhost:3000

### 2. Backend (Neural DSL)

```bash
cd neural
python -m venv .venv
.\.venv\Scripts\Activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -e ".[full]"
python no_code/no_code.py
```

Runs on http://localhost:8051

### 3. Configure API URL

Create `.env` file in `neural/aquarium/`:

```env
VITE_API_URL=http://localhost:8051
```

## API Endpoints

The frontend expects these endpoints on the backend:

### POST /api/compile

Compile DSL code to target backend (TensorFlow/PyTorch/ONNX).

**Request:**
```json
{
  "dsl": "network MyModel { ... }",
  "backend": "tensorflow"
}
```

**Response:**
```json
{
  "success": true,
  "code": "import tensorflow as tf...",
  "model_summary": { ... }
}
```

### POST /api/validate

Validate DSL syntax and semantics.

**Request:**
```json
{
  "dsl": "network MyModel { ... }"
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": []
}
```

### POST /api/parse

Parse DSL to AST/model structure.

**Request:**
```json
{
  "dsl": "network MyModel { ... }"
}
```

**Response:**
```json
{
  "model": {
    "name": "MyModel",
    "input_shape": [null, 28, 28, 1],
    "layers": [ ... ]
  }
}
```

### POST /api/export

Export model to file format (ONNX, SavedModel, etc.).

**Request:**
```json
{
  "dsl": "network MyModel { ... }",
  "backend": "onnx",
  "format": "onnx"
}
```

**Response:**
```json
{
  "success": true,
  "file_path": "/path/to/model.onnx",
  "file_size": 12345
}
```

## Backend Implementation Example

Add these routes to Flask/Dash backend (`neural/no_code/no_code.py`):

```python
from flask import Flask, request, jsonify
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code

app = Flask(__name__)

@app.route('/api/compile', methods=['POST'])
def compile_model():
    data = request.json
    dsl_code = data.get('dsl', '')
    backend = data.get('backend', 'tensorflow')
    
    try:
        # Parse DSL
        parser = create_parser()
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model = transformer.transform(tree)
        
        # Generate code
        code = generate_code(model, backend)
        
        return jsonify({
            'success': True,
            'code': code,
            'model_summary': {
                'name': model.get('name', 'MyModel'),
                'num_layers': len(model.get('layers', [])),
                'parameters': calculate_parameters(model)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/validate', methods=['POST'])
def validate_dsl():
    data = request.json
    dsl_code = data.get('dsl', '')
    
    try:
        parser = create_parser()
        tree = parser.parse(dsl_code)
        
        # Validate semantics
        errors = []
        warnings = []
        
        # Add validation logic here
        
        return jsonify({
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        })
    except Exception as e:
        return jsonify({
            'valid': False,
            'errors': [str(e)],
            'warnings': []
        }), 200

@app.route('/api/parse', methods=['POST'])
def parse_dsl():
    data = request.json
    dsl_code = data.get('dsl', '')
    
    try:
        parser = create_parser()
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model = transformer.transform(tree)
        
        return jsonify({
            'model': model
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=8051)
```

## Frontend Usage

Import API functions in React components:

```typescript
import { compileModel, validateDSL } from '../utils/api';

// In your component
const handleCompile = async () => {
  try {
    const result = await compileModel(dslCode);
    console.log('Compiled code:', result.code);
  } catch (error) {
    console.error('Compilation error:', error);
  }
};

const handleValidate = async () => {
  try {
    const result = await validateDSL(dslCode);
    if (!result.valid) {
      console.error('Validation errors:', result.errors);
    }
  } catch (error) {
    console.error('Validation error:', error);
  }
};
```

## CORS Configuration

Backend needs CORS enabled for frontend access:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
```

Install CORS:
```bash
pip install flask-cors
```

## Development Workflow

1. **Start Backend:**
   ```bash
   cd neural
   python no_code/no_code.py
   ```

2. **Start Frontend:**
   ```bash
   cd neural/aquarium
   npm run dev
   ```

3. **Test Integration:**
   - Create a network in visual designer
   - Click "Compile" (to be added)
   - Backend parses DSL and generates code
   - Frontend displays result

## Production Deployment

### Frontend Build

```bash
cd neural/aquarium
npm run build
```

Outputs to `dist/` directory.

### Serve with Backend

```python
from flask import send_from_directory

@app.route('/')
def serve_frontend():
    return send_from_directory('aquarium/dist', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('aquarium/dist', path)
```

### Docker Setup

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy and build frontend
COPY neural/aquarium /app/aquarium
WORKDIR /app/aquarium
RUN npm install && npm run build

# Copy backend code
COPY neural /app/neural
WORKDIR /app

# Run application
EXPOSE 8051
CMD ["python", "neural/no_code/no_code.py"]
```

## Real-time Sync Implementation

For bi-directional sync, the frontend already implements:

1. **Visual → DSL:**
   - Changes in designer trigger `nodesToDSL()`
   - Updates code editor in real-time

2. **DSL → Visual:**
   - Code editor changes trigger `parseDSLToNodes()`
   - Updates visual designer

3. **Optional: Backend Validation:**
   - Debounce code changes
   - Call `/api/validate` endpoint
   - Show validation errors in UI

```typescript
const validateCode = useCallback(
  debounce(async (code: string) => {
    try {
      const result = await validateDSL(code);
      setValidationErrors(result.errors);
      setValidationWarnings(result.warnings);
    } catch (error) {
      console.error('Validation error:', error);
    }
  }, 500),
  []
);

useEffect(() => {
  validateCode(dslCode);
}, [dslCode, validateCode]);
```

## Extending the Integration

### Add Model Templates

Backend provides templates:

```python
@app.route('/api/templates', methods=['GET'])
def get_templates():
    return jsonify({
        'templates': [
            {
                'name': 'MNIST CNN',
                'description': 'Convolutional network for MNIST',
                'dsl': '...'
            },
            # More templates...
        ]
    })
```

Frontend loads templates:

```typescript
const loadTemplates = async () => {
  const response = await fetch(`${API_CONFIG.baseURL}/api/templates`);
  const data = await response.json();
  setTemplates(data.templates);
};
```

### Add Shape Inference

Backend calculates shapes:

```python
@app.route('/api/infer_shapes', methods=['POST'])
def infer_shapes():
    data = request.json
    model = data.get('model')
    
    from neural.shape_propagation.shape_propagator import ShapePropagator
    propagator = ShapePropagator()
    
    shapes = []
    current_shape = model['input_shape']
    
    for layer in model['layers']:
        current_shape = propagator.propagate(current_shape, layer)
        shapes.append({
            'layer': layer['type'],
            'output_shape': current_shape
        })
    
    return jsonify({'shapes': shapes})
```

### Add Real-time Collaboration

Use WebSockets for multi-user editing:

```python
from flask_socketio import SocketIO, emit

socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

@socketio.on('model_update')
def handle_model_update(data):
    # Broadcast to all clients except sender
    emit('model_updated', data, broadcast=True, include_self=False)
```

Frontend:

```typescript
import io from 'socket.io-client';

const socket = io('http://localhost:8051');

socket.on('model_updated', (data) => {
  // Update local state
  setNodes(data.nodes);
  setEdges(data.edges);
});
```

## Troubleshooting

### CORS Errors

**Problem:** "Access to fetch blocked by CORS policy"
**Solution:** Enable CORS in Flask backend

### Connection Refused

**Problem:** "Failed to fetch"
**Solution:** 
- Ensure backend is running on port 8051
- Check `.env` file has correct API URL

### Parse Errors

**Problem:** Backend can't parse DSL
**Solution:**
- Verify DSL syntax matches grammar
- Check backend parser version
- Enable debug logging

## Testing Integration

Create integration tests:

```typescript
// frontend test
describe('API Integration', () => {
  it('should compile model', async () => {
    const dsl = 'network Test { ... }';
    const result = await compileModel(dsl);
    expect(result.success).toBe(true);
  });
});
```

```python
# backend test
def test_compile_endpoint():
    response = client.post('/api/compile', json={
        'dsl': 'network Test { ... }',
        'backend': 'tensorflow'
    })
    assert response.status_code == 200
    assert response.json['success'] == True
```

## Performance Optimization

1. **Caching:** Cache compilation results
2. **Debouncing:** Debounce validation calls
3. **Lazy Loading:** Load templates on demand
4. **Compression:** Enable gzip compression
5. **CDN:** Serve static assets from CDN

## Security Considerations

1. **Validation:** Validate all DSL input
2. **Sanitization:** Sanitize generated code
3. **Rate Limiting:** Limit API requests
4. **Authentication:** Add auth for production
5. **HTTPS:** Use HTTPS in production
