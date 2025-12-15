# API Server Removal Guide

## Overview

As of version 0.4.0, the Neural DSL API server module (`neural/api/`) has been removed as part of the strategic refocusing effort. The API server was identified as an "alternative tool feature" that did not align with the core mission of Neural DSL: **declarative neural network definition with multi-backend compilation and automatic shape validation**.

## What Was Removed

- **Module**: `neural/api/` directory and all its contents
- **Dependencies**: API-related dependencies (FastAPI, uvicorn, Celery, Redis, etc.) removed from setup.py
- **Installation extras**: `pip install neural-dsl[api]` is no longer available
- **Configuration**: `neural/config/settings/api.py` is deprecated (retained for backward compatibility only)

## Why Was It Removed?

The API server was removed because:

1. **Not Core Functionality**: Building a REST API server is not part of the core DSL compiler mission
2. **Maintenance Burden**: Added significant dependencies and complexity
3. **Better Alternatives Exist**: Users can easily wrap Neural in their preferred API framework
4. **Focus**: Resources better spent on improving DSL parsing, shape validation, and multi-backend support

## Migration Guide

### If You Were Using the API Server

You have several alternatives:

### Option 1: Use the CLI Directly

The most straightforward approach is to use Neural's CLI commands:

```bash
# Compile DSL to Python code
neural compile model.neural --backend tensorflow --output model.py

# Run and train a model
neural run model.neural --backend pytorch

# Debug a model
neural debug model.neural --dashboard

# Export for deployment
neural export model.neural --format onnx --optimize
```

### Option 2: Use the Unified Server

The unified server provides web interfaces for debugging, building, and monitoring:

```bash
# Start the unified server (includes dashboard, no-code builder, monitoring)
neural server --host localhost --port 8050
```

This provides:
- **Debug Dashboard (NeuralDbg)**: Real-time execution monitoring
- **Model Builder**: No-code interface for building models
- **Monitoring**: Production model monitoring

### Option 3: Build Your Own API Wrapper

If you need a REST API, you can easily wrap Neural in FastAPI or Flask:

#### Example with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import CodeGenerator

app = FastAPI(title="Neural DSL API")

class CompileRequest(BaseModel):
    dsl_code: str
    backend: str = "tensorflow"

class CompileResponse(BaseModel):
    compiled_code: str
    success: bool
    message: str

@app.post("/compile", response_model=CompileResponse)
async def compile_dsl(request: CompileRequest):
    try:
        # Parse DSL
        parser = create_parser('network')
        tree = parser.parse(request.dsl_code)
        model_data = ModelTransformer().transform(tree)
        
        # Generate code
        generator = CodeGenerator(model_data, backend=request.backend)
        code = generator.generate()
        
        return CompileResponse(
            compiled_code=code,
            success=True,
            message="Compilation successful"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Neural DSL API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

To run this:

```bash
# Install FastAPI
pip install fastapi uvicorn[standard]

# Run the server
python your_api.py
```

#### Example with Flask

```python
from flask import Flask, request, jsonify
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import CodeGenerator

app = Flask(__name__)

@app.route('/compile', methods=['POST'])
def compile_dsl():
    try:
        data = request.get_json()
        dsl_code = data.get('dsl_code')
        backend = data.get('backend', 'tensorflow')
        
        # Parse and generate
        parser = create_parser('network')
        tree = parser.parse(dsl_code)
        model_data = ModelTransformer().transform(tree)
        generator = CodeGenerator(model_data, backend=backend)
        code = generator.generate()
        
        return jsonify({
            'compiled_code': code,
            'success': True,
            'message': 'Compilation successful'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

To run this:

```bash
# Install Flask
pip install flask

# Run the server
python your_api.py
```

### Option 4: Python Library Integration

You can also use Neural as a Python library directly in your application:

```python
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import CodeGenerator
from neural.shape_propagation.shape_propagator import ShapePropagator

def compile_neural_dsl(dsl_code: str, backend: str = "tensorflow") -> str:
    """Compile Neural DSL code to target backend."""
    parser = create_parser('network')
    tree = parser.parse(dsl_code)
    model_data = ModelTransformer().transform(tree)
    
    # Validate shapes
    propagator = ShapePropagator(model_data)
    propagator.propagate()
    
    # Generate code
    generator = CodeGenerator(model_data, backend=backend)
    return generator.generate()

# Use in your application
dsl_code = """
network SimpleMLP {
    input: (784,)
    layers:
        Dense(128, "relu")
        Dropout(0.2)
        Dense(10, "softmax")
}
"""

tensorflow_code = compile_neural_dsl(dsl_code, backend="tensorflow")
pytorch_code = compile_neural_dsl(dsl_code, backend="pytorch")
```

## Benefits of Removal

1. **Reduced Dependencies**: ~12 fewer required packages (FastAPI, uvicorn, Celery, Redis, etc.)
2. **Faster Installation**: Smaller package size and faster `pip install`
3. **Clearer Focus**: Neural DSL is about DSL compilation, not API serving
4. **More Flexibility**: Users can build APIs exactly how they want with their preferred tools
5. **Easier Maintenance**: Less code to maintain, test, and document

## Need Help?

If you have questions about migrating from the API server or building your own wrapper:

- **Discord**: [Join our server](https://discord.gg/KFku4KvS)
- **GitHub Discussions**: [Ask a question](https://github.com/Lemniscate-world/Neural/discussions)
- **GitHub Issues**: [Report a problem](https://github.com/Lemniscate-world/Neural/issues)

## See Also

- [CHANGELOG.md](../CHANGELOG.md) - v0.4.0 release notes
- [REFOCUS.md](../REFOCUS.md) - Strategic refocusing rationale
- [README.md](../README.md) - Main documentation
- [examples/](../examples/) - DSL examples
