"""
API endpoints for real-time shape propagation integration.
Provides REST endpoints to fetch shape propagation data from the ShapePropagator.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.exceptions import ShapeException, ShapeMismatchError, InvalidShapeError


app = Flask(__name__)
CORS(app)

propagator = None
current_model = None


def initialize_propagator(model_data=None):
    global propagator, current_model
    
    propagator = ShapePropagator(debug=True)
    
    if model_data:
        current_model = model_data
        

@app.route('/api/shape-propagation', methods=['GET'])
def get_shape_propagation():
    global propagator
    
    if not propagator:
        return jsonify({
            'shape_history': [],
            'errors': [],
            'message': 'No propagator initialized'
        }), 200
    
    shape_history = []
    for i, (layer_name, output_shape) in enumerate(propagator.shape_history):
        input_shape = propagator.shape_history[i-1][1] if i > 0 else None
        
        if i < len(propagator.execution_trace):
            trace = propagator.execution_trace[i]
            flops = trace.get('flops', 0)
            memory = trace.get('memory', 0)
        else:
            flops = 0
            memory = 0
        
        shape_history.append({
            'layer_name': layer_name,
            'layer': layer_name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'parameters': calculate_parameters(layer_name, input_shape, output_shape),
            'flops': flops,
            'memory': memory,
            'error': False,
            'error_message': None
        })
    
    errors = []
    if hasattr(propagator, 'issues'):
        for issue in propagator.issues:
            errors.append({
                'layer': issue.get('layer', 'Unknown'),
                'message': issue.get('message', 'Unknown error'),
                'expected_shape': issue.get('expected_shape'),
                'actual_shape': issue.get('actual_shape')
            })
    
    return jsonify({
        'shape_history': shape_history,
        'errors': errors,
        'optimizations': propagator.optimizations if hasattr(propagator, 'optimizations') else []
    })


@app.route('/api/shape-propagation/propagate', methods=['POST'])
def propagate_model():
    global propagator
    
    data = request.json
    
    if not data or 'input_shape' not in data or 'layers' not in data:
        return jsonify({
            'error': 'Invalid request. Requires input_shape and layers.'
        }), 400
    
    initialize_propagator()
    
    input_shape = tuple(data['input_shape'])
    layers = data['layers']
    framework = data.get('framework', 'tensorflow')
    
    errors = []
    current_shape = input_shape
    
    try:
        for layer in layers:
            try:
                current_shape = propagator.propagate(current_shape, layer, framework)
            except (ShapeException, ShapeMismatchError, InvalidShapeError) as e:
                errors.append({
                    'layer': layer.get('type', 'Unknown'),
                    'message': str(e),
                    'expected_shape': getattr(e, 'expected_shape', None),
                    'actual_shape': getattr(e, 'actual_shape', None)
                })
                if hasattr(e, 'output_shape'):
                    current_shape = e.output_shape
    
    except Exception as e:
        return jsonify({
            'error': f'Propagation failed: {str(e)}'
        }), 500
    
    propagator.detect_issues()
    propagator.suggest_optimizations()
    
    return get_shape_propagation()


@app.route('/api/shape-propagation/reset', methods=['POST'])
def reset_propagator():
    global propagator
    initialize_propagator()
    return jsonify({'message': 'Propagator reset successfully'})


@app.route('/api/shape-propagation/layer/<int:layer_id>', methods=['GET'])
def get_layer_details(layer_id):
    global propagator
    
    if not propagator or layer_id >= len(propagator.shape_history):
        return jsonify({'error': 'Layer not found'}), 404
    
    layer_name, output_shape = propagator.shape_history[layer_id]
    input_shape = propagator.shape_history[layer_id-1][1] if layer_id > 0 else None
    
    trace_data = {}
    if layer_id < len(propagator.execution_trace):
        trace = propagator.execution_trace[layer_id]
        trace_data = {
            'execution_time': trace.get('execution_time', 0),
            'compute_time': trace.get('compute_time', 0),
            'transfer_time': trace.get('transfer_time', 0),
            'flops': trace.get('flops', 0),
            'memory': trace.get('memory', 0),
            'cpu_usage': trace.get('cpu_usage', 0),
            'memory_usage': trace.get('memory_usage', 0),
            'gpu_memory': trace.get('gpu_memory', 0),
        }
    
    return jsonify({
        'layer_id': layer_id,
        'layer_name': layer_name,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'parameters': calculate_parameters(layer_name, input_shape, output_shape),
        'trace': trace_data
    })


def calculate_parameters(layer_name, input_shape, output_shape):
    if not input_shape or not output_shape:
        return 0
    
    layer_type = layer_name.split('_')[0] if '_' in layer_name else layer_name
    
    if 'Conv' in layer_type:
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
            if isinstance(output_shape, (list, tuple)) and len(output_shape) >= 3:
                in_channels = input_shape[-1] if 'channels_last' else input_shape[1]
                out_channels = output_shape[-1] if 'channels_last' else output_shape[1]
                return int(3 * 3 * in_channels * out_channels + out_channels)
    
    elif 'Dense' in layer_type or 'Output' in layer_type:
        if isinstance(input_shape, (list, tuple)) and isinstance(output_shape, (list, tuple)):
            in_features = input_shape[-1] if len(input_shape) > 0 else 1
            out_features = output_shape[-1] if len(output_shape) > 0 else 1
            return int(in_features * out_features + out_features)
    
    return 0


def run_server(host='0.0.0.0', port=5002, debug=False):
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
