"""
Example usage of the Shape Propagation Panel with the ShapePropagator.
This demonstrates how to use the API to propagate shapes through a model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import requests
import json
from neural.shape_propagation.shape_propagator import ShapePropagator


def example_simple_cnn():
    """Example: Simple CNN for MNIST classification."""
    
    model_config = {
        "input_shape": [None, 28, 28, 1],
        "framework": "tensorflow",
        "layers": [
            {
                "type": "Conv2D",
                "params": {
                    "filters": 32,
                    "kernel_size": [3, 3],
                    "padding": "same",
                    "stride": 1
                }
            },
            {
                "type": "MaxPooling2D",
                "params": {
                    "pool_size": [2, 2]
                }
            },
            {
                "type": "Conv2D",
                "params": {
                    "filters": 64,
                    "kernel_size": [3, 3],
                    "padding": "same",
                    "stride": 1
                }
            },
            {
                "type": "MaxPooling2D",
                "params": {
                    "pool_size": [2, 2]
                }
            },
            {
                "type": "Flatten",
                "params": {}
            },
            {
                "type": "Dense",
                "params": {
                    "units": 128
                }
            },
            {
                "type": "Output",
                "params": {
                    "units": 10
                }
            }
        ]
    }
    
    response = requests.post(
        'http://localhost:5002/api/shape-propagation/propagate',
        json=model_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Shape propagation successful!")
        print(f"  Propagated through {len(result['shape_history'])} layers")
        print(f"  Detected {len(result['errors'])} errors")
        
        print("\nShape History:")
        for item in result['shape_history']:
            print(f"  {item['layer_name']:20s} {str(item['input_shape']):25s} -> {str(item['output_shape']):25s}")
        
        if result['errors']:
            print("\nErrors:")
            for error in result['errors']:
                print(f"  ⚠ {error['layer']}: {error['message']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def example_direct_propagator():
    """Example: Using ShapePropagator directly without API."""
    
    print("Using ShapePropagator directly:")
    
    propagator = ShapePropagator(debug=False)
    
    input_shape = (None, 28, 28, 1)
    
    layers = [
        {
            "type": "Conv2D",
            "params": {
                "filters": 32,
                "kernel_size": (3, 3),
                "padding": "same"
            }
        },
        {
            "type": "MaxPooling2D",
            "params": {
                "pool_size": (2, 2)
            }
        },
        {
            "type": "Flatten",
            "params": {}
        },
        {
            "type": "Dense",
            "params": {
                "units": 128
            }
        }
    ]
    
    current_shape = input_shape
    for layer in layers:
        try:
            current_shape = propagator.propagate(current_shape, layer, 'tensorflow')
            print(f"  ✓ {layer['type']:20s} -> {current_shape}")
        except Exception as e:
            print(f"  ✗ {layer['type']:20s} -> Error: {e}")
    
    print(f"\nTotal layers processed: {len(propagator.shape_history)}")
    print(f"Execution trace entries: {len(propagator.execution_trace)}")


if __name__ == '__main__':
    print("=" * 60)
    print("Shape Propagation Examples")
    print("=" * 60)
    
    print("\n1. Simple CNN Example")
    print("-" * 60)
    try:
        example_simple_cnn()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n2. Direct Propagator Example")
    print("-" * 60)
    try:
        example_direct_propagator()
    except Exception as e:
        print(f"Error: {e}")
