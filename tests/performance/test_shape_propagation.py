"""
Benchmark shape propagation with caching.
"""
import time
import pytest
from neural.shape_propagation.shape_propagator import ShapePropagator


def create_test_layers(count=100):
    """Create test layers for benchmarking."""
    layers = []
    for i in range(count):
        if i % 3 == 0:
            layers.append({
                'type': 'Conv2D',
                'params': {'filters': 32, 'kernel_size': (3, 3), 'padding': 0, 'stride': 1}
            })
        elif i % 3 == 1:
            layers.append({
                'type': 'MaxPooling2D',
                'params': {'pool_size': 2, 'stride': 2}
            })
        else:
            layers.append({
                'type': 'Dense',
                'params': {'units': 128}
            })
    return layers


def test_shape_propagation_performance():
    """Benchmark shape propagation speed."""
    propagator = ShapePropagator(debug=False)
    layers = create_test_layers(50)
    
    start = time.time()
    input_shape = (None, 28, 28, 3)
    
    for layer in layers[:10]:  # Test first 10 layers
        try:
            if layer['type'] in ['Conv2D', 'MaxPooling2D']:
                input_shape = propagator.propagate(input_shape, layer, 'tensorflow')
            elif layer['type'] == 'Dense':
                if len(input_shape) > 2:
                    flatten_layer = {'type': 'Flatten', 'params': {}}
                    input_shape = propagator.propagate(input_shape, flatten_layer, 'tensorflow')
                input_shape = propagator.propagate(input_shape, layer, 'tensorflow')
        except Exception as e:
            print(f"Skipping layer {layer['type']}: {e}")
            continue
    
    elapsed = time.time() - start
    
    assert elapsed < 1.0, f"Shape propagation took {elapsed:.3f}s, expected < 1.0s"
    print(f"\n✓ Shape propagation time (10 layers): {elapsed:.3f}s")


def test_cache_effectiveness():
    """Test that caching improves performance."""
    propagator = ShapePropagator(debug=False)
    
    layer = {
        'type': 'Conv2D',
        'params': {'filters': 64, 'kernel_size': (3, 3), 'padding': 0, 'stride': 1}
    }
    input_shape = (None, 224, 224, 3)
    
    start = time.time()
    for _ in range(100):
        _ = propagator._standardize_params(layer['params'], 'Conv2D', 'tensorflow')
    first_run = time.time() - start
    
    start = time.time()
    for _ in range(100):
        _ = propagator._standardize_params(layer['params'], 'Conv2D', 'tensorflow')
    cached_run = time.time() - start
    
    speedup = first_run / (cached_run + 1e-6)
    assert speedup > 2.0, f"Cache speedup only {speedup:.2f}x, expected > 2x"
    print(f"✓ Parameter cache speedup: {speedup:.2f}x")


def test_performance_computation_cache():
    """Test performance computation caching."""
    propagator = ShapePropagator(debug=False)
    
    layer = {
        'type': 'Conv2D',
        'params': {'filters': 64, 'kernel_size': (3, 3)}
    }
    input_shape = (1, 224, 224, 3)
    output_shape = (1, 222, 222, 64)
    
    start = time.time()
    for _ in range(1000):
        _ = propagator._compute_performance(layer, input_shape, output_shape)
    elapsed = time.time() - start
    
    assert elapsed < 0.5, f"Performance computation took {elapsed:.3f}s, expected < 0.5s"
    print(f"✓ Performance computation time (1000 iterations): {elapsed:.3f}s")


def test_layer_handler_performance():
    """Test layer handler dispatch performance."""
    propagator = ShapePropagator(debug=False)
    
    layer_types = ['Conv2D', 'Dense', 'MaxPooling2D', 'Flatten']
    test_configs = [
        ({'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'padding': 0}}, (None, 28, 28, 3)),
        ({'type': 'Dense', 'params': {'units': 128}}, (None, 784)),
        ({'type': 'MaxPooling2D', 'params': {'pool_size': 2, 'stride': 2}}, (None, 28, 28, 32)),
        ({'type': 'Flatten', 'params': {}}, (None, 14, 14, 32)),
    ]
    
    start = time.time()
    for layer, input_shape in test_configs:
        for _ in range(50):
            try:
                _ = propagator._process_layer(input_shape, layer, 'tensorflow')
            except Exception:
                pass
    elapsed = time.time() - start
    
    assert elapsed < 1.0, f"Layer handlers took {elapsed:.3f}s, expected < 1.0s"
    print(f"✓ Layer handler performance (200 operations): {elapsed:.3f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
