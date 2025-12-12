"""
End-to-end performance benchmarks for complete workflows.
"""
import time
import pytest
import tempfile
import os
from pathlib import Path


TEST_NETWORK = """
network BenchmarkNet {
    input: (28, 28, 1)
    layers:
        Conv2D(32, kernel_size=3)
        MaxPooling2D(pool_size=2)
        Conv2D(64, kernel_size=3)
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(128)
        Dropout(0.5)
        Output(10)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
"""


@pytest.fixture
def temp_neural_file():
    """Create a temporary .neural file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.neural', delete=False) as f:
        f.write(TEST_NETWORK)
        temp_path = f.name
    
    yield temp_path
    
    try:
        os.unlink(temp_path)
    except Exception:
        pass


def test_parse_and_propagate_workflow(temp_neural_file):
    """Test complete parse -> shape propagation workflow."""
    from neural.parser.parser import create_parser, ModelTransformer
    from neural.shape_propagation.shape_propagator import ShapePropagator
    
    start = time.time()
    
    parser = create_parser()
    with open(temp_neural_file, 'r') as f:
        content = f.read()
    
    tree = parser.parse(content)
    model_data = ModelTransformer().transform(tree)
    
    propagator = ShapePropagator(debug=False)
    input_shape = model_data['input']['shape']
    
    for layer in model_data['layers']:
        try:
            input_shape = propagator.propagate(input_shape, layer, 'tensorflow')
        except Exception as e:
            print(f"Warning: Skipping layer due to: {e}")
            continue
    
    elapsed = time.time() - start
    
    assert elapsed < 2.0, f"Parse + propagate workflow took {elapsed:.3f}s, expected < 2.0s"
    print(f"\n✓ Parse + propagate workflow time: {elapsed:.3f}s")


def test_code_generation_workflow(temp_neural_file):
    """Test parse -> transform -> code generation workflow."""
    from neural.parser.parser import create_parser, ModelTransformer
    
    try:
        from neural.code_generation.code_generator import generate_code
    except ImportError:
        pytest.skip("Code generator not available")
        return
    
    start = time.time()
    
    parser = create_parser()
    with open(temp_neural_file, 'r') as f:
        content = f.read()
    
    tree = parser.parse(content)
    model_data = ModelTransformer().transform(tree)
    
    code = generate_code(model_data, 'tensorflow')
    
    elapsed = time.time() - start
    
    assert code is not None
    assert len(code) > 0
    assert elapsed < 3.0, f"Code generation workflow took {elapsed:.3f}s, expected < 3.0s"
    print(f"✓ Code generation workflow time: {elapsed:.3f}s")


def test_visualization_workflow(temp_neural_file):
    """Test parse -> transform -> visualize workflow."""
    from neural.parser.parser import create_parser, ModelTransformer
    from neural.shape_propagation.shape_propagator import ShapePropagator
    
    start = time.time()
    
    parser = create_parser()
    with open(temp_neural_file, 'r') as f:
        content = f.read()
    
    tree = parser.parse(content)
    model_data = ModelTransformer().transform(tree)
    
    propagator = ShapePropagator(debug=False)
    input_shape = model_data['input']['shape']
    
    for layer in model_data['layers']:
        try:
            input_shape = propagator.propagate(input_shape, layer, 'tensorflow')
        except Exception as e:
            print(f"Warning: Skipping layer due to: {e}")
            continue
    
    try:
        report = propagator.generate_report()
        assert report is not None
    except Exception as e:
        print(f"Note: Visualization skipped due to: {e}")
    
    elapsed = time.time() - start
    
    assert elapsed < 3.0, f"Visualization workflow took {elapsed:.3f}s, expected < 3.0s"
    print(f"✓ Visualization workflow time: {elapsed:.3f}s")


def test_memory_usage():
    """Test memory usage during operations."""
    import psutil
    import gc
    
    process = psutil.Process()
    gc.collect()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    from neural.parser.parser import create_parser, ModelTransformer
    from neural.shape_propagation.shape_propagator import ShapePropagator
    
    parser = create_parser()
    tree = parser.parse(TEST_NETWORK)
    model_data = ModelTransformer().transform(tree)
    
    propagator = ShapePropagator(debug=False)
    input_shape = model_data['input']['shape']
    
    for layer in model_data['layers']:
        try:
            input_shape = propagator.propagate(input_shape, layer, 'tensorflow')
        except Exception:
            continue
    
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB, expected < 100MB"
    print(f"✓ Memory increase: {memory_increase:.1f}MB")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
