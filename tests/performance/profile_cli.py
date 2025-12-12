#!/usr/bin/env python
"""
Profile CLI startup and command execution.
"""
import sys
import time
import cProfile
import pstats
import io
from pathlib import Path


def profile_cli_import():
    """Profile importing the CLI module."""
    profiler = cProfile.Profile()
    
    print("Profiling CLI import...")
    profiler.enable()
    
    from neural.cli import cli
    
    profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    
    print("\n" + "="*60)
    print("CLI Import Profile (Top 30 by cumulative time)")
    print("="*60)
    print(s.getvalue())


def profile_parser_creation():
    """Profile parser creation."""
    profiler = cProfile.Profile()
    
    print("\nProfiling parser creation...")
    profiler.enable()
    
    from neural.parser.parser import create_parser
    parser = create_parser()
    
    profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    
    print("\n" + "="*60)
    print("Parser Creation Profile (Top 30 by cumulative time)")
    print("="*60)
    print(s.getvalue())


def profile_shape_propagation():
    """Profile shape propagation."""
    profiler = cProfile.Profile()
    
    print("\nProfiling shape propagation...")
    
    from neural.shape_propagation.shape_propagator import ShapePropagator
    
    propagator = ShapePropagator(debug=False)
    
    layers = [
        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'padding': 0, 'stride': 1}},
        {'type': 'MaxPooling2D', 'params': {'pool_size': 2, 'stride': 2}},
        {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (3, 3), 'padding': 0, 'stride': 1}},
        {'type': 'MaxPooling2D', 'params': {'pool_size': 2, 'stride': 2}},
    ]
    
    profiler.enable()
    
    input_shape = (None, 28, 28, 3)
    for layer in layers:
        try:
            input_shape = propagator.propagate(input_shape, layer, 'tensorflow')
        except Exception as e:
            print(f"Skipping layer: {e}")
    
    profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    
    print("\n" + "="*60)
    print("Shape Propagation Profile (Top 30 by cumulative time)")
    print("="*60)
    print(s.getvalue())


def profile_parse_and_transform():
    """Profile parsing and transformation."""
    profiler = cProfile.Profile()
    
    print("\nProfiling parse and transform...")
    
    from neural.parser.parser import create_parser, ModelTransformer
    
    test_network = """
network TestNet {
    input: (28, 28, 1)
    layers:
        Conv2D(32, kernel_size=3)
        MaxPooling2D(pool_size=2)
        Flatten()
        Dense(128)
        Output(10)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate=0.001)
}
"""
    
    parser = create_parser()
    
    profiler.enable()
    
    tree = parser.parse(test_network)
    transformer = ModelTransformer()
    model_data = transformer.transform(tree)
    
    profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    
    print("\n" + "="*60)
    print("Parse & Transform Profile (Top 30 by cumulative time)")
    print("="*60)
    print(s.getvalue())


def measure_startup_time():
    """Measure startup time without profiling overhead."""
    print("\nMeasuring startup times (no profiling overhead)...")
    print("="*60)
    
    iterations = 5
    
    print(f"\nCLI Import (average of {iterations} runs):")
    times = []
    for _ in range(iterations):
        start = time.time()
        # Use subprocess to get clean import time
        import subprocess
        result = subprocess.run(
            [sys.executable, '-c', 'from neural.cli import cli'],
            capture_output=True,
            timeout=10
        )
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Min:     {min_time:.3f}s")
    print(f"  Max:     {max_time:.3f}s")


def main():
    """Main entry point."""
    print("="*60)
    print("Neural Performance Profiling")
    print("="*60)
    
    try:
        measure_startup_time()
    except Exception as e:
        print(f"Error measuring startup time: {e}")
    
    try:
        profile_cli_import()
    except Exception as e:
        print(f"Error profiling CLI import: {e}")
    
    try:
        profile_parser_creation()
    except Exception as e:
        print(f"Error profiling parser creation: {e}")
    
    try:
        profile_shape_propagation()
    except Exception as e:
        print(f"Error profiling shape propagation: {e}")
    
    try:
        profile_parse_and_transform()
    except Exception as e:
        print(f"Error profiling parse and transform: {e}")
    
    print("\n" + "="*60)
    print("Profiling Complete")
    print("="*60)


if __name__ == '__main__':
    main()
