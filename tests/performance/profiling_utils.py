"""
Profiling utilities for Neural performance analysis.
"""
import time
import functools
import cProfile
import pstats
import io
from contextlib import contextmanager


@contextmanager
def time_block(name="Block"):
    """Context manager to time a block of code."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{name} took {elapsed:.3f}s")


def profile_function(func):
    """Decorator to profile a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        
        print(f"\nProfile for {func.__name__}:")
        print(s.getvalue())
        
        return result
    return wrapper


class PerformanceTracker:
    """Track performance metrics across multiple runs."""
    
    def __init__(self):
        self.timings = {}
        
    def record(self, name, elapsed):
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
        
    @contextmanager
    def track(self, name):
        """Context manager to track and record timing."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.record(name, elapsed)
            
    def summary(self):
        """Print summary statistics."""
        print("\nPerformance Summary:")
        print("-" * 60)
        
        for name, timings in self.timings.items():
            if timings:
                avg = sum(timings) / len(timings)
                min_time = min(timings)
                max_time = max(timings)
                
                print(f"{name:30} avg: {avg:.3f}s  min: {min_time:.3f}s  max: {max_time:.3f}s")


def benchmark_function(func, iterations=100, warmup=10):
    """Benchmark a function with warmup iterations."""
    for _ in range(warmup):
        func()
    
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'average': avg,
        'min': min_time,
        'max': max_time,
        'iterations': iterations
    }


def compare_implementations(implementations, iterations=100):
    """Compare multiple implementations of the same functionality."""
    results = {}
    
    print("\nBenchmarking implementations...")
    for name, func in implementations.items():
        print(f"  Testing {name}...")
        results[name] = benchmark_function(func, iterations)
    
    print("\nComparison Results:")
    print("-" * 60)
    
    baseline = None
    for name, metrics in results.items():
        avg = metrics['average']
        
        if baseline is None:
            baseline = avg
            speedup = 1.0
        else:
            speedup = baseline / avg
        
        print(f"{name:30} {avg*1000:8.3f}ms  ({speedup:5.2f}x)")
    
    return results


class MemoryTracker:
    """Track memory usage."""
    
    def __init__(self):
        try:
            import psutil
            self.process = psutil.Process()
            self.available = True
        except ImportError:
            self.available = False
            
    @contextmanager
    def track(self, name):
        """Track memory usage in a context."""
        if not self.available:
            yield
            return
            
        import gc
        gc.collect()
        
        initial = self.process.memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            gc.collect()
            final = self.process.memory_info().rss / 1024 / 1024
            increase = final - initial
            
            print(f"{name} memory: {initial:.1f}MB -> {final:.1f}MB ({increase:+.1f}MB)")


if __name__ == '__main__':
    @profile_function
    def example_function():
        """Example function to profile."""
        total = 0
        for i in range(1000000):
            total += i
        return total
    
    result = example_function()
    print(f"Result: {result}")
