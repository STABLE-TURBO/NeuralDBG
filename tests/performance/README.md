# Performance Benchmarks

This directory contains performance benchmarking tests for Neural.

## Test Files

- `test_cli_startup.py`: CLI startup and lazy import benchmarks
- `test_shape_propagation.py`: Shape propagation and caching benchmarks
- `test_parser_performance.py`: Parser grammar and optimization benchmarks
- `test_end_to_end.py`: End-to-end workflow performance tests

## Running Benchmarks

Run all performance tests:
```bash
pytest tests/performance/ -v
```

Run specific benchmark:
```bash
pytest tests/performance/test_cli_startup.py -v -s
```

Run with profiling:
```bash
pytest tests/performance/ -v --profile
```

## Performance Targets

### CLI Startup
- CLI import time: < 2.0s
- CLI --help time: < 5.0s
- Heavy dependencies should remain lazy

### Shape Propagation
- 10 layers propagation: < 1.0s
- Cache speedup: > 2x
- 1000 performance computations: < 0.5s

### Parser Performance
- Parser creation: < 1.0s
- Simple network parse: < 0.1s
- Complex network parse: < 0.2s
- Average parse time: < 0.15s

### End-to-End Workflows
- Parse + propagate: < 2.0s
- Code generation: < 3.0s
- Visualization: < 3.0s
- Memory increase: < 100MB

## Optimization Techniques Used

1. **Lazy Imports**: Heavy dependencies loaded on-demand
2. **LRU Caching**: Frequently used computations cached
3. **LALR Parser**: Efficient parser algorithm
4. **Grammar Optimization**: Reduced parser complexity
5. **Module Caching**: Reuse loaded modules
6. **Slots**: Memory-efficient classes

## Profiling

To profile a specific test:
```bash
python -m cProfile -o output.prof tests/performance/test_cli_startup.py
python -m pstats output.prof
```

Or use line_profiler:
```bash
kernprof -l -v tests/performance/test_cli_startup.py
```
