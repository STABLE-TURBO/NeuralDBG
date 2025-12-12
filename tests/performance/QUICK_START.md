# Performance Testing Quick Start

Quick guide to running and understanding Neural performance tests.

## Quick Commands

```bash
# Run all performance tests
pytest tests/performance/ -v

# Run specific test suite
pytest tests/performance/test_cli_startup.py -v -s

# Run benchmark runner with summary
python tests/performance/benchmark_runner.py

# Profile CLI operations
python tests/performance/profile_cli.py

# Profile with output to file
python tests/performance/profile_cli.py > profile_results.txt
```

## What Each Test Does

### test_cli_startup.py
Tests CLI import speed and lazy loading effectiveness.

**Key metrics:**
- CLI import time (target: < 2s)
- Help command time (target: < 5s)
- Verifies dependencies aren't loaded eagerly

### test_shape_propagation.py
Tests shape propagation performance and caching.

**Key metrics:**
- Propagation time for 10 layers (target: < 1s)
- Cache speedup (target: > 2x)
- Performance computation time (target: < 0.5s for 1000 ops)

### test_parser_performance.py
Tests parser creation and parsing speed.

**Key metrics:**
- Parser creation time (target: < 1s)
- Simple network parse (target: < 0.1s)
- Complex network parse (target: < 0.2s)

### test_end_to_end.py
Tests complete workflows end-to-end.

**Key metrics:**
- Parse + propagate (target: < 2s)
- Code generation (target: < 3s)
- Memory usage (target: < 100MB increase)

## Understanding Results

### Good Results
```
✓ CLI import time: 1.234s
✓ Simple network parse time: 0.045s
✓ Cache speedup: 3.45x
```

All metrics within targets, optimizations working well.

### Warning Signs
```
✗ CLI import time: 3.456s  # Too slow
✗ Cache speedup: 1.2x      # Cache not helping
✗ Memory increase: 150MB   # Too much memory
```

Performance regression or optimization not working.

## Common Issues

### Test Failures

**Issue:** Import time too slow
**Solution:** Check if heavy modules are being imported eagerly

**Issue:** Cache not speeding up
**Solution:** Verify cache keys are hashable and consistent

**Issue:** Parser slow
**Solution:** Check grammar complexity, ensure LALR parser is used

### Environment Issues

**Issue:** Tests fail on first run
**Solution:** Run with warmup: `pytest tests/performance/ -v --tb=short`

**Issue:** Inconsistent results
**Solution:** Close other applications, run multiple times, check CPU throttling

## Interpreting Profiling Output

### cProfile Output
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
1000    0.050    0.000    0.100    0.000 shape_propagator.py:123(_standardize_params)
```

- **ncalls**: Number of calls
- **tottime**: Time spent in function (excluding subcalls)
- **cumtime**: Time spent in function (including subcalls)
- **percall**: Time per call

Focus on:
1. High cumtime functions (most expensive)
2. High ncalls with low percall (good cache candidates)
3. High tottime (actual work being done)

### Benchmark Output
```
✓ CLI import time: 1.234s
✓ Parse + propagate workflow time: 1.567s
```

Direct time measurements for specific operations.

## Adding New Benchmarks

### Template
```python
def test_my_benchmark():
    """Test description."""
    import time
    
    # Setup
    setup_code()
    
    # Benchmark
    start = time.time()
    operation_to_benchmark()
    elapsed = time.time() - start
    
    # Assert
    assert elapsed < TARGET_TIME, f"Too slow: {elapsed:.3f}s"
    print(f"✓ My benchmark: {elapsed:.3f}s")
```

### Guidelines
1. Use realistic test data
2. Include warmup for cached operations
3. Set reasonable targets
4. Print results for visibility
5. Use descriptive test names

## Best Practices

### Running Tests
- Close unnecessary applications
- Run on consistent hardware
- Multiple runs for averages
- Note system load

### Writing Tests
- Focus on user-facing operations
- Test both cold and warm caches
- Include memory checks
- Document expected performance

### Debugging Slow Tests
1. Run with profiling: `python -m cProfile test_file.py`
2. Check for unintended imports
3. Verify caching is working
4. Look for O(n²) operations

## Quick Checklist

Before committing performance changes:

- [ ] All performance tests pass
- [ ] No regression in benchmark_runner.py
- [ ] Profile results show improvement
- [ ] Memory usage is reasonable
- [ ] Documentation updated

## Resources

- Main performance docs: `docs/PERFORMANCE.md`
- Profiling utilities: `tests/performance/profiling_utils.py`
- Benchmark runner: `tests/performance/benchmark_runner.py`
