# Testing Guide for Neural DSL

## Overview

This guide outlines the testing strategy, test organization, coverage goals, and best practices for the Neural DSL project.

## Test Coverage Goals

### Current Coverage Status (v0.3.0)

| Module | Coverage Target | Current Status | Priority |
|--------|----------------|----------------|----------|
| neural.parser | >90% | ✅ High | Critical |
| neural.code_generation | >90% | ✅ High | Critical |
| neural.shape_propagation | >90% | ✅ High | Critical |
| neural.cli | >85% | ✅ Good | High |
| neural.hpo | >80% | ✅ Good | High |
| neural.utils | >90% | ✅ High | High |
| neural.exceptions | >95% | ✅ High | High |
| neural.automl | >80% | ⚠️ Medium | Medium |
| neural.integrations | >75% | ⚠️ Medium | Medium |
| neural.teams | >80% | ⚠️ Medium | Medium |
| neural.federated | >80% | ⚠️ Medium | Medium |
| neural.mlops | >75% | ⚠️ Medium | Medium |
| neural.monitoring | >75% | ⚠️ Medium | Low |
| neural.dashboard | >70% | ⚠️ Medium | Medium |

### v1.0 Coverage Requirements

- **Core Modules**: Minimum 90% coverage
- **Feature Modules**: Minimum 80% coverage
- **Experimental Modules**: Minimum 70% coverage
- **Overall Project**: Minimum 80% coverage

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_*.py                      # Top-level test files
├── parser/                        # Parser tests
│   ├── test_parser.py
│   ├── test_validation.py
│   └── test_edge_cases.py
├── code_generator/                # Code generation tests
│   ├── test_tensorflow_generator.py
│   ├── test_pytorch_generator.py
│   └── test_onnx_generator.py
├── shape_propagation/             # Shape propagation tests
│   ├── test_basic_propagation.py
│   ├── test_complex_models.py
│   └── test_error_handling.py
├── hpo/                          # HPO tests
│   ├── test_hpo_integration.py
│   └── test_hpo_optimizers.py
├── integration_tests/            # End-to-end tests
│   ├── test_complete_workflow.py
│   ├── test_transformer_workflow.py
│   └── test_edge_cases.py
├── performance/                  # Performance benchmarks
│   ├── test_parser_performance.py
│   └── test_shape_propagation.py
└── ...
```

## Test Types

### 1. Unit Tests

**Purpose**: Test individual functions and classes in isolation.

**Characteristics**:
- Fast execution (<100ms per test)
- No external dependencies
- Mock external services
- Focus on single functionality

**Example**:
```python
def test_dense_layer_parsing(parser):
    """Test parsing a simple Dense layer."""
    dsl = "Dense(128, 'relu')"
    tree = parser.parse(dsl)
    assert tree is not None
    assert tree.data == "layer"
```

**Markers**: `@pytest.mark.unit`

### 2. Integration Tests

**Purpose**: Test interaction between multiple components.

**Characteristics**:
- Medium execution time (100ms-1s per test)
- Test component integration
- May use real dependencies
- Test complete workflows

**Example**:
```python
@pytest.mark.integration
def test_compile_and_execute_workflow():
    """Test complete compile and execute workflow."""
    dsl_code = """
    network TestNet {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(10, "softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """
    # Parse
    parser = create_parser()
    tree = parser.parse(dsl_code)
    
    # Generate
    generator = TensorFlowGenerator()
    code = generator.generate(tree)
    
    # Verify
    assert "tf.keras.layers.Flatten" in code
```

**Markers**: `@pytest.mark.integration`, `@pytest.mark.slow`

### 3. Performance Tests

**Purpose**: Measure and track performance metrics.

**Characteristics**:
- Measure execution time
- Track memory usage
- Identify performance regressions
- Set performance budgets

**Example**:
```python
@pytest.mark.performance
def test_parser_performance(benchmark):
    """Benchmark parser performance."""
    dsl_code = generate_large_model(layers=100)
    parser = create_parser()
    
    result = benchmark(parser.parse, dsl_code)
    
    # Should parse 100 layers in <100ms
    assert benchmark.stats['mean'] < 0.1
```

**Markers**: `@pytest.mark.performance`, `@pytest.mark.slow`

### 4. Regression Tests

**Purpose**: Prevent reintroduction of fixed bugs.

**Characteristics**:
- Test specific bug scenarios
- Reference issue numbers
- Document expected behavior
- Prevent regressions

**Example**:
```python
@pytest.mark.regression
def test_issue_123_conv2d_padding():
    """Test fix for issue #123: Conv2D padding parameter.
    
    Previously, 'same' padding was incorrectly parsed as string
    instead of being passed to the layer.
    """
    dsl = "Conv2D(32, (3, 3), padding='same')"
    # Test implementation
```

**Markers**: `@pytest.mark.regression`

### 5. Parameterized Tests

**Purpose**: Test multiple scenarios with different inputs.

**Characteristics**:
- Reduce code duplication
- Test edge cases systematically
- Easy to add new test cases
- Clear test documentation

**Example**:
```python
@pytest.mark.parametrize("units,activation", [
    (64, "relu"),
    (128, "sigmoid"),
    (256, "tanh"),
])
def test_dense_configurations(units, activation):
    """Test Dense layer with various configurations."""
    layer = f"Dense({units}, '{activation}')"
    # Test implementation
```

### 6. Property-Based Tests

**Purpose**: Test properties that should hold for all inputs.

**Characteristics**:
- Generate random test cases
- Find edge cases automatically
- Test invariants
- Use hypothesis library

**Example**:
```python
from hypothesis import given, strategies as st

@given(units=st.integers(min_value=1, max_value=1000))
def test_dense_units_property(units):
    """Test that Dense layer accepts any positive integer for units."""
    dsl = f"Dense({units})"
    parser = create_parser('layer')
    tree = parser.parse(dsl)
    assert tree is not None
```

## Test Markers

### Custom Markers

```python
# conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "backend(name): Backend-specific tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "regression: Regression tests")
```

### Running Specific Test Types

```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run GPU tests only
pytest -m gpu

# Run tests for specific backend
pytest -m "backend('tensorflow')"

# Run performance tests
pytest -m performance
```

## Fixtures

### Shared Fixtures (conftest.py)

```python
@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture
def parser():
    """Create a parser instance."""
    return create_parser()

@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
```

### Module-Specific Fixtures

```python
# tests/parser/conftest.py
@pytest.fixture
def sample_network():
    """Sample network definition for parser tests."""
    return """
    network TestNet {
        input: (28, 28, 1)
        layers:
            Dense(10)
    }
    """
```

## Coverage Measurement

### Running with Coverage

```bash
# Run tests with coverage
pytest --cov=neural --cov-report=html --cov-report=term

# Generate detailed HTML report
pytest --cov=neural --cov-report=html
open htmlcov/index.html

# Show missing lines
pytest --cov=neural --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=neural --cov-fail-under=80
```

### Coverage Configuration

```ini
# .coveragerc or pyproject.toml
[tool.coverage.run]
source = ["neural"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/aquarium/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

## Mocking and Patching

### External Dependencies

```python
from unittest.mock import Mock, patch, MagicMock

@patch('neural.integrations.boto3.client')
def test_sagemaker_integration(mock_boto3):
    """Test SageMaker integration with mocked AWS client."""
    mock_client = Mock()
    mock_boto3.return_value = mock_client
    
    # Test implementation
```

### File System Operations

```python
def test_file_writing(tmp_path):
    """Test file writing using temporary directory."""
    output_file = tmp_path / "output.py"
    generator = TensorFlowGenerator()
    generator.save(output_file)
    
    assert output_file.exists()
    assert output_file.read_text().startswith("import tensorflow")
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest --cov=neural --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Best Practices

### 1. Test Naming

```python
# Good: Descriptive test names
def test_dense_layer_with_relu_activation():
    pass

def test_conv2d_raises_error_on_negative_filters():
    pass

# Bad: Vague test names
def test_layer():
    pass

def test_error():
    pass
```

### 2. Test Documentation

```python
def test_shape_propagation_through_residual_connection():
    """
    Test that shape propagation correctly handles residual connections.
    
    A residual connection should:
    1. Propagate input shape through sublayers
    2. Add sublayer output to input
    3. Require matching shapes for addition
    
    Related: Issue #456
    """
    # Test implementation
```

### 3. Arrange-Act-Assert Pattern

```python
def test_model_compilation():
    # Arrange
    model_data = create_test_model()
    generator = TensorFlowGenerator()
    
    # Act
    code = generator.generate(model_data)
    
    # Assert
    assert "def build_model" in code
    assert "return model" in code
```

### 4. Test Data Management

```python
# Good: Use fixtures for shared test data
@pytest.fixture
def cnn_model():
    return {
        "name": "CNN",
        "layers": [...]
    }

def test_cnn_compilation(cnn_model):
    # Use fixture
    pass

# Bad: Duplicate test data
def test_1():
    model = {"name": "CNN", "layers": [...]}
    
def test_2():
    model = {"name": "CNN", "layers": [...]}
```

### 5. Avoid Test Interdependencies

```python
# Good: Independent tests
def test_parser_creation():
    parser = create_parser()
    assert parser is not None

def test_parsing_network():
    parser = create_parser()
    tree = parser.parse("network X {}")
    assert tree is not None

# Bad: Dependent tests
parser = None

def test_1_create_parser():
    global parser
    parser = create_parser()

def test_2_use_parser():
    # Depends on test_1
    tree = parser.parse("network X {}")
```

### 6. Test Edge Cases

```python
def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Empty input
    assert parse("") is None
    
    # Minimum valid input
    assert parse("Dense(1)") is not None
    
    # Maximum reasonable input
    assert parse("Dense(10000)") is not None
    
    # Invalid input
    with pytest.raises(DSLSyntaxError):
        parse("Dense(-1)")
```

## Testing Anti-Patterns

### ❌ Avoid These

1. **Testing Implementation Details**
   ```python
   # Bad: Testing private methods
   def test_private_method():
       obj = MyClass()
       result = obj._internal_method()
       assert result == expected
   ```

2. **Slow Tests Without Markers**
   ```python
   # Bad: No marker for slow test
   def test_large_model():
       model = create_model_with_1000_layers()
       # Takes 10 seconds
   
   # Good: Mark slow tests
   @pytest.mark.slow
   def test_large_model():
       model = create_model_with_1000_layers()
   ```

3. **Testing Multiple Things in One Test**
   ```python
   # Bad: Multiple assertions unrelated
   def test_everything():
       assert parser_works()
       assert generator_works()
       assert shape_propagation_works()
   
   # Good: Separate tests
   def test_parser():
       assert parser_works()
   
   def test_generator():
       assert generator_works()
   ```

## Debugging Failed Tests

### Verbose Output

```bash
# Show full output
pytest -v

# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb
```

### Test-Specific Debugging

```python
def test_complex_scenario():
    """Complex test that needs debugging."""
    import pdb; pdb.set_trace()  # Add breakpoint
    
    # Or use pytest's builtin
    pytest.set_trace()
```

## Performance Testing

### Benchmarking with pytest-benchmark

```python
def test_parser_benchmark(benchmark):
    """Benchmark parser performance."""
    dsl_code = "Dense(128, 'relu')"
    parser = create_parser()
    
    result = benchmark(parser.parse, dsl_code)
    
    # Assert performance requirements
    assert benchmark.stats['mean'] < 0.001  # < 1ms
```

### Memory Profiling

```python
import tracemalloc

def test_memory_usage():
    """Test memory usage of large model."""
    tracemalloc.start()
    
    model = create_large_model()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Assert memory constraints
    assert peak < 100 * 1024 * 1024  # < 100MB
```

## Test Maintenance

### Updating Tests for Breaking Changes

1. Update test expectations
2. Update fixtures
3. Update documentation
4. Run full test suite
5. Update CI/CD pipelines

### Deprecated Feature Testing

```python
@pytest.mark.deprecated
def test_legacy_api():
    """Test deprecated API - remove in v2.0."""
    with pytest.warns(DeprecationWarning):
        result = legacy_function()
    assert result is not None
```

## Summary

- **Coverage Goal**: >80% overall, >90% for core modules
- **Test Types**: Unit, integration, performance, regression
- **Best Practices**: Clear naming, AAA pattern, fixtures, markers
- **CI/CD**: Automated testing on all commits
- **Maintenance**: Regular updates, remove obsolete tests

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [hypothesis documentation](https://hypothesis.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
