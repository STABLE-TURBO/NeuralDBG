# Integration Tests

This directory contains comprehensive integration tests for the Neural DSL complete workflows.

## Overview

These integration tests validate the full pipeline from DSL parsing through shape propagation, code generation, and execution across TensorFlow, PyTorch, and ONNX backends, including HPO (Hyperparameter Optimization) and experiment tracking features.

## Test Files

### 1. `test_complete_workflow_integration.py`
**Purpose**: Tests complete end-to-end workflows for all backends.

**Key Test Areas**:
- DSL parsing → shape propagation workflows
- PyTorch code generation and execution
- TensorFlow code generation and execution
- ONNX export and validation
- HPO integration with code generation
- Experiment tracking integration
- Complex architectures (CNNs, RNNs)
- Multi-backend consistency
- Layer multiplication features
- Training configuration integration

**Test Classes**:
- `TestCompleteWorkflowIntegration`: Main workflow tests
- `TestMultiBackendExecution`: Backend execution tests

**Example Tests**:
- `test_simple_dsl_to_shape_propagation()`: Basic DSL → shape validation
- `test_pytorch_dsl_to_execution()`: Full PyTorch workflow with execution
- `test_tensorflow_dsl_to_execution()`: Full TensorFlow workflow with execution
- `test_pytorch_with_hpo_full_workflow()`: Complete HPO workflow
- `test_end_to_end_with_all_features()`: All features combined

### 2. `test_hpo_tracking_workflow.py`
**Purpose**: Integration tests for HPO and experiment tracking workflows.

**Key Test Areas**:
- HPO model creation and parameter resolution
- HPO training loop execution
- HPO objective function computation
- HPO optimization workflow
- Optimized DSL generation
- Experiment tracker initialization
- Hyperparameter logging
- Metrics logging
- Experiment metadata management
- Combined HPO and tracking workflows

**Test Classes**:
- `TestHPOWorkflowIntegration`: HPO-specific tests
- `TestTrackingWorkflowIntegration`: Experiment tracking tests
- `TestHPOTrackingCombinedWorkflow`: Combined workflows

**Example Tests**:
- `test_hpo_model_creation_pytorch()`: HPO model creation
- `test_hpo_objective_function()`: Objective function metrics
- `test_experiment_tracker_log_hyperparameters()`: Logging hyperparameters
- `test_complete_hpo_tracking_workflow()`: Full combined workflow

### 3. `test_onnx_workflow.py`
**Purpose**: ONNX backend-specific integration tests.

**Key Test Areas**:
- Simple ONNX export
- ONNX file export and loading
- ONNX model validation
- Convolutional layer export
- Multiple layer types in ONNX
- ONNX inference execution (when runtime available)
- ONNX shape inference
- ONNX opset version validation
- Model metadata in ONNX
- Complex architectures in ONNX
- Batch normalization in ONNX
- Different input shapes
- Cross-backend consistency

**Test Classes**:
- `TestONNXWorkflowIntegration`: ONNX-specific tests
- `TestONNXWithOtherBackends`: Cross-backend tests

**Example Tests**:
- `test_simple_onnx_export()`: Basic ONNX export
- `test_onnx_model_validation()`: Model validation
- `test_onnx_complex_architecture()`: Complex model export
- `test_same_model_multiple_exports()`: Multiple backend exports

### 4. `test_edge_cases_workflow.py`
**Purpose**: Error handling and edge case testing.

**Key Test Areas**:
- Invalid DSL syntax handling
- Missing required fields
- Invalid layer parameters
- Empty layers handling
- Invalid input shapes
- Incompatible layer sequences
- Invalid convolution parameters
- Shape mismatches
- Invalid backend errors
- Missing model data
- Invalid layer types
- HPO error handling
- Single layer networks
- Very deep networks
- Very wide networks
- Minimal/large input shapes
- Multiple HPO parameters
- Extreme parameter ranges
- Auto-flatten insertion

**Test Classes**:
- `TestParsingErrorHandling`: Parsing errors
- `TestShapePropagationErrorHandling`: Shape errors
- `TestCodeGenerationErrorHandling`: Code generation errors
- `TestHPOErrorHandling`: HPO errors
- `TestEdgeCases`: Edge cases
- `TestRecoveryAndGracefulDegradation`: Recovery mechanisms

**Example Tests**:
- `test_invalid_syntax_error()`: Invalid DSL syntax
- `test_incompatible_layer_sequence_error()`: Layer incompatibility
- `test_invalid_backend_error()`: Backend validation
- `test_very_deep_network()`: Deep network handling
- `test_auto_flatten_insertion()`: Auto-flatten feature

## Running the Tests

### Run all integration tests:
```bash
pytest tests/integration_tests/ -v
```

### Run specific test file:
```bash
pytest tests/integration_tests/test_complete_workflow_integration.py -v
pytest tests/integration_tests/test_hpo_tracking_workflow.py -v
pytest tests/integration_tests/test_onnx_workflow.py -v
pytest tests/integration_tests/test_edge_cases_workflow.py -v
```

### Run specific test class:
```bash
pytest tests/integration_tests/test_complete_workflow_integration.py::TestCompleteWorkflowIntegration -v
```

### Run specific test:
```bash
pytest tests/integration_tests/test_complete_workflow_integration.py::TestCompleteWorkflowIntegration::test_simple_dsl_to_shape_propagation -v
```

### Run with coverage:
```bash
pytest tests/integration_tests/ --cov=neural --cov-report=term -v
```

### Run tests for specific backend:
```bash
# PyTorch tests only
pytest tests/integration_tests/ -v -k "pytorch"

# TensorFlow tests only
pytest tests/integration_tests/ -v -k "tensorflow"

# ONNX tests only
pytest tests/integration_tests/ -v -k "onnx"
```

## Dependencies

These tests may require optional dependencies:

- **PyTorch**: `pip install torch`
- **TensorFlow**: `pip install tensorflow`
- **ONNX**: `pip install onnx onnxruntime`
- **Optuna** (for HPO): `pip install optuna`
- **Full install**: `pip install -e ".[full]"`

Tests will be skipped automatically if required dependencies are not available.

## Test Structure

Each test typically follows this pattern:

1. **Setup**: Create DSL code and temporary directories
2. **Parse**: Parse DSL using parser and transformer
3. **Propagate**: Validate shapes using ShapePropagator
4. **Generate**: Generate code for target backend(s)
5. **Execute**: (Optional) Execute generated code
6. **Verify**: Assert expected behavior and outputs
7. **Teardown**: Clean up temporary files

## Mocking

Tests use mocking for:
- **Data loaders**: Mock datasets to avoid downloading real data
- **HPO trials**: Mock Optuna trials for deterministic testing
- **Optimization**: Mock optimization results for faster tests

## Key Features Tested

### 1. Complete Workflow
- DSL → Parser → Transformer → Shape Propagation → Code Generation → Execution

### 2. Multi-Backend Support
- PyTorch code generation and execution
- TensorFlow code generation and execution
- ONNX model export and validation

### 3. HPO Integration
- HPO parameter parsing
- Model creation with HPO parameters
- Objective function computation
- Optimized DSL generation

### 4. Experiment Tracking
- Experiment initialization
- Hyperparameter logging
- Metrics logging
- Metadata management

### 5. Error Handling
- Parsing errors
- Validation errors
- Shape incompatibilities
- Runtime errors

### 6. Edge Cases
- Minimal networks
- Deep networks
- Wide networks
- Large input shapes
- Extreme parameter ranges

## Integration with CI/CD

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Integration Tests
  run: |
    pip install -e ".[full]"
    pytest tests/integration_tests/ -v --cov=neural
```

## Test Coverage

The integration tests aim to cover:
- ✅ All major backends (PyTorch, TensorFlow, ONNX)
- ✅ All major layer types (Dense, Conv2D, LSTM, etc.)
- ✅ All HPO parameter types (choice, range, log_range)
- ✅ All tracking features
- ✅ Error handling paths
- ✅ Edge cases and boundary conditions

## Contributing

When adding new features, please add corresponding integration tests:

1. Choose the appropriate test file based on the feature
2. Follow existing test patterns and naming conventions
3. Use appropriate mocking to keep tests fast
4. Document the test purpose in docstrings
5. Ensure tests are skipped gracefully if dependencies are missing

## Notes

- Tests create temporary directories and clean them up automatically
- Tests use deterministic mocking for reproducibility
- Tests are designed to be independent and can run in any order
- Failed tests should provide clear error messages
- Tests skip gracefully when optional dependencies are not available
