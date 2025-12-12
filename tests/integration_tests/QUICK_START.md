# Integration Tests - Quick Start Guide

Quick reference for running and understanding the integration tests.

## 🚀 Quick Run Commands

### Run Everything
```bash
# All integration tests
pytest tests/integration_tests/ -v

# With coverage report
pytest tests/integration_tests/ --cov=neural --cov-report=term -v
```

### Run by File
```bash
# Complete workflows
pytest tests/integration_tests/test_complete_workflow_integration.py -v

# HPO and tracking
pytest tests/integration_tests/test_hpo_tracking_workflow.py -v

# ONNX backend
pytest tests/integration_tests/test_onnx_workflow.py -v

# Edge cases and errors
pytest tests/integration_tests/test_edge_cases_workflow.py -v

# Real-world scenarios
pytest tests/integration_tests/test_end_to_end_scenarios.py -v
```

### Run by Backend
```bash
# PyTorch only
pytest tests/integration_tests/ -v -k "pytorch"

# TensorFlow only
pytest tests/integration_tests/ -v -k "tensorflow"

# ONNX only
pytest tests/integration_tests/ -v -k "onnx"
```

### Run by Feature
```bash
# HPO tests
pytest tests/integration_tests/ -v -k "hpo"

# Tracking tests
pytest tests/integration_tests/ -v -k "tracking"

# Shape propagation tests
pytest tests/integration_tests/ -v -k "shape"

# Execution tests
pytest tests/integration_tests/ -v -k "execution"
```

## 📋 Test File Overview

| File | Focus | Tests | Time |
|------|-------|-------|------|
| `test_complete_workflow_integration.py` | End-to-end workflows | 20+ | ~30s |
| `test_hpo_tracking_workflow.py` | HPO & tracking | 25+ | ~40s |
| `test_onnx_workflow.py` | ONNX export | 20+ | ~20s |
| `test_edge_cases_workflow.py` | Error handling | 30+ | ~25s |
| `test_end_to_end_scenarios.py` | Real scenarios | 10+ | ~35s |

## 🎯 Common Test Scenarios

### 1. Test Complete PyTorch Workflow
```bash
pytest tests/integration_tests/test_complete_workflow_integration.py::TestCompleteWorkflowIntegration::test_pytorch_dsl_to_execution -v
```

### 2. Test HPO Optimization
```bash
pytest tests/integration_tests/test_hpo_tracking_workflow.py::TestHPOWorkflowIntegration::test_hpo_optimization_workflow -v
```

### 3. Test ONNX Export
```bash
pytest tests/integration_tests/test_onnx_workflow.py::TestONNXWorkflowIntegration::test_simple_onnx_export -v
```

### 4. Test Error Handling
```bash
pytest tests/integration_tests/test_edge_cases_workflow.py::TestParsingErrorHandling -v
```

### 5. Test Production Pipeline
```bash
pytest tests/integration_tests/test_end_to_end_scenarios.py::TestEndToEndScenarios::test_scenario_production_pipeline -v
```

## 🔧 Setup Requirements

### Minimal Setup (Core Tests)
```bash
pip install -e .
```

### Full Setup (All Tests)
```bash
pip install -e ".[full]"
```

### Individual Backends
```bash
# PyTorch
pip install torch

# TensorFlow
pip install tensorflow

# ONNX
pip install onnx onnxruntime

# HPO
pip install optuna
```

## 📊 What Gets Tested

### ✅ DSL Parsing
- Valid syntax parsing
- Error detection
- HPO parameter extraction
- Training config parsing

### ✅ Shape Propagation
- All layer types
- Shape validation
- Error detection
- Auto-flatten insertion

### ✅ Code Generation
- PyTorch code
- TensorFlow code
- ONNX models
- Training loops
- Inference code

### ✅ Execution
- Model creation
- Forward pass
- Training
- Inference
- Multi-backend

### ✅ HPO
- Parameter parsing
- Optimization
- DSL generation
- Multi-metric tracking

### ✅ Tracking
- Experiment init
- Hyperparameter logging
- Metrics logging
- Metadata management

## 🐛 Debugging Tests

### Run Single Test with Output
```bash
pytest tests/integration_tests/test_complete_workflow_integration.py::TestCompleteWorkflowIntegration::test_simple_dsl_to_shape_propagation -v -s
```

### Run with Python Debugger
```bash
pytest tests/integration_tests/test_complete_workflow_integration.py::TestCompleteWorkflowIntegration::test_simple_dsl_to_shape_propagation -v --pdb
```

### Show Print Statements
```bash
pytest tests/integration_tests/ -v -s
```

### Stop on First Failure
```bash
pytest tests/integration_tests/ -v -x
```

### Show Locals on Failure
```bash
pytest tests/integration_tests/ -v -l
```

## 📝 Test Patterns

### Standard Test Structure
```python
def test_feature_name(self):
    """Test: Clear description of what's being tested."""
    # 1. Setup
    dsl_code = """network TestNet { ... }"""
    
    # 2. Parse
    parser = create_parser("network")
    tree = parser.parse(dsl_code)
    model_config = transformer.transform(tree)
    
    # 3. Validate
    propagator = ShapePropagator()
    current_shape = propagator.propagate(input_shape, layer)
    
    # 4. Generate
    code = generate_code(model_config, 'pytorch')
    
    # 5. Assert
    assert 'expected_string' in code
    assert output_shape == expected_shape
```

### HPO Test Pattern
```python
@patch('neural.hpo.hpo.get_data', mock_data_loader)
def test_hpo_feature(self):
    """Test: HPO feature description."""
    dsl_with_hpo = """network HPONet { 
        layers: Dense(HPO(choice(64, 128))) 
    }"""
    
    model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_with_hpo)
    best_params = optimize_and_return(dsl_with_hpo, n_trials=2)
    optimized_dsl = generate_optimized_dsl(dsl_with_hpo, best_params)
    
    assert 'HPO' not in optimized_dsl
```

### Tracking Test Pattern
```python
def test_tracking_feature(self):
    """Test: Tracking feature description."""
    tracker = ExperimentTracker(experiment_name="test")
    tracker.log_hyperparameters({'lr': 0.001})
    tracker.log_metrics({'loss': 0.5}, step=1)
    tracker.save_metadata()
    
    assert os.path.exists(tracker.experiment_dir)
```

## 🎓 Understanding Test Results

### Test Passed ✅
```
test_simple_dsl_to_shape_propagation PASSED [100%]
```
Everything worked as expected!

### Test Skipped ⚠️
```
test_pytorch_dsl_to_execution SKIPPED (PyTorch not available) [50%]
```
Test was skipped because required dependency is missing.

### Test Failed ❌
```
test_invalid_syntax_error FAILED
AssertionError: Expected exception not raised
```
Test failed - check the error message and traceback.

## 🔍 Common Issues

### Issue: Tests Skipped
**Cause**: Missing optional dependencies  
**Solution**: Install required packages
```bash
pip install torch tensorflow onnx optuna
```

### Issue: Import Errors
**Cause**: Package not installed in editable mode  
**Solution**: Install package
```bash
pip install -e .
```

### Issue: Temporary File Errors
**Cause**: Previous test didn't cleanup  
**Solution**: Remove temp directories
```bash
rm -rf tests/tmp_path/
rm -rf neural_experiments/
```

### Issue: Slow Tests
**Cause**: Running actual training loops  
**Solution**: Tests use mocking, check mock patches are active

## 📚 Further Reading

- **README.md**: Complete test documentation
- **TEST_SUMMARY.md**: Detailed test coverage summary
- **AGENTS.md**: Project setup and commands
- **CONTRIBUTING.md**: Contribution guidelines

## 💡 Tips

1. **Start with fast tests**: Run parsing and validation tests first
2. **Use markers**: Skip slow tests during development with `-m "not slow"`
3. **Check coverage**: Ensure new features have test coverage
4. **Read docstrings**: Each test has a clear description
5. **Use fixtures**: Leverage setup/teardown for cleaner tests

## 🤝 Contributing Tests

When adding new tests:
1. Choose appropriate test file
2. Follow existing patterns
3. Add clear docstrings
4. Use appropriate mocking
5. Ensure cleanup in teardown
6. Update documentation

## 📞 Getting Help

- Check existing tests for examples
- Read test docstrings for clarity
- Review README.md for detailed info
- Check TEST_SUMMARY.md for coverage
- Ask in project discussions/issues

---

**Quick reminder**: These tests validate the complete Neural DSL workflow from parsing to execution. Run them before committing changes! 🚀
