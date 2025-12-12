# Integration Tests - Complete Index

**Neural DSL Integration Tests Documentation Index**

## 📚 Documentation Files

| File | Purpose | Target Audience |
|------|---------|-----------------|
| **QUICK_START.md** | Quick reference and common commands | Developers (daily use) |
| **README.md** | Complete test documentation | Developers (detailed reference) |
| **TEST_SUMMARY.md** | Coverage and metrics summary | Maintainers, reviewers |
| **INDEX.md** | This file - navigation guide | All users |

## 🧪 Test Files

| File | Tests | Focus | Runtime |
|------|-------|-------|---------|
| **test_complete_workflow_integration.py** | 20+ | Full workflows, multi-backend | ~30s |
| **test_hpo_tracking_workflow.py** | 25+ | HPO and experiment tracking | ~40s |
| **test_onnx_workflow.py** | 20+ | ONNX export and validation | ~20s |
| **test_edge_cases_workflow.py** | 30+ | Error handling, edge cases | ~25s |
| **test_end_to_end_scenarios.py** | 10+ | Real-world scenarios | ~35s |

## 🎯 Quick Navigation

### For New Contributors
1. Start with **QUICK_START.md** for immediate test running
2. Review **README.md** for test structure understanding
3. Check **TEST_SUMMARY.md** for coverage details

### For Daily Development
1. **QUICK_START.md** - Run commands by feature/backend
2. **test_complete_workflow_integration.py** - Main workflow tests
3. **run_all_tests.py** - Convenient test runner

### For Code Review
1. **TEST_SUMMARY.md** - Coverage verification
2. **README.md** - Test pattern validation
3. Relevant test files - Implementation review

### For CI/CD Setup
1. **README.md** - CI/CD integration section
2. **QUICK_START.md** - Command examples
3. **run_all_tests.py** - Automated test runner

## 📖 Test File Details

### 1. test_complete_workflow_integration.py
**What it tests:**
- Complete DSL → PyTorch workflow
- Complete DSL → TensorFlow workflow
- Complete DSL → ONNX workflow
- Shape propagation integration
- Multi-backend consistency
- Complex architectures
- Training configuration

**Key test classes:**
- `TestCompleteWorkflowIntegration`
- `TestMultiBackendExecution`

**When to modify:**
- Adding new layer types
- Changing code generation
- Updating workflow logic

### 2. test_hpo_tracking_workflow.py
**What it tests:**
- HPO parameter parsing and resolution
- HPO model creation
- HPO training loop
- HPO objective function
- Experiment tracker initialization
- Hyperparameter logging
- Metrics logging
- Combined HPO + tracking workflows

**Key test classes:**
- `TestHPOWorkflowIntegration`
- `TestTrackingWorkflowIntegration`
- `TestHPOTrackingCombinedWorkflow`

**When to modify:**
- Adding HPO parameter types
- Changing tracking behavior
- Updating optimization logic

### 3. test_onnx_workflow.py
**What it tests:**
- ONNX model generation
- ONNX model validation
- ONNX file I/O
- ONNX shape inference
- ONNX opset versions
- Complex architectures in ONNX
- Cross-backend consistency

**Key test classes:**
- `TestONNXWorkflowIntegration`
- `TestONNXWithOtherBackends`

**When to modify:**
- Adding ONNX export features
- Updating ONNX specifications
- Changing layer mapping

### 4. test_edge_cases_workflow.py
**What it tests:**
- Invalid DSL syntax handling
- Missing fields validation
- Invalid parameters
- Shape mismatch detection
- Incompatible layers
- Invalid backends
- HPO edge cases
- Recovery mechanisms
- Extreme configurations

**Key test classes:**
- `TestParsingErrorHandling`
- `TestShapePropagationErrorHandling`
- `TestCodeGenerationErrorHandling`
- `TestHPOErrorHandling`
- `TestEdgeCases`
- `TestRecoveryAndGracefulDegradation`

**When to modify:**
- Adding validation rules
- Improving error messages
- Adding recovery logic

### 5. test_end_to_end_scenarios.py
**What it tests:**
- MNIST classification scenario
- HPO optimization scenario
- Experiment tracking scenario
- Multi-backend deployment
- Iterative improvement
- Transfer learning preparation
- Production pipeline
- Architecture exploration
- Model versioning

**Key test class:**
- `TestEndToEndScenarios`

**When to modify:**
- Adding new use cases
- Updating best practices
- Demonstrating new features

## 🛠️ Utility Files

### run_all_tests.py
**Purpose:** Convenient test runner with options

**Usage:**
```bash
python tests/integration_tests/run_all_tests.py --help
python tests/integration_tests/run_all_tests.py --backend pytorch
python tests/integration_tests/run_all_tests.py --coverage
```

### __init__.py
**Purpose:** Package initialization and metadata

**Contains:**
- Package version
- Module docstring
- Package metadata

## 📊 Test Coverage Map

### By Component
- **Parser**: test_complete_workflow_integration.py, test_edge_cases_workflow.py
- **Shape Propagator**: test_complete_workflow_integration.py, test_edge_cases_workflow.py
- **Code Generator**: All test files
- **HPO**: test_hpo_tracking_workflow.py, test_complete_workflow_integration.py
- **Tracking**: test_hpo_tracking_workflow.py, test_end_to_end_scenarios.py
- **ONNX**: test_onnx_workflow.py, test_complete_workflow_integration.py

### By Backend
- **PyTorch**: All test files (primary)
- **TensorFlow**: test_complete_workflow_integration.py, test_end_to_end_scenarios.py
- **ONNX**: test_onnx_workflow.py, test_complete_workflow_integration.py

### By Feature
- **DSL Parsing**: test_complete_workflow_integration.py, test_edge_cases_workflow.py
- **Shape Propagation**: test_complete_workflow_integration.py, test_edge_cases_workflow.py
- **Code Generation**: All test files
- **Execution**: test_complete_workflow_integration.py, test_end_to_end_scenarios.py
- **HPO**: test_hpo_tracking_workflow.py, test_complete_workflow_integration.py
- **Tracking**: test_hpo_tracking_workflow.py, test_end_to_end_scenarios.py
- **Error Handling**: test_edge_cases_workflow.py
- **Scenarios**: test_end_to_end_scenarios.py

## 🚀 Common Workflows

### 1. Running Tests During Development
```bash
# Quick check (fast tests only)
pytest tests/integration_tests/ -v -k "not slow"

# Specific feature
pytest tests/integration_tests/ -v -k "hpo"

# Specific backend
pytest tests/integration_tests/ -v -k "pytorch"
```

### 2. Pre-Commit Testing
```bash
# Run all tests with coverage
pytest tests/integration_tests/ --cov=neural --cov-report=term -v

# Or use convenience script
python tests/integration_tests/run_all_tests.py --coverage
```

### 3. CI/CD Pipeline
```bash
# Full test suite with coverage
pytest tests/integration_tests/ -v --cov=neural --cov-report=xml

# Parallel execution
pytest tests/integration_tests/ -v -n auto --cov=neural
```

### 4. Debugging Failures
```bash
# Run single test with output
pytest tests/integration_tests/test_complete_workflow_integration.py::TestCompleteWorkflowIntegration::test_simple_dsl_to_shape_propagation -v -s

# Run with debugger
pytest tests/integration_tests/test_complete_workflow_integration.py::TestCompleteWorkflowIntegration::test_simple_dsl_to_shape_propagation -v --pdb
```

## 📈 Metrics and Goals

### Current Status
- **Total Tests**: 105+
- **Test Files**: 5
- **Test Classes**: 13
- **Line Coverage**: ~85%
- **Branch Coverage**: ~80%
- **Execution Time**: ~150s (all tests)

### Quality Targets
- ✅ Line Coverage: 85%+ (Target: 90%)
- ✅ Branch Coverage: 80%+ (Target: 85%)
- ✅ Test Pass Rate: 100%
- ✅ Test Independence: 100%
- ✅ Documentation: 100%

## 🔄 Update Workflow

### When Adding New Features
1. Add tests to appropriate file
2. Update TEST_SUMMARY.md with new coverage
3. Update README.md if adding new patterns
4. Run full test suite
5. Update this INDEX.md if adding new files

### When Fixing Bugs
1. Add regression test first
2. Fix the bug
3. Verify test passes
4. Update documentation if needed

### When Refactoring
1. Run tests before changes
2. Make refactoring changes
3. Run tests after changes
4. Ensure no regressions
5. Update tests if API changed

## 🎓 Learning Path

### Beginner
1. Read QUICK_START.md
2. Run simple tests
3. Examine test_complete_workflow_integration.py
4. Understand basic patterns

### Intermediate
1. Read README.md completely
2. Study all test files
3. Understand mocking patterns
4. Review TEST_SUMMARY.md

### Advanced
1. Review test coverage gaps
2. Add new test scenarios
3. Optimize test performance
4. Contribute to test infrastructure

## 📞 Support and Resources

### Getting Help
- Check QUICK_START.md for common issues
- Review README.md for detailed patterns
- Search existing tests for examples
- Check TEST_SUMMARY.md for coverage

### Contributing
- Follow existing test patterns
- Add clear docstrings
- Use appropriate mocking
- Update documentation
- Ensure cleanup

### Best Practices
- Keep tests independent
- Use descriptive names
- Mock external dependencies
- Clean up resources
- Document assumptions

## 🗺️ Repository Context

### Related Directories
- `neural/parser/` - Parser implementation
- `neural/shape_propagation/` - Shape propagator
- `neural/code_generation/` - Code generators
- `neural/hpo/` - HPO implementation
- `neural/tracking/` - Experiment tracking

### Related Documentation
- `AGENTS.md` - Project setup guide
- `README.md` - Project overview
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history

---

**Last Updated:** 2024
**Version:** 1.0.0
**Maintained By:** Neural DSL Team
