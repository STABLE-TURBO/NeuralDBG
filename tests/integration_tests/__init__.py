"""
Neural DSL Integration Tests

Complete integration tests for Neural DSL workflows including:
- DSL parsing and validation
- Shape propagation
- Code generation (PyTorch, TensorFlow, ONNX)
- Model execution
- Hyperparameter optimization (HPO)
- Experiment tracking
- Error handling and edge cases

Test Files:
-----------
- test_complete_workflow_integration.py: End-to-end workflows
- test_hpo_tracking_workflow.py: HPO and tracking features
- test_onnx_workflow.py: ONNX backend integration
- test_edge_cases_workflow.py: Error handling and edge cases
- test_end_to_end_scenarios.py: Real-world scenarios

Quick Start:
-----------
Run all tests:
    pytest tests/integration_tests/ -v

Run with coverage:
    pytest tests/integration_tests/ --cov=neural --cov-report=term -v

Run specific backend:
    pytest tests/integration_tests/ -v -k "pytorch"
    pytest tests/integration_tests/ -v -k "tensorflow"
    pytest tests/integration_tests/ -v -k "onnx"

Documentation:
-------------
- README.md: Comprehensive test documentation
- QUICK_START.md: Quick reference guide
- TEST_SUMMARY.md: Detailed coverage summary
"""

__version__ = "1.0.0"
__all__ = []
