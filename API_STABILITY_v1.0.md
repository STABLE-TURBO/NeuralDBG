# API Stability Guide for Neural DSL v1.0

## Overview

This document outlines the API stability commitments for Neural DSL v1.0. It defines which APIs are stable, which are experimental, and the deprecation policy leading up to and beyond v1.0.

## Version Policy

### Semantic Versioning

Neural DSL follows [Semantic Versioning 2.0.0](https://semver.org/):
- **MAJOR** version (1.x.x): Breaking changes to stable APIs
- **MINOR** version (x.1.x): New features, backwards compatible
- **PATCH** version (x.x.1): Bug fixes, backwards compatible

### Pre-v1.0 Status

Currently at v0.3.0, APIs may change between minor versions. This is the last major opportunity for breaking changes before v1.0.

## API Classification

### Stable APIs (v1.0)

These APIs will be supported with backwards compatibility guarantees:

#### Core DSL Syntax
- **Network definitions**: `network`, `input`, `layers`, `loss`, `optimizer`
- **Layer types**: All documented layer types (Dense, Conv2D, LSTM, etc.)
- **Activation functions**: Standard activations (relu, sigmoid, tanh, etc.)
- **Optimizers**: Adam, SGD, RMSprop, AdamW
- **Loss functions**: All documented loss functions

**Stability Commitment**: No breaking changes in v1.x releases.

#### CLI Commands
```bash
neural compile <file>         # Compile DSL to code
neural run <file>              # Execute compiled code
neural visualize <file>        # Generate visualizations
neural debug <file>            # Launch debugger
neural test <file>             # Run tests
```

**Stability Commitment**: Command signatures and primary options will remain stable.

#### Parser API
```python
from neural.parser import create_parser, ModelTransformer

parser = create_parser()
tree = parser.parse(dsl_code)
transformer = ModelTransformer()
model = transformer.transform(tree)
```

**Stability Commitment**: Core parsing API will remain stable.

#### Code Generation API
```python
from neural.code_generation import TensorFlowGenerator, PyTorchGenerator, ONNXGenerator

generator = TensorFlowGenerator()
code = generator.generate(model_data)
```

**Stability Commitment**: Main generation methods will remain stable.

#### Shape Propagation API
```python
from neural.shape_propagation import ShapePropagator

propagator = ShapePropagator()
shapes = propagator.propagate(model_data, input_shape)
```

**Stability Commitment**: Core propagation API will remain stable.

### Experimental APIs

These APIs may change in v1.x releases with deprecation warnings:

#### AutoML & NAS
```python
from neural.automl import AutoMLEngine, ArchitectureSpace
```

**Status**: Experimental. API may evolve based on usage patterns.

#### Federated Learning
```python
from neural.federated import FederatedClient, FederatedServer
```

**Status**: Experimental. Security and privacy features are still evolving.

#### Cloud Integrations
```python
from neural.integrations import SageMakerIntegration, DatabricksIntegration
```

**Status**: Experimental. Cloud provider APIs may require changes.

#### AI Assistant
```python
from neural.ai import AIAssistant, NaturalLanguageProcessor
```

**Status**: Experimental. LLM integration patterns are evolving.

#### Dashboard & Visualization
```python
from neural.dashboard import NeuralDebugger
from neural.visualization import NetworkVisualizer
```

**Status**: Mostly stable, but UI/visualization APIs may have minor changes.

### Internal APIs

These APIs are not part of the public API surface:

- `neural.parser.parser_utils.*` (internal utilities)
- `neural.code_generation.base_generator._internal_*` (private methods)
- Any module or function prefixed with `_` (private)

**Stability Commitment**: None. These may change at any time.

## Breaking Changes Policy

### Before v1.0 (Current: v0.3.0)

**Now is the time for breaking changes.** We will make necessary breaking changes to stabilize APIs before v1.0.

#### Planned Breaking Changes for v0.4.0 → v1.0

1. **DSL Syntax Standardization**
   - Standardize parameter names across all layers
   - Enforce consistent naming conventions (snake_case for parameters)
   - Remove deprecated syntax variants

2. **API Consolidation**
   - Consolidate duplicate functionality
   - Remove deprecated functions and classes
   - Simplify import paths

3. **Configuration Format**
   - Standardize YAML configuration format
   - Remove legacy configuration options
   - Enforce schema validation

4. **Error Handling**
   - Standardize exception hierarchy
   - Improve error messages with actionable suggestions
   - Add error codes for programmatic handling

5. **Type Annotations**
   - Complete type annotations for all public APIs
   - Enable strict type checking with mypy
   - Add runtime type validation where appropriate

### After v1.0

**Backwards Compatibility Guarantee**: Once v1.0 is released, stable APIs will maintain backwards compatibility throughout the v1.x series.

#### Deprecation Process

1. **Deprecation Announcement** (v1.x)
   - Feature marked as deprecated
   - Deprecation warning added to documentation
   - Runtime warning issued when deprecated feature is used
   - Alternative approach documented

2. **Deprecation Period** (minimum 2 minor versions)
   - Deprecated feature continues to work with warnings
   - Users have time to migrate
   - Migration guide provided

3. **Removal** (v2.0)
   - Deprecated feature removed in next major version
   - Breaking change documented in migration guide

#### Example Deprecation
```python
# v1.5: Deprecation warning
@deprecated(
    version="1.5.0",
    reason="Use create_parser() instead",
    removal_version="2.0.0"
)
def get_parser():
    warnings.warn(
        "get_parser() is deprecated. Use create_parser() instead. "
        "This will be removed in v2.0.0",
        DeprecationWarning,
        stacklevel=2
    )
    return create_parser()

# v1.6-1.9: Still available with warnings

# v2.0: Removed
```

## Stability by Module

### High Stability (v1.0 ready)
- ✅ `neural.parser` - Core parsing functionality
- ✅ `neural.code_generation` - Code generation for TF/PyTorch/ONNX
- ✅ `neural.shape_propagation` - Shape validation and propagation
- ✅ `neural.cli` - Command-line interface
- ✅ `neural.exceptions` - Exception hierarchy
- ✅ `neural.utils` - Core utilities

### Medium Stability (v1.0 candidate)
- ⚠️ `neural.hpo` - Hyperparameter optimization
- ⚠️ `neural.dashboard` - Debugging dashboard
- ⚠️ `neural.tracking` - Experiment tracking
- ⚠️ `neural.training` - Training utilities
- ⚠️ `neural.benchmarks` - Benchmarking tools
- ⚠️ `neural.data` - Data management

### Low Stability (Experimental)
- 🔬 `neural.automl` - AutoML and NAS
- 🔬 `neural.federated` - Federated learning
- 🔬 `neural.integrations` - Cloud integrations
- 🔬 `neural.ai` - AI assistant features
- 🔬 `neural.cloud` - Cloud execution
- 🔬 `neural.monitoring` - Production monitoring
- 🔬 `neural.mlops` - MLOps features
- 🔬 `neural.collaboration` - Collaborative editing

## Pre-v1.0 Action Items

### Required for v1.0

1. **Complete API Review**
   - [ ] Review all public APIs for consistency
   - [ ] Remove duplicate functionality
   - [ ] Standardize naming conventions
   - [ ] Add comprehensive docstrings

2. **Type Safety**
   - [x] Expand mypy coverage to core modules
   - [ ] Add type hints to all public functions
   - [ ] Enable strict type checking
   - [ ] Document type requirements

3. **Error Handling**
   - [x] Implement comprehensive exception hierarchy
   - [x] Add actionable error messages
   - [ ] Add error codes for all exceptions
   - [ ] Document error handling patterns

4. **Testing**
   - [x] Increase test coverage to >80% for core modules
   - [ ] Add integration tests for all major workflows
   - [ ] Add regression tests for known issues
   - [ ] Add performance benchmarks

5. **Documentation**
   - [ ] Complete API reference documentation
   - [ ] Add migration guides for breaking changes
   - [ ] Create comprehensive examples
   - [ ] Document best practices

6. **Performance**
   - [ ] Optimize parser performance
   - [ ] Optimize code generation
   - [ ] Add performance benchmarks
   - [ ] Document performance characteristics

## Breaking Changes Process (Pre-v1.0)

### Announcement
All breaking changes will be announced in:
- CHANGELOG.md
- GitHub release notes
- Documentation updates
- Migration guides

### Communication Channels
- GitHub Issues (label: `breaking-change`)
- GitHub Discussions
- Release notes
- Documentation

## API Guarantee Matrix

| Module | v1.0 Status | Breaking Changes After v1.0 | Deprecation Required |
|--------|-------------|----------------------------|---------------------|
| neural.parser | ✅ Stable | ❌ No | ✅ Yes |
| neural.code_generation | ✅ Stable | ❌ No | ✅ Yes |
| neural.shape_propagation | ✅ Stable | ❌ No | ✅ Yes |
| neural.cli | ✅ Stable | ❌ No | ✅ Yes |
| neural.exceptions | ✅ Stable | ❌ No | ✅ Yes |
| neural.hpo | ⚠️ Candidate | ⚠️ Maybe | ✅ Yes |
| neural.dashboard | ⚠️ Candidate | ⚠️ Maybe | ✅ Yes |
| neural.automl | 🔬 Experimental | ✅ Yes | ⚠️ Maybe |
| neural.federated | 🔬 Experimental | ✅ Yes | ⚠️ Maybe |
| neural.ai | 🔬 Experimental | ✅ Yes | ⚠️ Maybe |

## Extension Points

These extension points are designed for long-term stability:

### Custom Layers
```python
from neural.layers import CustomLayer

class MyLayer(CustomLayer):
    def __init__(self, units: int):
        super().__init__()
        self.units = units
```

### Custom Backends
```python
from neural.code_generation import BaseGenerator

class MyBackendGenerator(BaseGenerator):
    def generate(self, model_data: dict) -> str:
        # Generate code for custom backend
        pass
```

### Custom HPO Strategies
```python
from neural.hpo import BaseStrategy

class MyStrategy(BaseStrategy):
    def suggest(self, trial) -> dict:
        # Custom hyperparameter suggestion logic
        pass
```

## Feedback and Evolution

### How to Provide Feedback

1. **GitHub Issues**: Report bugs, request features
2. **GitHub Discussions**: Discuss API design
3. **Pull Requests**: Propose changes with implementation

### API Evolution Process

1. **Proposal**: Create GitHub issue with API proposal
2. **Discussion**: Community feedback and iteration
3. **Implementation**: Prototype in experimental module
4. **Stabilization**: Usage patterns inform final API
5. **Documentation**: Comprehensive docs before stability declaration
6. **Commitment**: API marked as stable in release

## Compatibility Testing

### Test Suite Requirements

- ✅ Unit tests for all public APIs
- ✅ Integration tests for common workflows
- ✅ Backwards compatibility tests
- ✅ Performance regression tests
- ✅ Cross-platform tests (Windows, Linux, macOS)
- ✅ Cross-version tests (Python 3.8-3.12)

### Continuous Integration

- Run full test suite on every commit
- Test against all supported Python versions
- Test with minimum and maximum dependency versions
- Run static type checking (mypy)
- Run linting (ruff, pylint)

## Version Support Policy

### Python Versions
- **Supported**: Python 3.8, 3.9, 3.10, 3.11, 3.12
- **Testing**: All supported versions in CI
- **Minimum**: Python 3.8+ required

### Dependency Versions
- **Policy**: Support latest stable versions
- **Minimum Versions**: Documented in setup.py
- **Testing**: Test with both minimum and latest versions

### Long-Term Support (LTS)
- **v1.0 LTS**: Security fixes for 2 years after v2.0 release
- **Critical Bugs**: Fixed in all supported versions
- **Security Issues**: Backported to LTS versions

## Summary

- **Current Phase**: Pre-v1.0 (v0.3.0) - Breaking changes allowed
- **Action Required**: Complete API stabilization tasks
- **Timeline**: Target v1.0 release after completing action items
- **Commitment**: Strong backwards compatibility after v1.0
- **Communication**: Transparent about breaking changes and deprecations

## Questions?

For questions about API stability:
- Open a GitHub Discussion
- Check the documentation
- Review CHANGELOG.md for recent changes
- See MIGRATION_GUIDE.md for breaking change details
