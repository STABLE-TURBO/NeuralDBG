# Deprecated Features - Neural DSL

## Deprecation Policy

Features marked as deprecated will:
1. Continue to work in current version with warnings
2. Be removed or extracted in next major version
3. Have migration paths documented

## Currently Deprecated Features

### 1. Aquarium IDE (`neural/aquarium/`)
**Status**: Extraction planned  
**Reason**: Full IDE development is out of scope for core DSL project  
**Timeline**: Extract to separate `Neural-Aquarium` repository by v0.4.0  
**Migration**: Use VS Code/PyCharm with Neural language server instead  
**Warning**: `DeprecationWarning` when importing `neural.aquarium`

```python
# Deprecated
from neural.aquarium import AquariumIDE

# Recommended
# Use VS Code with .neural file support
# Or use neural CLI: neural compile model.neural
```

### 2. Neural Chat (`neural/neural_chat/`)
**Status**: Deprecated  
**Reason**: LLM-based conversational interface doesn't align with DSL-first approach  
**Timeline**: Remove in v0.4.0  
**Migration**: Use clear DSL syntax and documentation instead  
**Warning**: `DeprecationWarning` when importing `neural.neural_chat`

```python
# Deprecated
from neural.neural_chat import NeuralChat

# Recommended
# Write explicit DSL code
# Use neural --help for CLI guidance
```

### 3. Neural LLM (`neural/neural_llm/`)
**Status**: Deprecated (consolidate with AI assistant)  
**Reason**: Redundant with `neural/ai/` module  
**Timeline**: Remove in v0.4.0  
**Migration**: Use `neural.ai.generate_model()` instead  
**Warning**: `DeprecationWarning` when importing `neural.neural_llm`

```python
# Deprecated
from neural.neural_llm import NeuralLLM

# Recommended
from neural.ai import generate_model
model_code = generate_model("CNN for MNIST")
```

### 4. Collaboration Tools (`neural/collaboration/`)
**Status**: Deprecated  
**Reason**: Version control and collaboration better handled by Git  
**Timeline**: Remove in v0.4.0  
**Migration**: Use Git for version control, GitHub/GitLab for collaboration  
**Warning**: `DeprecationWarning` when importing `neural.collaboration`

```python
# Deprecated
from neural.collaboration import Workspace

# Recommended
# Use Git: git init, git commit, git push
# Use GitHub/GitLab for team collaboration
```

### 5. Marketplace (`neural/marketplace/`)
**Status**: Deprecated  
**Reason**: Premature - HuggingFace Hub provides better model sharing  
**Timeline**: Remove in v0.4.0  
**Migration**: Use HuggingFace Hub integration  
**Warning**: `DeprecationWarning` when importing `neural.marketplace`

```python
# Deprecated
from neural.marketplace import ModelRegistry

# Recommended
from neural.marketplace.huggingface_integration import HFIntegration
# Or use HuggingFace Hub directly
```

### 6. Federated Learning (`neural/federated/`)
**Status**: ✅ REMOVED in v0.3.0+  
**Reason**: Too specialized for core DSL, deserves dedicated project  
**Migration**: Module has been removed. Will be available as separate `neural-federated` package in future if needed  

The federated learning module was not essential to core DSL functionality and had no integration with primary workflows or examples.

### 7. Advanced MLOps (`neural/mlops/`)
**Status**: Simplified (partial deprecation)  
**Reason**: Production MLOps better handled by specialized platforms  
**Timeline**: Simplify to basic deployment helpers by v0.4.0  
**Migration**: Use platform-specific MLOps tools  
**Warning**: Some features will show `DeprecationWarning`

```python
# Deprecated (complex workflows)
from neural.mlops import CIPipeline, ABTesting

# Retained (basic helpers)
from neural.mlops import ModelRegistry, BasicDeployment

# Recommended
# Use platform tools: SageMaker Pipelines, Kubeflow, MLflow
```

### 8. Data Versioning (`neural/data/`)
**Status**: Simplified (partial deprecation)  
**Reason**: DVC and similar tools are more mature  
**Timeline**: Simplify to basic DVC integration by v0.4.0  
**Migration**: Use DVC directly  
**Warning**: Complex features will show `DeprecationWarning`

```python
# Deprecated (full implementation)
from neural.data import DataVersioning, FeatureStore

# Retained (simple adapter)
from neural.data import DVCIntegration

# Recommended
# Use DVC directly: dvc init, dvc add, dvc push
```

## Experimental Features (May Be Deprecated)

These features are experimental and may be deprecated in future versions:

### 1. Cost Optimization (`neural/cost/`)
**Status**: Experimental  
**Evaluation**: v0.4.0  
**Risk**: May be too specialized, better handled by cloud platform tools

### 2. Monitoring (`neural/monitoring/`)
**Status**: Experimental  
**Evaluation**: v0.4.0  
**Risk**: Prometheus/Grafana integration may be sufficient

### 3. Research Paper Generation (`neural/research_generation/`)
**Status**: Experimental  
**Evaluation**: v0.4.0  
**Risk**: Very niche use case

## Retained Features (Confirmed Scope)

These features are in scope and actively maintained:

### Core (Priority 1)
- ✅ DSL Parser (`neural/parser/`)
- ✅ Code Generation (`neural/code_generation/`)
- ✅ Shape Propagation (`neural/shape_propagation/`)
- ✅ CLI Interface (`neural/cli/`)
- ✅ Dashboard/Debugger (`neural/dashboard/`)

### Semi-Core (Priority 2)
- ✅ HPO (simplified) (`neural/hpo/`)
- ✅ AutoML (educational) (`neural/automl/`)
- ✅ Tracking (basic) (`neural/tracking/`)
- ✅ Visualization (`neural/visualization/`)
- ✅ Cloud Integration (major platforms) (`neural/integrations/`)

### Utilities (Priority 3)
- ✅ Training helpers (`neural/training/`)
- ✅ Utils (`neural/utils/`)
- ✅ Pretrained models (basic) (`neural/pretrained_models/`)
- ✅ No-code GUI (simple) (`neural/no_code/`)

## How to Check Deprecation Status

```python
import warnings

# Enable deprecation warnings
warnings.filterwarnings('default', category=DeprecationWarning)

# Check if module is deprecated
from neural.aquarium import AquariumIDE  # Shows warning
```

## Deprecation Warning Format

```
DeprecationWarning: neural.aquarium is deprecated and will be removed in v0.4.0.
Use VS Code or PyCharm with Neural language server instead.
See docs/DEPRECATIONS.md for migration guide.
```

## Migration Timeline

| Version | Action | Features Affected |
|---------|--------|-------------------|
| v0.3.0 | Mark deprecated | Add warnings to deprecated modules |
| v0.3.1 | Documentation | Update docs with migration guides |
| v0.4.0 | Remove/Extract | Remove deprecated code, extract to separate repos |

## Developer Guidelines

### Adding Deprecation Warnings

```python
# In module __init__.py
import warnings

warnings.warn(
    "neural.feature_name is deprecated and will be removed in v0.4.0. "
    "Use alternative_solution instead. "
    "See docs/DEPRECATIONS.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)
```

### Testing Deprecated Features

```python
import pytest

def test_deprecated_feature():
    with pytest.warns(DeprecationWarning):
        from neural.deprecated_module import DeprecatedClass
```

## Support Policy

- **Deprecated features**: Bug fixes only, no new features
- **Experimental features**: May change API without notice
- **Core features**: Full backward compatibility within major version

## Questions?

If you rely on a deprecated feature and need help migrating:
1. Check migration guide in this document
2. Open a discussion on GitHub Discussions
3. Ask on Discord (#migrations channel)

We're here to help make the transition smooth!
