# Dependency and Import Issues Fix Summary

## Overview
Fixed all dependency and import issues across the Neural DSL codebase, ensuring optional dependencies (torch, tensorflow, optuna) are properly handled with try/except blocks and appropriate DependencyError exceptions.

## Changes Made

### 1. Removed Wildcard Imports

#### `neural/benchmarks/framework_implementations.py`
- **Fixed**: Removed `from fastai.vision.all import *` and `from fastai.data.all import *` (lines 423-424)
- **Replaced with**: Explicit imports in the specific methods that need them:
  - `from fastai.vision.all import Learner, CrossEntropyLossFlat, accuracy`
  - `from fastai.data.all import DataLoaders`
- **Verified**: No other wildcard imports found in the mentioned files

### 2. Added Proper Dependency Handling

#### `neural/benchmarks/framework_implementations.py`
- Added DependencyError import from neural.exceptions
- Added try/except blocks with DependencyError for:
  - TensorFlow imports in NeuralDSLImplementation, KerasImplementation
  - PyTorch imports in PyTorchLightningImplementation, FastAIImplementation
  - Ludwig imports in LudwigImplementation
  - Pandas imports for data handling
- All dependency checks now provide helpful error messages with install hints

#### `neural/hpo/hpo.py`
- Added comprehensive try/except blocks for all optional dependencies at module level:
  - TensorFlow (tf, keras) with HAS_TENSORFLOW flag
  - PyTorch (torch, nn, optim) with HAS_TORCH flag
  - Optuna (optuna, Trial) with HAS_OPTUNA flag
  - scikit-learn (precision_score, recall_score) with HAS_SKLEARN flag
  - torchvision (CIFAR10, MNIST, ToTensor) with HAS_TORCHVISION flag
- Added dependency checks in key functions:
  - `get_data()`: Checks for torchvision, torch/tensorflow depending on backend
  - `create_dynamic_model()`: Checks for optuna, torch/tensorflow
  - `resolve_hpo_params()`: Checks for optuna
  - `DynamicPTModel.__init__()`: Checks for torch
  - `DynamicTFModel.__init__()`: Checks for tensorflow
  - `train_model()`: Checks for torch and sklearn
  - `objective()`: Checks for optuna and torch
  - `optimize_and_return()`: Checks for optuna
- All type hints updated to use string quotes for optional types (e.g., 'Trial', 'torch.Tensor')

#### `neural/training/training.py`
- Complete rewrite with proper dependency handling
- Added try/except blocks for TensorFlow and PyTorch imports
- TensorBoardLogger now checks for torch before creating SummaryWriter
- All operations properly guarded with dependency checks

#### `neural/explainability/feature_importance.py`
- Added DependencyError import
- Updated `_get_prediction_function()` to check for tensorflow/torch before using them
- Updated `plot_importance()` to raise DependencyError instead of logging warning for matplotlib

#### `neural/execution_optimization/execution.py`
- Added try/except block for torch import with HAS_TORCH flag
- Updated TensorRT availability check to also check HAS_TORCH
- Added dependency checks in all functions:
  - `get_device()`: Checks for torch
  - `run_inference()`: Checks for torch
  - `optimize_model_with_tensorrt()`: Checks for torch
  - `run_optimized_inference()`: Checks for torch
- All type hints updated to use string quotes for torch types

#### `neural/pretrained_models/pretrained.py`
- Added try/except block for torch import with HAS_TORCH flag
- Updated function signatures to use quoted type hints
- Added dependency check in `fuse_conv_bn_weights()` function
- Already had proper handling for huggingface_hub and triton

### 3. No Changes Needed

#### `neural/ai/model_optimizer.py`
- **Status**: No changes needed
- **Reason**: This file does not import any optional dependencies directly
- **Verified**: No torch, tensorflow, or optuna imports found

#### `neural/ai/debugging_assistant.py`
- **Status**: No changes needed
- **Reason**: This file does not import any optional dependencies
- **Verified**: Pure Python implementation with no ML framework dependencies

## Consistency Patterns Applied

1. **Import Pattern**:
   ```python
   try:
       import torch
       HAS_TORCH = True
   except ImportError:
       HAS_TORCH = False
       torch = None
   ```

2. **Function Guard Pattern**:
   ```python
   def some_function():
       if not HAS_TORCH:
           raise DependencyError(
               dependency="torch",
               feature="feature description",
               install_hint="pip install torch"
           )
       # function implementation
   ```

3. **Type Hint Pattern for Optional Dependencies**:
   ```python
   def function(model: 'torch.nn.Module') -> 'torch.Tensor':
       # Using quotes to avoid NameError when torch is not installed
   ```

## Benefits

1. **Clear Error Messages**: Users now get helpful error messages with install hints when optional dependencies are missing
2. **No Runtime Crashes**: Import errors are caught and handled gracefully
3. **Better User Experience**: Install hints guide users to install the correct packages
4. **Maintainability**: Consistent pattern across all modules makes it easy to add new optional dependencies
5. **Type Safety**: Type hints preserved using quoted strings for forward references

## Testing Recommendations

1. Test each module with and without optional dependencies installed
2. Verify error messages are helpful and accurate
3. Ensure functions fail gracefully with DependencyError when dependencies are missing
4. Check that all features work correctly when dependencies are available

## Files Modified

1. `neural/benchmarks/framework_implementations.py` - Removed wildcard imports, added dependency checks
2. `neural/hpo/hpo.py` - Added comprehensive dependency handling throughout
3. `neural/training/training.py` - Complete rewrite with proper dependency handling
4. `neural/explainability/feature_importance.py` - Added dependency checks
5. `neural/execution_optimization/execution.py` - Added torch dependency handling
6. `neural/pretrained_models/pretrained.py` - Added torch dependency check

## No Modifications Required

1. `neural/ai/model_optimizer.py` - No optional dependencies used
2. `neural/ai/debugging_assistant.py` - No optional dependencies used
