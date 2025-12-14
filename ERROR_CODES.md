# Neural DSL Error Codes Reference

## Overview

This document provides a comprehensive reference of error codes used in Neural DSL. Error codes enable programmatic error handling and provide consistent error identification across the framework.

## Error Code Format

Error codes follow the format: `NEURAL-XXXX-YY`

- `NEURAL`: Framework prefix
- `XXXX`: Category code (4 digits)
- `YY`: Specific error code (2 digits)

### Category Codes

- `1000`: Parser errors
- `2000`: Code generation errors
- `3000`: Shape propagation errors
- `4000`: Parameter validation errors
- `5000`: HPO errors
- `6000`: Tracking errors
- `7000`: Cloud execution errors
- `8000`: MLOps errors
- `9000`: System errors

---

## Parser Errors (1000-1099)

### NEURAL-1000-01: Syntax Error
**Exception**: `DSLSyntaxError`  
**Description**: Invalid DSL syntax detected  
**Example**:
```python
# Missing colon after network definition
network MyNet {  # Error: missing ':'
```
**Resolution**: Check DSL syntax, ensure proper punctuation

### NEURAL-1000-02: Invalid Token
**Exception**: `DSLSyntaxError`  
**Description**: Unexpected or invalid token in DSL  
**Example**:
```python
Dense(128 "relu")  # Error: missing comma
```
**Resolution**: Check for missing commas, quotes, or parentheses

### NEURAL-1000-03: Unmatched Delimiter
**Exception**: `DSLSyntaxError`  
**Description**: Unmatched brackets, parentheses, or braces  
**Example**:
```python
network MyNet {
    layers:
        Dense(128, "relu"  # Error: missing closing parenthesis
}
```
**Resolution**: Ensure all delimiters are properly matched

### NEURAL-1001-01: Validation Error
**Exception**: `DSLValidationError`  
**Description**: DSL is syntactically correct but semantically invalid  
**Example**:
```python
network MyNet {
    # Error: input shape missing
    layers: Dense(10)
}
```
**Resolution**: Ensure all required fields are present

### NEURAL-1001-02: Unknown Layer Type
**Exception**: `DSLValidationError`  
**Description**: Layer type not recognized  
**Example**:
```python
InvalidLayer(128)  # Error: unknown layer type
```
**Resolution**: Check layer name spelling, refer to documentation

---

## Code Generation Errors (2000-2099)

### NEURAL-2000-01: Unsupported Layer
**Exception**: `UnsupportedLayerError`  
**Description**: Layer not supported by target backend  
**Example**:
```python
# TensorFlow-specific layer used with PyTorch backend
TimeDistributed(Dense(10))  # Not supported in PyTorch
```
**Resolution**: Use backend-compatible layers or switch backend

### NEURAL-2000-02: Backend Not Available
**Exception**: `UnsupportedBackendError`  
**Description**: Requested backend not installed or invalid  
**Example**:
```python
generate_code(backend='jax')  # Error: unsupported backend
```
**Resolution**: Install backend or use supported backend (tensorflow, pytorch, onnx)

### NEURAL-2001-01: Generation Failed
**Exception**: `CodeGenException`  
**Description**: Code generation failed for unknown reason  
**Resolution**: Check logs, report issue if persists

---

## Shape Propagation Errors (3000-3099)

### NEURAL-3000-01: Shape Mismatch
**Exception**: `ShapeMismatchError`  
**Description**: Incompatible tensor shapes between layers  
**Example**:
```python
# Dense expects 2D input, got 4D
Conv2D(32, (3, 3))  # Output: (N, H, W, C)
Dense(128)          # Error: expects (N, features)
```
**Resolution**: Add Flatten() or reshape layer

### NEURAL-3000-02: Invalid Shape
**Exception**: `InvalidShapeError`  
**Description**: Shape is malformed or invalid  
**Example**:
```python
input: (-1, 28, 28)  # Error: negative dimensions not allowed
```
**Resolution**: Use None for variable dimensions, positive integers for fixed

### NEURAL-3001-01: Dimension Mismatch
**Exception**: `ShapeMismatchError`  
**Description**: Operation requires specific number of dimensions  
**Example**:
```python
Conv2D on 1D input  # Error: Conv2D requires 3D or 4D input
```
**Resolution**: Check layer requirements, adjust input shape

### NEURAL-3001-02: Kernel Size Too Large
**Exception**: `ShapeMismatchError`  
**Description**: Kernel size exceeds input dimensions  
**Example**:
```python
input: (28, 28, 1)
Conv2D(32, (32, 32))  # Error: kernel larger than input
```
**Resolution**: Reduce kernel size or increase input size

---

## Parameter Validation Errors (4000-4099)

### NEURAL-4000-01: Invalid Parameter Value
**Exception**: `InvalidParameterError`  
**Description**: Parameter value is invalid  
**Example**:
```python
Dense(-10)  # Error: units must be positive
```
**Resolution**: Use valid parameter value (check documentation)

### NEURAL-4000-02: Missing Required Parameter
**Exception**: `InvalidParameterError`  
**Description**: Required parameter not provided  
**Example**:
```python
Conv2D()  # Error: filters parameter required
```
**Resolution**: Provide required parameters

### NEURAL-4000-03: Parameter Type Mismatch
**Exception**: `InvalidParameterError`  
**Description**: Parameter has wrong type  
**Example**:
```python
Dense("not_a_number")  # Error: units must be int
```
**Resolution**: Use correct parameter type

### NEURAL-4001-01: Invalid Range
**Exception**: `InvalidParameterError`  
**Description**: Parameter value outside valid range  
**Example**:
```python
Dropout(1.5)  # Error: rate must be in [0, 1)
```
**Resolution**: Use value within valid range

---

## HPO Errors (5000-5099)

### NEURAL-5000-01: Invalid HPO Config
**Exception**: `InvalidHPOConfigError`  
**Description**: HPO configuration is invalid  
**Example**:
```python
HPO(range(10, 1))  # Error: start > end
```
**Resolution**: Check HPO parameter ranges and constraints

### NEURAL-5000-02: Search Method Not Supported
**Exception**: `InvalidHPOConfigError`  
**Description**: HPO search method not recognized  
**Example**:
```python
train { search_method: "invalid_method" }
```
**Resolution**: Use supported method (random, grid, bayesian)

### NEURAL-5001-01: Search Failed
**Exception**: `HPOSearchError`  
**Description**: HPO search failed to find valid trials  
**Resolution**: Check search space, increase trials, adjust constraints

### NEURAL-5001-02: Optimization Timeout
**Exception**: `HPOSearchError`  
**Description**: HPO search exceeded time limit  
**Resolution**: Increase timeout or reduce search space

---

## Tracking Errors (6000-6099)

### NEURAL-6000-01: Experiment Not Found
**Exception**: `ExperimentNotFoundError`  
**Description**: Experiment ID does not exist  
**Resolution**: Check experiment ID, ensure experiment was created

### NEURAL-6001-01: Metric Logging Failed
**Exception**: `MetricLoggingError`  
**Description**: Failed to log metrics  
**Resolution**: Check tracking backend connectivity, permissions

---

## Cloud Execution Errors (7000-7099)

### NEURAL-7000-01: Connection Failed
**Exception**: `CloudConnectionError`  
**Description**: Failed to connect to cloud service  
**Resolution**: Check credentials, network connectivity, service status

### NEURAL-7000-02: Authentication Failed
**Exception**: `CloudConnectionError`  
**Description**: Cloud authentication failed  
**Resolution**: Check credentials, API keys, IAM permissions

### NEURAL-7001-01: Execution Failed
**Exception**: `CloudExecutionError`  
**Description**: Cloud execution failed  
**Resolution**: Check logs, resource availability, quotas

### NEURAL-7001-02: Resource Unavailable
**Exception**: `CloudExecutionError`  
**Description**: Required cloud resources not available  
**Resolution**: Check quotas, instance availability, region

---

## MLOps Errors (8000-8099)

### NEURAL-8000-01: Model Registry Error
**Exception**: `ModelRegistryError`  
**Description**: Model registry operation failed  
**Resolution**: Check registry connectivity, model ID, permissions

### NEURAL-8000-02: Model Not Found
**Exception**: `ModelRegistryError`  
**Description**: Model not found in registry  
**Resolution**: Check model ID, ensure model was registered

### NEURAL-8001-01: Deployment Failed
**Exception**: `DeploymentError`  
**Description**: Model deployment failed  
**Resolution**: Check deployment configuration, resources, permissions

### NEURAL-8001-02: Health Check Failed
**Exception**: `DeploymentError`  
**Description**: Deployment health check failed  
**Resolution**: Check deployment status, logs, endpoint availability

---

## System Errors (9000-9099)

### NEURAL-9000-01: Dependency Missing
**Exception**: `DependencyError`  
**Description**: Required dependency not installed  
**Example**:
```python
# PyTorch backend requested but not installed
generate_code(backend='pytorch')
```
**Resolution**: Install missing dependency (pip install torch)

### NEURAL-9000-02: File Operation Failed
**Exception**: `FileOperationError`  
**Description**: File read/write operation failed  
**Resolution**: Check file path, permissions, disk space

### NEURAL-9000-03: Configuration Error
**Exception**: `ConfigurationError`  
**Description**: Configuration is invalid or missing  
**Resolution**: Check configuration file, ensure all required fields present

### NEURAL-9001-01: Execution Error
**Exception**: `ExecutionError`  
**Description**: Model execution failed  
**Resolution**: Check model configuration, input data, resources

---

## Programmatic Error Handling

### Using Error Codes

```python
from neural.exceptions import InvalidParameterError

try:
    layer = Dense(-10)
except InvalidParameterError as e:
    if e.error_code == "NEURAL-4000-01":
        # Handle invalid parameter value
        print(f"Invalid value: {e.value}")
    elif e.error_code == "NEURAL-4000-02":
        # Handle missing parameter
        print(f"Missing: {e.parameter}")
```

### Error Code in Exception

```python
class NeuralException(Exception):
    """Base exception with error code support."""
    
    error_code: str = "NEURAL-0000-00"
    
    def __init__(self, message: str, **kwargs):
        self.message = message
        super().__init__(f"[{self.error_code}] {message}")
```

### Checking Error Categories

```python
def is_parser_error(error_code: str) -> bool:
    """Check if error code is parser-related."""
    return error_code.startswith("NEURAL-1")

def is_recoverable_error(error_code: str) -> bool:
    """Check if error is potentially recoverable."""
    # Most 4000 series (parameter errors) are recoverable
    return error_code.startswith("NEURAL-4")
```

---

## Error Severity Levels

| Level | Description | Example Codes |
|-------|-------------|---------------|
| **CRITICAL** | System failure, cannot continue | 9000-xx |
| **ERROR** | Operation failed, may be recoverable | 1000-xx, 2000-xx |
| **WARNING** | Potential issue, operation continues | Validation warnings |
| **INFO** | Informational message | Debug info |

---

## Best Practices

### 1. Always Include Error Codes

```python
# Good
raise InvalidParameterError(
    parameter="units",
    value=-10,
    layer_type="Dense"
)

# Error code automatically assigned by exception class
```

### 2. Log Error Codes

```python
import logging

try:
    # Operation
    pass
except NeuralException as e:
    logging.error(f"Error {e.error_code}: {e.message}")
```

### 3. Handle Specific Errors

```python
from neural.exceptions import (
    InvalidParameterError,
    ShapeMismatchError,
    DependencyError
)

try:
    model = build_model()
except InvalidParameterError as e:
    # Fix parameter and retry
    pass
except ShapeMismatchError as e:
    # Add reshape layer
    pass
except DependencyError as e:
    # Install dependency
    print(f"Install: {e.install_hint}")
```

### 4. Document Error Codes

```python
def parse_dsl(dsl_code: str) -> Model:
    """
    Parse DSL code into model.
    
    Raises:
        DSLSyntaxError (NEURAL-1000-01): Invalid syntax
        DSLValidationError (NEURAL-1001-01): Validation failed
    """
    pass
```

---

## Future Error Codes

Reserved ranges for future use:

- `10000-10999`: Visualization errors
- `11000-11999`: Collaboration errors
- `12000-12999`: AutoML errors
- `13000-13999`: Federated learning errors
- `14000-14999`: Monitoring errors
- `15000-15999`: Data management errors

---

## See Also

- [ERROR_MESSAGES_GUIDE.md](ERROR_MESSAGES_GUIDE.md) - Error message formatting
- [neural/exceptions.py](neural/exceptions.py) - Exception implementation
- [neural/error_suggestions.py](neural/error_suggestions.py) - Error suggestions
