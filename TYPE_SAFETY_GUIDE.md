# Type Safety Guide for Neural DSL

## Overview

This guide documents the type safety strategy, type annotation conventions, and mypy configuration for the Neural DSL project.

## Type Safety Goals

### v1.0 Requirements
- ✅ 100% type coverage for all public APIs
- ✅ Strict mypy checking for core modules
- ✅ Runtime type validation for critical paths
- ✅ Comprehensive type documentation

## Mypy Configuration

### Strict Modules (v0.3.0+)

The following modules have strict type checking enabled:

```ini
# Core modules - Full type safety
neural.parser.*
neural.code_generation.*
neural.shape_propagation.*
neural.cli.*
neural.utils.*
neural.hpo.*
neural.dashboard.*
neural.exceptions.*

# Feature modules - Full type safety (v0.3.0+)
neural.automl.*
neural.integrations.*
neural.teams.*
neural.federated.*
neural.mlops.*
neural.monitoring.*
neural.api.*
neural.data.*
neural.cost.*
neural.benchmarks.*
```

### Strict Mode Settings

```ini
disallow_untyped_defs = True     # All functions must have type hints
warn_return_any = True           # Warn when returning Any
warn_redundant_casts = True      # Catch unnecessary casts
strict_optional = True           # Strict None checking
```

## Type Annotation Conventions

### Function Signatures

```python
from __future__ import annotations  # Enable forward references
from typing import Optional, List, Dict, Any, Union

def parse_layer(
    layer_def: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Parse a layer definition.
    
    Args:
        layer_def: Layer definition string
        context: Optional parsing context
        
    Returns:
        Parsed layer dictionary
        
    Raises:
        DSLSyntaxError: If layer_def is invalid
    """
    pass
```

### Class Definitions

```python
from typing import ClassVar, Protocol
from dataclasses import dataclass

@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    
    layer_type: str
    parameters: Dict[str, Any]
    input_shape: Optional[tuple[int, ...]] = None
    output_shape: Optional[tuple[int, ...]] = None
    
    # Class variable
    _registry: ClassVar[Dict[str, type]] = {}
    
    def validate(self) -> bool:
        """Validate the configuration."""
        return self.layer_type in self._registry
```

### Protocol Types

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Generator(Protocol):
    """Protocol for code generators."""
    
    def generate(self, model_data: Dict[str, Any]) -> str:
        """Generate code from model data."""
        ...
    
    def save(self, output_path: str) -> None:
        """Save generated code to file."""
        ...
```

### Generic Types

```python
from typing import TypeVar, Generic, Callable

T = TypeVar('T')
ModelData = TypeVar('ModelData', bound=dict)

class Cache(Generic[T]):
    """Generic cache implementation."""
    
    def __init__(self) -> None:
        self._cache: Dict[str, T] = {}
    
    def get(self, key: str) -> Optional[T]:
        return self._cache.get(key)
    
    def set(self, key: str, value: T) -> None:
        self._cache[key] = value
```

### Type Aliases

```python
from typing import NewType, TypeAlias

# Type aliases for clarity
LayerName: TypeAlias = str
ParameterValue: TypeAlias = Union[int, float, str, bool]
ShapeTuple: TypeAlias = tuple[Optional[int], ...]

# Distinct types (not just aliases)
UserId = NewType('UserId', int)
ModelId = NewType('ModelId', str)

def get_user(user_id: UserId) -> User:
    pass

# Type checker enforces distinction
user_id = UserId(123)
model_id = ModelId("model_v1")
# get_user(model_id)  # Type error!
```

## Common Type Patterns

### Optional Parameters

```python
# Use Optional for parameters that can be None
def create_layer(
    layer_type: str,
    units: int,
    activation: Optional[str] = None  # Clear that None is valid
) -> Layer:
    pass
```

### Union Types

```python
# Use Union for multiple possible types
def process_input(
    data: Union[list, dict, str]
) -> ProcessedData:
    if isinstance(data, list):
        # Type narrowing
        return process_list(data)
    elif isinstance(data, dict):
        return process_dict(data)
    else:
        return process_string(data)
```

### Return Type Annotations

```python
from typing import NoReturn, Never

def always_raises() -> NoReturn:
    """Function that never returns normally."""
    raise RuntimeError("Always fails")

def infinite_loop() -> Never:
    """Function that never returns at all."""
    while True:
        pass

def maybe_returns() -> Optional[str]:
    """Function that may return None."""
    if some_condition():
        return "value"
    return None
```

### Callable Types

```python
from typing import Callable

def apply_transform(
    data: List[int],
    transform: Callable[[int], int]
) -> List[int]:
    """Apply transformation function to each element."""
    return [transform(x) for x in data]

# With more specific signature
TransformFunc: TypeAlias = Callable[[int, str], bool]

def process(func: TransformFunc) -> None:
    result = func(42, "test")  # Type checked!
```

### Context Managers

```python
from typing import ContextManager, Iterator
from contextlib import contextmanager

@contextmanager
def managed_resource() -> Iterator[Resource]:
    """Type-safe context manager."""
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)

# Usage
with managed_resource() as res:
    res.use()  # Type checker knows res is Resource
```

## Advanced Type Features

### Literal Types

```python
from typing import Literal

Backend = Literal["tensorflow", "pytorch", "onnx"]

def generate_code(backend: Backend) -> str:
    """Generate code for specific backend."""
    if backend == "tensorflow":
        return generate_tf()
    elif backend == "pytorch":
        return generate_pytorch()
    else:
        return generate_onnx()

# Type error: generate_code("keras")  # Not a valid literal
```

### TypedDict

```python
from typing import TypedDict, NotRequired

class LayerDict(TypedDict):
    """Typed dictionary for layer data."""
    type: str
    params: Dict[str, Any]
    sublayers: list[LayerDict]  # Recursive!
    
class OptionalLayerDict(TypedDict, total=False):
    """Layer dict with optional fields."""
    type: str  # Still required
    activation: NotRequired[str]  # Optional
    dropout: NotRequired[float]  # Optional
```

### Generics with Constraints

```python
from typing import TypeVar

# Constrained type variable
Number = TypeVar('Number', int, float)

def add_numbers(a: Number, b: Number) -> Number:
    """Add two numbers of the same type."""
    return a + b

result1 = add_numbers(1, 2)      # int
result2 = add_numbers(1.0, 2.0)  # float
# result3 = add_numbers(1, 2.0)  # Type error!
```

### Self Type

```python
from typing import Self  # Python 3.11+

class Builder:
    """Builder pattern with proper types."""
    
    def with_layer(self, layer: str) -> Self:
        """Add layer and return self."""
        self.layers.append(layer)
        return self
    
    def build(self) -> Model:
        """Build the model."""
        return Model(self.layers)

# Chainable with type safety
builder = Builder().with_layer("Dense").with_layer("Conv2D").build()
```

## Runtime Type Checking

### Using isinstance for Type Narrowing

```python
def process_value(value: Union[int, str, list]) -> str:
    """Process value with type narrowing."""
    if isinstance(value, int):
        # Type checker knows value is int here
        return str(value * 2)
    elif isinstance(value, str):
        # Type checker knows value is str here
        return value.upper()
    else:
        # Type checker knows value is list here
        return ",".join(str(x) for x in value)
```

### Type Guards

```python
from typing import TypeGuard

def is_string_list(val: List[Any]) -> TypeGuard[List[str]]:
    """Check if all elements are strings."""
    return all(isinstance(x, str) for x in val)

def process(items: List[Any]) -> None:
    if is_string_list(items):
        # Type checker knows items is List[str] here
        for item in items:
            print(item.upper())  # Safe!
```

### Pydantic for Runtime Validation

```python
from pydantic import BaseModel, Field, validator

class LayerConfig(BaseModel):
    """Layer configuration with runtime validation."""
    
    layer_type: str = Field(..., min_length=1)
    units: int = Field(..., gt=0)
    activation: Optional[str] = None
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    
    @validator('activation')
    def validate_activation(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_ACTIVATIONS:
            raise ValueError(f"Invalid activation: {v}")
        return v

# Runtime validation
try:
    config = LayerConfig(
        layer_type="Dense",
        units=-10  # Validation error!
    )
except ValidationError as e:
    print(e)
```

## Type Checking Workflow

### Development

```bash
# Check types during development
python -m mypy neural/

# Check specific module
python -m mypy neural/parser/

# Show error codes
python -m mypy --show-error-codes neural/

# Generate coverage report
python -m mypy --html-report mypy-report neural/
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: 'v1.0.0'
    hooks:
      - id: mypy
        args: [--strict, --show-error-codes]
        additional_dependencies: [types-pyyaml, types-requests]
```

### CI/CD Integration

```yaml
# .github/workflows/type-check.yml
name: Type Check

on: [push, pull_request]

jobs:
  mypy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: mypy neural/ --strict
```

## Common Type Errors and Solutions

### Error: Incompatible return type

```python
# ❌ Wrong
def get_user(user_id: int) -> User:
    return None  # Error: incompatible return type

# ✅ Correct
def get_user(user_id: int) -> Optional[User]:
    return None  # OK
```

### Error: Missing return statement

```python
# ❌ Wrong
def process(data: str) -> int:
    if data:
        return len(data)
    # Error: missing return statement

# ✅ Correct
def process(data: str) -> int:
    if data:
        return len(data)
    return 0
```

### Error: Argument has incompatible type

```python
# ❌ Wrong
def add(a: int, b: int) -> int:
    return a + b

add(1, "2")  # Error: incompatible type

# ✅ Correct
add(1, int("2"))  # OK
```

### Error: Cannot determine type of expression

```python
# ❌ Wrong
items = []  # Type: List[Any]
items.append(1)

# ✅ Correct
items: List[int] = []  # Type: List[int]
items.append(1)
```

## Migration Strategy

### Phase 1: Core Modules (Complete ✅)
- [x] neural.parser
- [x] neural.code_generation
- [x] neural.shape_propagation
- [x] neural.cli
- [x] neural.utils

### Phase 2: Feature Modules (Complete ✅)
- [x] neural.hpo
- [x] neural.dashboard
- [x] neural.automl
- [x] neural.integrations
- [x] neural.teams
- [x] neural.federated
- [x] neural.mlops
- [x] neural.monitoring

### Phase 3: Experimental Modules (In Progress)
- [ ] neural.ai
- [ ] neural.cloud
- [ ] neural.visualization
- [ ] neural.tracking
- [ ] neural.collaboration

### Migration Process

1. **Add type stubs**
   ```python
   # Create .pyi stub files for gradual typing
   def function(x, y):  # .py file
       pass
   
   # function.pyi
   def function(x: int, y: str) -> bool: ...
   ```

2. **Add inline types**
   ```python
   # Migrate from stubs to inline
   def function(x: int, y: str) -> bool:
       pass
   ```

3. **Enable strict checking**
   ```ini
   # mypy.ini
   [mypy-neural.new_module.*]
   disallow_untyped_defs = True
   ```

4. **Fix type errors**
   - Run mypy
   - Fix reported errors
   - Add ignore comments for false positives
   - Document why ignores are needed

## Best Practices

1. **Always use type hints for public APIs**
2. **Use Optional explicitly for None values**
3. **Prefer Union over Any when possible**
4. **Use Literal types for string constants**
5. **Use Protocol for structural subtyping**
6. **Add docstring parameter types for documentation**
7. **Use TypedDict for dictionary structures**
8. **Run mypy regularly during development**
9. **Keep type stubs (.pyi) for third-party code**
10. **Document complex type relationships**

## Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 526 - Variable Annotations](https://peps.python.org/pep-0526/)
- [PEP 544 - Protocols](https://peps.python.org/pep-0544/)
- [typing-extensions](https://pypi.org/project/typing-extensions/)
