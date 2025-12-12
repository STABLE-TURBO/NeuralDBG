# Parser Refactoring Usage Examples

This document provides examples of how to use the refactored parser modules.

## Using the Main Parser

The main parser interface remains unchanged:

```python
from neural.parser import NeuralParser, DSLValidationError

parser = NeuralParser()

# Parse a simple network
code = """
network SimpleNet {
    input: (28, 28, 1)
    layers:
        Dense(128)
        Dropout(0.2)
        Dense(10)
    optimizer: "adam"
    loss: "categorical_crossentropy"
}
"""

try:
    model = parser.parse(code)
    print(f"Model: {model['name']}")
    print(f"Layers: {len(model['layers'])}")
except DSLValidationError as e:
    print(f"Error at line {e.line}, column {e.column}: {e.message}")
```

## Using Layer Processors

For custom validation or parameter processing:

```python
from neural.parser import layer_processors as lp

# Validate a positive integer
units = 128
is_valid, error_msg = lp.validate_positive_integer(units, 'units', 'Dense')
if not is_valid:
    print(f"Validation error: {error_msg}")

# Validate dropout rate
rate = 0.5
is_valid, error_msg = lp.validate_dropout_rate(rate)
if is_valid:
    print(f"Dropout rate {rate} is valid")

# Extract and map parameters
param_values = [128, 'relu']
ordered, named = lp.extract_ordered_and_named_params(param_values)
params = lp.map_positional_to_dense_params(ordered)
# params = {'units': 128, 'activation': 'relu'}
```

## Using HPO Utilities

For hyperparameter optimization:

```python
from neural.parser import hpo_utils

# Parse HPO expressions
hpo_text = "Adam(learning_rate=HPO(log_range(0.001, 0.1)))"
hpo_exprs = hpo_utils.extract_hpo_expressions(hpo_text)
print(f"Found HPO expressions: {hpo_exprs}")

for expr in hpo_exprs:
    hpo_config = hpo_utils.parse_hpo_expression(expr)
    if hpo_config:
        print(f"HPO Config: {hpo_config}")
        # Output: {'type': 'log_range', 'min': 0.001, 'max': 0.1}

# Create categorical HPO from a list
values = [16, 32, 64, 128]
hpo_config = hpo_utils.create_categorical_hpo_from_list(values)
# hpo_config = {'hpo': {'type': 'categorical', 'values': [16, 32, 64, 128], ...}}
```

## Using Layer Handlers

For processing specific layer types:

```python
from neural.parser import layer_handlers as lh

def my_error_handler(msg, node):
    raise ValueError(msg)

def my_hpo_tracker(layer_type, param_name, hpo_data, node):
    print(f"Tracking HPO: {layer_type}.{param_name} = {hpo_data}")

# Process Dense layer parameters
param_values = {'units': 128, 'activation': 'relu'}
params = lh.process_dense_params(
    param_values,
    my_error_handler,
    my_hpo_tracker,
    None  # node
)
print(f"Dense params: {params}")

# Process Conv2D layer parameters
param_values = [32, (3, 3), 'relu']
params = lh.process_conv2d_params(
    param_values,
    my_error_handler,
    my_hpo_tracker,
    None  # node
)
print(f"Conv2D params: {params}")
# Output: {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}
```

## Using Network Processors

For network-level operations:

```python
from neural.parser import network_processors as np

# Detect framework
model = {
    'layers': [
        {'type': 'Dense', 'params': {'units': 128, 'framework': 'torch'}},
        {'type': 'Dense', 'params': {'units': 10}}
    ]
}
framework = np.detect_framework(model)
print(f"Detected framework: {framework}")  # Output: pytorch

# Process execution config
model = {'name': 'TPUModel', 'layers': []}
model = np.process_execution_config(model)
print(f"Execution config: {model.get('execution')}")
# Output: {'device': 'tpu'}

# Expand repeated layers
layers = [
    {'type': 'Dense', 'params': {'units': 128}},
    ({'type': 'Dropout', 'params': {'rate': 0.2}}, 3)  # Repeat 3 times
]
expanded = np.expand_repeated_layers(layers)
print(f"Expanded to {len(expanded)} layers")  # Output: 4 layers
```

## Using Value Extractors

For parse tree value extraction:

```python
from neural.parser import value_extractors as ve
from lark import Token

# Extract token value
token = Token('INT', '42')
value = ve.extract_token_value(token)
print(f"Extracted value: {value}")  # Output: 42

# Shift if token (remove leading token from items list)
items = [Token('DENSE', 'dense'), {'units': 128}]
items = ve.shift_if_token(items)
print(f"After shift: {items}")  # Output: [{'units': 128}]

# Validate parameter count
ordered_params = [128, 'relu', 'extra']
is_valid, error_msg = ve.validate_param_count(ordered_params, 2, 'Dense')
if not is_valid:
    print(f"Error: {error_msg}")
    # Output: Dense accepts at most 2 positional arguments
```

## Using Parser Utils

For error handling and parsing utilities:

```python
from neural.parser.parser_utils import (
    log_by_severity, DSLValidationError, safe_parse, split_params
)
from neural.parser.hpo_utils import Severity
from neural.parser import create_parser

# Severity-based logging
log_by_severity(Severity.INFO, "Parsing network...")
log_by_severity(Severity.WARNING, "Deprecated syntax detected")

# Safe parsing with better errors
parser = create_parser('network')
code = "network BadNet { invalid syntax }"
try:
    result = safe_parse(parser, code)
except DSLValidationError as e:
    print(f"Parse error at line {e.line}, col {e.column}")
    print(f"Message: {e.message}")

# Split parameters
param_string = "128, activation='relu', kernel_initializer='he_normal'"
params = split_params(param_string)
print(f"Parameters: {params}")
# Output: ['128', "activation='relu'", "kernel_initializer='he_normal'"]
```

## Custom Layer Processor Example

Creating a custom layer processor:

```python
from neural.parser import layer_processors as lp

def process_custom_layer_params(param_values, raise_error_fn, node):
    """Process parameters for a custom layer."""
    ordered, named = lp.extract_ordered_and_named_params(param_values)
    
    # Map positional parameters
    params = {}
    if len(ordered) >= 1:
        params['size'] = ordered[0]
    if len(ordered) >= 2:
        params['mode'] = ordered[1]
    
    # Merge named parameters
    params.update(named)
    
    # Validate required parameters
    if 'size' not in params:
        raise_error_fn("CustomLayer requires 'size' parameter", node)
    
    # Validate size
    is_valid, error_msg = lp.validate_positive_integer(params['size'], 'size', 'CustomLayer')
    if not is_valid:
        raise_error_fn(error_msg, node)
    
    return params

# Usage
try:
    params = process_custom_layer_params(
        [256, 'standard'],
        lambda msg, node: print(f"Error: {msg}"),
        None
    )
    print(f"Custom layer params: {params}")
except Exception as e:
    print(f"Failed: {e}")
```

## Error Handling Best Practices

```python
from neural.parser import NeuralParser, DSLValidationError
from neural.parser.hpo_utils import Severity

parser = NeuralParser()

code = """
network MyNet {
    input: (28, 28, 1)
    layers:
        Dense(-128)  # Invalid: negative units
        Dropout(0.5)
}
"""

try:
    model = parser.parse(code)
except DSLValidationError as e:
    # Error has line/column information
    if e.line and e.column:
        print(f"Error at line {e.line}, column {e.column}:")
        print(f"  {e.message}")
    else:
        print(f"Error: {e.message}")
    
    # Check severity
    if e.severity == Severity.ERROR:
        print("This is a non-recoverable error")
    elif e.severity == Severity.WARNING:
        print("This is a warning, parsing may continue")
```

## Testing Custom Validators

```python
import pytest
from neural.parser import layer_processors as lp

def test_validate_positive_integer():
    # Valid cases
    assert lp.validate_positive_integer(10, 'units', 'Dense') == (True, None)
    assert lp.validate_positive_integer(1, 'filters', 'Conv2D') == (True, None)
    
    # Invalid cases
    is_valid, error = lp.validate_positive_integer(-5, 'units', 'Dense')
    assert not is_valid
    assert 'positive' in error.lower()
    
    is_valid, error = lp.validate_positive_integer('abc', 'units', 'Dense')
    assert not is_valid
    assert 'integer' in error.lower()
    
    # HPO parameters should pass validation
    hpo_param = {'hpo': {'type': 'range', 'start': 1, 'end': 100}}
    assert lp.validate_positive_integer(hpo_param, 'units', 'Dense') == (True, None)

def test_extract_ordered_and_named_params():
    # Mixed parameters
    param_values = [128, 'relu', {'dropout': 0.2}, {'activation': 'tanh'}]
    ordered, named = lp.extract_ordered_and_named_params(param_values)
    
    assert ordered == [128, 'relu']
    assert named == {'dropout': 0.2, 'activation': 'tanh'}
    
    # Only positional
    param_values = [64, (3, 3)]
    ordered, named = lp.extract_ordered_and_named_params(param_values)
    
    assert ordered == [64, (3, 3)]
    assert named == {}

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

## Integration with Existing Code

The refactoring maintains backward compatibility:

```python
# Old code continues to work
from neural.parser import NeuralParser

parser = NeuralParser()
model = parser.parse("""
network OldStyleNet {
    input: (784,)
    layers:
        Dense(256)
        Dropout(0.3)
        Dense(128)
        Dense(10, activation='softmax')
    optimizer: "adam"
    loss: "categorical_crossentropy"
}
""")

# New code can use utilities
from neural.parser import layer_processors as lp

# Validate model layer parameters
for layer in model['layers']:
    if layer['type'] == 'Dense':
        units = layer['params'].get('units')
        if units:
            is_valid, _ = lp.validate_positive_integer(units, 'units', 'Dense')
            assert is_valid, f"Invalid units in Dense layer: {units}"
```
