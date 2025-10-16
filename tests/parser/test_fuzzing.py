"""
Fuzz testing infrastructure for the Neural parser.
Uses Hypothesis for property-based testing and fuzz testing.
"""

from hypothesis import given, strategies as st
from hypothesis.strategies import sampled_from, text, lists, integers
from neural.parser.parser import NeuralParser
import pytest

# Define common Neural DSL components
LAYER_TYPES = [
    "Dense", "Conv2D", "MaxPool2D", "Flatten", "Dropout", 
    "BatchNormalization", "Output"
]

ACTIVATION_FUNCTIONS = [
    "relu", "sigmoid", "tanh", "softmax", "linear"
]

@st.composite
def valid_layer_names(draw):
    """Strategy to generate valid layer names."""
    return draw(st.from_regex(r'[a-zA-Z][a-zA-Z0-9_]{0,29}'))

@st.composite
def valid_shapes(draw):
    """Strategy to generate valid input shapes."""
    dims = draw(st.integers(min_value=1, max_value=4))
    return draw(lists(integers(min_value=1, max_value=1000), min_size=dims, max_size=dims))

@st.composite
def valid_neural_programs(draw):
    """Strategy to generate valid Neural programs."""
    # Generate input shape
    shape = draw(valid_shapes())
    shape_str = ", ".join(str(x) for x in shape)
    
    # Generate layers
    num_layers = draw(st.integers(min_value=1, max_value=10))
    layers = []
    
    for _ in range(num_layers):
        layer_type = draw(sampled_from(LAYER_TYPES))
        layer_name = draw(valid_layer_names())
        activation = draw(sampled_from(ACTIVATION_FUNCTIONS))
        
        if layer_type == "Dense":
            units = draw(st.integers(min_value=1, max_value=1000))
            layers.append(f"{layer_name} = Dense({units}, activation={activation})")
        elif layer_type == "Conv2D":
            filters = draw(st.integers(min_value=1, max_value=128))
            kernel = draw(st.integers(min_value=1, max_value=7))
            layers.append(f"{layer_name} = Conv2D({filters}, {kernel}, activation={activation})")
        elif layer_type in ["MaxPool2D"]:
            pool_size = draw(st.integers(min_value=1, max_value=4))
            layers.append(f"{layer_name} = {layer_type}({pool_size})")
        elif layer_type in ["Flatten", "Dropout", "BatchNormalization"]:
            if layer_type == "Dropout":
                rate = draw(st.floats(min_value=0.1, max_value=0.9))
                layers.append(f"{layer_name} = {layer_type}({rate})")
            else:
                layers.append(f"{layer_name} = {layer_type}()")
        elif layer_type == "Output":
            units = draw(st.integers(min_value=1, max_value=1000))
            layers.append(f"{layer_name} = Output({units}, activation={activation})")
    
    # Combine into a valid program
    program = f"input = Input({shape_str})\n" + "\n".join(layers)
    return program

@given(valid_neural_programs())
def test_parser_valid_programs(program):
    """Test that valid generated programs can be parsed."""
    parser = NeuralParser()
    try:
        result = parser.parse(program)
        assert result is not None
    except Exception as e:
        pytest.fail(f"Parser failed on valid program:\n{program}\nError: {str(e)}")

@given(text())
def test_parser_arbitrary_input(s):
    """Test that parser handles arbitrary input gracefully."""
    parser = NeuralParser()
    try:
        parser.parse(s)
    except Exception as e:
        # We expect exceptions for invalid input, but they should be our custom exceptions
        assert "Neural" in str(e) or "Lark" in str(e), f"Unexpected exception type: {type(e)}"