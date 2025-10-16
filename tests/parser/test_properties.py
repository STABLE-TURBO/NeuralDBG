"""
Property-based tests for the Neural parser.
Tests invariants and properties that should always hold true.
"""

import pytest
from hypothesis import given, strategies as st
from neural.parser.parser import NeuralParser
from typing import Dict, Any

def verify_layer_structure(layer: Dict[str, Any]) -> bool:
    """Verify that a layer dictionary has the required structure."""
    required_keys = {'type', 'name'}
    return all(key in layer for key in required_keys)

def verify_model_structure(model: Dict[str, Any]) -> bool:
    """Verify that a parsed model has the required structure."""
    if not isinstance(model, dict):
        return False
    
    required_keys = {'input', 'layers'}
    if not all(key in model for key in required_keys):
        return False
        
    if not isinstance(model['layers'], list):
        return False
        
    return all(verify_layer_structure(layer) for layer in model['layers'])

class TestParserProperties:
    """Property-based tests for the Neural parser."""
    
    def setup_method(self):
        self.parser = NeuralParser()
    
    @given(st.from_regex(r'input\s*=\s*Input\(\s*\d+\s*(,\s*\d+\s*)*\)'))
    def test_input_declaration(self, input_decl: str):
        """Test that valid input declarations are parsed correctly."""
        result = self.parser.parse(input_decl)
        assert 'input' in result
        assert 'shape' in result['input']
        assert isinstance(result['input']['shape'], list)
        
    def test_layer_name_uniqueness(self):
        """Test that duplicate layer names are caught."""
        program = """
        input = Input(28, 28)
        layer1 = Dense(128)
        layer1 = Dense(64)  # Duplicate name
        """
        with pytest.raises(Exception) as exc_info:
            self.parser.parse(program)
        assert "duplicate" in str(exc_info.value).lower()
        
    def test_output_layer_requirement(self):
        """Test that models require an Output layer."""
        program = """
        input = Input(28, 28)
        layer1 = Dense(128)
        layer2 = Dense(64)
        """
        with pytest.raises(Exception) as exc_info:
            self.parser.parse(program)
        assert "output" in str(exc_info.value).lower()
        
    @given(st.from_regex(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*Dense\(\s*\d+\s*(,\s*activation\s*=\s*[a-zA-Z_]+\s*)?\)'))
    def test_dense_layer_declaration(self, dense_decl: str):
        """Test that valid Dense layer declarations are parsed correctly."""
        program = f"""
        input = Input(10)
        {dense_decl}
        output = Output(1)
        """
        result = self.parser.parse(program)
        assert verify_model_structure(result)
        
    def test_shape_propagation_validation(self):
        """Test that incompatible shapes are caught."""
        program = """
        input = Input(28, 28)
        conv = Conv2D(32, 3)
        dense = Dense(128)  # Missing Flatten
        output = Output(10)
        """
        with pytest.raises(Exception) as exc_info:
            self.parser.parse(program)
        assert "shape" in str(exc_info.value).lower()
        
    @given(st.text())
    def test_parser_robustness(self, random_text: str):
        """Test that the parser handles arbitrary input without crashing."""
        try:
            self.parser.parse(random_text)
        except Exception as e:
            # Should only raise expected exception types
            assert isinstance(e, (ValueError, SyntaxError)) or "Neural" in str(e)