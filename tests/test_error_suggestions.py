"""
Test suite for error suggestion and formatting functionality.
"""

import pytest
from neural.error_suggestions import ErrorSuggestion, ErrorFormatter


class TestErrorSuggestion:
    """Test error suggestion generation."""
    
    def test_parameter_typo_suggestion(self):
        """Test parameter name typo suggestions."""
        assert "units" in ErrorSuggestion.suggest_parameter_fix("unit", "Dense")
        assert "filters" in ErrorSuggestion.suggest_parameter_fix("filter", "Conv2D")
        assert "kernel_size" in ErrorSuggestion.suggest_parameter_fix("kernal_size", "Conv2D")
        assert "activation" in ErrorSuggestion.suggest_parameter_fix("activaton", "Dense")
    
    def test_layer_typo_suggestion(self):
        """Test layer name typo suggestions."""
        assert "Conv2D" in ErrorSuggestion.suggest_layer_fix("Convolution2D")
        assert "MaxPooling2D" in ErrorSuggestion.suggest_layer_fix("MaxPool2D")
        assert "BatchNormalization" in ErrorSuggestion.suggest_layer_fix("BatchNorm")
        assert "LSTM" in ErrorSuggestion.suggest_layer_fix("Lstm")
    
    def test_layer_case_sensitivity(self):
        """Test case sensitivity suggestions for layers."""
        suggestion = ErrorSuggestion.suggest_layer_fix("dense")
        assert suggestion is not None
        assert "case-sensitive" in suggestion.lower()
        assert "Dense" in suggestion
    
    def test_activation_typo_suggestion(self):
        """Test activation function typo suggestions."""
        assert "relu" in ErrorSuggestion.suggest_activation_fix("Relu")
        assert "sigmoid" in ErrorSuggestion.suggest_activation_fix("Sigmoid")
        assert "leaky_relu" in ErrorSuggestion.suggest_activation_fix("leakyrelu")
    
    def test_optimizer_typo_suggestion(self):
        """Test optimizer name typo suggestions."""
        assert "Adam" in ErrorSuggestion.suggest_optimizer_fix("adam")
        assert "SGD" in ErrorSuggestion.suggest_optimizer_fix("sgd")
        assert "RMSprop" in ErrorSuggestion.suggest_optimizer_fix("rmsprop")
    
    def test_loss_typo_suggestion(self):
        """Test loss function typo suggestions."""
        assert "categorical_crossentropy" in ErrorSuggestion.suggest_loss_fix("crossentropy")
        assert "binary_crossentropy" in ErrorSuggestion.suggest_loss_fix("binarycrossentropy")
        assert "mse" in ErrorSuggestion.suggest_loss_fix("mean_squared_error")
    
    def test_shape_fix_dense_layer(self):
        """Test shape fix suggestions for Dense layers."""
        suggestion = ErrorSuggestion.suggest_shape_fix((None, 28, 28, 1), "Dense")
        assert suggestion is not None
        assert "Flatten" in suggestion
    
    def test_shape_fix_conv2d_layer(self):
        """Test shape fix suggestions for Conv2D layers."""
        suggestion = ErrorSuggestion.suggest_shape_fix((None, 28), "Conv2D")
        assert suggestion is not None
        assert "3D" in suggestion or "4D" in suggestion
    
    def test_shape_fix_negative_dimensions(self):
        """Test detection of negative dimensions."""
        suggestion = ErrorSuggestion.suggest_shape_fix((None, -28, 28), "Dense")
        assert suggestion is not None
        assert "negative" in suggestion.lower()
    
    def test_parameter_value_validation(self):
        """Test parameter value validation suggestions."""
        # Negative units
        suggestion = ErrorSuggestion.suggest_parameter_value_fix("units", -10, "Dense")
        assert suggestion is not None
        assert "positive" in suggestion.lower()
        
        # Invalid dropout rate
        suggestion = ErrorSuggestion.suggest_parameter_value_fix("rate", 1.5, "Dropout")
        assert suggestion is not None
        assert "between 0 and 1" in suggestion
        
        # Invalid kernel size
        suggestion = ErrorSuggestion.suggest_parameter_value_fix("kernel_size", -3, "Conv2D")
        assert suggestion is not None
        assert "positive" in suggestion.lower()
    
    def test_dependency_install_suggestion(self):
        """Test dependency installation suggestions."""
        suggestion = ErrorSuggestion.suggest_dependency_install("torch")
        assert "pip install" in suggestion
        assert "torch" in suggestion
        
        suggestion = ErrorSuggestion.suggest_dependency_install("optuna")
        assert "pip install" in suggestion
        assert "hpo" in suggestion.lower()
    
    def test_backend_fix_suggestion(self):
        """Test backend name fix suggestions."""
        suggestion = ErrorSuggestion.suggest_backend_fix("tf")
        assert "tensorflow" in suggestion.lower()
        
        suggestion = ErrorSuggestion.suggest_backend_fix("torch")
        assert "pytorch" in suggestion.lower()
        
        suggestion = ErrorSuggestion.suggest_backend_fix("invalid")
        assert "Available backends" in suggestion
    
    def test_syntax_fix_suggestion(self):
        """Test syntax error fix suggestions."""
        suggestion = ErrorSuggestion.suggest_syntax_fix("expected :")
        assert suggestion is not None
        assert "colon" in suggestion.lower()
        
        suggestion = ErrorSuggestion.suggest_syntax_fix("unexpected {")
        assert suggestion is not None
        assert "brace" in suggestion.lower()


class TestErrorFormatter:
    """Test error message formatting."""
    
    def test_parser_error_formatting(self):
        """Test parser error formatting with location."""
        error = ErrorFormatter.format_parser_error(
            "Missing colon",
            line=10,
            column=25,
            suggestion="Add ':' after the layer name"
        )
        assert "line 10" in error
        assert "column 25" in error
        assert "Missing colon" in error
        assert "💡" in error
    
    def test_parser_error_with_code_snippet(self):
        """Test parser error with code snippet."""
        error = ErrorFormatter.format_parser_error(
            "Syntax error",
            line=5,
            column=10,
            code_snippet="    Dense(128)",
            suggestion="Add activation function"
        )
        assert "Dense(128)" in error
        assert "💡" in error
        assert "line 5" in error
    
    def test_shape_error_formatting(self):
        """Test shape error formatting."""
        error = ErrorFormatter.format_shape_error(
            "Incompatible shapes",
            input_shape=(None, 28, 28, 1),
            expected_shape=(None, 784),
            layer_type="Dense",
            suggestion="Add Flatten() layer"
        )
        assert "Shape Error" in error
        assert "(None, 28, 28, 1)" in error
        assert "(None, 784)" in error
        assert "Dense" in error
        assert "Flatten" in error
    
    def test_parameter_error_formatting(self):
        """Test parameter error formatting."""
        error = ErrorFormatter.format_parameter_error(
            parameter="units",
            value=-10,
            layer_type="Dense",
            reason="Cannot be negative",
            expected="positive integer"
        )
        assert "Invalid Parameter" in error
        assert "units" in error
        assert "-10" in error
        assert "Dense" in error
        assert "positive integer" in error
    
    def test_parameter_error_auto_suggestion(self):
        """Test automatic suggestion generation for parameter errors."""
        error = ErrorFormatter.format_parameter_error(
            parameter="rate",
            value=1.5,
            layer_type="Dropout"
        )
        assert "💡" in error
        assert "between 0 and 1" in error
    
    def test_dependency_error_formatting(self):
        """Test dependency error formatting."""
        error = ErrorFormatter.format_dependency_error(
            dependency="torch",
            feature="PyTorch code generation",
            import_error="No module named 'torch'"
        )
        assert "Missing Dependency" in error
        assert "torch" in error
        assert "PyTorch" in error
        assert "pip install" in error
        assert "💡" in error
    
    def test_validation_error_formatting(self):
        """Test validation error formatting."""
        error = ErrorSuggestion.format_validation_error(
            "Invalid configuration",
            {"layer": "Dense", "parameter": "units", "value": -10},
            "Use a positive integer for units"
        )
        assert "Validation Error" in error
        assert "Dense" in error
        assert "units" in error
        assert "-10" in error
        assert "💡" in error


class TestErrorSuggestionIntegration:
    """Test integration of error suggestions with actual errors."""
    
    def test_complete_error_workflow(self):
        """Test complete error detection and suggestion workflow."""
        # Simulate a common error: wrong parameter name
        param_name = "unit"
        layer_type = "Dense"
        
        suggestion = ErrorSuggestion.suggest_parameter_fix(param_name, layer_type)
        assert suggestion is not None
        assert "units" in suggestion
        
        # Format the error
        error = ErrorFormatter.format_parameter_error(
            parameter=param_name,
            value=128,
            layer_type=layer_type,
            reason="Unknown parameter",
            suggestion=suggestion
        )
        
        assert "Invalid Parameter" in error
        assert "unit" in error
        assert "units" in error
        assert "💡" in error
    
    def test_multiple_suggestions(self):
        """Test handling of multiple potential suggestions."""
        # Test parameter with multiple possible issues
        value = -10
        param = "units"
        layer = "Dense"
        
        value_suggestion = ErrorSuggestion.suggest_parameter_value_fix(param, value, layer)
        assert value_suggestion is not None
        assert "positive" in value_suggestion.lower()
    
    def test_no_suggestion_available(self):
        """Test behavior when no suggestion is available."""
        # Test with valid input that shouldn't generate suggestions
        suggestion = ErrorSuggestion.suggest_parameter_fix("units", "Dense")
        assert suggestion is None
        
        suggestion = ErrorSuggestion.suggest_layer_fix("Dense")
        assert suggestion is None


@pytest.mark.parametrize("typo,correct", [
    ("unit", "units"),
    ("filter", "filters"),
    ("kernal_size", "kernel_size"),
    ("activaton", "activation"),
])
def test_parameter_typo_corrections(typo, correct):
    """Parameterized test for parameter typo corrections."""
    suggestion = ErrorSuggestion.suggest_parameter_fix(typo, "Conv2D")
    assert suggestion is not None
    assert correct in suggestion


@pytest.mark.parametrize("typo,correct", [
    ("Convolution2D", "Conv2D"),
    ("MaxPool2D", "MaxPooling2D"),
    ("BatchNorm", "BatchNormalization"),
    ("Lstm", "LSTM"),
])
def test_layer_typo_corrections(typo, correct):
    """Parameterized test for layer typo corrections."""
    suggestion = ErrorSuggestion.suggest_layer_fix(typo)
    assert suggestion is not None
    assert correct in suggestion


@pytest.mark.parametrize("activation,correct", [
    ("Relu", "relu"),
    ("Sigmoid", "sigmoid"),
    ("Softmax", "softmax"),
])
def test_activation_case_corrections(activation, correct):
    """Parameterized test for activation case corrections."""
    suggestion = ErrorSuggestion.suggest_activation_fix(activation)
    assert suggestion is not None
    assert correct in suggestion
