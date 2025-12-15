"""
Edge cases and error handling integration tests.

Tests error handling and edge cases in the complete workflow across all backends.
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.parser.parser import create_parser, ModelTransformer, DSLValidationError
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.code_generation.code_generator import generate_code, generate_optimized_dsl
from neural.hpo.hpo import create_dynamic_model
from lark.exceptions import UnexpectedToken, VisitError

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


class MockTrial:
    """Mock trial for testing."""
    def suggest_categorical(self, name, choices):
        return choices[0]
    
    def suggest_float(self, name, low, high, step=None, log=False):
        return low
    
    def suggest_int(self, name, low, high):
        return low


class TestParsingErrorHandling:
    """Test error handling in DSL parsing."""

    def test_invalid_syntax_error(self):
        """Test: Invalid DSL syntax raises parsing error."""
        invalid_dsl = """
        network InvalidSyntax {
            input: (28, 28, 1)
            layers:
                Dense(64) @@@ invalid
        }
        """
        
        parser = create_parser("network")
        
        with pytest.raises(Exception):
            tree = parser.parse(invalid_dsl)

    def test_missing_required_field_error(self):
        """Test: Missing required fields raises error."""
        invalid_dsl = """
        network MissingFields {
            layers:
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        
        with pytest.raises(Exception):
            tree = parser.parse(invalid_dsl)

    def test_invalid_layer_parameter_error(self):
        """Test: Invalid layer parameters raise validation error."""
        invalid_dsl = """
        network InvalidParams {
            input: (28, 28, 1)
            layers:
                Dense(-64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        with pytest.raises((VisitError, DSLValidationError)):
            tree = parser.parse(invalid_dsl)
            model_config = transformer.transform(tree)

    def test_empty_layers_error(self):
        """Test: Empty layers list raises error."""
        invalid_dsl = """
        network EmptyLayers {
            input: (28, 28, 1)
            layers:
        }
        """
        
        parser = create_parser("network")
        
        with pytest.raises(Exception):
            tree = parser.parse(invalid_dsl)

    def test_invalid_input_shape_error(self):
        """Test: Invalid input shape raises error."""
        invalid_dsl = """
        network InvalidInputShape {
            input: (0, 0, 0)
            layers:
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        with pytest.raises((VisitError, DSLValidationError, ValueError)):
            tree = parser.parse(invalid_dsl)
            model_config = transformer.transform(tree)


class TestShapePropagationErrorHandling:
    """Test error handling in shape propagation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        os.chdir(self.original_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_incompatible_layer_sequence_error(self):
        """Test: Incompatible layer sequence raises shape error."""
        dsl_code = """
        network IncompatibleSequence {
            input: (28, 28, 1)
            layers:
                Conv2D(filters=32, kernel_size=3)
                Dense(64)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        with pytest.raises(Exception):
            for layer in model_config['layers']:
                current_shape = propagator.propagate(current_shape, layer)

    def test_invalid_conv_parameters_error(self):
        """Test: Invalid convolution parameters raise error."""
        dsl_code = """
        network InvalidConvParams {
            input: (5, 5, 3)
            layers:
                Conv2D(filters=32, kernel_size=10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        with pytest.raises(Exception):
            for layer in model_config['layers']:
                current_shape = propagator.propagate(current_shape, layer)

    def test_shape_mismatch_error(self):
        """Test: Shape mismatch between layers raises error."""
        dsl_code = """
        network ShapeMismatch {
            input: (10,)
            layers:
                Conv2D(filters=32, kernel_size=3)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        with pytest.raises(Exception):
            for layer in model_config['layers']:
                current_shape = propagator.propagate(current_shape, layer)


class TestCodeGenerationErrorHandling:
    """Test error handling in code generation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        os.chdir(self.original_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_invalid_backend_error(self):
        """Test: Invalid backend raises error."""
        dsl_code = """
        network ValidNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        with pytest.raises((ValueError, KeyError)):
            code = generate_code(model_config, 'invalid_backend')

    def test_missing_model_data_error(self):
        """Test: Missing model data raises error."""
        with pytest.raises((ValueError, KeyError, TypeError)):
            code = generate_code({}, 'pytorch')

    def test_invalid_layer_type_error(self):
        """Test: Invalid layer type in code generation."""
        model_config = {
            'input': {'shape': (28, 28, 1)},
            'layers': [
                {'type': 'InvalidLayerType', 'params': {}}
            ]
        }
        
        code = generate_code(model_config, 'pytorch')
        assert code is not None


class TestHPOErrorHandling:
    """Test error handling in HPO workflows."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        os.chdir(self.original_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_invalid_hpo_parameter_type(self):
        """Test: Invalid HPO parameter type handling."""
        dsl_code = """
        network InvalidHPO {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(invalid_type(64, 128)))
                Output(10)
        }
        """
        
        with pytest.raises(Exception):
            parser = create_parser("network")
            tree = parser.parse(dsl_code)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_missing_hpo_parameters_in_optimization(self):
        """Test: Handle missing HPO parameters gracefully."""
        dsl_code = """
        network MissingHPOParams {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Output(10)
        }
        """
        
        best_params = {}
        
        optimized_dsl = generate_optimized_dsl(dsl_code, best_params)
        
        assert 'HPO' in optimized_dsl

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_hpo_with_no_layers(self):
        """Test: HPO with minimal network."""
        dsl_code = """
        network MinimalHPO {
            input: (28, 28, 1)
            layers:
                Output(10)
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        assert len(model_dict['layers']) == 1
        
        trial = MockTrial()
        model = create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch')
        
        assert model is not None


class TestEdgeCases:
    """Test edge cases in workflows."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        os.chdir(self.original_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_single_layer_network(self):
        """Test: Network with single output layer."""
        dsl_code = """
        network SingleLayer {
            input: (784,)
            layers:
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        code = generate_code(model_config, 'pytorch')
        assert code is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_very_deep_network(self):
        """Test: Very deep network with many layers."""
        layers_str = "\n".join([f"                Dense(64, \"relu\")" for _ in range(50)])
        
        dsl_code = f"""
        network DeepNetwork {{
            input: (784,)
            layers:
{layers_str}
                Output(10)
        }}
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert len(model_config['layers']) == 51
        
        code = generate_code(model_config, 'pytorch')
        assert code is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_very_wide_network(self):
        """Test: Network with very large layer sizes."""
        dsl_code = """
        network WideNetwork {
            input: (784,)
            layers:
                Dense(4096, "relu")
                Dense(4096, "relu")
                Dense(4096, "relu")
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        code = generate_code(model_config, 'pytorch')
        assert 'nn.Linear' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_minimal_input_shape(self):
        """Test: Network with minimal input shape."""
        dsl_code = """
        network MinimalInput {
            input: (1,)
            layers:
                Dense(10)
                Output(2)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        assert current_shape == (None, 2)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_large_input_shape(self):
        """Test: Network with large input shape."""
        dsl_code = """
        network LargeInput {
            input: (512, 512, 3)
            layers:
                Conv2D(filters=32, kernel_size=3, strides=2)
                MaxPooling2D(pool_size=2)
                Conv2D(filters=64, kernel_size=3, strides=2)
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(128)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        assert 'nn.Conv2d' in code

    def test_special_characters_in_network_name(self):
        """Test: Network name with special characters."""
        dsl_code = """
        network My_Network_2024 {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert model_config is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_multiple_hpo_same_layer_type(self):
        """Test: Multiple HPO parameters on same layer type."""
        dsl_code = """
        network MultipleHPOSameType {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Dense(HPO(choice(32, 64)), "relu")
                Dense(HPO(choice(16, 32)), "relu")
                Output(10)
        }
        """
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        assert len(hpo_params) >= 3

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_hpo_with_extreme_ranges(self):
        """Test: HPO with extreme parameter ranges."""
        dsl_code = """
        network ExtremeRanges {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(1, 10000)), "relu")
                Dropout(HPO(range(0.01, 0.99, step=0.01)))
                Output(10)
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-6, 1.0)))
        }
        """
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        assert len(hpo_params) >= 2

    def test_empty_optimizer_config(self):
        """Test: Network without optimizer configuration."""
        dsl_code = """
        network NoOptimizer {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        assert code is not None

    def test_all_activations(self):
        """Test: Network with different activation functions."""
        dsl_code = """
        network AllActivations {
            input: (784,)
            layers:
                Dense(64, "relu")
                Dense(64, "sigmoid")
                Dense(64, "tanh")
                Dense(64, "softmax")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        assert code is not None


class TestRecoveryAndGracefulDegradation:
    """Test recovery from errors and graceful degradation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        os.chdir(self.original_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_partial_hpo_optimization(self):
        """Test: Partial HPO parameter optimization."""
        dsl_code = """
        network PartialHPO {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10)
        }
        """
        
        best_params = {'dense_units': 128}
        
        optimized_dsl = generate_optimized_dsl(dsl_code, best_params)
        
        assert 'Dense(128)' in optimized_dsl or 'Dense(128,' in optimized_dsl

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_auto_flatten_insertion(self):
        """Test: Auto-flatten insertion for incompatible layers."""
        dsl_code = """
        network AutoFlatten {
            input: (28, 28, 1)
            layers:
                Conv2D(filters=32, kernel_size=3)
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(64)
                Output(10)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        model_config["auto_flatten_output"] = True
        
        code = generate_code(model_config, 'pytorch', auto_flatten_output=True)
        
        assert code is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
