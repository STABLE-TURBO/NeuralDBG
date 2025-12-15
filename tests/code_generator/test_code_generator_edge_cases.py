import pytest
import os
import warnings
from neural.code_generation.code_generator import (
    generate_code, save_file, load_file, export_onnx, to_number,
    generate_tensorflow_layer, generate_pytorch_layer,
    _policy_ensure_2d_before_dense_tf, _policy_ensure_2d_before_dense_pt,
    generate_optimized_dsl
)
from neural.parser.parser import create_parser, ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator


class TestToNumberConversion:
    """Test to_number helper function edge cases"""
    
    def test_to_number_positive_int(self):
        """Test conversion of positive integer string"""
        assert to_number("42") == 42
        assert isinstance(to_number("42"), int)
    
    def test_to_number_negative_int(self):
        """Test conversion of negative integer string"""
        assert to_number("-15") == -15
    
    def test_to_number_positive_float(self):
        """Test conversion of positive float string"""
        assert abs(to_number("3.14") - 3.14) < 1e-9
        assert isinstance(to_number("3.14"), float)
    
    def test_to_number_negative_float(self):
        """Test conversion of negative float string"""
        assert abs(to_number("-2.5") - (-2.5)) < 1e-9
    
    def test_to_number_scientific_notation(self):
        """Test conversion of scientific notation"""
        assert abs(to_number("1e-3") - 0.001) < 1e-9
        assert abs(to_number("2.5e2") - 250.0) < 1e-9
    
    def test_to_number_invalid_string(self):
        """Test conversion of invalid string"""
        with pytest.raises(ValueError):
            to_number("invalid")
    
    def test_to_number_empty_string(self):
        """Test conversion of empty string"""
        with pytest.raises(ValueError):
            to_number("")


class TestFileOperations:
    """Test file saving and loading edge cases"""
    
    def test_save_file_success(self, tmp_path):
        """Test successful file save"""
        file_path = tmp_path / "test.py"
        content = "# Test code"
        save_file(str(file_path), content)
        assert file_path.exists()
        assert file_path.read_text() == content
    
    def test_save_file_invalid_path(self, tmp_path):
        """Test save to invalid path"""
        invalid_path = tmp_path / "nonexistent" / "subdir" / "test.py"
        with pytest.raises(IOError):
            save_file(str(invalid_path), "content")
    
    def test_save_file_empty_content(self, tmp_path):
        """Test saving empty content"""
        file_path = tmp_path / "empty.py"
        save_file(str(file_path), "")
        assert file_path.exists()
        assert file_path.read_text() == ""
    
    def test_load_file_unsupported_extension(self, tmp_path):
        """Test loading file with unsupported extension"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")
        with pytest.raises(ValueError) as exc_info:
            load_file(str(file_path))
        assert "Unsupported file type" in str(exc_info.value)
    
    def test_load_file_nonexistent(self):
        """Test loading nonexistent file"""
        with pytest.raises(FileNotFoundError):
            load_file("nonexistent.neural")


class TestModelDataValidation:
    """Test model data validation edge cases"""
    
    def test_generate_code_invalid_model_data_type(self):
        """Test generate_code with invalid model_data type"""
        with pytest.raises(ValueError) as exc_info:
            generate_code("invalid", "tensorflow")
        assert "Invalid model_data format" in str(exc_info.value)
    
    def test_generate_code_missing_layers_key(self):
        """Test generate_code with missing layers key"""
        model_data = {"input": {"shape": (10,)}}
        with pytest.raises(ValueError) as exc_info:
            generate_code(model_data, "tensorflow")
        assert "Invalid model_data format" in str(exc_info.value)
    
    def test_generate_code_missing_input_key(self):
        """Test generate_code with missing input key"""
        model_data = {"layers": []}
        with pytest.raises(ValueError) as exc_info:
            generate_code(model_data, "tensorflow")
        assert "Invalid model_data format" in str(exc_info.value)
    
    def test_generate_code_invalid_layer_format(self):
        """Test generate_code with invalid layer format"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": ["invalid_layer"],
            "loss": "mse",
            "optimizer": "Adam"
        }
        with pytest.raises(ValueError) as exc_info:
            generate_code(model_data, "tensorflow")
        assert "Invalid layer format" in str(exc_info.value)
    
    def test_generate_code_layer_missing_type(self):
        """Test generate_code with layer missing type"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"params": {"units": 64}}],
            "loss": "mse",
            "optimizer": "Adam"
        }
        with pytest.raises(ValueError) as exc_info:
            generate_code(model_data, "tensorflow")
        assert "Invalid layer format" in str(exc_info.value)
    
    def test_generate_code_unsupported_backend(self):
        """Test generate_code with unsupported backend"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": "mse",
            "optimizer": "Adam"
        }
        with pytest.raises(ValueError) as exc_info:
            generate_code(model_data, "unsupported_backend")
        assert "Unsupported backend" in str(exc_info.value)


class TestLayerMultiplication:
    """Test layer multiplication edge cases"""
    
    def test_multiply_value_zero(self):
        """Test layer with multiply value of 0"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}, "multiply": 0}],
            "loss": "mse",
            "optimizer": "Adam"
        }
        with pytest.raises(ValueError) as exc_info:
            generate_code(model_data, "tensorflow")
        assert "Invalid 'multiply' value" in str(exc_info.value)
    
    def test_multiply_value_negative(self):
        """Test layer with negative multiply value"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}, "multiply": -1}],
            "loss": "mse",
            "optimizer": "Adam"
        }
        with pytest.raises(ValueError) as exc_info:
            generate_code(model_data, "tensorflow")
        assert "Invalid 'multiply' value" in str(exc_info.value)
    
    def test_multiply_value_non_integer(self):
        """Test layer with non-integer multiply value"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}, "multiply": 2.5}],
            "loss": "mse",
            "optimizer": "Adam"
        }
        with pytest.raises(ValueError) as exc_info:
            generate_code(model_data, "tensorflow")
        assert "Invalid 'multiply' value" in str(exc_info.value)
    
    def test_multiply_value_one(self):
        """Test layer with multiply value of 1 (default)"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}, "multiply": 1}],
            "loss": "mse",
            "optimizer": "Adam"
        }
        code = generate_code(model_data, "tensorflow")
        # Should generate code successfully
        assert "Dense(units=64)" in code
    
    def test_multiply_value_large(self):
        """Test layer with large multiply value"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}, "multiply": 100}],
            "loss": "mse",
            "optimizer": "Adam"
        }
        code = generate_code(model_data, "tensorflow")
        # Should create 100 copies
        assert code.count("Dense(units=64)") == 100


class TestTensorFlowLayerGeneration:
    """Test TensorFlow layer generation edge cases"""
    
    def test_transformer_encoder_default_params(self):
        """Test TransformerEncoder with default parameters"""
        layer_code = generate_tensorflow_layer("TransformerEncoder", {})
        assert "MultiHeadAttention" in layer_code
        assert "num_heads=8" in layer_code
    
    def test_transformer_encoder_custom_params(self):
        """Test TransformerEncoder with custom parameters"""
        layer_code = generate_tensorflow_layer("TransformerEncoder", {
            "num_heads": 4,
            "ff_dim": 256,
            "dropout": 0.2
        })
        assert "num_heads=4" in layer_code
        assert "ff_dim=256" in layer_code or "Dense(256" in layer_code
    
    def test_batch_normalization_default_params(self):
        """Test BatchNormalization with default parameters"""
        layer_code = generate_tensorflow_layer("BatchNormalization", {})
        assert layer_code == "layers.BatchNormalization()"
    
    def test_batch_normalization_custom_params(self):
        """Test BatchNormalization with custom parameters"""
        layer_code = generate_tensorflow_layer("BatchNormalization", {
            "momentum": 0.95,
            "epsilon": 0.002
        })
        assert "momentum=0.95" in layer_code
        assert "epsilon=0.002" in layer_code
    
    def test_conv2d_with_list_kernel(self):
        """Test Conv2D with kernel_size as list"""
        layer_code = generate_tensorflow_layer("Conv2D", {
            "filters": 32,
            "kernel_size": [3, 3]
        })
        assert "filters=32" in layer_code
        assert "kernel_size=3" in layer_code
    
    def test_maxpooling2d_with_strides(self):
        """Test MaxPooling2D with strides"""
        layer_code = generate_tensorflow_layer("MaxPooling2D", {
            "pool_size": (2, 2),
            "strides": 2
        })
        assert "pool_size=(2, 2)" in layer_code
        assert "strides=2" in layer_code
    
    def test_lstm_with_return_sequences(self):
        """Test LSTM with return_sequences"""
        layer_code = generate_tensorflow_layer("LSTM", {
            "units": 128,
            "return_sequences": True
        })
        assert "units=128" in layer_code
        assert "return_sequences=True" in layer_code
    
    def test_unsupported_layer_type_warning(self):
        """Test warning for unsupported layer type"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer_code = generate_tensorflow_layer("UnsupportedLayer", {})
            assert len(w) == 1
            assert "Unsupported layer type" in str(w[0].message)
            assert layer_code is None


class TestPyTorchLayerGeneration:
    """Test PyTorch layer generation edge cases"""
    
    def test_conv2d_with_dict_filters(self):
        """Test Conv2D with filters as dict"""
        layer_code = generate_pytorch_layer("Conv2D", {
            "filters": {"value": 32},
            "kernel_size": 3
        }, input_shape=(1, 3, 28, 28))
        assert "out_channels=32" in layer_code
    
    def test_dense_with_dict_activation(self):
        """Test Dense with activation as dict"""
        layer_code = generate_pytorch_layer("Dense", {
            "units": {"value": 64},
            "activation": {"value": "relu"}
        }, input_shape=(1, 128))
        assert "out_features=64" in layer_code
        assert "ReLU" in layer_code
    
    def test_dropout_with_dict_rate(self):
        """Test Dropout with rate as dict"""
        layer_code = generate_pytorch_layer("Dropout", {
            "rate": {"value": 0.5}
        })
        assert "p=0.5" in layer_code
    
    def test_lstm_with_dict_input_size(self):
        """Test LSTM with input_size as dict"""
        layer_code = generate_pytorch_layer("LSTM", {
            "units": {"value": 128}
        }, input_shape=(1, 10, 32))
        assert "hidden_size=128" in layer_code
    
    def test_batch_normalization_default_features(self):
        """Test BatchNormalization with default num_features"""
        layer_code = generate_pytorch_layer("BatchNormalization", {})
        assert "BatchNorm2d" in layer_code
    
    def test_dense_invalid_activation(self):
        """Test Dense with invalid activation (should use Identity)"""
        layer_code = generate_pytorch_layer("Dense", {
            "units": 64,
            "activation": "invalid"
        }, input_shape=(1, 128))
        assert "Identity" in layer_code
    
    def test_transformer_encoder_with_dict_params(self):
        """Test TransformerEncoder with dict parameters"""
        layer_code = generate_pytorch_layer("TransformerEncoder", {
            "d_model": {"value": 512},
            "num_heads": {"value": 8},
            "ff_dim": {"value": 2048},
            "dropout": {"value": 0.1}
        })
        assert "d_model=512" in layer_code
        assert "nhead=8" in layer_code


class TestPolicyHelpers:
    """Test policy helper functions for 2D input enforcement"""
    
    def test_ensure_2d_tf_already_2d(self):
        """Test TF policy with already 2D input"""
        propagator = ShapePropagator()
        current_shape = (None, 128)
        insert_code, output_shape = _policy_ensure_2d_before_dense_tf(
            1, False, propagator, current_shape
        )
        assert insert_code == ""
        assert output_shape == current_shape
    
    def test_ensure_2d_tf_higher_rank_auto_flatten(self):
        """Test TF policy with higher rank and auto_flatten"""
        propagator = ShapePropagator()
        current_shape = (None, 7, 7, 64)
        insert_code, output_shape = _policy_ensure_2d_before_dense_tf(
            3, True, propagator, current_shape
        )
        assert "Flatten" in insert_code
        assert len(output_shape) == 2
    
    def test_ensure_2d_tf_higher_rank_no_auto_flatten(self):
        """Test TF policy with higher rank without auto_flatten"""
        propagator = ShapePropagator()
        current_shape = (None, 7, 7, 64)
        with pytest.raises(ValueError) as exc_info:
            _policy_ensure_2d_before_dense_tf(3, False, propagator, current_shape)
        assert "expects 2D input" in str(exc_info.value)
    
    def test_ensure_2d_pt_already_2d(self):
        """Test PT policy with already 2D input"""
        propagator = ShapePropagator()
        current_shape = (None, 128)
        forward_code_body = []
        output_shape = _policy_ensure_2d_before_dense_pt(
            1, False, forward_code_body, propagator, current_shape
        )
        assert len(forward_code_body) == 0
        assert output_shape == current_shape
    
    def test_ensure_2d_pt_higher_rank_auto_flatten(self):
        """Test PT policy with higher rank and auto_flatten"""
        propagator = ShapePropagator()
        current_shape = (None, 64, 7, 7)
        forward_code_body = []
        output_shape = _policy_ensure_2d_before_dense_pt(
            3, True, forward_code_body, propagator, current_shape
        )
        assert len(forward_code_body) == 1
        assert "view" in forward_code_body[0]
    
    def test_ensure_2d_pt_higher_rank_no_auto_flatten(self):
        """Test PT policy with higher rank without auto_flatten"""
        propagator = ShapePropagator()
        current_shape = (None, 64, 7, 7)
        forward_code_body = []
        with pytest.raises(ValueError) as exc_info:
            _policy_ensure_2d_before_dense_pt(3, False, forward_code_body, propagator, current_shape)
        assert "expects 2D input" in str(exc_info.value)


class TestOptimizerHandling:
    """Test optimizer configuration edge cases"""
    
    def test_optimizer_as_string(self):
        """Test optimizer specified as simple string"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": "mse",
            "optimizer": "SGD"
        }
        code = generate_code(model_data, "tensorflow")
        assert "SGD" in code
    
    def test_optimizer_as_dict_no_params(self):
        """Test optimizer as dict with no parameters"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": "mse",
            "optimizer": {"type": "Adam"}
        }
        code = generate_code(model_data, "tensorflow")
        assert "Adam()" in code
    
    def test_optimizer_with_string_param(self):
        """Test optimizer with string parameter"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": "mse",
            "optimizer": {"type": "Adam", "params": {"decay": "exponential"}}
        }
        code = generate_code(model_data, "tensorflow")
        assert "decay='exponential'" in code


class TestLossHandling:
    """Test loss function handling edge cases"""
    
    def test_loss_as_none(self):
        """Test loss specified as None (should use default)"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": None,
            "optimizer": "Adam"
        }
        code = generate_code(model_data, "tensorflow")
        assert "loss='categorical_crossentropy'" in code
    
    def test_loss_as_dict(self):
        """Test loss specified as dict"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": {"value": "mse"},
            "optimizer": "Adam"
        }
        code = generate_code(model_data, "tensorflow")
        assert "loss='mse'" in code
    
    def test_loss_as_empty_dict(self):
        """Test loss as empty dict (should use default)"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": {},
            "optimizer": "Adam"
        }
        code = generate_code(model_data, "tensorflow")
        assert "loss=" in code


class TestTrainingConfiguration:
    """Test training configuration edge cases"""
    
    def test_training_config_with_mixed_precision(self):
        """Test training config with mixed precision"""
        model_data = {
            "input": {"shape": (28, 28, 1)},
            "layers": [{"type": "Dense", "params": {"units": 10}}],
            "loss": "mse",
            "optimizer": "Adam",
            "training_config": {"mixed_precision": True}
        }
        code = generate_code(model_data, "tensorflow")
        assert "mixed_precision" in code
        assert "set_global_policy" in code
    
    def test_training_config_with_save_path(self):
        """Test training config with save path"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": "mse",
            "optimizer": "Adam",
            "training_config": {"save_path": "model.h5", "epochs": 10, "batch_size": 32}
        }
        code = generate_code(model_data, "tensorflow")
        assert "model.save('model.h5')" in code
    
    def test_pytorch_training_config(self):
        """Test PyTorch training configuration"""
        model_data = {
            "input": {"shape": (10,)},
            "layers": [{"type": "Dense", "params": {"units": 64}}],
            "loss": "mse",
            "optimizer": "Adam",
            "training_config": {"epochs": 5, "batch_size": 64, "save_path": "model.pt"}
        }
        code = generate_code(model_data, "pytorch")
        assert "for epoch in range(5)" in code
        assert "batch_size=64" in code
        assert "torch.save" in code


class TestResidualLayers:
    """Test residual layer generation edge cases"""
    
    def test_residual_tensorflow(self):
        """Test residual block in TensorFlow"""
        model_data = {
            "input": {"shape": (32, 32, 3)},
            "layers": [
                {
                    "type": "Residual",
                    "sub_layers": [
                        {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3}},
                        {"type": "BatchNormalization"}
                    ]
                }
            ],
            "loss": "mse",
            "optimizer": "Adam"
        }
        code = generate_code(model_data, "tensorflow")
        assert "Residual block" in code
        assert "Add()" in code
    
    def test_residual_empty_sublayers(self):
        """Test residual block with empty sublayers"""
        model_data = {
            "input": {"shape": (32, 32, 3)},
            "layers": [
                {"type": "Residual", "sub_layers": []}
            ],
            "loss": "mse",
            "optimizer": "Adam",
            "auto_flatten_output": True
        }
        code = generate_code(model_data, "tensorflow")
        # Should still generate residual structure
        assert "residual_input" in code


class TestGenerateOptimizedDSL:
    """Test optimized DSL generation with HPO results"""
    
    def test_generate_optimized_dsl_basic(self):
        """Test basic optimized DSL generation"""
        config = """
        network Test {
            input: (10,)
            layers: Dense(HPO(choice(64, 128)))
            optimizer: Adam(learning_rate=HPO(log_range(0.001, 0.1)))
        }
        """
        best_params = {
            'dense_units': 128,
            'learning_rate': 0.01
        }
        result = generate_optimized_dsl(config, best_params)
        assert "128" in result
        assert "0.01" in result
    
    def test_generate_optimized_dsl_no_hpo(self):
        """Test optimized DSL with no HPO parameters"""
        config = """
        network Test {
            input: (10,)
            layers: Dense(64)
        }
        """
        best_params = {}
        result = generate_optimized_dsl(config, best_params)
        assert "Dense(64)" in result


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
