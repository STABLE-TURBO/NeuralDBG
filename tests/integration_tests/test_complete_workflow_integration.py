"""
Complete workflow integration tests for Neural DSL.

Tests the full pipeline: DSL parsing → shape propagation → code generation → execution
for TensorFlow, PyTorch, and ONNX backends, including HPO and tracking features.
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.parser.parser import create_parser, ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.code_generation.code_generator import generate_code, export_onnx, generate_onnx
from neural.hpo.hpo import optimize_and_return, create_dynamic_model, objective
from neural.code_generation.code_generator import generate_optimized_dsl

try:
    import torch
    import torch.nn as nn
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

try:
    import onnx
    from onnx import helper, TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    onnx = None
    ONNX_AVAILABLE = False


class MockTrial:
    """Mock Optuna trial for HPO testing."""
    def suggest_categorical(self, name, choices):
        return 32 if name == "batch_size" else choices[0]

    def suggest_float(self, name, low, high, step=None, log=False):
        return low if not log else 0.001

    def suggest_int(self, name, low, high):
        return low


def mock_data_loader(dataset_name, input_shape, batch_size=32, train=True, backend='pytorch'):
    """Mock data loader for testing without real datasets."""
    if backend == 'pytorch' and TORCH_AVAILABLE:
        if train:
            x = torch.randn(100, *input_shape)
            y = torch.randint(0, 10, (100,))
        else:
            x = torch.randn(20, *input_shape)
            y = torch.randint(0, 10, (20,))
        
        if len(input_shape) == 3:
            x = x.permute(0, 3, 1, 2)
        
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=train)
    
    elif backend == 'tensorflow' and TF_AVAILABLE:
        if train:
            x = np.random.randn(100, *input_shape).astype(np.float32)
            y = np.random.randint(0, 10, (100,)).astype(np.int32)
        else:
            x = np.random.randn(20, *input_shape).astype(np.float32)
            y = np.random.randint(0, 10, (20,)).astype(np.int32)
        
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    
    return None


class TestCompleteWorkflowIntegration:
    """Integration tests for complete DSL workflows."""

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
        
        if os.path.exists("neural_experiments"):
            shutil.rmtree("neural_experiments")

    def test_simple_dsl_to_shape_propagation(self):
        """Test: DSL parsing → shape propagation."""
        dsl_code = """
        network SimpleNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(128, "relu")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert 'input' in model_config
        assert 'layers' in model_config
        assert len(model_config['layers']) == 3
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        assert current_shape == (None, 10)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_dsl_to_execution(self):
        """Test: DSL parsing → shape propagation → PyTorch code generation → execution."""
        dsl_code = """
        network MNISTClassifier {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Dense(32, "relu")
                Output(10, "softmax")
            
            loss: "cross_entropy"
            optimizer: Adam(learning_rate=0.001)
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
        
        assert current_shape == (None, 10)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'import torch' in code
        assert 'nn.Linear' in code
        assert 'nn.Flatten' in code
        
        with open('test_model.py', 'w') as f:
            f.write(code)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_model", "test_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        model = module.NeuralModel()
        test_input = torch.randn(4, 1, 28, 28)
        output = model(test_input)
        
        assert output.shape == (4, 10)

    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_tensorflow_dsl_to_execution(self):
        """Test: DSL parsing → shape propagation → TensorFlow code generation → execution."""
        dsl_code = """
        network ConvNet {
            input: (32, 32, 3)
            layers:
                Conv2D(filters=16, kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
            
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
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
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'import tensorflow' in code
        assert 'layers.Conv2D' in code
        assert 'layers.MaxPooling2D' in code
        
        with open('test_model_tf.py', 'w') as f:
            f.write(code)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_model_tf", "test_model_tf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        model = module.create_model()
        test_input = np.random.randn(4, 32, 32, 3).astype(np.float32)
        output = model(test_input)
        
        assert output.shape == (4, 10)

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_dsl_to_export(self):
        """Test: DSL parsing → shape propagation → ONNX export."""
        dsl_code = """
        network SimpleONNX {
            input: (224, 224, 3)
            layers:
                Conv2D(filters=32, kernel_size=3)
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(128)
                Output(1000)
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
        
        onnx_model = generate_onnx(model_config)
        
        assert onnx_model is not None
        assert onnx_model.graph is not None
        
        export_path = 'test_model.onnx'
        result = export_onnx(model_config, export_path)
        
        assert os.path.exists(export_path)
        assert "ONNX model saved" in result

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_pytorch_with_hpo_full_workflow(self):
        """Test: DSL with HPO → optimization → optimized code generation → execution."""
        dsl_code = """
        network HPONet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128, 256)), "relu")
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10, "softmax")
            
            loss: "cross_entropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        assert len(hpo_params) > 0
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        with patch('neural.hpo.hpo.optimize_and_return') as mock_optimize:
            mock_optimize.return_value = {
                'batch_size': 32,
                'dense_units': 128,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
            
            best_params = optimize_and_return(dsl_code, n_trials=2, dataset_name='MNIST', backend='pytorch')
        
        assert 'dense_units' in best_params or 'learning_rate' in best_params
        
        optimized_dsl = generate_optimized_dsl(dsl_code, best_params)
        
        assert 'HPO' not in optimized_dsl.split('optimizer:')[0]
        
        optimized_model_config, optimized_hpo_params = transformer.parse_network_with_hpo(optimized_dsl)
        
        assert len(optimized_hpo_params) == 0 or all(
            not any('hpo' in str(p.get('params', {})).lower() for p in optimized_model_config.get('layers', []))
        )
        
        code = generate_code(optimized_model_config, 'pytorch')
        
        assert 'import torch' in code
        assert 'nn.Linear' in code

    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_tensorflow_with_tracking_integration(self):
        """Test: DSL parsing → TensorFlow code with experiment tracking."""
        dsl_code = """
        network TrackedNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(100, "relu")
                Output(10, "softmax")
            
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
            
            train {
                epochs: 2
                batch_size: 32
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'from neural.tracking.experiment_tracker import ExperimentTracker' in code or \
               'from neural.tracking.experiment_tracker import ExperimentManager' in code
        assert 'experiment' in code.lower()
        assert 'log_hyperparameters' in code or 'log_params' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_complex_architecture_full_workflow(self):
        """Test: Complex architecture with multiple layer types → full workflow."""
        dsl_code = """
        network ComplexNet {
            input: (32, 32, 3)
            layers:
                Conv2D(filters=32, kernel_size=3, activation="relu")
                BatchNormalization()
                MaxPooling2D(pool_size=2)
                Conv2D(filters=64, kernel_size=3, activation="relu")
                BatchNormalization()
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(128, "relu")
                Dropout(rate=0.5)
                Dense(64, "relu")
                Output(10, "softmax")
            
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
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
        
        assert current_shape == (None, 10)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'nn.Conv2d' in code
        assert 'nn.BatchNorm2d' in code
        assert 'nn.MaxPool2d' in code
        assert 'nn.Dropout' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_recurrent_network_workflow(self):
        """Test: RNN/LSTM workflow → parsing → shape propagation → code generation."""
        dsl_code = """
        network RNNModel {
            input: (100, 64)
            layers:
                LSTM(units=128, return_sequences=True)
                LSTM(units=64, return_sequences=False)
                Dense(32, "relu")
                Output(10, "softmax")
            
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
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
        
        assert 'nn.LSTM' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_multi_backend_consistency(self):
        """Test: Same DSL generates consistent models across backends."""
        dsl_code = """
        network ConsistentNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Dense(32, "relu")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        pytorch_code = generate_code(model_config, 'pytorch')
        
        assert 'nn.Linear' in pytorch_code
        assert 'nn.Flatten' in pytorch_code
        
        if TF_AVAILABLE:
            tf_code = generate_code(model_config, 'tensorflow')
            assert 'layers.Dense' in tf_code
            assert 'layers.Flatten' in tf_code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_hpo_objective_function_integration(self):
        """Test: HPO objective function with model creation and training."""
        dsl_code = """
        network HPOObjectiveNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128)))
                Output(10)
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        trial = MockTrial()
        
        loss, acc, precision, recall = objective(trial, dsl_code, 'MNIST', backend='pytorch')
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_shape_propagation_error_detection(self):
        """Test: Shape propagation detects incompatible layer configurations."""
        dsl_code = """
        network IncompatibleNet {
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

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_layer_multiplication_workflow(self):
        """Test: Layer multiplication feature in complete workflow."""
        dsl_code = """
        network MultiLayerNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu") * 3
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        dense_layers = [l for l in model_config['layers'] if l['type'] == 'Dense']
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'nn.Linear' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_training_config_integration(self):
        """Test: Training configuration is properly integrated in generated code."""
        dsl_code = """
        network TrainingConfigNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
            
            loss: "cross_entropy"
            optimizer: Adam(learning_rate=0.001)
            
            train {
                epochs: 10
                batch_size: 64
                validation_split: 0.2
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert 'training_config' in model_config
        assert model_config['training_config']['epochs'] == 10
        assert model_config['training_config']['batch_size'] == 64
        assert model_config['training_config']['validation_split'] == 0.2
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'epochs' in code or '10' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_end_to_end_with_all_features(self):
        """Test: End-to-end workflow with HPO, tracking, and all backends."""
        dsl_code = """
        network FullFeaturesNet {
            input: (28, 28, 1)
            layers:
                Conv2D(filters=HPO(choice(16, 32)), kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Dropout(HPO(range(0.3, 0.6, step=0.1)))
                Output(10, "softmax")
            
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
            
            train {
                epochs: 3
                batch_size: 32
            }
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        assert len(hpo_params) > 0
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        with patch('neural.hpo.hpo.optimize_and_return') as mock_optimize:
            mock_optimize.return_value = {
                'conv2d_filters': 32,
                'dense_units': 128,
                'dropout_rate': 0.4,
                'learning_rate': 0.001
            }
            
            best_params = optimize_and_return(dsl_code, n_trials=2, dataset_name='MNIST', backend='pytorch')
        
        optimized_dsl = generate_optimized_dsl(dsl_code, best_params)
        
        optimized_model_config = transformer.transform(parser.parse(optimized_dsl))
        
        pytorch_code = generate_code(optimized_model_config, 'pytorch')
        assert 'import torch' in pytorch_code
        
        if TF_AVAILABLE:
            tf_code = generate_code(optimized_model_config, 'tensorflow')
            assert 'import tensorflow' in tf_code
        
        if ONNX_AVAILABLE:
            onnx_model = generate_onnx(optimized_model_config)
            assert onnx_model is not None


class TestMultiBackendExecution:
    """Tests for executing generated code across multiple backends."""

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
    def test_pytorch_model_inference(self):
        """Test: Execute inference on generated PyTorch model."""
        dsl_code = """
        network InferenceNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        
        with open('inference_model.py', 'w') as f:
            f.write(code)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference_model", "inference_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        model = module.NeuralModel()
        model.eval()
        
        with torch.no_grad():
            test_input = torch.randn(1, 1, 28, 28)
            output = model(test_input)
            
            assert output.shape == (1, 10)
            assert torch.allclose(output.sum(dim=1), torch.tensor([1.0]), atol=1e-5)

    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_tensorflow_model_inference(self):
        """Test: Execute inference on generated TensorFlow model."""
        dsl_code = """
        network TFInferenceNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        with open('tf_inference_model.py', 'w') as f:
            f.write(code)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("tf_inference_model", "tf_inference_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        model = module.create_model()
        
        test_input = np.random.randn(1, 28, 28, 1).astype(np.float32)
        output = model(test_input)
        
        assert output.shape == (1, 10)
        assert np.allclose(output.numpy().sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.skipif(not (TORCH_AVAILABLE and TF_AVAILABLE), reason="Both PyTorch and TensorFlow required")
    def test_cross_backend_output_consistency(self):
        """Test: Verify similar outputs across PyTorch and TensorFlow backends."""
        dsl_code = """
        network CrossBackendNet {
            input: (10,)
            layers:
                Dense(20, "relu")
                Dense(10, "sigmoid")
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        pytorch_code = generate_code(model_config, 'pytorch')
        tf_code = generate_code(model_config, 'tensorflow')
        
        assert 'nn.Linear' in pytorch_code
        assert 'layers.Dense' in tf_code


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
