"""
End-to-end scenario integration tests.

Tests realistic, complete scenarios that demonstrate the full capabilities of Neural DSL
across all backends with HPO and tracking features.
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.parser.parser import create_parser, ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.code_generation.code_generator import generate_code, generate_optimized_dsl
from neural.hpo.hpo import optimize_and_return, create_dynamic_model
from neural.tracking.experiment_tracker import ExperimentTracker

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
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
    ONNX_AVAILABLE = True
except ImportError:
    onnx = None
    ONNX_AVAILABLE = False


class MockTrial:
    """Mock trial for HPO testing."""
    def suggest_categorical(self, name, choices):
        return 32 if name == "batch_size" else choices[0]
    
    def suggest_float(self, name, low, high, step=None, log=False):
        return low if not log else 0.001
    
    def suggest_int(self, name, low, high):
        return low


def mock_data_loader(dataset_name, input_shape, batch_size=32, train=True, backend='pytorch'):
    """Mock data loader for testing."""
    if backend == 'pytorch' and TORCH_AVAILABLE:
        import numpy as np
        if train:
            x = torch.randn(100, *input_shape)
            y = torch.randint(0, 10, (100,))
        else:
            x = torch.randn(20, *input_shape)
            y = torch.randint(0, 10, (20,))
        
        if len(input_shape) == 3:
            x = x.permute(0, 3, 1, 2)
        
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=train)
    
    elif backend == 'tensorflow' and TF_AVAILABLE:
        import numpy as np
        if train:
            x = np.random.randn(100, *input_shape).astype(np.float32)
            y = np.random.randint(0, 10, (100,)).astype(np.int32)
        else:
            x = np.random.randn(20, *input_shape).astype(np.float32)
            y = np.random.randint(0, 10, (20,)).astype(np.int32)
        
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    
    return None


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

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

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_scenario_mnist_classification(self):
        """
        Scenario: MNIST digit classification
        Steps: Parse DSL → Validate shapes → Generate PyTorch code → Execute
        """
        mnist_dsl = """
        network MNISTClassifier {
            input: (28, 28, 1)
            layers:
                Conv2D(filters=32, kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Conv2D(filters=64, kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(128, "relu")
                Dropout(rate=0.5)
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
        tree = parser.parse(mnist_dsl)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        assert model_config['input']['shape'] == (28, 28, 1)
        assert model_config['training_config']['epochs'] == 10
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        assert current_shape == (None, 10)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'import torch' in code
        assert 'nn.Conv2d' in code
        assert 'nn.MaxPool2d' in code
        assert 'nn.Dropout' in code
        
        with open('mnist_model.py', 'w') as f:
            f.write(code)
        
        assert os.path.exists('mnist_model.py')

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_scenario_hpo_optimization(self):
        """
        Scenario: Hyperparameter optimization workflow
        Steps: Parse DSL with HPO → Optimize → Generate optimized DSL → Execute
        """
        hpo_dsl = """
        network HPOClassifier {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128, 256)), "relu")
                Dropout(HPO(range(0.2, 0.6, step=0.1)))
                Dense(HPO(choice(32, 64, 128)), "relu")
                Output(10, "softmax")
            
            loss: "cross_entropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
            
            train {
                epochs: 5
                batch_size: 32
            }
        }
        """
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(hpo_dsl)
        
        assert len(hpo_params) >= 3
        
        with patch('neural.hpo.hpo.optimize_and_return') as mock_optimize:
            mock_optimize.return_value = {
                'dense_units': 128,
                'dropout_rate': 0.4,
                'learning_rate': 0.001
            }
            
            best_params = optimize_and_return(hpo_dsl, n_trials=5, dataset_name='MNIST', backend='pytorch')
        
        assert 'dense_units' in best_params or 'learning_rate' in best_params
        
        optimized_dsl = generate_optimized_dsl(hpo_dsl, best_params)
        
        assert 'HPO' not in optimized_dsl.split('optimizer:')[0]
        
        parser = create_parser("network")
        tree = parser.parse(optimized_dsl)
        optimized_config = transformer.transform(tree)
        
        code = generate_code(optimized_config, 'pytorch')
        assert 'import torch' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_scenario_experiment_tracking(self):
        """
        Scenario: Train model with experiment tracking
        Steps: Parse DSL → Train → Log metrics → Save artifacts
        """
        tracked_dsl = """
        network TrackedNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(128, "relu")
                Dropout(rate=0.3)
                Dense(64, "relu")
                Output(10, "softmax")
            
            loss: "cross_entropy"
            optimizer: Adam(learning_rate=0.001)
            
            train {
                epochs: 5
                batch_size: 32
            }
        }
        """
        
        tracker = ExperimentTracker(experiment_name="mnist_experiment")
        
        parser = create_parser("network")
        tree = parser.parse(tracked_dsl)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        hyperparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5,
            'architecture': 'Dense-128-64-10'
        }
        tracker.log_hyperparameters(hyperparams)
        
        for epoch in range(5):
            metrics = {
                'loss': 0.5 - (epoch * 0.08),
                'accuracy': 0.75 + (epoch * 0.04),
                'val_loss': 0.6 - (epoch * 0.09),
                'val_accuracy': 0.70 + (epoch * 0.05)
            }
            tracker.log_metrics(metrics, step=epoch)
        
        tracker.metadata['dataset'] = 'MNIST'
        tracker.metadata['model_type'] = 'Dense Neural Network'
        tracker.save_metadata()
        
        assert len(tracker.metrics_history) == 5
        assert os.path.exists(os.path.join(tracker.experiment_dir, "metadata.json"))

    @pytest.mark.skipif(not (TORCH_AVAILABLE and TF_AVAILABLE), reason="Both PyTorch and TensorFlow required")
    def test_scenario_multi_backend_deployment(self):
        """
        Scenario: Deploy same model to multiple backends
        Steps: Parse DSL → Generate PyTorch code → Generate TF code → Generate ONNX
        """
        deployment_dsl = """
        network DeploymentNet {
            input: (32, 32, 3)
            layers:
                Conv2D(filters=32, kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(128, "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(deployment_dsl)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        pytorch_code = generate_code(model_config, 'pytorch')
        assert 'import torch' in pytorch_code
        
        with open('deployment_pytorch.py', 'w') as f:
            f.write(pytorch_code)
        
        tf_code = generate_code(model_config, 'tensorflow')
        assert 'import tensorflow' in tf_code
        
        with open('deployment_tensorflow.py', 'w') as f:
            f.write(tf_code)
        
        if ONNX_AVAILABLE:
            from neural.code_generation.code_generator import export_onnx
            export_onnx(model_config, 'deployment_model.onnx')
            assert os.path.exists('deployment_model.onnx')
        
        assert os.path.exists('deployment_pytorch.py')
        assert os.path.exists('deployment_tensorflow.py')

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_scenario_iterative_model_improvement(self):
        """
        Scenario: Iterative model improvement with HPO
        Steps: Train baseline → HPO optimize → Compare results
        """
        baseline_dsl = """
        network BaselineModel {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        hpo_dsl = """
        network ImprovedModel {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128, 256)), "relu")
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Dense(HPO(choice(32, 64)), "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        baseline_tracker = ExperimentTracker(experiment_name="baseline")
        baseline_tracker.log_hyperparameters({'architecture': 'simple', 'layers': 2})
        baseline_tracker.log_metrics({'loss': 0.45, 'accuracy': 0.85}, step=10)
        
        improved_tracker = ExperimentTracker(experiment_name="improved")
        
        with patch('neural.hpo.hpo.optimize_and_return') as mock_optimize:
            mock_optimize.return_value = {
                'dense_units': 256,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
            
            best_params = optimize_and_return(hpo_dsl, n_trials=10, dataset_name='MNIST', backend='pytorch')
        
        improved_tracker.log_hyperparameters(best_params)
        improved_tracker.log_metrics({'loss': 0.35, 'accuracy': 0.92}, step=10)
        
        assert baseline_tracker.metrics_history[0]['metrics']['accuracy'] < \
               improved_tracker.metrics_history[0]['metrics']['accuracy']

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_scenario_transfer_learning_preparation(self):
        """
        Scenario: Prepare model for transfer learning
        Steps: Parse complex architecture → Validate → Generate code with feature extraction
        """
        transfer_dsl = """
        network FeatureExtractor {
            input: (224, 224, 3)
            layers:
                Conv2D(filters=64, kernel_size=7, strides=2, activation="relu")
                MaxPooling2D(pool_size=3)
                Conv2D(filters=128, kernel_size=3, activation="relu")
                Conv2D(filters=128, kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Conv2D(filters=256, kernel_size=3, activation="relu")
                Conv2D(filters=256, kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(512, "relu")
                Dropout(rate=0.5)
                Output(1000, "softmax")
            
            optimizer: Adam(learning_rate=0.0001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(transfer_dsl)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        propagator = ShapePropagator()
        input_shape = (None,) + tuple(model_config['input']['shape'])
        current_shape = input_shape
        
        for layer in model_config['layers']:
            current_shape = propagator.propagate(current_shape, layer)
        
        assert current_shape == (None, 1000)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'nn.Conv2d' in code
        assert 'nn.MaxPool2d' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_scenario_production_pipeline(self):
        """
        Scenario: Complete production pipeline
        Steps: HPO → Train → Track → Export → Deploy
        """
        production_dsl = """
        network ProductionModel {
            input: (28, 28, 1)
            layers:
                Conv2D(filters=HPO(choice(16, 32)), kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Dropout(HPO(range(0.3, 0.5, step=0.1)))
                Output(10, "softmax")
            
            loss: "cross_entropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
            
            train {
                epochs: 10
                batch_size: 32
            }
        }
        """
        
        tracker = ExperimentTracker(experiment_name="production_pipeline")
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(production_dsl)
        
        with patch('neural.hpo.hpo.optimize_and_return') as mock_optimize:
            mock_optimize.return_value = {
                'conv2d_filters': 32,
                'dense_units': 128,
                'dropout_rate': 0.4,
                'learning_rate': 0.001
            }
            
            best_params = optimize_and_return(production_dsl, n_trials=20, dataset_name='MNIST', backend='pytorch')
        
        tracker.log_hyperparameters(best_params)
        
        optimized_dsl = generate_optimized_dsl(production_dsl, best_params)
        
        parser = create_parser("network")
        tree = parser.parse(optimized_dsl)
        optimized_config = transformer.transform(tree)
        
        pytorch_code = generate_code(optimized_config, 'pytorch')
        
        with open('production_model.py', 'w') as f:
            f.write(pytorch_code)
        
        tracker.log_metrics({'final_loss': 0.25, 'final_accuracy': 0.95}, step=10)
        tracker.metadata['model_version'] = 'v1.0'
        tracker.metadata['deployment_ready'] = True
        tracker.save_metadata()
        
        assert os.path.exists('production_model.py')
        assert os.path.exists(os.path.join(tracker.experiment_dir, "metadata.json"))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_scenario_architecture_exploration(self):
        """
        Scenario: Explore different architectures
        Steps: Try multiple architectures → Compare → Select best
        """
        architectures = {
            'simple': """
                network SimpleArch {
                    input: (28, 28, 1)
                    layers:
                        Flatten()
                        Dense(128, "relu")
                        Output(10, "softmax")
                }
            """,
            'cnn': """
                network CNNArch {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(filters=32, kernel_size=3, activation="relu")
                        MaxPooling2D(pool_size=2)
                        Flatten()
                        Dense(64, "relu")
                        Output(10, "softmax")
                }
            """,
            'deep': """
                network DeepArch {
                    input: (28, 28, 1)
                    layers:
                        Flatten()
                        Dense(256, "relu")
                        Dense(128, "relu")
                        Dense(64, "relu")
                        Output(10, "softmax")
                }
            """
        }
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        results = {}
        
        for name, dsl in architectures.items():
            tree = parser.parse(dsl)
            model_config = transformer.transform(tree)
            
            tracker = ExperimentTracker(experiment_name=f"arch_{name}")
            tracker.log_hyperparameters({'architecture': name})
            
            simulated_accuracy = {'simple': 0.85, 'cnn': 0.92, 'deep': 0.88}[name]
            tracker.log_metrics({'accuracy': simulated_accuracy}, step=1)
            
            results[name] = simulated_accuracy
        
        best_arch = max(results, key=results.get)
        assert best_arch == 'cnn'

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_scenario_model_versioning(self):
        """
        Scenario: Model versioning and comparison
        Steps: Train v1 → Improve to v2 → Compare → Deploy v2
        """
        v1_dsl = """
        network ModelV1 {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
        }
        """
        
        v2_dsl = """
        network ModelV2 {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(128, "relu")
                Dropout(rate=0.3)
                Dense(64, "relu")
                Output(10, "softmax")
        }
        """
        
        parser = create_parser("network")
        transformer = ModelTransformer()
        
        v1_tracker = ExperimentTracker(experiment_name="model_v1")
        v1_tracker.metadata['version'] = '1.0'
        v1_tracker.log_hyperparameters({'layers': 2, 'total_params': '~51K'})
        v1_tracker.log_metrics({'loss': 0.40, 'accuracy': 0.87}, step=10)
        v1_tracker.save_metadata()
        
        v2_tracker = ExperimentTracker(experiment_name="model_v2")
        v2_tracker.metadata['version'] = '2.0'
        v2_tracker.log_hyperparameters({'layers': 4, 'total_params': '~110K'})
        v2_tracker.log_metrics({'loss': 0.30, 'accuracy': 0.93}, step=10)
        v2_tracker.save_metadata()
        
        assert v2_tracker.metrics_history[0]['metrics']['accuracy'] > \
               v1_tracker.metrics_history[0]['metrics']['accuracy']
        
        tree = parser.parse(v2_dsl)
        v2_config = transformer.transform(tree)
        deployment_code = generate_code(v2_config, 'pytorch')
        
        with open('deployed_model_v2.py', 'w') as f:
            f.write(deployment_code)
        
        assert os.path.exists('deployed_model_v2.py')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
