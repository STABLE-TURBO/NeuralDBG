"""
Integration tests for HPO and Tracking workflows.

Tests the complete HPO workflow with experiment tracking across all backends.
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.parser.parser import create_parser, ModelTransformer
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
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        from neural.hpo.hpo import optimize_and_return, create_dynamic_model, objective, train_model
    except ImportError:
        optimize_and_return = None
        create_dynamic_model = None
        objective = None
        train_model = None
else:
    optimize_and_return = None
    create_dynamic_model = None
    objective = None
    train_model = None

try:
    from neural.code_generation.code_generator import generate_code, generate_optimized_dsl
except ImportError:
    generate_code = None
    generate_optimized_dsl = None


class MockTrial:
    """Mock Optuna trial for testing."""
    def __init__(self):
        self.params = {}
    
    def suggest_categorical(self, name, choices):
        value = 32 if name == "batch_size" else choices[0]
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, step=None, log=False):
        value = low if not log else 0.001
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high):
        value = low
        self.params[name] = value
        return value


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


@pytest.fixture
def temp_workspace():
    """Fixture to provide a temporary workspace for tests."""
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    os.chdir(original_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    if os.path.exists("neural_experiments"):
        shutil.rmtree("neural_experiments")


class TestHPOWorkflowIntegration:
    """Integration tests for HPO workflows."""

    @pytest.mark.skipif(not TORCH_AVAILABLE or create_dynamic_model is None, reason="PyTorch or HPO module not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_hpo_model_creation_pytorch(self, temp_workspace):
        """Test: HPO model creation and parameter resolution for PyTorch."""
        dsl_code = """
        network HPOModelTest {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128, 256)), "relu")
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        assert len(hpo_params) >= 2
        assert any('dense' in str(p).lower() for p in hpo_params)
        assert any('dropout' in str(p).lower() for p in hpo_params)
        
        trial = MockTrial()
        model = create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch')
        
        assert model is not None
        test_input = torch.randn(4, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (4, 10)

    @pytest.mark.skipif(not TORCH_AVAILABLE or train_model is None, reason="PyTorch or HPO module not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_hpo_training_loop(self, temp_workspace):
        """Test: HPO training loop execution."""
        dsl_code = """
        network TrainingLoopTest {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        trial = MockTrial()
        model = create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_loader = mock_data_loader('MNIST', model_dict['input']['shape'], batch_size=32, train=True)
        val_loader = mock_data_loader('MNIST', model_dict['input']['shape'], batch_size=32, train=False)
        
        result = train_model(model, optimizer, train_loader, val_loader, backend='pytorch',
                           execution_config=model_dict.get('execution_config', {}))
        
        assert result is not None
        assert len(result) == 2
        loss, acc = result
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1

    @pytest.mark.skipif(not TORCH_AVAILABLE or objective is None, reason="PyTorch or HPO module not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_hpo_objective_function(self, temp_workspace):
        """Test: HPO objective function computes all metrics."""
        dsl_code = """
        network ObjectiveTest {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        trial = MockTrial()
        
        loss, acc, precision, recall = objective(trial, dsl_code, 'MNIST', backend='pytorch')
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        
        assert 0 <= acc <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    @pytest.mark.skipif(not (TORCH_AVAILABLE and OPTUNA_AVAILABLE) or optimize_and_return is None, 
                       reason="PyTorch, Optuna, or HPO module not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_hpo_optimization_workflow(self, temp_workspace):
        """Test: Complete HPO optimization workflow."""
        dsl_code = """
        network OptimizationTest {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(32, 64)), "relu")
                Dropout(HPO(range(0.2, 0.4, step=0.1)))
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        with patch('optuna.create_study') as mock_create_study:
            mock_study = MagicMock()
            mock_trial = MockTrial()
            mock_study.best_params = {
                'Dense_units': 64,
                'Dropout_rate': 0.3,
                'opt_learning_rate': 0.001
            }
            mock_create_study.return_value = mock_study
            
            best_params = optimize_and_return(dsl_code, n_trials=2, dataset_name='MNIST', backend='pytorch')
            
            assert isinstance(best_params, dict)

    @pytest.mark.skipif(not TORCH_AVAILABLE or generate_optimized_dsl is None, 
                       reason="PyTorch or code generator not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_hpo_dsl_generation(self, temp_workspace):
        """Test: Generate optimized DSL from HPO results."""
        dsl_code = """
        network DSLGenTest {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128, 256)), "relu")
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        best_params = {
            'dense_units': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
        
        optimized_dsl = generate_optimized_dsl(dsl_code, best_params)
        
        assert 'Dense(128)' in optimized_dsl or 'Dense(128,' in optimized_dsl
        assert 'Dropout(0.3)' in optimized_dsl or 'Dropout(0.3,' in optimized_dsl
        assert 'learning_rate=0.001' in optimized_dsl
        
        assert 'HPO' not in optimized_dsl.split('optimizer:')[0]

    @pytest.mark.skipif(not TORCH_AVAILABLE or create_dynamic_model is None, 
                       reason="PyTorch or HPO module not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_hpo_multiple_layer_types(self, temp_workspace):
        """Test: HPO with multiple layer types and parameters."""
        dsl_code = """
        network MultiLayerHPO {
            input: (32, 32, 3)
            layers:
                Conv2D(filters=HPO(choice(16, 32)), kernel_size=3, activation="relu")
                MaxPooling2D(pool_size=2)
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Dense(HPO(choice(32, 64)), "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        assert len(hpo_params) >= 3
        
        trial = MockTrial()
        model = create_dynamic_model(model_dict, trial, hpo_params, backend='pytorch')
        
        assert model is not None


class TestTrackingWorkflowIntegration:
    """Integration tests for experiment tracking workflows."""

    def test_experiment_tracker_initialization(self, temp_workspace):
        """Test: Experiment tracker initialization and directory creation."""
        tracker = ExperimentTracker(experiment_name="test_experiment")
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.experiment_id is not None
        assert os.path.exists(tracker.experiment_dir)
        assert os.path.exists(os.path.join(tracker.experiment_dir, "artifacts"))
        assert os.path.exists(os.path.join(tracker.experiment_dir, "plots"))

    def test_experiment_tracker_log_hyperparameters(self, temp_workspace):
        """Test: Log hyperparameters to experiment tracker."""
        tracker = ExperimentTracker(experiment_name="hyperparams_test")
        
        hyperparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'optimizer': 'Adam'
        }
        
        tracker.log_hyperparameters(hyperparams)
        
        assert tracker.hyperparameters == hyperparams

    def test_experiment_tracker_log_metrics(self, temp_workspace):
        """Test: Log metrics to experiment tracker."""
        tracker = ExperimentTracker(experiment_name="metrics_test")
        
        metrics = {
            'loss': 0.5,
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88
        }
        
        tracker.log_metrics(metrics, step=1)
        
        assert len(tracker.metrics_history) == 1
        assert tracker.metrics_history[0]['step'] == 1

    def test_experiment_tracker_save_metadata(self, temp_workspace):
        """Test: Save experiment metadata."""
        tracker = ExperimentTracker(experiment_name="metadata_test")
        
        tracker.metadata['model_architecture'] = 'CNN'
        tracker.metadata['dataset'] = 'MNIST'
        
        tracker.save_metadata()
        
        metadata_file = os.path.join(tracker.experiment_dir, "metadata.json")
        assert os.path.exists(metadata_file)
        
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['model_architecture'] == 'CNN'
        assert loaded_metadata['dataset'] == 'MNIST'

    @pytest.mark.skipif(not TORCH_AVAILABLE or generate_code is None, 
                       reason="PyTorch or code generator not available")
    def test_pytorch_code_with_tracking(self, temp_workspace):
        """Test: Generated PyTorch code includes tracking."""
        dsl_code = """
        network TrackedPyTorchNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'pytorch')
        
        assert 'from neural.tracking.experiment_tracker import' in code or \
               'ExperimentTracker' in code or \
               'ExperimentManager' in code

    @pytest.mark.skipif(not TF_AVAILABLE or generate_code is None, 
                       reason="TensorFlow or code generator not available")
    def test_tensorflow_code_with_tracking(self, temp_workspace):
        """Test: Generated TensorFlow code includes tracking."""
        dsl_code = """
        network TrackedTFNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(64, "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        parser = create_parser("network")
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_config = transformer.transform(tree)
        
        code = generate_code(model_config, 'tensorflow')
        
        assert 'from neural.tracking.experiment_tracker import' in code or \
               'ExperimentTracker' in code or \
               'ExperimentManager' in code

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_hpo_with_tracking_integration(self, temp_workspace):
        """Test: HPO workflow with experiment tracking."""
        dsl_code = """
        network HPOTrackedNet {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        tracker = ExperimentTracker(experiment_name="hpo_tracked_test")
        
        hyperparams = {
            'dense_units': 128,
            'learning_rate': 0.001
        }
        tracker.log_hyperparameters(hyperparams)
        
        assert tracker.hyperparameters == hyperparams
        
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        tracker.log_metrics(metrics, step=1)
        
        assert len(tracker.metrics_history) == 1

    def test_experiment_comparison_workflow(self, temp_workspace):
        """Test: Compare multiple experiments workflow."""
        tracker1 = ExperimentTracker(experiment_name="experiment_1")
        tracker1.log_hyperparameters({'learning_rate': 0.001})
        tracker1.log_metrics({'loss': 0.5, 'accuracy': 0.85}, step=1)
        tracker1.save_metadata()
        
        tracker2 = ExperimentTracker(experiment_name="experiment_2")
        tracker2.log_hyperparameters({'learning_rate': 0.01})
        tracker2.log_metrics({'loss': 0.4, 'accuracy': 0.88}, step=1)
        tracker2.save_metadata()
        
        experiments = [tracker1, tracker2]
        
        for exp in experiments:
            assert os.path.exists(exp.experiment_dir)
            assert os.path.exists(os.path.join(exp.experiment_dir, "metadata.json"))


class TestHPOTrackingCombinedWorkflow:
    """Integration tests combining HPO and tracking."""

    @pytest.mark.skipif(not TORCH_AVAILABLE or optimize_and_return is None or generate_code is None, 
                       reason="PyTorch, HPO module, or code generator not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_complete_hpo_tracking_workflow(self, temp_workspace):
        """Test: Complete workflow with HPO optimization and experiment tracking."""
        dsl_code = """
        network CompleteWorkflow {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Dropout(HPO(range(0.2, 0.4, step=0.1)))
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
            
            train {
                epochs: 3
                batch_size: 32
            }
        }
        """
        
        tracker = ExperimentTracker(experiment_name="complete_workflow")
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        with patch('neural.hpo.hpo.optimize_and_return') as mock_optimize:
            mock_optimize.return_value = {
                'dense_units': 128,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
            
            best_params = optimize_and_return(dsl_code, n_trials=2, dataset_name='MNIST', backend='pytorch')
        
        tracker.log_hyperparameters(best_params)
        
        if generate_optimized_dsl:
            optimized_dsl = generate_optimized_dsl(dsl_code, best_params)
            optimized_model_dict = transformer.transform(create_parser("network").parse(optimized_dsl))
            code = generate_code(optimized_model_dict, 'pytorch')
            assert 'import torch' in code
        
        tracker.log_metrics({'final_loss': 0.3, 'final_accuracy': 0.90}, step=3)
        tracker.save_metadata()
        
        assert os.path.exists(os.path.join(tracker.experiment_dir, "metadata.json"))
        assert len(tracker.metrics_history) == 1

    @pytest.mark.skipif(not TORCH_AVAILABLE or optimize_and_return is None, 
                       reason="PyTorch or HPO module not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_multiple_hpo_runs_with_tracking(self, temp_workspace):
        """Test: Multiple HPO runs with separate experiment tracking."""
        dsl_code = """
        network MultiRunTest {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(32, 64)), "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """
        
        trackers = []
        
        for run_id in range(3):
            tracker = ExperimentTracker(experiment_name=f"run_{run_id}")
            trackers.append(tracker)
            
            with patch('neural.hpo.hpo.optimize_and_return') as mock_optimize:
                mock_optimize.return_value = {
                    'dense_units': 32 + (run_id * 16),
                    'learning_rate': 0.001 * (run_id + 1)
                }
                
                best_params = optimize_and_return(dsl_code, n_trials=1, dataset_name='MNIST', backend='pytorch')
            
            tracker.log_hyperparameters(best_params)
            tracker.log_metrics({'loss': 0.5 - (run_id * 0.1), 'accuracy': 0.8 + (run_id * 0.03)}, step=1)
            tracker.save_metadata()
        
        for tracker in trackers:
            assert os.path.exists(tracker.experiment_dir)

    @pytest.mark.skipif(not (TORCH_AVAILABLE and TF_AVAILABLE) or generate_code is None or generate_optimized_dsl is None, 
                       reason="Both PyTorch and TensorFlow, or code generator not available")
    @patch('neural.hpo.hpo.get_data', mock_data_loader)
    def test_cross_backend_hpo_tracking(self, temp_workspace):
        """Test: HPO and tracking across multiple backends."""
        dsl_code = """
        network CrossBackendHPO {
            input: (28, 28, 1)
            layers:
                Flatten()
                Dense(HPO(choice(64, 128)), "relu")
                Output(10, "softmax")
            
            optimizer: Adam(learning_rate=0.001)
        }
        """
        
        transformer = ModelTransformer()
        model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_code)
        
        pytorch_tracker = ExperimentTracker(experiment_name="pytorch_backend")
        tf_tracker = ExperimentTracker(experiment_name="tf_backend")
        
        best_params = {'dense_units': 128, 'learning_rate': 0.001}
        
        pytorch_tracker.log_hyperparameters({**best_params, 'backend': 'pytorch'})
        tf_tracker.log_hyperparameters({**best_params, 'backend': 'tensorflow'})
        
        optimized_dsl = generate_optimized_dsl(dsl_code, best_params)
        optimized_model_dict = transformer.transform(create_parser("network").parse(optimized_dsl))
        
        pytorch_code = generate_code(optimized_model_dict, 'pytorch')
        tf_code = generate_code(optimized_model_dict, 'tensorflow')
        
        assert 'import torch' in pytorch_code
        assert 'import tensorflow' in tf_code
        
        pytorch_tracker.save_metadata()
        tf_tracker.save_metadata()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
