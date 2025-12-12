import os
import sys
import pytest
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural.parser.parser import create_parser, ModelTransformer


pytest_plugins = []


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for component interactions")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line("markers", "gpu: Tests that require GPU/CUDA")
    config.addinivalue_line("markers", "backend(name): Tests for specific backend (tensorflow, pytorch, onnx)")
    config.addinivalue_line("markers", "parser: Parser-related tests")
    config.addinivalue_line("markers", "codegen: Code generation tests")
    config.addinivalue_line("markers", "shape: Shape propagation tests")
    config.addinivalue_line("markers", "hpo: Hyperparameter optimization tests")
    config.addinivalue_line("markers", "dashboard: Dashboard tests")
    config.addinivalue_line("markers", "cloud: Cloud execution tests")


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def neural_module_path(project_root):
    return project_root / "neural"


@pytest.fixture
def parser():
    return create_parser()


@pytest.fixture
def layer_parser():
    return create_parser('layer')


@pytest.fixture
def network_parser():
    return create_parser('network')


@pytest.fixture
def research_parser():
    return create_parser('research')


@pytest.fixture
def define_parser():
    return create_parser('define')


@pytest.fixture
def transformer():
    return ModelTransformer()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dsl_simple():
    return """
    network SimpleNet {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(128, "relu")
            Dense(10, "softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """


@pytest.fixture
def sample_dsl_cnn():
    return """
    network CNNModel {
        input: (32, 32, 3)
        layers:
            Conv2D(32, (3, 3), "relu")
            MaxPooling2D((2, 2))
            Conv2D(64, (3, 3), "relu")
            MaxPooling2D((2, 2))
            Flatten()
            Dense(128, "relu")
            Dropout(0.5)
            Dense(10, "softmax")
        loss: "categorical_crossentropy"
        optimizer: Adam(learning_rate=0.001)
    }
    """


@pytest.fixture
def sample_dsl_rnn():
    return """
    network RNNModel {
        input: (None, 100, 64)
        layers:
            LSTM(128, return_sequences=true)
            LSTM(64)
            Dense(32, "relu")
            Dense(10, "softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """


@pytest.fixture
def sample_dsl_transformer():
    return """
    network TransformerNet {
        input: (128, 512)
        layers:
            TransformerEncoder(num_heads=8, ff_dim=2048) {
                Dense(512, "relu")
                Dropout(0.1)
            }
            GlobalAveragePooling1D()
            Dense(256, "relu")
            Dense(10, "softmax")
        loss: "categorical_crossentropy"
        optimizer: Adam(learning_rate=1e-4)
    }
    """


@pytest.fixture
def sample_dsl_residual():
    return """
    network ResidualNet {
        input: (64, 64, 3)
        layers:
            Conv2D(64, (3, 3), padding="same", activation="relu")
            ResidualConnection() {
                Conv2D(64, (3, 3), padding="same", activation="relu")
                BatchNormalization()
                Conv2D(64, (3, 3), padding="same")
            }
            GlobalAveragePooling2D()
            Dense(10, "softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """


@pytest.fixture
def sample_dsl_hpo():
    return """
    network HPOModel {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(HPO(choice(64, 128, 256)), "relu")
            Dropout(HPO(range(0.3, 0.7, step=0.1)))
            Dense(10, "softmax")
        loss: "categorical_crossentropy"
        optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        train {
            epochs: 10
            search_method: "bayesian"
        }
    }
    """


@pytest.fixture
def sample_dsl_device():
    return """
    network DeviceModel {
        input: (32, 32, 3)
        layers:
            Conv2D(32, (3, 3), "relu") @ "cuda:0"
            MaxPooling2D((2, 2))
            Flatten() @ "cpu"
            Dense(128, "relu") @ "cuda:0"
            Dense(10, "softmax")
        execution {
            device: "auto"
        }
    }
    """


@pytest.fixture
def simple_model_data():
    return {
        "type": "model",
        "name": "SimpleModel",
        "input": {"type": "Input", "shape": (None, 32, 32, 3)},
        "layers": [
            {"type": "Conv2D", "params": {"filters": 16, "kernel_size": 3}, "sublayers": []},
            {"type": "Flatten", "params": None, "sublayers": []},
            {"type": "Dense", "params": {"units": 10}, "sublayers": []}
        ],
        "loss": "mse",
        "optimizer": {"type": "Adam", "params": {}},
        "framework": "tensorflow"
    }


@pytest.fixture
def complex_model_data():
    return {
        "type": "model",
        "name": "ComplexNet",
        "input": {"type": "Input", "shape": (None, 64, 64, 3)},
        "layers": [
            {
                "type": "Residual",
                "params": {},
                "sublayers": [
                    {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3, "padding": "same"}, "sublayers": []},
                    {"type": "BatchNormalization", "params": None, "sublayers": []}
                ]
            },
            {"type": "MaxPooling2D", "params": {"pool_size": 2}, "sublayers": []},
            {"type": "Flatten", "params": None, "sublayers": []},
            {"type": "Dense", "params": {"units": 256, "activation": "relu"}, "sublayers": []},
            {"type": "Dropout", "params": {"rate": 0.5}, "sublayers": []},
            {"type": "Dense", "params": {"units": 10, "activation": "softmax"}, "sublayers": []}
        ],
        "loss": "categorical_crossentropy",
        "optimizer": {"type": "Adam", "params": {"learning_rate": 0.001}},
        "framework": "tensorflow"
    }


@pytest.fixture
def channels_first_model_data():
    return {
        "type": "model",
        "name": "ChannelsFirstNet",
        "input": {"type": "Input", "shape": (None, 3, 32, 32)},
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "data_format": "channels_first"}, "sublayers": []},
            {"type": "MaxPooling2D", "params": {"pool_size": 2}, "sublayers": []},
            {"type": "Flatten", "params": None, "sublayers": []},
            {"type": "Dense", "params": {"units": 10}, "sublayers": []}
        ],
        "loss": "mse",
        "optimizer": {"type": "SGD", "params": {}},
        "framework": "pytorch"
    }


@pytest.fixture
def transformer_model_data():
    return {
        "type": "model",
        "name": "TransformerModel",
        "input": {"type": "Input", "shape": (None, 128)},
        "layers": [
            {
                "type": "TransformerEncoder",
                "params": {"num_heads": 4, "ff_dim": 256, "dropout": 0.1},
                "sublayers": []
            },
            {"type": "Dense", "params": {"units": 10}, "sublayers": []}
        ],
        "loss": "categorical_crossentropy",
        "optimizer": {"type": "Adam", "params": {}},
        "framework": "tensorflow"
    }


@pytest.fixture
def rnn_model_data():
    return {
        "type": "model",
        "name": "RNNModel",
        "input": {"type": "Input", "shape": (None, 10, 32)},
        "layers": [
            {"type": "LSTM", "params": {"units": 128, "return_sequences": True}, "sublayers": []},
            {"type": "GRU", "params": {"units": 64}, "sublayers": []},
            {"type": "Dense", "params": {"units": 10}, "sublayers": []}
        ],
        "loss": "mse",
        "optimizer": {"type": "Adam", "params": {}},
        "framework": "tensorflow"
    }


@pytest.fixture
def multiplied_layers_model():
    return {
        "type": "model",
        "name": "MultipliedModel",
        "input": {"type": "Input", "shape": (None, 32)},
        "layers": [
            {"type": "Dense", "params": {"units": 64}, "multiply": 3, "sublayers": []},
            {"type": "Dropout", "params": {"rate": 0.5}, "multiply": 2, "sublayers": []}
        ],
        "loss": "mse",
        "optimizer": {"type": "Adam", "params": {}},
        "framework": "tensorflow"
    }


@pytest.fixture(params=["tensorflow", "pytorch", "onnx"])
def backend(request):
    return request.param


@pytest.fixture(params=["tensorflow", "pytorch"])
def ml_backend(request):
    return request.param


@pytest.fixture
def sample_shapes():
    return {
        "image_1d": (None, 28),
        "image_2d_grayscale": (None, 28, 28, 1),
        "image_2d_rgb": (None, 32, 32, 3),
        "image_2d_large": (None, 224, 224, 3),
        "sequence": (None, 100, 64),
        "sequence_variable": (None, None, 128),
        "embedding": (None, 512),
        "batch_only": (32,),
        "channels_first": (None, 3, 64, 64),
        "3d_video": (None, 16, 112, 112, 3),
    }


@pytest.fixture
def sample_layer_configs():
    return {
        "dense": {"type": "Dense", "params": {"units": 128, "activation": "relu"}, "sublayers": []},
        "conv2d": {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "activation": "relu"}, "sublayers": []},
        "lstm": {"type": "LSTM", "params": {"units": 64, "return_sequences": True}, "sublayers": []},
        "gru": {"type": "GRU", "params": {"units": 64}, "sublayers": []},
        "dropout": {"type": "Dropout", "params": {"rate": 0.5}, "sublayers": []},
        "batchnorm": {"type": "BatchNormalization", "params": None, "sublayers": []},
        "maxpool2d": {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}, "sublayers": []},
        "avgpool2d": {"type": "AveragePooling2D", "params": {"pool_size": (2, 2)}, "sublayers": []},
        "flatten": {"type": "Flatten", "params": None, "sublayers": []},
        "globalavgpool2d": {"type": "GlobalAveragePooling2D", "params": {}, "sublayers": []},
    }


@pytest.fixture
def sample_optimizer_configs():
    return {
        "adam": {"type": "Adam", "params": {}},
        "adam_lr": {"type": "Adam", "params": {"learning_rate": 0.001}},
        "sgd": {"type": "SGD", "params": {}},
        "sgd_momentum": {"type": "SGD", "params": {"learning_rate": 0.01, "momentum": 0.9}},
        "rmsprop": {"type": "RMSprop", "params": {}},
        "adamw": {"type": "AdamW", "params": {"learning_rate": 0.001, "weight_decay": 0.01}},
    }


@pytest.fixture
def sample_loss_functions():
    return [
        "mse",
        "mae",
        "categorical_crossentropy",
        "binary_crossentropy",
        "sparse_categorical_crossentropy",
        "huber",
        "hinge",
    ]


@pytest.fixture
def sample_activations():
    return [
        "relu",
        "sigmoid",
        "tanh",
        "softmax",
        "elu",
        "selu",
        "leaky_relu",
        "swish",
        "gelu",
    ]


@pytest.fixture
def sample_hpo_configs():
    return {
        "categorical": {"hpo": {"type": "categorical", "values": [64, 128, 256]}},
        "range": {"hpo": {"type": "range", "start": 0.1, "end": 0.9, "step": 0.1}},
        "log_range": {"hpo": {"type": "log_range", "low": 1e-5, "high": 1e-1}},
    }


@pytest.fixture
def sample_training_configs():
    return {
        "minimal": {"epochs": 10, "batch_size": 32},
        "with_validation": {"epochs": 10, "batch_size": 32, "validation_split": 0.2},
        "with_callbacks": {"epochs": 50, "batch_size": 64, "early_stopping": True, "patience": 5},
        "with_search": {"epochs": 10, "search_method": "bayesian", "n_trials": 20},
    }


@pytest.fixture
def sample_execution_configs():
    return {
        "auto": {"device": "auto"},
        "cpu": {"device": "cpu"},
        "cuda": {"device": "cuda:0"},
        "multi_gpu": {"device": "auto", "strategy": "mirrored"},
    }


@pytest.fixture
def sample_device_specs():
    return ["cpu", "cuda:0", "cuda:1", "tpu", "auto"]


@pytest.fixture
def expected_tensorflow_imports():
    return [
        "import tensorflow as tf",
        "from tensorflow import keras",
        "from tensorflow.keras import layers",
    ]


@pytest.fixture
def expected_pytorch_imports():
    return [
        "import torch",
        "import torch.nn as nn",
        "import torch.optim as optim",
    ]


@pytest.fixture
def expected_onnx_imports():
    return [
        "import onnx",
        "from onnx import helper",
    ]


@pytest.fixture
def validation_error_cases():
    return {
        "negative_units": {
            "dsl": "Dense(-10)",
            "error": "must be a positive",
        },
        "invalid_dropout": {
            "dsl": "Dropout(1.5)",
            "error": "should be between 0 and 1",
        },
        "zero_filters": {
            "dsl": "Conv2D(0, (3, 3))",
            "error": "must be a positive integer",
        },
        "negative_kernel": {
            "dsl": "Conv2D(32, (-3, -3))",
            "error": "should be positive",
        },
        "missing_units": {
            "dsl": "Dense()",
            "error": "requires 'units' parameter",
        },
    }


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "integration_tests" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif "test_device" in item.name or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        if "parser" in str(item.fspath):
            item.add_marker(pytest.mark.parser)
        elif "code_generator" in str(item.fspath):
            item.add_marker(pytest.mark.codegen)
        elif "shape_propagation" in str(item.fspath):
            item.add_marker(pytest.mark.shape)
        elif "hpo" in str(item.fspath):
            item.add_marker(pytest.mark.hpo)
        elif "dashboard" in str(item.fspath):
            item.add_marker(pytest.mark.dashboard)
        elif "cloud" in str(item.fspath):
            item.add_marker(pytest.mark.cloud)
        
        if not any(marker.name in ["unit", "integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
