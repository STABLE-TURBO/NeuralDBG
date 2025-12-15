"""
Framework-specific implementations for benchmarking.
"""

import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from neural.exceptions import DependencyError


class FrameworkImplementation(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.code_content = ""
        self.temp_files = []

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, dataset: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict_single(self) -> Any:
        pass

    def count_lines_of_code(self) -> int:
        if not self.code_content:
            return 0
        lines = self.code_content.strip().split("\n")
        non_empty = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        return len(non_empty)

    def get_code_complexity(self) -> Dict[str, Any]:
        loc = self.count_lines_of_code()
        imports = self.code_content.count("import")
        classes = self.code_content.count("class ")
        functions = self.code_content.count("def ")
        
        setup_complexity = imports + classes * 2 + functions
        readability_score = max(0, 10 - (loc / 10))
        
        return {
            "setup_complexity": setup_complexity,
            "readability_score": min(10, readability_score),
            "imports_count": imports,
            "classes_count": classes,
            "functions_count": functions,
        }

    def get_model_size_mb(self) -> float:
        if self.model is None:
            return 0.0
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
            self.temp_files.append(temp_path)
            
            try:
                self._save_model(temp_path)
                size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                return size_mb
            except Exception:
                return 0.0

    @abstractmethod
    def _save_model(self, path: str):
        pass

    @abstractmethod
    def get_parameter_count(self) -> int:
        pass

    def cleanup(self):
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass


class NeuralDSLImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("Neural DSL")
        self.dsl_code = ""
        self.backend = "tensorflow"

    def setup(self):
        self.dsl_code = """network MNISTClassifier {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3,3), activation="relu")
        MaxPooling2D(pool_size=(2,2))
        Flatten()
        Dense(units=128, activation="relu")
        Dropout(rate=0.5)
        Output(units=10, activation="softmax")
    
    loss: "sparse_categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}"""
        self.code_content = self.dsl_code

    def build_model(self):
        try:
            from neural.parser.parser import create_parser, ModelTransformer
            from neural.code_generation.code_generator import generate_code
        except ImportError as e:
            raise DependencyError(
                dependency="neural.parser",
                feature="Neural DSL model building",
                install_hint="pip install -e ."
            ) from e
        
        parser = create_parser(start_rule="network")
        tree = parser.parse(self.dsl_code)
        model_data = ModelTransformer().transform(tree)
        
        generated_code = generate_code(model_data, self.backend)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(generated_code)
            temp_path = f.name
            self.temp_files.append(temp_path)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("generated_model", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        self.model = module.model

    def train(self, dataset: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        try:
            import tensorflow as tf
        except ImportError as e:
            raise DependencyError(
                dependency="tensorflow",
                feature="Neural DSL training with TensorFlow backend",
                install_hint="pip install tensorflow"
            ) from e
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        
        x_train = x_train[:5000]
        y_train = y_train[:5000]
        x_test = x_test[:1000]
        y_test = y_test[:1000]
        
        start_time = time.time()
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        training_time = time.time() - start_time
        
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        return {
            "training_time": training_time,
            "accuracy": accuracy,
            "val_accuracy": history.history["val_accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "training_loss": history.history["loss"][-1],
            "peak_memory_mb": 0,
            "error_rate": 1.0 - accuracy,
        }

    def predict_single(self):
        sample = np.random.randn(1, 28, 28, 1).astype("float32")
        return self.model.predict(sample, verbose=0)

    def _save_model(self, path: str):
        self.model.save(path)

    def get_parameter_count(self) -> int:
        return self.model.count_params()


class KerasImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("Keras")

    def setup(self):
        self.code_content = """import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)"""

    def build_model(self):
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as e:
            raise DependencyError(
                dependency="tensorflow",
                feature="Keras model building",
                install_hint="pip install tensorflow"
            ) from e
        
        self.model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, dataset: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        try:
            import tensorflow as tf
        except ImportError as e:
            raise DependencyError(
                dependency="tensorflow",
                feature="Keras training",
                install_hint="pip install tensorflow"
            ) from e
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        
        x_train = x_train[:5000]
        y_train = y_train[:5000]
        x_test = x_test[:1000]
        y_test = y_test[:1000]
        
        start_time = time.time()
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        training_time = time.time() - start_time
        
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        return {
            "training_time": training_time,
            "accuracy": accuracy,
            "val_accuracy": history.history["val_accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "training_loss": history.history["loss"][-1],
            "peak_memory_mb": 0,
            "error_rate": 1.0 - accuracy,
        }

    def predict_single(self):
        sample = np.random.randn(1, 28, 28, 1).astype("float32")
        return self.model.predict(sample, verbose=0)

    def _save_model(self, path: str):
        self.model.save(path)

    def get_parameter_count(self) -> int:
        return self.model.count_params()


class PyTorchLightningImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("PyTorch Lightning")

    def setup(self):
        self.code_content = """import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = MNISTClassifier()
trainer = pl.Trainer(max_epochs=5)"""

    def build_model(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import pytorch_lightning as pl
        except ImportError as e:
            raise DependencyError(
                dependency="torch and pytorch_lightning",
                feature="PyTorch Lightning model building",
                install_hint="pip install torch pytorch-lightning"
            ) from e
        
        class MNISTClassifier(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(32 * 13 * 13, 128)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(128, 10)
            
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = x.view(-1, 32 * 13 * 13)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
            
            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                self.log('train_loss', loss)
                return loss
            
            def validation_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                self.log('val_loss', loss)
                self.log('val_acc', acc)
            
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)
        
        self.model = MNISTClassifier()

    def train(self, dataset: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        try:
            import torch
            import pytorch_lightning as pl
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            raise DependencyError(
                dependency="torch and pytorch_lightning",
                feature="PyTorch Lightning training",
                install_hint="pip install torch pytorch-lightning"
            ) from e
        
        try:
            from tensorflow.keras.datasets import mnist
        except ImportError as e:
            raise DependencyError(
                dependency="tensorflow",
                feature="MNIST dataset loading",
                install_hint="pip install tensorflow"
            ) from e
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = torch.FloatTensor(x_train[:5000]).reshape(-1, 1, 28, 28) / 255.0
        y_train = torch.LongTensor(y_train[:5000])
        x_test = torch.FloatTensor(x_test[:1000]).reshape(-1, 1, 28, 28) / 255.0
        y_test = torch.LongTensor(y_test[:1000])
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, logger=False)
        
        start_time = time.time()
        trainer.fit(self.model, train_loader, val_loader)
        training_time = time.time() - start_time
        
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total
        
        return {
            "training_time": training_time,
            "accuracy": accuracy,
            "val_accuracy": accuracy,
            "val_loss": 0.0,
            "training_loss": 0.0,
            "peak_memory_mb": 0,
            "error_rate": 1.0 - accuracy,
        }

    def predict_single(self):
        try:
            import torch
        except ImportError as e:
            raise DependencyError(
                dependency="torch",
                feature="PyTorch prediction",
                install_hint="pip install torch"
            ) from e
        
        self.model.eval()
        sample = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            return self.model(sample)

    def _save_model(self, path: str):
        try:
            import torch
        except ImportError as e:
            raise DependencyError(
                dependency="torch",
                feature="PyTorch model saving",
                install_hint="pip install torch"
            ) from e
        
        torch.save(self.model.state_dict(), path)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class FastAIImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("Fast.ai")

    def setup(self):
        self.code_content = """from fastai.vision.all import Learner, CrossEntropyLossFlat, accuracy
from fastai.data.all import DataLoaders
import torch.nn as nn

def create_cnn_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 13 * 13, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10)
    )

dls = DataLoaders(train_dl, valid_dl)
learn = Learner(dls, create_cnn_model(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)"""

    def build_model(self):
        try:
            from fastai.vision.all import Learner
            from fastai.data.all import DataLoaders
            import torch.nn as nn
        except ImportError as e:
            raise DependencyError(
                dependency="fastai and torch",
                feature="Fast.ai model building",
                install_hint="pip install fastai torch"
            ) from e
        
        def create_cnn_model():
            return nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(32 * 13 * 13, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )
        
        self.model_fn = create_cnn_model

    def train(self, dataset: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        try:
            from fastai.vision.all import Learner, CrossEntropyLossFlat, accuracy
            from fastai.data.all import DataLoaders
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            raise DependencyError(
                dependency="fastai and torch",
                feature="Fast.ai training",
                install_hint="pip install fastai torch"
            ) from e
        
        try:
            from tensorflow.keras.datasets import mnist
        except ImportError as e:
            raise DependencyError(
                dependency="tensorflow",
                feature="MNIST dataset loading",
                install_hint="pip install tensorflow"
            ) from e
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = torch.FloatTensor(x_train[:5000]).reshape(-1, 1, 28, 28) / 255.0
        y_train = torch.LongTensor(y_train[:5000])
        x_test = torch.FloatTensor(x_test[:1000]).reshape(-1, 1, 28, 28) / 255.0
        y_test = torch.LongTensor(y_test[:1000])
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_test, y_test)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size)
        
        dls = DataLoaders(train_dl, val_dl)
        self.model = Learner(dls, self.model_fn(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        
        start_time = time.time()
        self.model.fit(epochs)
        training_time = time.time() - start_time
        
        val_metrics = self.model.validate()
        val_loss = float(val_metrics[0])
        val_acc = float(val_metrics[1])
        
        return {
            "training_time": training_time,
            "accuracy": val_acc,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "training_loss": 0.0,
            "peak_memory_mb": 0,
            "error_rate": 1.0 - val_acc,
        }

    def predict_single(self):
        try:
            import torch
        except ImportError as e:
            raise DependencyError(
                dependency="torch",
                feature="Fast.ai prediction",
                install_hint="pip install torch"
            ) from e
        
        sample = torch.randn(1, 1, 28, 28)
        return self.model.model(sample)

    def _save_model(self, path: str):
        self.model.export(path)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.model.parameters())


class LudwigImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("Ludwig")

    def setup(self):
        self.code_content = """from ludwig.api import LudwigModel
import pandas as pd

config = {
    'input_features': [
        {'name': 'image', 'type': 'image', 'encoder': 'stacked_cnn'}
    ],
    'output_features': [
        {'name': 'label', 'type': 'category'}
    ],
    'trainer': {
        'epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.001
    }
}

model = LudwigModel(config)
model.train(dataset=train_df)"""

    def build_model(self):
        try:
            from ludwig.api import LudwigModel
        except ImportError as e:
            raise DependencyError(
                dependency="ludwig",
                feature="Ludwig model building",
                install_hint="pip install ludwig"
            ) from e
        
        config = {
            'input_features': [
                {'name': 'image', 'type': 'image', 'encoder': 'stacked_cnn',
                 'preprocessing': {'height': 28, 'width': 28, 'num_channels': 1}}
            ],
            'output_features': [
                {'name': 'label', 'type': 'category'}
            ],
            'trainer': {
                'epochs': 5,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        self.model = LudwigModel(config)
        self.config = config

    def train(self, dataset: str, epochs: int, batch_size: int) -> Dict[str, Any]:
        try:
            import pandas as pd
        except ImportError as e:
            raise DependencyError(
                dependency="pandas",
                feature="Ludwig training",
                install_hint="pip install pandas"
            ) from e
        
        try:
            from tensorflow.keras.datasets import mnist
        except ImportError as e:
            raise DependencyError(
                dependency="tensorflow",
                feature="MNIST dataset loading",
                install_hint="pip install tensorflow"
            ) from e
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train[:1000]
        y_train = y_train[:1000]
        x_test = x_test[:200]
        y_test = y_test[:200]
        
        train_df = pd.DataFrame({
            'image': [x.reshape(28, 28, 1) for x in x_train],
            'label': y_train
        })
        
        self.config['trainer']['epochs'] = epochs
        self.config['trainer']['batch_size'] = batch_size
        
        start_time = time.time()
        try:
            train_stats, _, _ = self.model.train(dataset=train_df)
            training_time = time.time() - start_time
            
            test_df = pd.DataFrame({
                'image': [x.reshape(28, 28, 1) for x in x_test],
                'label': y_test
            })
            
            predictions, _ = self.model.predict(dataset=test_df)
            predicted_labels = predictions['label_predictions'].values
            accuracy = (predicted_labels == y_test).mean()
            
            return {
                "training_time": training_time,
                "accuracy": accuracy,
                "val_accuracy": accuracy,
                "val_loss": 0.0,
                "training_loss": 0.0,
                "peak_memory_mb": 0,
                "error_rate": 1.0 - accuracy,
            }
        except Exception as e:
            return {
                "training_time": 0,
                "accuracy": 0,
                "val_accuracy": 0,
                "val_loss": 0.0,
                "training_loss": 0.0,
                "peak_memory_mb": 0,
                "error_rate": 1.0,
            }

    def predict_single(self):
        try:
            import pandas as pd
            import numpy as np
        except ImportError as e:
            raise DependencyError(
                dependency="pandas and numpy",
                feature="Ludwig prediction",
                install_hint="pip install pandas numpy"
            ) from e
        
        sample = np.random.randn(28, 28, 1)
        test_df = pd.DataFrame({'image': [sample]})
        predictions, _ = self.model.predict(dataset=test_df)
        return predictions

    def _save_model(self, path: str):
        self.model.save(path)

    def get_parameter_count(self) -> int:
        return 0
