from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class FederatedClient:
    def __init__(
        self,
        client_id: str,
        model: Any = None,
        local_data: Union[Tuple, Any] = None,
        backend: str = 'tensorflow',
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        device: Optional[str] = None,
        compute_capability: float = 1.0,
        bandwidth: float = 1.0,
    ):
        self.client_id = client_id
        self.model = model
        self.local_data = local_data
        self.backend = backend
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if backend == 'pytorch' else 'GPU')
        self.compute_capability = compute_capability
        self.bandwidth = bandwidth
        
        self.num_samples = 0
        self.training_history = []
        self.communication_cost = 0
        self.computation_time = 0
        
        self._initialize_data()
    
    def train(self, *args, **kwargs):
        return self.local_train()
    
    def _initialize_data(self):
        if isinstance(self.local_data, tuple) and len(self.local_data) == 2:
            X, y = self.local_data
            if hasattr(X, 'shape'):
                self.num_samples = len(X)
            elif hasattr(X, '__len__'):
                self.num_samples = len(X)
        else:
            if hasattr(self.local_data, '__len__'):
                self.num_samples = len(self.local_data)
    
    def set_weights(self, weights: List[np.ndarray]) -> None:
        if self.backend == 'tensorflow':
            try:
                import tensorflow as tf
                self.model.set_weights(weights)
            except ImportError:
                logger.warning("TensorFlow not available")
        elif self.backend == 'pytorch':
            try:
                import torch
                state_dict = self.model.state_dict()
                for i, (key, param) in enumerate(state_dict.items()):
                    if i < len(weights):
                        state_dict[key] = torch.tensor(weights[i])
                self.model.load_state_dict(state_dict)
            except ImportError:
                logger.warning("PyTorch not available")
    
    def get_weights(self) -> List[np.ndarray]:
        if self.backend == 'tensorflow':
            try:
                import tensorflow as tf
                return [w.numpy() for w in self.model.get_weights()]
            except ImportError:
                logger.warning("TensorFlow not available")
                return []
        elif self.backend == 'pytorch':
            try:
                import torch
                return [param.detach().cpu().numpy() for param in self.model.parameters()]
            except ImportError:
                logger.warning("PyTorch not available")
                return []
        return []
    
    def local_train(
        self,
        global_weights: Optional[List[np.ndarray]] = None,
        proximal_mu: float = 0.0,
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        if global_weights is not None:
            self.set_weights(global_weights)
        
        initial_weights = self.get_weights() if proximal_mu > 0 else None
        
        if self.backend == 'tensorflow':
            metrics = self._train_tensorflow(initial_weights, proximal_mu)
        elif self.backend == 'pytorch':
            metrics = self._train_pytorch(initial_weights, proximal_mu)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        self.computation_time = time.time() - start_time
        
        updated_weights = self.get_weights()
        
        return {
            'client_id': self.client_id,
            'weights': updated_weights,
            'num_samples': self.num_samples,
            'metrics': metrics,
            'computation_time': self.computation_time,
        }
    
    def _train_tensorflow(
        self,
        initial_weights: Optional[List[np.ndarray]],
        proximal_mu: float,
    ) -> Dict[str, float]:
        try:
            import tensorflow as tf
            
            if isinstance(self.local_data, tuple) and len(self.local_data) == 2:
                X, y = self.local_data
            else:
                raise ValueError("Invalid data format for TensorFlow training")
            
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            
            train_loss = []
            train_accuracy = []
            
            for epoch in range(self.local_epochs):
                epoch_loss = []
                epoch_acc = []
                
                num_batches = int(np.ceil(len(X) / self.batch_size))
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(X))
                    
                    X_batch = X[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx]
                    
                    with tf.GradientTape() as tape:
                        predictions = self.model(X_batch, training=True)
                        loss = loss_fn(y_batch, predictions)
                        
                        if proximal_mu > 0 and initial_weights is not None:
                            current_weights = self.model.get_weights()
                            proximal_term = 0
                            for w_current, w_initial in zip(current_weights, initial_weights):
                                proximal_term += tf.reduce_sum(tf.square(w_current - w_initial))
                            loss += (proximal_mu / 2) * proximal_term
                    
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    epoch_loss.append(float(loss))
                    
                    pred_labels = tf.argmax(predictions, axis=1)
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_labels, y_batch), tf.float32))
                    epoch_acc.append(float(accuracy))
                
                train_loss.append(np.mean(epoch_loss))
                train_accuracy.append(np.mean(epoch_acc))
            
            return {
                'loss': np.mean(train_loss),
                'accuracy': np.mean(train_accuracy),
            }
        
        except ImportError:
            logger.error("TensorFlow not available")
            return {'loss': 0.0, 'accuracy': 0.0}
    
    def _train_pytorch(
        self,
        initial_weights: Optional[List[np.ndarray]],
        proximal_mu: float,
    ) -> Dict[str, float]:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            if isinstance(self.local_data, tuple) and len(self.local_data) == 2:
                X, y = self.local_data
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.long)
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            else:
                dataloader = self.local_data
            
            device = torch.device(self.device if torch.cuda.is_available() and 'cuda' in self.device else 'cpu')
            self.model.to(device)
            
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            train_loss = []
            train_accuracy = []
            
            for epoch in range(self.local_epochs):
                epoch_loss = []
                epoch_acc = []
                
                self.model.train()
                for X_batch, y_batch in dataloader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    
                    if proximal_mu > 0 and initial_weights is not None:
                        proximal_term = 0
                        for param, initial_param in zip(self.model.parameters(), initial_weights):
                            proximal_term += torch.sum(torch.square(param - torch.tensor(initial_param).to(device)))
                        loss += (proximal_mu / 2) * proximal_term
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss.append(loss.item())
                    
                    pred_labels = torch.argmax(predictions, dim=1)
                    accuracy = (pred_labels == y_batch).float().mean()
                    epoch_acc.append(accuracy.item())
                
                train_loss.append(np.mean(epoch_loss))
                train_accuracy.append(np.mean(epoch_acc))
            
            return {
                'loss': np.mean(train_loss),
                'accuracy': np.mean(train_accuracy),
            }
        
        except ImportError:
            logger.error("PyTorch not available")
            return {'loss': 0.0, 'accuracy': 0.0}
    
    def evaluate(self, test_data: Optional[Tuple] = None) -> Dict[str, float]:
        if test_data is None:
            test_data = self.local_data
        
        if self.backend == 'tensorflow':
            return self._evaluate_tensorflow(test_data)
        elif self.backend == 'pytorch':
            return self._evaluate_pytorch(test_data)
        else:
            return {'loss': 0.0, 'accuracy': 0.0}
    
    def _evaluate_tensorflow(self, test_data: Tuple) -> Dict[str, float]:
        try:
            import tensorflow as tf
            
            if isinstance(test_data, tuple) and len(test_data) == 2:
                X_test, y_test = test_data
                predictions = self.model(X_test, training=False)
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                loss = float(loss_fn(y_test, predictions))
                
                pred_labels = tf.argmax(predictions, axis=1)
                accuracy = float(tf.reduce_mean(tf.cast(tf.equal(pred_labels, y_test), tf.float32)))
                
                return {'loss': loss, 'accuracy': accuracy}
        except ImportError:
            pass
        return {'loss': 0.0, 'accuracy': 0.0}
    
    def _evaluate_pytorch(self, test_data: Tuple) -> Dict[str, float]:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            device = torch.device(self.device if torch.cuda.is_available() and 'cuda' in self.device else 'cpu')
            self.model.to(device)
            self.model.eval()
            
            if isinstance(test_data, tuple) and len(test_data) == 2:
                X_test, y_test = test_data
                X_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_tensor = torch.tensor(y_test, dtype=torch.long)
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            else:
                dataloader = test_data
            
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in dataloader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    total_loss += loss.item() * len(X_batch)
                    
                    pred_labels = torch.argmax(predictions, dim=1)
                    correct += (pred_labels == y_batch).sum().item()
                    total += len(y_batch)
            
            return {
                'loss': total_loss / total if total > 0 else 0.0,
                'accuracy': correct / total if total > 0 else 0.0,
            }
        except ImportError:
            pass
        return {'loss': 0.0, 'accuracy': 0.0}
