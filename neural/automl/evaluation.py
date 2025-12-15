"""
Architecture evaluation and performance prediction for AutoML.

Provides tools to evaluate architectures and predict their performance.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MetricTracker:
    """Track metrics across training steps."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.step_metrics: Dict[int, Dict[str, float]] = {}
        self.current_step = 0
    
    def log(self, step: int, **metrics):
        """Log metrics for a step."""
        self.current_step = step
        self.step_metrics[step] = metrics
        
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def get_metric(self, name: str, step: Optional[int] = None) -> Optional[float]:
        """Get metric value at a specific step."""
        if step is None:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1]
            return None
        
        return self.step_metrics.get(step, {}).get(name)
    
    def get_best(self, name: str, mode: str = 'max') -> Tuple[int, float]:
        """Get best metric value and its step."""
        if name not in self.metrics or not self.metrics[name]:
            return -1, float('-inf') if mode == 'max' else float('inf')
        
        values = self.metrics[name]
        
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return best_idx, values[best_idx]
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """Get all tracked metrics."""
        return self.metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'last': values[-1]
                }
        
        return summary


class ArchitectureEvaluator:
    """Evaluate architectures on training tasks."""
    
    def __init__(
        self,
        backend: str = 'pytorch',
        device: str = 'auto',
        max_epochs: int = 10
    ):
        self.backend = backend
        self.device = device
        self.max_epochs = max_epochs
    
    def evaluate(
        self,
        architecture: Dict[str, Any],
        train_data,
        val_data,
        metric_tracker: Optional[MetricTracker] = None
    ) -> Dict[str, float]:
        """
        Evaluate an architecture.
        
        Args:
            architecture: Architecture configuration
            train_data: Training data loader
            val_data: Validation data loader
            metric_tracker: Optional metric tracker for step-wise logging
        
        Returns:
            Dictionary of evaluation metrics
        """
        if metric_tracker is None:
            metric_tracker = MetricTracker()
        
        try:
            model = self._build_model(architecture)
            optimizer = self._create_optimizer(model, architecture)
            
            start_time = time.time()
            
            for epoch in range(self.max_epochs):
                train_metrics = self._train_epoch(model, optimizer, train_data, epoch)
                val_metrics = self._validate(model, val_data, epoch)
                
                metric_tracker.log(
                    epoch,
                    train_loss=train_metrics['loss'],
                    train_acc=train_metrics['accuracy'],
                    val_loss=val_metrics['loss'],
                    val_acc=val_metrics['accuracy']
                )
            
            training_time = time.time() - start_time
            
            final_metrics = {
                'accuracy': val_metrics['accuracy'],
                'loss': val_metrics['loss'],
                'training_time': training_time,
                'num_params': self._count_parameters(model)
            }
            
            return final_metrics
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'accuracy': 0.0,
                'loss': float('inf'),
                'training_time': 0.0,
                'num_params': 0,
                'error': str(e)
            }
    
    def _build_model(self, architecture: Dict[str, Any]):
        """Build model from architecture."""
        from neural.automl.architecture_space import ArchitectureBuilder
        
        builder = ArchitectureBuilder(backend=self.backend)
        return builder.build(architecture)
    
    def _create_optimizer(self, model, architecture: Dict[str, Any]):
        """Create optimizer for model."""
        opt_config = architecture.get('optimizer', {'type': 'Adam', 'params': {'learning_rate': 0.001}})
        
        if self.backend == 'pytorch':
            import torch.optim as optim
            opt_type = opt_config['type']
            lr = opt_config.get('params', {}).get('learning_rate', 0.001)
            
            optimizer_class = getattr(optim, opt_type, optim.Adam)
            return optimizer_class(model.parameters(), lr=lr)
        
        elif self.backend == 'tensorflow':
            import tensorflow as tf
            opt_type = opt_config['type'].lower()
            lr = opt_config.get('params', {}).get('learning_rate', 0.001)
            
            if opt_type == 'adam':
                return tf.keras.optimizers.Adam(learning_rate=lr)
            elif opt_type == 'sgd':
                return tf.keras.optimizers.SGD(learning_rate=lr)
            else:
                return tf.keras.optimizers.Adam(learning_rate=lr)
    
    def _train_epoch(self, model, optimizer, train_data, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        if self.backend == 'pytorch':
            return self._train_epoch_pytorch(model, optimizer, train_data)
        elif self.backend == 'tensorflow':
            return self._train_epoch_tensorflow(model, optimizer, train_data)
    
    def _train_epoch_pytorch(self, model, optimizer, train_data) -> Dict[str, float]:
        """PyTorch training epoch."""
        import torch
        import torch.nn as nn
        
        from neural.execution_optimization.execution import get_device
        
        device = get_device(self.device)
        model.to(device)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_data:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return {
            'loss': total_loss / len(train_data),
            'accuracy': correct / total
        }
    
    def _train_epoch_tensorflow(self, model, optimizer, train_data) -> Dict[str, float]:
        """TensorFlow training epoch."""
        import tensorflow as tf
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_data:
            with tf.GradientTape() as tape:
                output = model(data, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(target, output)
                loss = tf.reduce_mean(loss)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            total_loss += loss.numpy()
            pred = tf.argmax(output, axis=1)
            correct += tf.reduce_sum(tf.cast(tf.equal(pred, target), tf.int32)).numpy()
            total += len(target)
        
        return {
            'loss': total_loss / len(train_data),
            'accuracy': correct / total
        }
    
    def _validate(self, model, val_data, epoch: int) -> Dict[str, float]:
        """Validate model."""
        if self.backend == 'pytorch':
            return self._validate_pytorch(model, val_data)
        elif self.backend == 'tensorflow':
            return self._validate_tensorflow(model, val_data)
    
    def _validate_pytorch(self, model, val_data) -> Dict[str, float]:
        """PyTorch validation."""
        import torch
        import torch.nn as nn
        
        from neural.execution_optimization.execution import get_device
        
        device = get_device(self.device)
        model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(val_data),
            'accuracy': correct / total
        }
    
    def _validate_tensorflow(self, model, val_data) -> Dict[str, float]:
        """TensorFlow validation."""
        import tensorflow as tf
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in val_data:
            output = model(data, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(target, output)
            loss = tf.reduce_mean(loss)
            
            total_loss += loss.numpy()
            pred = tf.argmax(output, axis=1)
            correct += tf.reduce_sum(tf.cast(tf.equal(pred, target), tf.int32)).numpy()
            total += len(target)
        
        return {
            'loss': total_loss / len(val_data),
            'accuracy': correct / total
        }
    
    def _count_parameters(self, model) -> int:
        """Count model parameters."""
        if self.backend == 'pytorch':
            return sum(p.numel() for p in model.parameters())
        elif self.backend == 'tensorflow':
            return model.count_params()
        return 0


class PerformancePredictor:
    """Predict architecture performance without full training."""
    
    def __init__(self, predictor_type: str = 'learning_curve'):
        self.predictor_type = predictor_type
        self.history: List[Dict[str, Any]] = []
    
    def predict(
        self,
        architecture: Dict[str, Any],
        partial_metrics: Optional[Dict[str, List[float]]] = None
    ) -> float:
        """
        Predict final performance of an architecture.
        
        Args:
            architecture: Architecture configuration
            partial_metrics: Partial training metrics (if available)
        
        Returns:
            Predicted final accuracy
        """
        if self.predictor_type == 'learning_curve' and partial_metrics:
            return self._predict_from_learning_curve(partial_metrics)
        elif self.predictor_type == 'zero_cost':
            return self._zero_cost_prediction(architecture)
        else:
            return 0.5
    
    def _predict_from_learning_curve(self, partial_metrics: Dict[str, List[float]]) -> float:
        """Predict from partial learning curve."""
        if 'val_acc' not in partial_metrics or not partial_metrics['val_acc']:
            return 0.5
        
        accuracies = partial_metrics['val_acc']
        
        if len(accuracies) < 3:
            return accuracies[-1]
        
        x = np.arange(len(accuracies))
        y = np.array(accuracies)
        
        try:
            from scipy.optimize import curve_fit
            
            def power_law(x, a, b, c):
                return a - b * np.power(x + 1, -c)
            
            popt, _ = curve_fit(
                power_law,
                x, y,
                p0=[0.9, 0.5, 0.5],
                maxfev=1000,
                bounds=([0, 0, 0], [1, 1, 2])
            )
            
            prediction = power_law(100, *popt)
            return float(np.clip(prediction, 0, 1))
        
        except Exception:
            recent_trend = np.mean(accuracies[-3:])
            return float(recent_trend)
    
    def _zero_cost_prediction(self, architecture: Dict[str, Any]) -> float:
        """Zero-cost proxy prediction."""
        from neural.automl.nas_operations import estimate_model_size
        
        num_params = estimate_model_size(architecture)
        num_layers = len(architecture.get('layers', []))
        
        score = 0.5
        
        if 30 <= num_layers <= 100:
            score += 0.1
        
        if 1e6 <= num_params <= 1e7:
            score += 0.1
        
        has_skip = any(
            layer.get('type') == 'Identity' or 'skip' in layer.get('type', '').lower()
            for layer in architecture.get('layers', [])
        )
        if has_skip:
            score += 0.05
        
        has_bn = any(
            layer.get('type') == 'BatchNormalization'
            for layer in architecture.get('layers', [])
        )
        if has_bn:
            score += 0.05
        
        return min(score, 0.95)
    
    def update(self, architecture: Dict[str, Any], final_accuracy: float):
        """Update predictor with actual results."""
        self.history.append({
            'architecture': architecture,
            'accuracy': final_accuracy
        })

class ModelEvaluator:
    def __init__(self, metric: str = 'accuracy'):
        self.metric = metric
        self._evaluator = ArchitectureEvaluator()
    
    def evaluate(self, model, train_data, val_data):
        return {'accuracy': 0.0, 'loss': 0.0}
