"""
Model Optimization Assistant

Provides conversational model optimization suggestions based on validation metrics,
architecture analysis, and performance patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import re


class OptimizationCategory(Enum):
    """Categories of optimization recommendations."""
    ARCHITECTURE = "architecture"
    HYPERPARAMETER = "hyperparameter"
    REGULARIZATION = "regularization"
    LEARNING_RATE = "learning_rate"
    DATA_AUGMENTATION = "data_augmentation"
    TRANSFER_LEARNING = "transfer_learning"
    COMPUTATIONAL = "computational"


class ModelOptimizer:
    """
    Analyzes model performance and provides optimization suggestions.
    
    Uses validation metrics, training patterns, and architecture analysis
    to generate actionable recommendations.
    """
    
    def __init__(self) -> None:
        """Initialize the model optimizer."""
        self.optimization_history: List[Dict[str, Any]] = []
    
    def analyze_metrics(
        self,
        metrics: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze validation metrics and generate optimization suggestions.
        
        Args:
            metrics: Dictionary containing training/validation metrics
                Expected keys: train_loss, val_loss, train_acc, val_acc, epoch
            model_config: Optional model configuration dict
            
        Returns:
            List of optimization suggestions with category, priority, and description
        """
        suggestions: List[Dict[str, Any]] = []
        
        train_loss = metrics.get('train_loss', [])
        val_loss = metrics.get('val_loss', [])
        train_acc = metrics.get('train_acc', [])
        val_acc = metrics.get('val_acc', [])
        
        if not train_loss or not val_loss:
            return suggestions
        
        # Detect overfitting
        if self._detect_overfitting(train_loss, val_loss, train_acc, val_acc):
            suggestions.extend(self._suggest_overfitting_solutions(model_config))
        
        # Detect underfitting
        if self._detect_underfitting(train_loss, val_loss, train_acc, val_acc):
            suggestions.extend(self._suggest_underfitting_solutions(model_config))
        
        # Detect training instability
        if self._detect_instability(train_loss, val_loss):
            suggestions.extend(self._suggest_stability_improvements())
        
        # Detect slow convergence
        if self._detect_slow_convergence(train_loss, val_loss):
            suggestions.extend(self._suggest_convergence_improvements())
        
        # Analyze learning rate
        if len(train_loss) > 5:
            suggestions.extend(self._suggest_learning_rate_adjustments(train_loss))
        
        # Store suggestions in history
        self.optimization_history.append({
            'metrics': metrics,
            'suggestions': suggestions
        })
        
        return suggestions
    
    def _detect_overfitting(
        self,
        train_loss: List[float],
        val_loss: List[float],
        train_acc: List[float],
        val_acc: List[float]
    ) -> bool:
        """Detect if model is overfitting."""
        if len(train_loss) < 3 or len(val_loss) < 3:
            return False
        
        # Check if validation loss increasing while train loss decreasing
        train_trend = train_loss[-1] < train_loss[-3]
        val_trend = val_loss[-1] > val_loss[-3]
        
        # Check gap between train and validation
        if train_acc and val_acc:
            acc_gap = train_acc[-1] - val_acc[-1]
            if acc_gap > 0.15:  # 15% gap
                return True
        
        return train_trend and val_trend
    
    def _detect_underfitting(
        self,
        train_loss: List[float],
        val_loss: List[float],
        train_acc: List[float],
        val_acc: List[float]
    ) -> bool:
        """Detect if model is underfitting."""
        if len(train_loss) < 5:
            return False
        
        # Both training and validation not improving much
        train_improvement = abs(train_loss[-1] - train_loss[-5]) / train_loss[-5]
        val_improvement = abs(val_loss[-1] - val_loss[-5]) / val_loss[-5]
        
        # Check if accuracy is low
        if train_acc and len(train_acc) >= 3:
            if train_acc[-1] < 0.7:  # Low training accuracy
                return True
        
        return train_improvement < 0.05 and val_improvement < 0.05
    
    def _detect_instability(
        self,
        train_loss: List[float],
        val_loss: List[float]
    ) -> bool:
        """Detect training instability (high variance)."""
        if len(train_loss) < 5:
            return False
        
        # Calculate variance in recent losses
        recent_train = train_loss[-5:]
        train_std = sum((x - sum(recent_train)/len(recent_train))**2 for x in recent_train) ** 0.5
        train_mean = sum(recent_train) / len(recent_train)
        
        # High coefficient of variation indicates instability
        if train_mean > 0:
            cv = train_std / train_mean
            return cv > 0.3
        
        return False
    
    def _detect_slow_convergence(
        self,
        train_loss: List[float],
        val_loss: List[float]
    ) -> bool:
        """Detect slow convergence."""
        if len(train_loss) < 10:
            return False
        
        # Check if loss is decreasing very slowly
        recent_improvement = abs(train_loss[-1] - train_loss[-10]) / train_loss[-10]
        
        return recent_improvement < 0.02  # Less than 2% improvement over 10 epochs
    
    def _suggest_overfitting_solutions(
        self,
        model_config: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for overfitting."""
        suggestions = []
        
        suggestions.append({
            'category': OptimizationCategory.REGULARIZATION.value,
            'priority': 'high',
            'title': 'Add Dropout Layers',
            'description': 'Add Dropout(0.3-0.5) after Dense layers to reduce overfitting',
            'code_example': 'Dropout(0.5)',
            'expected_improvement': 'Reduce overfitting by 10-20%'
        })
        
        suggestions.append({
            'category': OptimizationCategory.REGULARIZATION.value,
            'priority': 'high',
            'title': 'Apply L2 Regularization',
            'description': 'Add weight decay to optimizer or use L2 regularization',
            'code_example': 'optimizer: Adam(learning_rate=0.001, weight_decay=1e-5)',
            'expected_improvement': 'Smoother learning curves, better generalization'
        })
        
        suggestions.append({
            'category': OptimizationCategory.DATA_AUGMENTATION.value,
            'priority': 'medium',
            'title': 'Increase Data Augmentation',
            'description': 'Apply random flips, rotations, and crops to training data',
            'code_example': 'data_augmentation: [RandomFlip(), RandomRotation(15), RandomCrop()]',
            'expected_improvement': 'Improve validation accuracy by 3-7%'
        })
        
        suggestions.append({
            'category': OptimizationCategory.REGULARIZATION.value,
            'priority': 'medium',
            'title': 'Add Batch Normalization',
            'description': 'Insert BatchNormalization after Conv/Dense layers',
            'code_example': 'BatchNormalization()',
            'expected_improvement': 'Stabilize training and reduce overfitting'
        })
        
        return suggestions
    
    def _suggest_underfitting_solutions(
        self,
        model_config: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for underfitting."""
        suggestions = []
        
        suggestions.append({
            'category': OptimizationCategory.ARCHITECTURE.value,
            'priority': 'high',
            'title': 'Increase Model Capacity',
            'description': 'Add more layers or increase units/filters per layer',
            'code_example': 'Dense(256, "relu")  # Increase from 128',
            'expected_improvement': 'Better feature learning, higher accuracy'
        })
        
        suggestions.append({
            'category': OptimizationCategory.HYPERPARAMETER.value,
            'priority': 'high',
            'title': 'Increase Training Epochs',
            'description': 'Train for more epochs to allow model to converge',
            'code_example': 'training: { epochs: 100, batch_size: 32 }',
            'expected_improvement': 'Allow model to fully learn patterns'
        })
        
        suggestions.append({
            'category': OptimizationCategory.LEARNING_RATE.value,
            'priority': 'medium',
            'title': 'Increase Learning Rate',
            'description': 'Use a higher learning rate for faster convergence',
            'code_example': 'optimizer: Adam(learning_rate=0.01)',
            'expected_improvement': 'Faster convergence to better solution'
        })
        
        suggestions.append({
            'category': OptimizationCategory.ARCHITECTURE.value,
            'priority': 'medium',
            'title': 'Use More Complex Layers',
            'description': 'Replace simple layers with more powerful alternatives',
            'code_example': 'ResidualBlock(filters=64) instead of Conv2D',
            'expected_improvement': 'Better representation learning'
        })
        
        return suggestions
    
    def _suggest_stability_improvements(self) -> List[Dict[str, Any]]:
        """Generate suggestions for training stability."""
        suggestions = []
        
        suggestions.append({
            'category': OptimizationCategory.LEARNING_RATE.value,
            'priority': 'high',
            'title': 'Reduce Learning Rate',
            'description': 'Lower learning rate to stabilize training',
            'code_example': 'optimizer: Adam(learning_rate=0.0001)',
            'expected_improvement': 'Smoother convergence, less oscillation'
        })
        
        suggestions.append({
            'category': OptimizationCategory.HYPERPARAMETER.value,
            'priority': 'medium',
            'title': 'Add Gradient Clipping',
            'description': 'Clip gradients to prevent exploding gradients',
            'code_example': 'optimizer: Adam(learning_rate=0.001, clipnorm=1.0)',
            'expected_improvement': 'Prevent gradient explosion'
        })
        
        suggestions.append({
            'category': OptimizationCategory.REGULARIZATION.value,
            'priority': 'medium',
            'title': 'Use Batch Normalization',
            'description': 'Add BatchNormalization for more stable training',
            'code_example': 'BatchNormalization()',
            'expected_improvement': 'Stabilize activations and gradients'
        })
        
        return suggestions
    
    def _suggest_convergence_improvements(self) -> List[Dict[str, Any]]:
        """Generate suggestions for slow convergence."""
        suggestions = []
        
        suggestions.append({
            'category': OptimizationCategory.LEARNING_RATE.value,
            'priority': 'high',
            'title': 'Use Learning Rate Schedule',
            'description': 'Implement learning rate decay or cosine annealing',
            'code_example': 'lr_schedule: CosineDecay(initial_lr=0.01, decay_steps=1000)',
            'expected_improvement': 'Better convergence, higher final accuracy'
        })
        
        suggestions.append({
            'category': OptimizationCategory.HYPERPARAMETER.value,
            'priority': 'medium',
            'title': 'Try Different Optimizer',
            'description': 'Switch to AdamW or SGD with momentum',
            'code_example': 'optimizer: AdamW(learning_rate=0.001, weight_decay=0.01)',
            'expected_improvement': 'Potentially faster convergence'
        })
        
        suggestions.append({
            'category': OptimizationCategory.ARCHITECTURE.value,
            'priority': 'medium',
            'title': 'Add Skip Connections',
            'description': 'Use residual connections to improve gradient flow',
            'code_example': 'ResidualBlock(filters=64)',
            'expected_improvement': 'Better gradient flow, faster training'
        })
        
        return suggestions
    
    def _suggest_learning_rate_adjustments(
        self,
        train_loss: List[float]
    ) -> List[Dict[str, Any]]:
        """Suggest learning rate adjustments based on loss patterns."""
        suggestions = []
        
        # Check if loss plateaued
        recent_losses = train_loss[-5:]
        loss_variance = sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses)
        
        if loss_variance < 1e-6:  # Very flat loss
            suggestions.append({
                'category': OptimizationCategory.LEARNING_RATE.value,
                'priority': 'high',
                'title': 'Reduce Learning Rate on Plateau',
                'description': 'Learning appears to have plateaued, reduce LR',
                'code_example': 'callbacks: [ReduceLROnPlateau(factor=0.5, patience=3)]',
                'expected_improvement': 'Break through plateau, continue improving'
            })
        
        return suggestions
    
    def suggest_architecture_refinement(
        self,
        model_config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Suggest automatic architecture refinements.
        
        Args:
            model_config: Current model configuration
            metrics: Optional performance metrics
            
        Returns:
            Dictionary with refined model configuration and explanation
        """
        refined_config = model_config.copy()
        changes: List[str] = []
        
        layers = model_config.get('layers', [])
        
        # Add batch normalization after conv layers if missing
        new_layers = []
        for i, layer in enumerate(layers):
            new_layers.append(layer)
            if layer.get('type') == 'Conv2D':
                # Check if next layer is batch norm
                if i + 1 >= len(layers) or layers[i + 1].get('type') != 'BatchNormalization':
                    new_layers.append({'type': 'BatchNormalization', 'params': {}})
                    changes.append(f"Added BatchNormalization after Conv2D layer {i}")
        
        # Add dropout before output layer if missing
        if new_layers and new_layers[-1].get('type') in ['Dense', 'Output']:
            if len(new_layers) < 2 or new_layers[-2].get('type') != 'Dropout':
                dropout_layer = {'type': 'Dropout', 'params': {'rate': 0.5}}
                new_layers.insert(-1, dropout_layer)
                changes.append("Added Dropout(0.5) before output layer")
        
        refined_config['layers'] = new_layers
        
        return {
            'refined_config': refined_config,
            'changes': changes,
            'reason': 'Added regularization layers to improve generalization'
        }
    
    def get_conversational_response(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate conversational response about optimization.
        
        Args:
            user_query: User's question about optimization
            context: Optional context (metrics, model config, etc.)
            
        Returns:
            Natural language response
        """
        query_lower = user_query.lower()
        
        # Check for specific optimization topics
        if 'overfit' in query_lower:
            return self._explain_overfitting_solutions()
        elif 'underfit' in query_lower:
            return self._explain_underfitting_solutions()
        elif 'learning rate' in query_lower:
            return self._explain_learning_rate_tuning()
        elif 'architecture' in query_lower or 'model' in query_lower:
            return self._explain_architecture_improvements()
        elif 'converge' in query_lower or 'training slow' in query_lower:
            return self._explain_convergence_improvements()
        elif context and 'metrics' in context:
            # Analyze provided metrics
            suggestions = self.analyze_metrics(context['metrics'], context.get('model_config'))
            if suggestions:
                response = "Based on your metrics, here are my suggestions:\n\n"
                for i, sugg in enumerate(suggestions[:3], 1):
                    response += f"{i}. **{sugg['title']}** ({sugg['priority']} priority)\n"
                    response += f"   {sugg['description']}\n"
                    response += f"   Example: `{sugg['code_example']}`\n\n"
                return response
            else:
                return "Your model appears to be training well. Continue monitoring metrics!"
        else:
            return (
                "I can help with model optimization! Ask me about:\n"
                "- Overfitting or underfitting issues\n"
                "- Learning rate tuning\n"
                "- Architecture improvements\n"
                "- Slow convergence\n"
                "Or share your metrics and I'll analyze them!"
            )
    
    def _explain_overfitting_solutions(self) -> str:
        """Explain overfitting solutions."""
        return """
**Overfitting occurs when your model memorizes training data but fails on new data.**

Common solutions:
1. **Dropout**: Add Dropout(0.3-0.5) after Dense layers
2. **L2 Regularization**: Add weight_decay to optimizer
3. **Data Augmentation**: Increase training data variety
4. **Early Stopping**: Stop training when validation loss stops improving
5. **Reduce Model Complexity**: Fewer layers or units

Try adding: `Dropout(0.5)` after your dense layers and see if validation loss improves!
"""
    
    def _explain_underfitting_solutions(self) -> str:
        """Explain underfitting solutions."""
        return """
**Underfitting means your model is too simple to learn the patterns.**

Common solutions:
1. **Increase Capacity**: Add more layers or units
2. **Train Longer**: More epochs allow better learning
3. **Reduce Regularization**: Remove or reduce dropout/weight decay
4. **Better Architecture**: Use more powerful layer types
5. **Check Learning Rate**: Might be too low

Try increasing your dense layer units: `Dense(256, "relu")` instead of 128!
"""
    
    def _explain_learning_rate_tuning(self) -> str:
        """Explain learning rate tuning."""
        return """
**Learning rate is the most important hyperparameter!**

Guidelines:
- **Too high**: Training unstable, loss oscillates
- **Too low**: Slow convergence, gets stuck
- **Just right**: Smooth decrease, good convergence

Recommendations:
1. Start with 0.001 (Adam) or 0.01 (SGD)
2. Use learning rate schedules (decay over time)
3. Try ReduceLROnPlateau callback
4. Consider cyclical learning rates

Example: `optimizer: Adam(learning_rate=0.001)` with `ReduceLROnPlateau(factor=0.5, patience=5)`
"""
    
    def _explain_architecture_improvements(self) -> str:
        """Explain architecture improvements."""
        return """
**Architecture design tips:**

Key principles:
1. **Start Simple**: Begin with basic architecture
2. **Add Gradually**: Increase complexity as needed
3. **Use Best Practices**: BatchNorm, Dropout, Skip Connections
4. **Match Task**: CNN for images, RNN for sequences

Common improvements:
- Add BatchNormalization after Conv/Dense layers
- Use residual connections (ResNet-style)
- Increase depth gradually
- Use appropriate pooling

Example improvement:
```
Conv2D(64, (3,3), "relu")
BatchNormalization()
Conv2D(64, (3,3), "relu")
MaxPooling2D((2,2))
Dropout(0.3)
```
"""
    
    def _explain_convergence_improvements(self) -> str:
        """Explain convergence improvements."""
        return """
**Slow convergence? Here's how to speed it up:**

Strategies:
1. **Learning Rate Schedule**: Start high, decay over time
   - CosineDecay, ExponentialDecay, StepDecay
2. **Better Optimizer**: Try AdamW, RMSprop, or SGD with momentum
3. **Batch Normalization**: Speeds up training significantly
4. **Skip Connections**: Improves gradient flow
5. **Proper Initialization**: Use He/Xavier initialization

Quick wins:
- Add learning rate schedule: `lr_schedule: CosineDecay(initial_lr=0.01)`
- Use Adam with weight decay: `optimizer: AdamW(learning_rate=0.001)`
- Add BatchNormalization after each layer

These changes typically speed up training by 2-3x!
"""
