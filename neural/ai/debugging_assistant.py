"""
Debugging Assistant

Provides intelligent debugging assistance for common neural network issues,
training problems, and architecture errors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import re


class IssueType(Enum):
    """Types of debugging issues."""
    LOSS_NOT_DECREASING = "loss_not_decreasing"
    LOSS_EXPLODING = "loss_exploding"
    LOSS_NAN = "loss_nan"
    GRADIENT_VANISHING = "gradient_vanishing"
    GRADIENT_EXPLODING = "gradient_exploding"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    SLOW_TRAINING = "slow_training"
    ACCURACY_LOW = "accuracy_low"
    MEMORY_ERROR = "memory_error"
    SHAPE_MISMATCH = "shape_mismatch"
    CONVERGENCE_ISSUE = "convergence_issue"


class DebuggingAssistant:
    """
    Provides debugging assistance for neural network training issues.
    """
    
    def __init__(self) -> None:
        """Initialize debugging assistant."""
        self.debug_history: List[Dict[str, Any]] = []
        self.common_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common error patterns and solutions."""
        return {
            'loss_nan': {
                'symptoms': ['loss is NaN', 'loss becomes inf', 'NaN in gradients'],
                'causes': [
                    'Learning rate too high',
                    'Numerical instability',
                    'Division by zero',
                    'Log of zero or negative'
                ],
                'solutions': [
                    'Reduce learning rate (try 0.0001)',
                    'Add gradient clipping',
                    'Check for numerical stability in loss function',
                    'Add small epsilon to log operations',
                    'Use mixed precision training carefully'
                ]
            },
            'loss_not_decreasing': {
                'symptoms': ['loss stuck', 'no improvement', 'plateau'],
                'causes': [
                    'Learning rate too low',
                    'Poor initialization',
                    'Dead neurons',
                    'Wrong loss function',
                    'Data normalization issue'
                ],
                'solutions': [
                    'Increase learning rate',
                    'Try different weight initialization',
                    'Check activation functions (avoid saturating)',
                    'Verify loss function matches task',
                    'Normalize/standardize input data',
                    'Add batch normalization'
                ]
            },
            'gradient_vanishing': {
                'symptoms': ['gradients very small', 'early layers not learning', 'slow convergence'],
                'causes': [
                    'Deep network with sigmoid/tanh',
                    'Poor initialization',
                    'No batch normalization'
                ],
                'solutions': [
                    'Use ReLU instead of sigmoid/tanh',
                    'Add batch normalization',
                    'Use residual connections',
                    'Use He initialization for ReLU',
                    'Reduce network depth'
                ]
            },
            'gradient_exploding': {
                'symptoms': ['gradients very large', 'loss oscillating', 'unstable training'],
                'causes': [
                    'Learning rate too high',
                    'Large weights',
                    'Unstable architecture'
                ],
                'solutions': [
                    'Add gradient clipping (clipnorm=1.0)',
                    'Reduce learning rate',
                    'Use batch normalization',
                    'Check weight initialization',
                    'Add L2 regularization'
                ]
            },
            'memory_error': {
                'symptoms': ['OOM', 'CUDA out of memory', 'allocation failed'],
                'causes': [
                    'Batch size too large',
                    'Model too large',
                    'Accumulating gradients',
                    'Memory leak'
                ],
                'solutions': [
                    'Reduce batch size',
                    'Use gradient accumulation',
                    'Enable mixed precision (FP16)',
                    'Reduce model size',
                    'Clear cache between batches',
                    'Use gradient checkpointing'
                ]
            },
            'shape_mismatch': {
                'symptoms': ['shape error', 'dimension mismatch', 'incompatible shapes'],
                'causes': [
                    'Wrong input shape',
                    'Incorrect layer configuration',
                    'Batch dimension issue'
                ],
                'solutions': [
                    'Print shapes at each layer',
                    'Verify input shape matches model',
                    'Check flatten/reshape operations',
                    'Ensure batch dimension consistency',
                    'Use shape propagation tools'
                ]
            }
        }
    
    def diagnose_issue(
        self,
        symptoms: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Diagnose training or architecture issues.
        
        Args:
            symptoms: List of observed symptoms
            metrics: Training metrics (loss, accuracy, gradients, etc.)
            error_message: Optional error message
            
        Returns:
            Diagnosis with identified issues and recommended solutions
        """
        identified_issues: List[Dict[str, Any]] = []
        
        # Analyze error message if provided
        if error_message:
            issue = self._analyze_error_message(error_message)
            if issue:
                identified_issues.append(issue)
        
        # Analyze metrics if provided
        if metrics:
            metric_issues = self._analyze_metrics(metrics)
            identified_issues.extend(metric_issues)
        
        # Analyze symptoms if provided
        if symptoms:
            for symptom in symptoms:
                issue = self._match_symptom_to_issue(symptom)
                if issue and issue not in identified_issues:
                    identified_issues.append(issue)
        
        # Generate diagnosis report
        diagnosis = {
            'identified_issues': identified_issues,
            'primary_issue': identified_issues[0] if identified_issues else None,
            'recommended_actions': self._prioritize_solutions(identified_issues),
            'additional_tips': self._generate_general_tips(identified_issues)
        }
        
        # Store in history
        self.debug_history.append({
            'symptoms': symptoms,
            'metrics': metrics,
            'error': error_message,
            'diagnosis': diagnosis
        })
        
        return diagnosis
    
    def _analyze_error_message(self, error_message: str) -> Optional[Dict[str, Any]]:
        """Analyze error message and identify issue."""
        error_lower = error_message.lower()
        
        if 'nan' in error_lower or 'inf' in error_lower:
            return {
                'type': IssueType.LOSS_NAN,
                'pattern': 'loss_nan',
                'confidence': 0.95,
                'details': self.common_patterns['loss_nan']
            }
        
        if 'out of memory' in error_lower or 'oom' in error_lower:
            return {
                'type': IssueType.MEMORY_ERROR,
                'pattern': 'memory_error',
                'confidence': 0.95,
                'details': self.common_patterns['memory_error']
            }
        
        if 'shape' in error_lower or 'dimension' in error_lower:
            return {
                'type': IssueType.SHAPE_MISMATCH,
                'pattern': 'shape_mismatch',
                'confidence': 0.90,
                'details': self.common_patterns['shape_mismatch']
            }
        
        return None
    
    def _analyze_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze training metrics to identify issues."""
        issues = []
        
        train_loss = metrics.get('train_loss', [])
        val_loss = metrics.get('val_loss', [])
        gradients = metrics.get('gradients', {})
        
        if not train_loss:
            return issues
        
        # Check for NaN/Inf
        if any(x != x for x in train_loss[-3:]):  # NaN check
            issues.append({
                'type': IssueType.LOSS_NAN,
                'pattern': 'loss_nan',
                'confidence': 0.95,
                'details': self.common_patterns['loss_nan']
            })
        
        # Check for exploding loss
        if len(train_loss) >= 3:
            if train_loss[-1] > train_loss[0] * 10:
                issues.append({
                    'type': IssueType.LOSS_EXPLODING,
                    'pattern': 'gradient_exploding',
                    'confidence': 0.85,
                    'details': self.common_patterns['gradient_exploding']
                })
        
        # Check for stagnant loss
        if len(train_loss) >= 10:
            recent_variance = sum(
                (x - sum(train_loss[-10:]) / 10) ** 2 
                for x in train_loss[-10:]
            ) / 10
            if recent_variance < 1e-6:
                issues.append({
                    'type': IssueType.LOSS_NOT_DECREASING,
                    'pattern': 'loss_not_decreasing',
                    'confidence': 0.80,
                    'details': self.common_patterns['loss_not_decreasing']
                })
        
        # Check for vanishing gradients
        if gradients:
            avg_grad = sum(gradients.values()) / len(gradients)
            if avg_grad < 1e-6:
                issues.append({
                    'type': IssueType.GRADIENT_VANISHING,
                    'pattern': 'gradient_vanishing',
                    'confidence': 0.85,
                    'details': self.common_patterns['gradient_vanishing']
                })
        
        # Check for exploding gradients
        if gradients:
            max_grad = max(gradients.values())
            if max_grad > 100:
                issues.append({
                    'type': IssueType.GRADIENT_EXPLODING,
                    'pattern': 'gradient_exploding',
                    'confidence': 0.85,
                    'details': self.common_patterns['gradient_exploding']
                })
        
        return issues
    
    def _match_symptom_to_issue(self, symptom: str) -> Optional[Dict[str, Any]]:
        """Match symptom description to known issue pattern."""
        symptom_lower = symptom.lower()
        
        for pattern_name, pattern in self.common_patterns.items():
            if any(s in symptom_lower for s in pattern['symptoms']):
                return {
                    'type': IssueType[pattern_name.upper()],
                    'pattern': pattern_name,
                    'confidence': 0.75,
                    'details': pattern
                }
        
        return None
    
    def _prioritize_solutions(
        self,
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioritize and consolidate solutions."""
        if not issues:
            return []
        
        # Get primary issue
        primary = issues[0]
        solutions = []
        
        for i, sol in enumerate(primary['details']['solutions'], 1):
            solutions.append({
                'priority': 'high' if i <= 2 else 'medium',
                'action': sol,
                'issue_type': primary['type'].value
            })
        
        return solutions
    
    def _generate_general_tips(
        self,
        issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate general debugging tips."""
        tips = [
            "Monitor training metrics closely (loss, accuracy, gradients)",
            "Use TensorBoard or similar tools for visualization",
            "Start with a simple model and gradually increase complexity",
            "Verify data preprocessing and normalization",
            "Check for data leakage or label errors"
        ]
        
        if any(issue['type'] == IssueType.LOSS_NAN for issue in issues):
            tips.append("Add print statements to track where NaN first appears")
        
        if any(issue['type'] in [IssueType.GRADIENT_VANISHING, IssueType.GRADIENT_EXPLODING] 
               for issue in issues):
            tips.append("Use gradient clipping as a safety measure")
            tips.append("Consider using batch normalization throughout the network")
        
        return tips
    
    def suggest_debugging_code(
        self,
        issue_type: IssueType
    ) -> Dict[str, str]:
        """
        Suggest debugging code snippets.
        
        Args:
            issue_type: Type of issue
            
        Returns:
            Dictionary with debugging code examples
        """
        debugging_code = {
            IssueType.LOSS_NAN: """
# Debug NaN loss
import tensorflow as tf
import numpy as np

# Check for NaN in data
assert not np.any(np.isnan(x_train)), "NaN in training data!"
assert not np.any(np.isnan(y_train)), "NaN in labels!"

# Add numerical stability
epsilon = 1e-7
# Example: tf.math.log(predictions + epsilon) instead of tf.math.log(predictions)

# Monitor gradients
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check for NaN in gradients
    for grad, var in zip(gradients, model.trainable_variables):
        if tf.reduce_any(tf.math.is_nan(grad)):
            print(f"NaN gradient in {var.name}")
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
""",
            IssueType.GRADIENT_VANISHING: """
# Debug vanishing gradients
import tensorflow as tf

# Add gradient monitoring callback
class GradientMonitor(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    grads = tf.keras.backend.gradients(
                        self.model.total_loss, layer.kernel
                    )[0]
                    if grads is not None:
                        grad_norm = tf.norm(grads)
                        print(f"{layer.name}: gradient norm = {grad_norm:.6f}")

# Use callback during training
model.fit(x_train, y_train, callbacks=[GradientMonitor()])

# Solution: Add batch normalization
network MyModel {
    layers:
        Conv2D(64, (3,3), "relu")
        BatchNormalization()  # Add this
        Conv2D(64, (3,3), "relu")
        BatchNormalization()  # Add this
}
""",
            IssueType.MEMORY_ERROR: """
# Debug memory issues
import tensorflow as tf

# Check GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Use gradient accumulation for large models
accumulation_steps = 4
batch_size = 32 // accumulation_steps  # Effective batch size 32

# Reduce batch size in config
network MyModel {
    training: {
        batch_size: 16  # Reduce from 32
        epochs: 100
    }
}

# Clear memory between training runs
import gc
gc.collect()
tf.keras.backend.clear_session()
""",
            IssueType.SHAPE_MISMATCH: """
# Debug shape issues
import tensorflow as tf

# Print shapes at each layer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    # Add debug layer
    tf.keras.layers.Lambda(lambda x: tf.print("After Conv:", tf.shape(x)) or x),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Lambda(lambda x: tf.print("After Flatten:", tf.shape(x)) or x),
    tf.keras.layers.Dense(10)
])

# Use model.summary() to check shapes
model.summary()

# Verify input shape
print(f"Input shape: {x_train.shape}")
print(f"Expected: (batch_size, height, width, channels)")
"""
        }
        
        return {
            'code': debugging_code.get(issue_type, "# No specific debugging code available"),
            'explanation': self._generate_code_explanation(issue_type)
        }
    
    def _generate_code_explanation(self, issue_type: IssueType) -> str:
        """Generate explanation for debugging code."""
        explanations = {
            IssueType.LOSS_NAN: (
                "This code helps identify where NaN values appear in your training. "
                "Check data first, then gradients, then add numerical stability."
            ),
            IssueType.GRADIENT_VANISHING: (
                "Monitor gradient norms to confirm vanishing gradients. "
                "Add batch normalization to stabilize gradient flow."
            ),
            IssueType.MEMORY_ERROR: (
                "Enable memory growth, reduce batch size, and use gradient accumulation "
                "to train large models with limited memory."
            ),
            IssueType.SHAPE_MISMATCH: (
                "Print shapes at each layer to identify where the mismatch occurs. "
                "Use model.summary() to verify architecture."
            )
        }
        return explanations.get(issue_type, "Use this code to debug the issue.")
    
    def get_conversational_response(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate conversational debugging assistance.
        
        Args:
            user_query: User's debugging question
            context: Optional context (error message, metrics, etc.)
            
        Returns:
            Natural language response with debugging guidance
        """
        query_lower = user_query.lower()
        
        # Check for specific issues
        if 'nan' in query_lower or 'inf' in query_lower:
            return self._explain_nan_debugging()
        elif 'gradient' in query_lower and ('vanish' in query_lower or 'small' in query_lower):
            return self._explain_vanishing_gradients()
        elif 'gradient' in query_lower and ('explod' in query_lower or 'large' in query_lower):
            return self._explain_exploding_gradients()
        elif 'memory' in query_lower or 'oom' in query_lower:
            return self._explain_memory_debugging()
        elif 'loss' in query_lower and ('stuck' in query_lower or 'not' in query_lower):
            return self._explain_stuck_loss()
        elif 'shape' in query_lower or 'dimension' in query_lower:
            return self._explain_shape_debugging()
        elif context:
            # Diagnose based on provided context
            diagnosis = self.diagnose_issue(
                symptoms=context.get('symptoms'),
                metrics=context.get('metrics'),
                error_message=context.get('error')
            )
            
            if diagnosis['primary_issue']:
                issue = diagnosis['primary_issue']
                response = f"**Diagnosed Issue:** {issue['type'].value}\n\n"
                response += f"**Confidence:** {issue['confidence']:.0%}\n\n"
                response += "**Recommended Solutions:**\n"
                for i, action in enumerate(diagnosis['recommended_actions'][:3], 1):
                    response += f"{i}. {action['action']}\n"
                return response
            else:
                return "I couldn't identify a specific issue. Please provide more details!"
        else:
            return (
                "I can help debug your neural network! Tell me about:\n"
                "- NaN or Inf losses\n"
                "- Vanishing or exploding gradients\n"
                "- Memory errors\n"
                "- Loss not decreasing\n"
                "- Shape mismatches\n"
                "Or share your error message and I'll diagnose it!"
            )
    
    def _explain_nan_debugging(self) -> str:
        """Explain NaN debugging."""
        return """
**Debugging NaN Loss:**

Common causes:
1. **Learning rate too high** → Reduce to 0.0001
2. **Numerical instability** → Add epsilon to log operations
3. **Invalid data** → Check for NaN/Inf in dataset
4. **Division by zero** → Add small constant to denominators

Step-by-step debugging:
```python
# 1. Check data
assert not np.any(np.isnan(x_train)), "NaN in data!"

# 2. Add numerical stability
predictions = model(x)
predictions = tf.clip_by_value(predictions, 1e-7, 1.0)
loss = -tf.reduce_sum(y * tf.math.log(predictions + 1e-7))

# 3. Reduce learning rate
optimizer: Adam(learning_rate=0.0001)  # Instead of 0.001

# 4. Add gradient clipping
optimizer: Adam(learning_rate=0.0001, clipnorm=1.0)
```

Quick fixes:
- ✓ Reduce learning rate by 10x
- ✓ Add gradient clipping
- ✓ Check data preprocessing
- ✓ Use stable loss functions (e.g., sparse_categorical_crossentropy)
"""
    
    def _explain_vanishing_gradients(self) -> str:
        """Explain vanishing gradients."""
        return """
**Vanishing Gradients Problem:**

Symptoms:
- Early layers not learning
- Gradients < 1e-6
- Loss decreasing very slowly

Root causes:
- Sigmoid/tanh activations (saturate)
- Deep networks without normalization
- Poor weight initialization

Solutions:
1. **Use ReLU activation:**
   ```
   Dense(128, "relu")  # Instead of sigmoid
   ```

2. **Add Batch Normalization:**
   ```
   Conv2D(64, (3,3), "relu")
   BatchNormalization()  # Normalizes activations
   ```

3. **Use residual connections:**
   ```
   ResidualBlock(filters=64)  # Skip connections
   ```

4. **Proper initialization:**
   - He initialization for ReLU
   - Xavier initialization for tanh

Best practice architecture:
```
network MyModel {
    layers:
        Conv2D(64, (3,3), "relu")
        BatchNormalization()  # Add this!
        Conv2D(64, (3,3), "relu")
        BatchNormalization()  # And this!
        MaxPooling2D((2,2))
}
```

This should fix 90% of vanishing gradient issues!
"""
    
    def _explain_exploding_gradients(self) -> str:
        """Explain exploding gradients."""
        return """
**Exploding Gradients Problem:**

Symptoms:
- Loss becomes NaN or Inf
- Loss oscillating wildly
- Very large gradient values (>100)

Quick fixes:
1. **Gradient Clipping:**
   ```
   optimizer: Adam(learning_rate=0.001, clipnorm=1.0)
   ```

2. **Reduce Learning Rate:**
   ```
   optimizer: Adam(learning_rate=0.0001)  # 10x smaller
   ```

3. **Batch Normalization:**
   ```
   Conv2D(64, (3,3))
   BatchNormalization()  # Stabilizes training
   ```

4. **L2 Regularization:**
   ```
   optimizer: Adam(learning_rate=0.001, weight_decay=0.01)
   ```

Complete solution:
```python
network StableModel {
    layers:
        Conv2D(64, (3,3), "relu")
        BatchNormalization()
        Dropout(0.3)
    
    optimizer: Adam(
        learning_rate=0.0001,  # Lower LR
        clipnorm=1.0,          # Clip gradients
        weight_decay=0.01      # L2 regularization
    )
}
```

Apply all three fixes for maximum stability!
"""
    
    def _explain_memory_debugging(self) -> str:
        """Explain memory debugging."""
        return """
**Out of Memory (OOM) Errors:**

Quick fixes:
1. **Reduce batch size:**
   ```
   training: { batch_size: 16 }  # Instead of 32
   ```

2. **Enable memory growth (TensorFlow/GPU):**
   ```python
   physical_devices = tf.config.list_physical_devices('GPU')
   tf.config.experimental.set_memory_growth(device, True)
   ```

3. **Use gradient accumulation:**
   ```python
   # Simulate batch_size=32 with batch_size=8
   accumulation_steps = 4
   for step in range(accumulation_steps):
       # Forward pass
       # Accumulate gradients
   # Apply gradients once
   ```

4. **Mixed precision training (FP16):**
   ```python
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

5. **Reduce model size:**
   - Fewer layers
   - Fewer filters/units per layer
   - Use depthwise separable convolutions

Memory usage estimates:
- Batch size: 32 → ~4GB
- Batch size: 16 → ~2GB
- Batch size: 8 → ~1GB

Start with small batch size and increase gradually!
"""
    
    def _explain_stuck_loss(self) -> str:
        """Explain stuck loss debugging."""
        return """
**Loss Not Decreasing:**

Possible causes & solutions:

1. **Learning rate too low:**
   ```
   optimizer: Adam(learning_rate=0.01)  # Try higher
   ```

2. **Dead neurons (ReLU dying):**
   ```
   # Use Leaky ReLU instead
   Dense(128, "leaky_relu")
   ```

3. **Poor initialization:**
   ```
   # Use He initialization (automatic with most frameworks)
   # Or try different random seed
   ```

4. **Wrong loss function:**
   ```
   # For classification: categorical_crossentropy
   # For regression: mse or mae
   ```

5. **Data not normalized:**
   ```python
   # Normalize inputs to [0, 1] or standardize
   x_train = x_train / 255.0  # Images
   x_train = (x_train - mean) / std  # General
   ```

6. **Need batch normalization:**
   ```
   Conv2D(64, (3,3), "relu")
   BatchNormalization()  # Helps convergence
   ```

Debugging checklist:
☐ Verify data is normalized
☐ Check loss function matches task
☐ Try learning rate 10x higher
☐ Add batch normalization
☐ Monitor gradient magnitudes
☐ Verify labels are correct

If still stuck, try training on a tiny subset to verify the model CAN learn!
"""
    
    def _explain_shape_debugging(self) -> str:
        """Explain shape debugging."""
        return """
**Shape Mismatch Debugging:**

Common scenarios:

1. **Input shape mismatch:**
   ```python
   # Model expects: (batch, 28, 28, 1)
   # You provide: (batch, 28, 28)
   
   # Fix:
   x_train = x_train.reshape(-1, 28, 28, 1)
   ```

2. **Forgot to flatten before Dense:**
   ```
   Conv2D(64, (3,3))
   MaxPooling2D((2,2))
   Flatten()  # ← Don't forget this!
   Dense(128)
   ```

3. **Wrong output shape:**
   ```
   # For 10 classes:
   Output(10, "softmax")  # Not Dense(1)
   ```

4. **Batch dimension issues:**
   ```python
   # Single sample needs batch dimension
   x = x_train[0]  # Shape: (28, 28, 1)
   x = np.expand_dims(x, 0)  # Shape: (1, 28, 28, 1)
   predictions = model.predict(x)
   ```

Debugging tools:
```python
# 1. Print model summary
model.summary()

# 2. Check actual shapes
print(f"Input: {x_train.shape}")
print(f"Output: {y_train.shape}")

# 3. Add debug prints
model.add(Lambda(lambda x: tf.print("Shape:", tf.shape(x)) or x))
```

Pro tip: Always check `model.summary()` before training!
"""
