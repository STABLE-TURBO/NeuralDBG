# Enhanced AI Assistant - Quick Start Guide

Get started with Neural DSL's enhanced AI assistant in 5 minutes.

## Installation

The enhanced AI assistant is included with Neural DSL. No additional installation required for core features.

**Optional LLM Integration** (for advanced DSL generation):
```bash
# For OpenAI
pip install openai
export OPENAI_API_KEY='your-key-here'

# For Anthropic
pip install anthropic
export ANTHROPIC_API_KEY='your-key-here'

# For local LLMs (Ollama)
# Install Ollama from https://ollama.ai
ollama pull llama2
```

## 5-Minute Tutorial

### 1. Basic Usage

```python
from neural.ai import EnhancedAIAssistant

# Initialize (works without LLM)
assistant = EnhancedAIAssistant()

# Ask questions
response = assistant.chat("Why is my model overfitting?")
print(response['response'])
```

### 2. Analyze Training Metrics

```python
# Your training metrics
metrics = {
    'train_loss': [2.3, 1.8, 1.2, 0.8, 0.5, 0.3],
    'val_loss': [2.2, 1.7, 1.3, 1.2, 1.3, 1.4],
    'train_acc': [0.3, 0.5, 0.7, 0.85, 0.92, 0.95],
    'val_acc': [0.35, 0.52, 0.68, 0.72, 0.70, 0.68]
}

# Get optimization suggestions
analysis = assistant.analyze_training_metrics(metrics)

print(analysis['overall_assessment'])
for suggestion in analysis['optimization_suggestions']:
    print(f"\n{suggestion['title']}")
    print(f"→ {suggestion['description']}")
    print(f"Code: {suggestion['code_example']}")
```

**Output:**
```
Primary issue detected: regularization. I've identified 4 improvement opportunities.

Add Dropout Layers
→ Add Dropout(0.3-0.5) after Dense layers to reduce overfitting
Code: Dropout(0.5)

Apply L2 Regularization
→ Add weight decay to optimizer or use L2 regularization
Code: optimizer: Adam(learning_rate=0.001, weight_decay=1e-5)
```

### 3. Get Transfer Learning Recommendations

```python
# For a small image classification dataset
recommendations = assistant.recommend_transfer_learning(
    task_type='image_classification',
    dataset_size=5000
)

top_rec = recommendations[0]
print(f"Recommended: {top_rec['model_name']}")
print(f"Strategy: {top_rec['strategy']}")
print(f"Expected accuracy: {top_rec['expected_accuracy']:.1%}")
print(f"\nCode example:\n{top_rec['code_example']}")
```

**Output:**
```
Recommended: EfficientNetB0
Strategy: fine_tune_top_layers
Expected accuracy: 65.6%

Code example:
network TransferModel {
    input: (224, 224, 3)
    base_model: EfficientNetB0(weights="imagenet", freeze_until=-10)
    layers:
        GlobalAveragePooling2D()
        Dense(512, "relu")
        BatchNormalization()
        Dropout(0.5)
        Output(num_classes, "softmax")
    ...
}
```

### 4. Generate Data Augmentation Pipeline

```python
# For a small dataset
pipeline = assistant.generate_augmentation_pipeline(
    data_type='image',
    dataset_size=3000,
    level='moderate'
)

print(f"Expected improvement: {pipeline['expected_improvement']}")
print(f"\n{pipeline['code']}")
```

**Output:**
```
Expected improvement: 10% improvement in validation accuracy expected

data_augmentation: {
    pipeline: [
        RandomFlip(mode=horizontal),
        RandomRotation(degrees=15),
        RandomCrop(scale=(0.8, 1.0)),
        ColorJitter(brightness=0.2),
        RandomErasing(probability=0.5),
    ]
    probability: 0.8
}
```

### 5. Debug Training Issues

```python
# Diagnose a NaN loss issue
diagnosis = assistant.diagnose_issue(
    error_message="Loss became NaN at epoch 10",
    metrics={'train_loss': [2.3, 1.8, 1.2, 0.8, float('nan')]}
)

print(f"Issue: {diagnosis['primary_issue']['type'].value}")
print(f"Confidence: {diagnosis['primary_issue']['confidence']:.0%}")
print("\nRecommended actions:")
for action in diagnosis['recommended_actions'][:3]:
    print(f"• {action['action']}")

# Get debugging code
debug_code = assistant.get_debugging_code('loss_nan')
print(f"\nDebugging code:\n{debug_code['code'][:300]}...")
```

**Output:**
```
Issue: loss_nan
Confidence: 95%

Recommended actions:
• Reduce learning rate (try 0.0001)
• Add gradient clipping
• Check for numerical stability in loss function

Debugging code:
# Debug NaN loss
import tensorflow as tf
import numpy as np

# Check for NaN in data
assert not np.any(np.isnan(x_train)), "NaN in training data!"
...
```

### 6. Interactive Session with Context

```python
# Start a session
assistant = EnhancedAIAssistant(persistence_dir='./.neural_sessions')
session_id = assistant.start_session()

# Have a conversation
questions = [
    "My model is overfitting, what should I do?",
    "What about data augmentation?",
    "How do I add dropout in Neural DSL?"
]

for q in questions:
    response = assistant.chat(q)
    print(f"\nYou: {q}")
    print(f"AI: {response['response'][:200]}...")

# Save for later
assistant.save_session()

# Resume later
assistant.resume_session(session_id)
```

## Common Use Cases

### Case 1: Model Not Learning
```python
diagnosis = assistant.diagnose_issue(
    symptoms=["loss stuck at 2.3", "no improvement after 20 epochs"]
)
```

### Case 2: Choose Pre-trained Model
```python
recommendations = assistant.recommend_transfer_learning(
    task_type='image_classification',
    dataset_size=2000  # Small dataset
)
```

### Case 3: Improve Small Dataset Performance
```python
# Get augmentation
pipeline = assistant.generate_augmentation_pipeline(
    data_type='image',
    dataset_size=2000,
    level='aggressive'
)

# Get transfer learning
recommendations = assistant.recommend_transfer_learning(
    task_type='image_classification',
    dataset_size=2000
)

# Combined approach gives best results
```

### Case 4: Complete Task Guidance
```python
advice = assistant.get_comprehensive_advice(
    task_description="Classify medical X-rays",
    dataset_info={
        'type': 'image',
        'size': 3000,
        'task_type': 'image_classification'
    }
)

print("Transfer Learning:", advice['transfer_learning'][0]['model_name'])
print("Augmentation:", advice['data_augmentation']['level'])
print("Tips:", advice['architecture_tips'])
```

## Tips & Best Practices

### 1. Always Start with Metrics Analysis
```python
# After each training run
analysis = assistant.analyze_training_metrics(metrics)
# Follow top 2-3 suggestions
```

### 2. Use Transfer Learning for Small Datasets
```python
if dataset_size < 10000:
    recommendations = assistant.recommend_transfer_learning(...)
    # Use recommended model
```

### 3. Apply Augmentation Aggressively for Tiny Datasets
```python
if dataset_size < 1000:
    pipeline = assistant.generate_augmentation_pipeline(
        data_type='image',
        dataset_size=dataset_size,
        level='aggressive'  # Use aggressive for tiny datasets
    )
```

### 4. Debug Issues Immediately
```python
if 'nan' in str(loss) or loss > 100:
    diagnosis = assistant.diagnose_issue(error_message=f"Loss: {loss}")
    # Apply fixes immediately
```

### 5. Maintain Sessions for Complex Projects
```python
assistant = EnhancedAIAssistant(persistence_dir='./project_sessions')
session_id = assistant.start_session('my_project')
# Assistant remembers context across runs
```

## Troubleshooting

### LLM Not Working
```python
# Works without LLM
assistant = EnhancedAIAssistant(use_llm=False)

# Check LLM availability
if assistant.llm and assistant.llm.is_available():
    print("LLM ready!")
else:
    print("Using rule-based processing (still works!)")
```

### Session Not Persisting
```python
# Ensure directory exists and has write permissions
import os
os.makedirs('./.neural_sessions', exist_ok=True)

assistant = EnhancedAIAssistant(persistence_dir='./.neural_sessions')
session_id = assistant.start_session()
# ... use assistant ...
success = assistant.save_session()
print(f"Saved: {success}")
```

### Memory Issues
```python
# Don't store too many messages in session
# Save and start new session periodically
if len(assistant.context_manager.current_session.messages) > 100:
    assistant.save_session()
    assistant.start_session()
```

## Next Steps

1. **Read Full Documentation**: Check `neural/ai/README.md`
2. **Run Examples**: `python -m neural.ai.examples`
3. **Explore API**: Check individual module documentation
4. **Integrate with Training**: Use in training loops
5. **Customize**: Extend with custom advisors

## Quick Reference

```python
from neural.ai import EnhancedAIAssistant

assistant = EnhancedAIAssistant()

# Core methods
assistant.chat(user_input, context)
assistant.analyze_training_metrics(metrics, model_config)
assistant.recommend_transfer_learning(task_type, dataset_size, constraints)
assistant.generate_augmentation_pipeline(data_type, dataset_size, level)
assistant.diagnose_issue(symptoms, metrics, error_message)
assistant.get_comprehensive_advice(task_description, dataset_info)

# Session management
assistant.start_session(session_id)
assistant.save_session()
assistant.resume_session(session_id)
assistant.get_session_summary()
```

## Support

- **Documentation**: `neural/ai/README.md`
- **Examples**: `neural/ai/examples.py`
- **Implementation Details**: `neural/ai/IMPLEMENTATION_SUMMARY.md`
- **Issues**: Report on GitHub

Happy training! 🚀
