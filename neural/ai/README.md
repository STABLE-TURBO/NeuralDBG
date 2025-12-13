# Neural DSL - Enhanced AI Assistant

Comprehensive AI-powered assistance for neural network development with conversational optimization suggestions, architecture refinement, transfer learning recommendations, data augmentation strategies, and debugging help.

## Features

### 🎯 Model Optimization Assistant

Analyzes training metrics and provides actionable suggestions:

- **Overfitting Detection**: Identifies when model memorizes training data
- **Underfitting Detection**: Detects when model is too simple
- **Learning Rate Tuning**: Suggests optimal learning rates and schedules
- **Convergence Analysis**: Identifies slow or stalled training
- **Architecture Refinement**: Automatically suggests improvements

```python
from neural.ai import EnhancedAIAssistant

assistant = EnhancedAIAssistant()

# Analyze training metrics
metrics = {
    'train_loss': [2.3, 1.8, 1.2, 0.8, 0.5],
    'val_loss': [2.2, 1.7, 1.3, 1.2, 1.3],
    'train_acc': [0.3, 0.5, 0.7, 0.85, 0.92],
    'val_acc': [0.35, 0.52, 0.68, 0.72, 0.70]
}

analysis = assistant.analyze_training_metrics(metrics)

# Get actionable suggestions
for suggestion in analysis['optimization_suggestions']:
    print(f"{suggestion['title']}: {suggestion['description']}")
    print(f"Example: {suggestion['code_example']}")
```

### 🚀 Transfer Learning Advisor

Recommends pre-trained models and fine-tuning strategies:

- **Model Recommendations**: Based on task, dataset size, and constraints
- **Fine-tuning Strategies**: Feature extraction, layer-wise fine-tuning, full fine-tuning
- **Code Generation**: Generates complete DSL code for transfer learning
- **Task Similarity Analysis**: Evaluates source/target task compatibility

```python
# Get transfer learning recommendations
recommendations = assistant.recommend_transfer_learning(
    task_type='image_classification',
    dataset_size=5000,
    constraints={'max_params': 50e6}
)

for rec in recommendations:
    print(f"Model: {rec['model_name']}")
    print(f"Strategy: {rec['strategy']}")
    print(f"Expected accuracy: {rec['expected_accuracy']:.1%}")
    print(f"Code:\n{rec['code_example']}")
```

**Available Models:**
- **Images**: ResNet50, EfficientNet, VGG16, MobileNetV2, InceptionV3
- **Text**: BERT-base, DistilBERT

### 📊 Data Augmentation Advisor

Suggests optimal augmentation strategies:

- **Image Augmentation**: Flip, rotate, crop, color jitter, MixUp, CutMix, AutoAugment
- **Text Augmentation**: Synonym replacement, back-translation, random operations
- **Time Series**: Jitter, warping, window slicing
- **Pipeline Generation**: Creates complete augmentation pipelines

```python
# Generate augmentation pipeline
pipeline = assistant.generate_augmentation_pipeline(
    data_type='image',
    dataset_size=3000,
    level='moderate'  # light, moderate, aggressive
)

print(f"Expected improvement: {pipeline['expected_improvement']}")
print(f"Generated code:\n{pipeline['code']}")
```

**Augmentation Techniques by Effectiveness:**
- **High (>85%)**: RandomFlip, RandomCrop, MixUp, CutMix, AutoAugment
- **Medium (70-85%)**: RandomRotation, ColorJitter, TimeWarping
- **Utility**: RandomErasing, GaussianNoise, Dropout-like effects

### 🐛 Debugging Assistant

Diagnoses and provides solutions for common issues:

- **NaN/Inf Loss**: Identifies numerical instability causes
- **Gradient Issues**: Vanishing and exploding gradient detection
- **Memory Errors**: OOM debugging and optimization
- **Shape Mismatches**: Layer compatibility issues
- **Training Instability**: Loss oscillation and divergence

```python
# Diagnose issue
diagnosis = assistant.diagnose_issue(
    error_message="Loss is NaN after 5 epochs",
    metrics={'train_loss': [2.3, 1.8, 1.2, 0.8, float('nan')]}
)

print(f"Issue: {diagnosis['primary_issue']['type'].value}")
for action in diagnosis['recommended_actions']:
    print(f"- {action['action']}")

# Get debugging code
debug_code = assistant.get_debugging_code('loss_nan')
print(debug_code['code'])
```

**Common Issues Covered:**
- Loss not decreasing
- Loss exploding or NaN
- Gradient vanishing/exploding
- Out of memory errors
- Shape mismatches
- Slow convergence

### 💬 Context-Aware Conversations

Maintains conversation history and session state:

- **Session Management**: Persist conversations across sessions
- **Context Retention**: Remembers model state and previous discussions
- **Conversational Interface**: Natural language interaction
- **Multi-topic Support**: Handles optimization, debugging, and architecture questions

```python
# Start a session
assistant = EnhancedAIAssistant(persistence_dir='./.neural_sessions')
session_id = assistant.start_session()

# Ask questions with context
response1 = assistant.chat("Why is my model overfitting?")
response2 = assistant.chat("What about data augmentation?")  # Contextual
response3 = assistant.chat("How do I implement that?")  # Follows previous

# Save and resume
assistant.save_session()
assistant.resume_session(session_id)

# Get conversation summary
summary = assistant.get_session_summary()
```

### 🤖 LLM Integration with Few-Shot Examples

Enhanced prompts with examples for better DSL generation:

```python
from neural.ai import NeuralAIAssistant

assistant = NeuralAIAssistant(
    use_llm=True,
    llm_provider='openai'  # or 'anthropic', 'ollama'
)

# LLM automatically uses few-shot examples
result = assistant.chat("Create a ResNet-style CNN with skip connections")
print(result['dsl_code'])
```

**Supported Providers:**
- **OpenAI**: GPT-4, GPT-3.5 (requires API key)
- **Anthropic**: Claude 3 (requires API key)
- **Ollama**: Local LLMs (requires Ollama server)

## Complete Example

```python
from neural.ai import EnhancedAIAssistant

# Initialize with all features
assistant = EnhancedAIAssistant(
    use_llm=True,
    llm_provider='openai',
    persistence_dir='./.neural_sessions'
)

# Start interactive session
session_id = assistant.start_session()

# Get comprehensive advice
advice = assistant.get_comprehensive_advice(
    task_description="Medical image classification",
    dataset_info={
        'type': 'image',
        'size': 2000,
        'task_type': 'image_classification'
    }
)

print("Transfer Learning:", advice['transfer_learning'][0]['model_name'])
print("Augmentation:", advice['data_augmentation']['level'])
print("Architecture Tips:", advice['architecture_tips'])

# Analyze training progress
metrics = {
    'train_loss': [2.3, 1.8, 1.2, 0.8, 0.5],
    'val_loss': [2.2, 1.7, 1.3, 1.2, 1.3]
}
analysis = assistant.analyze_training_metrics(metrics)

for suggestion in analysis['optimization_suggestions']:
    print(f"\n{suggestion['title']}")
    print(f"Priority: {suggestion['priority']}")
    print(f"Action: {suggestion['description']}")
    print(f"Code: {suggestion['code_example']}")

# Interactive assistance
response = assistant.chat("My loss is stuck at 1.5, what should I do?")
print(response['response'])

# Save session
assistant.save_session()
```

## API Reference

### EnhancedAIAssistant

Main interface for all AI features.

**Methods:**
- `chat(user_input, context)` - Conversational interface
- `analyze_training_metrics(metrics, model_config)` - Analyze metrics
- `recommend_transfer_learning(task_type, dataset_size, constraints)` - Get recommendations
- `generate_augmentation_pipeline(data_type, dataset_size, level)` - Generate pipeline
- `diagnose_issue(symptoms, metrics, error_message)` - Debug issues
- `get_comprehensive_advice(task_description, dataset_info)` - Complete guidance
- `start_session(session_id)` - Start conversation session
- `save_session()` - Persist session
- `resume_session(session_id)` - Resume previous session

### Specialized Assistants

Individual components for specific tasks:

```python
from neural.ai import (
    ModelOptimizer,
    TransferLearningAdvisor,
    DataAugmentationAdvisor,
    DebuggingAssistant,
    ContextManager
)

# Use individually
optimizer = ModelOptimizer()
suggestions = optimizer.analyze_metrics(metrics)

transfer_advisor = TransferLearningAdvisor()
models = transfer_advisor.recommend_model(task_type, dataset_size)

augmentation_advisor = DataAugmentationAdvisor()
pipeline = augmentation_advisor.generate_augmentation_pipeline(...)

debugger = DebuggingAssistant()
diagnosis = debugger.diagnose_issue(...)

context_manager = ContextManager(persistence_dir='./sessions')
session = context_manager.start_session()
```

## Examples

See `neural/ai/examples.py` for comprehensive usage examples:

```bash
python -m neural.ai.examples
```

Examples include:
1. Optimization suggestions from metrics
2. Transfer learning recommendations
3. Data augmentation pipeline generation
4. Debugging assistance
5. Conversational assistance with context
6. Comprehensive task advice
7. Real-time metric-based optimization
8. Automatic architecture refinement

## Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY='your-key-here'

# Anthropic
export ANTHROPIC_API_KEY='your-key-here'

# Ollama (if using local LLMs)
# Ensure Ollama server is running on http://localhost:11434
```

### Persistence

Enable session persistence:

```python
assistant = EnhancedAIAssistant(persistence_dir='~/.neural_sessions')
```

Sessions are saved as JSON files and can be resumed later.

## Architecture

```
neural/ai/
├── ai_assistant.py           # Main assistant interface
├── enhanced_assistant.py     # Enhanced features coordinator
├── model_optimizer.py        # Optimization suggestions
├── transfer_learning.py      # Transfer learning advisor
├── data_augmentation.py      # Augmentation strategies
├── debugging_assistant.py    # Debug diagnostics
├── context_manager.py        # Session & context management
├── llm_integration.py        # LLM provider abstraction (enhanced)
├── natural_language_processor.py  # NLP utilities
├── multi_language.py         # Multi-language support
├── examples.py              # Usage examples
└── README.md                # This file
```

## Best Practices

1. **Start Simple**: Begin with basic features, add complexity as needed
2. **Monitor Metrics**: Continuously analyze training metrics for issues
3. **Use Transfer Learning**: Almost always beneficial for small datasets
4. **Apply Augmentation**: Essential for datasets < 10K samples
5. **Debug Early**: Address NaN/gradient issues immediately
6. **Save Sessions**: Persist conversations for future reference
7. **Iterate**: Use suggestions, evaluate results, refine approach

## Performance

- **Optimization Analysis**: < 100ms for typical metric arrays
- **Transfer Learning Recommendations**: < 50ms (catalog lookup)
- **Augmentation Pipeline Generation**: < 30ms
- **Debugging Diagnosis**: < 100ms for pattern matching
- **LLM Integration**: 1-5s (depends on provider and model)
- **Session Persistence**: < 50ms (JSON serialization)

## Limitations

- LLM features require API keys or local Ollama installation
- Transfer learning catalog limited to common models
- Augmentation recommendations based on general best practices
- Debugging patterns cover common issues (not exhaustive)
- Context retention limited by session storage

## Future Enhancements

- [ ] More pre-trained model integrations
- [ ] Custom augmentation policy learning
- [ ] Advanced metric analysis (learning curves, gradient flow)
- [ ] Interactive debugging with code execution
- [ ] Multi-modal model support
- [ ] Automated hyperparameter tuning integration
- [ ] Real-time training monitoring
- [ ] Community-contributed optimization patterns

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Submit PR with clear description

## License

Same as Neural DSL project.
