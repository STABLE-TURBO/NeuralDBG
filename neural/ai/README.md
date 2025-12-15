# Neural DSL - AI Assistant

Basic AI-powered assistance for neural network development with natural language to DSL conversion.

## Features

### 🤖 Natural Language to DSL Conversion

Convert natural language descriptions into Neural DSL code using rule-based processing:

```python
from neural.ai import NeuralAIAssistant

assistant = NeuralAIAssistant()

# Create a model from natural language
result = assistant.chat("Create a CNN for MNIST digit classification")
print(result['dsl_code'])

# Add layers conversationally
result = assistant.chat("Add a dense layer with 128 units")
print(result['response'])

# Get current model
model = assistant.get_current_model()
print(model)
```

### 📝 Supported Intents

The assistant understands various natural language intents:

- **Create Model**: "Create a CNN for image classification"
- **Add Layer**: "Add a conv2d layer with 64 filters"
- **Modify Configuration**: "Set optimizer to Adam with learning rate 0.001"
- **Compile**: "Compile the model"
- **Visualize**: "Show me the model architecture"

### 🔧 Layer Types

Supported layer types and their parameters:

- **Conv2D**: Filters, kernel size, activation
  - Example: "Add a convolutional layer with 32 filters, kernel size 3, and relu activation"
- **Dense**: Units, activation
  - Example: "Add a fully connected layer with 128 units"
- **Dropout**: Rate
  - Example: "Add dropout with rate 0.5"
- **MaxPooling2D**: Pool size
  - Example: "Add max pooling with size 2"
- **Flatten**: No parameters
  - Example: "Add a flatten layer"

## Complete Example

```python
from neural.ai import NeuralAIAssistant

# Initialize assistant
assistant = NeuralAIAssistant()

# Build a model conversationally
assistant.chat("Create a model named MNISTClassifier for MNIST")
assistant.chat("Add a conv2d layer with 32 filters")
assistant.chat("Add max pooling")
assistant.chat("Add a conv2d layer with 64 filters")
assistant.chat("Add max pooling")
assistant.chat("Add flatten")
assistant.chat("Add a dense layer with 128 units")
assistant.chat("Add dropout with rate 0.5")

# Get the complete model
model_dsl = assistant.get_current_model()
print(model_dsl)

# Reset for a new model
assistant.reset()
```

## API Reference

### NeuralAIAssistant

Main interface for natural language to DSL conversion.

**Methods:**
- `chat(user_input, context=None)` - Process natural language input
  - Returns: Dictionary with `response`, `dsl_code`, `intent`, and `success`
- `get_current_model()` - Get the current model as DSL code
- `reset()` - Reset the assistant state

### NaturalLanguageProcessor

Low-level natural language processing utilities.

**Methods:**
- `extract_intent(text)` - Extract intent and parameters from text
  - Returns: Tuple of (IntentType, parameters dict)
- `detect_language(text)` - Detect the language of input text
  - Returns: Language code (e.g., 'en')

### DSLGenerator

Generate Neural DSL code from intents and parameters.

**Methods:**
- `generate_from_intent(intent_type, params)` - Generate DSL from intent
- `get_full_model()` - Get complete model as DSL

## Architecture

```
neural/ai/
├── ai_assistant.py                # Main assistant interface
├── natural_language_processor.py  # NLP utilities and DSL generation
└── README.md                      # This file
```

## Usage in Other Modules

The AI assistant is used by:
- `neural.neural_chat.neural_chat` - Conversational model building interface

## Limitations

- Rule-based processing only (no LLM integration)
- Limited to predefined intents and patterns
- Basic parameter extraction using regex
- English language only

## Future Enhancements

- LLM integration for more natural conversation
- Multi-language support
- More sophisticated intent detection
- Context-aware suggestions
- Integration with CLI commands

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Submit PR with clear description

## License

Same as Neural DSL project.
