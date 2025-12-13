# Neural AI Integration Guide

## What This Actually Does

The AI integration translates natural language into Neural DSL code. Think of it as a convenience layer—you can say "add a conv layer with 64 filters" instead of typing out the DSL syntax. It's useful when you're prototyping or learning the DSL, but you'll still need to understand what you're building.

Two modes:
- **Rule-based** (default): Pattern matching for common operations. Fast, no setup, but limited.
- **LLM-powered** (optional): More flexible, but requires API keys or local setup, and the output still needs review.

## Getting Started

### Rule-Based Mode (Start Here)

This works out of the box with no external dependencies:

```python
from neural.ai.ai_assistant import NeuralAIAssistant

assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Create a CNN for MNIST classification")
print(result['dsl_code'])
```

The rule-based mode looks for keywords like "CNN", "MNIST", "convolutional", "dense", etc. It works well for straightforward requests but won't understand complex descriptions or nuanced requirements.

### When to Use What

**Use rule-based when:**
- Building standard architectures (CNNs, simple MLPs)
- Adding common layers (conv, dense, dropout)
- Learning the DSL syntax
- You want fast, predictable responses

**Use LLM when:**
- Describing complex architectures in natural language
- Experimenting with newer architectures (transformers, GANs)
- You don't mind reviewing and fixing the generated code

**Don't use AI when:**
- You already know the DSL well (just write it directly)
- Building production models (write explicit code)
- Fine-tuning specific hyperparameters (be explicit)

## Real Workflow Example

Here's how you might actually use this:

```python
assistant = NeuralAIAssistant(use_llm=False)

# Quick prototype: describe what you want
result = assistant.chat("Create a model for MNIST")
print(result['dsl_code'])
# Output:
# model MNIST {
#   input shape=[28, 28, 1]
#   Conv2D filters=32 kernel=[3,3] activation=relu
#   MaxPooling2D pool_size=[2,2]
#   Flatten
#   Dense units=128 activation=relu
#   Dense units=10 activation=softmax
# }

# That's a reasonable starting point, but let's refine it
result = assistant.chat("Add dropout after the dense layer")
# Now check if it put dropout in the right place...
```

The AI gives you a scaffold. You still need to:
- Verify the architecture makes sense for your data
- Adjust hyperparameters based on your needs
- Add proper training configuration
- Test and iterate

## Common Requests That Work

These work reliably with rule-based mode:

```python
assistant = NeuralAIAssistant(use_llm=False)

# Creating models
assistant.chat("Create a CNN for image classification")
assistant.chat("Create a model named ResNet")

# Adding layers
assistant.chat("Add Conv2D with 64 filters")
assistant.chat("Add MaxPooling2D")
assistant.chat("Add Dense layer with 256 units")
assistant.chat("Add dropout 0.5")
assistant.chat("Add batch normalization")

# Configuring training
assistant.chat("Set optimizer to Adam with learning rate 0.001")
assistant.chat("Use categorical crossentropy loss")
assistant.chat("Train for 50 epochs with batch size 32")
```

## What Usually Needs Fixing

The AI generates reasonable starting points, but here are things that often need manual adjustment:

**Problem: Generic hyperparameters**
```python
# AI generates this
result = assistant.chat("Create a CNN for my dataset")
# You get: Conv2D filters=32 kernel=[3,3]
# But your images are 512x512, so you probably want more filters
```

**Problem: Missing details**
```python
# AI generates this
result = assistant.chat("Add dense layer")
# You get: Dense units=128
# But it doesn't know if you want activation, dropout, regularization, etc.
```

**Problem: Order matters**
```python
# AI might put layers in suboptimal order
# You ask: "Add dropout and batch norm"
# You might get BatchNorm → Dropout
# But you wanted Dropout → BatchNorm
# Always review the generated DSL
```

**Fix: Be specific and then edit**
```python
# Better approach
result = assistant.chat("Add Conv2D 128 filters 3x3 relu")
# Then manually edit the DSL to add what's missing
```

## Building Models Incrementally

You can build up a model step by step, which is useful for exploration:

```python
assistant = NeuralAIAssistant(use_llm=False)

assistant.chat("Create a model named TextClassifier")
assistant.chat("Add embedding layer with 10000 vocab size")
assistant.chat("Add LSTM with 128 units")
assistant.chat("Add dense layer with 64 units and relu")
assistant.chat("Add output layer with 5 classes")

# Get the complete model
final_dsl = assistant.get_current_model()
```

This is handy when you're experimenting, but for real work, just write the DSL file directly. It's clearer and easier to version control.

## LLM Mode: More Flexible, More Review Needed

If you want to describe complex architectures, LLMs can help, but the output quality varies:

```python
# Setup required - pick one:
# 1. Local: Install Ollama (https://ollama.ai), run `ollama pull llama2`
# 2. OpenAI: pip install openai, set OPENAI_API_KEY
# 3. Anthropic: pip install anthropic, set ANTHROPIC_API_KEY

assistant = NeuralAIAssistant(use_llm=True, llm_provider='ollama')

# Complex request
result = assistant.chat(
    "Create a small transformer encoder with 4 attention heads "
    "and 2 layers for sequence classification"
)
print(result['dsl_code'])
```

**What to expect:**
- The LLM understands context better than pattern matching
- It might generate valid DSL for complex architectures
- It might also hallucinate layer types or parameters that don't exist
- You'll need to validate and often fix the output

**Reality check:**
LLMs are trained on general programming knowledge, not specifically on Neural DSL. They'll try to help, but they don't "know" your DSL syntax perfectly. Treat the output as a suggestion, not production-ready code.

## Actual Limitations

**Rule-based mode:**
- Only recognizes patterns it's been programmed to handle
- Can't understand context (e.g., "make it bigger" doesn't work)
- Won't suggest better architectures for your use case
- No conversation memory—each request is independent

**LLM mode:**
- Might generate invalid DSL syntax
- Can hallucinate features that don't exist
- Inconsistent quality across different requests
- Costs money (OpenAI/Anthropic) or requires local setup (Ollama)
- Slower than rule-based

**Both modes:**
- Won't optimize your model's performance
- Can't tell you if your architecture is appropriate for your data
- Don't understand your specific domain requirements
- Won't debug training issues

## When Not to Use AI

**Just use the DSL directly when:**

1. **You know what you want:**
   ```
   # Writing this directly is faster than describing it:
   model MyModel {
     input shape=[224, 224, 3]
     Conv2D filters=64 kernel=[3,3] activation=relu
     MaxPooling2D pool_size=[2,2]
     ...
   }
   ```

2. **You're using an existing architecture:**
   Use the examples in `examples/` as templates. Copy and modify them—it's more reliable than asking AI to recreate standard architectures.

3. **You need precise control:**
   If you're tuning hyperparameters or adjusting architecture based on experiments, direct editing is better. AI guessing at your intent adds unnecessary friction.

4. **Production code:**
   Don't rely on AI-generated code for production models. Write explicit, well-tested DSL files that you understand completely.

## Multi-Language Support

The system can handle inputs in other languages, but honestly, just use English. The translation layer adds another point of failure, and technical terms translate poorly.

If you do need it:
```python
# Requires: pip install deep-translator
assistant = NeuralAIAssistant(use_llm=False)
result = assistant.chat("Créer un CNN pour classification")  # French
result = assistant.chat("创建一个CNN模型")  # Chinese
```

It works by translating to English first, then processing. Direct English input is more reliable.

## Integration with Chat Interface

The AI assistant plugs into the neural chat interface:

```python
from neural.neural_chat.neural_chat import NeuralChat

chat = NeuralChat(use_ai=True)
response = chat.process_command("Create a CNN for image classification")
```

This is the same AI assistant, just wrapped in the interactive chat. Use it when you want a conversation-like interface for exploring the DSL.

## Response Format

The `chat()` method returns a dictionary:

```python
{
    'response': 'Human-readable response',
    'dsl_code': 'Generated DSL (if applicable)',
    'intent': 'What the system thinks you wanted',
    'success': True,  # or False if something went wrong
    'language': 'en'  # detected language
}
```

Always check `success` before using `dsl_code`. If intent detection fails, the DSL might be empty or incorrect.

## Debugging Tips

**"Nothing happens" or "Intent not recognized":**
- The rule-based system doesn't understand your phrasing
- Try simpler, more direct language: "add conv layer 32 filters" not "I'd like to include a convolutional operation"
- Check `examples/ai_examples.py` for working examples
- Consider using LLM mode if you need natural phrasing

**"LLM generates invalid DSL":**
- LLMs approximate based on training data—they don't have your grammar spec
- Validate the output: run it through the Neural DSL parser
- Use the generated code as a starting point and fix it manually
- Report common errors so we can improve the system prompt

**"Rule-based is too limited, LLM is too unreliable":**
- That's... accurate
- Use the AI for quick prototypes and learning
- Write DSL directly for anything important
- Think of AI mode as training wheels, not autopilot

## Examples

See `examples/ai_examples.py` for working code you can run.

## How It Works

```
Your input
    ↓
Language detection (if not English)
    ↓
Translation (if needed)
    ↓
Intent extraction (patterns or LLM)
    ↓
DSL generation
    ↓
Basic validation
    ↓
Returns DSL code + response
```

The system doesn't execute or train models—it just generates DSL text. You still need to compile and run it through the normal Neural workflow.

## Improving the AI

If you want to enhance the AI capabilities:

1. **Add patterns**: Edit `neural/ai/natural_language_processor.py` to handle new keywords
2. **Better prompts**: Improve LLM prompts in `neural/ai/llm_integration.py`
3. **Validation**: Add checks to catch common generation errors
4. **Examples**: More examples help both humans and LLMs learn the DSL

The rule-based system is regex and keyword matching. It's simple but predictable. The LLM system sends your request along with DSL documentation and examples to a language model, which attempts to generate valid DSL.

## Bottom Line

The AI integration is a convenience tool for prototyping and learning, not a replacement for understanding your models. Use it to get started quickly, but always review and refine the output. For production work or when you know exactly what you want, just write the DSL directly—it's clearer, faster, and more maintainable.
