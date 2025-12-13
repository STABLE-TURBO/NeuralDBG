"""
LLM Integration Layer for Neural AI

Provides abstraction for different LLM providers (OpenAI, Anthropic, Open-Source).
Supports both API-based and local LLM execution.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Any
import os
import json
from neural.exceptions import DependencyError, ConfigurationError


class LLMProvider:
    """Base class for LLM providers."""
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return False


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4/GPT-3.5 integration."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4") -> None:
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        self.api_key: Optional[str] = api_key or os.getenv('OPENAI_API_KEY')
        self.model: str = model
        self._client: Any = None
        
        if self.api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                pass
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self._client is not None
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using OpenAI."""
        if not self.is_available():
            raise DependencyError(
                dependency='openai',
                feature='OpenAI LLM integration',
                install_hint='pip install openai and set OPENAI_API_KEY environment variable'
            )
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Neural DSL generation."""
        return """You are an expert in neural networks and the Neural DSL language.
Your task is to convert natural language descriptions into valid Neural DSL code.

Neural DSL syntax:
- network ModelName { input: (height, width, channels) layers: ... }
- Layers: Conv2D(filters, kernel_size, activation), Dense(units, activation), etc.
- Always provide complete, valid Neural DSL code.

Few-shot examples:

Example 1:
Input: Create a simple CNN for MNIST digit classification
Output:
```
network MNISTClassifier {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Conv2D(64, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Dropout(0.5)
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

Example 2:
Input: Build a ResNet-style model with skip connections for image classification
Output:
```
network ResNetClassifier {
    input: (224, 224, 3)
    layers:
        Conv2D(64, (7, 7), "relu", stride=2)
        BatchNormalization()
        MaxPooling2D((3, 3), stride=2)
        
        ResidualBlock(filters=64, blocks=3)
        ResidualBlock(filters=128, blocks=4, stride=2)
        ResidualBlock(filters=256, blocks=6, stride=2)
        
        GlobalAveragePooling2D()
        Dense(512, "relu")
        Dropout(0.5)
        Output(1000, "softmax")
    
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.0001)
    
    training: {
        epochs: 100
        batch_size: 32
        callbacks: [
            EarlyStopping(patience=10),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    }
}
```

Example 3:
Input: Create an LSTM model for text classification with 5 classes
Output:
```
network TextClassifier {
    input: (100,)  # Sequence length
    embedding: {
        vocab_size: 10000
        embedding_dim: 128
    }
    layers:
        Embedding(vocab_size=10000, embedding_dim=128)
        LSTM(64, return_sequences=True)
        Dropout(0.3)
        LSTM(32)
        Dense(64, "relu")
        Dropout(0.5)
        Output(5, "softmax")
    
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}
```

Respond only with the Neural DSL code in the format shown above."""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229") -> None:
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            model: Model to use
        """
        self.api_key: Optional[str] = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model: str = model
        self._client: Any = None
        
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                pass
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return self._client is not None
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using Anthropic Claude."""
        if not self.is_available():
            raise RuntimeError("Anthropic client not available. Install anthropic package and set API key.")
        
        response = self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', 1024),
            messages=[
                {"role": "user", "content": f"{self._get_system_prompt()}\n\n{prompt}"}
            ]
        )
        
        return response.content[0].text
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Neural DSL generation."""
        return """You are an expert in neural networks and the Neural DSL language.
Convert natural language descriptions into valid Neural DSL code.

Few-shot examples:

Example 1 - Simple CNN:
Input: Create a CNN for MNIST
Output:
network MNISTClassifier {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}

Example 2 - LSTM:
Input: Text classifier with LSTM
Output:
network TextClassifier {
    input: (100,)
    layers:
        Embedding(vocab_size=10000, embedding_dim=128)
        LSTM(64)
        Dense(64, "relu")
        Output(5, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}

Respond only with the Neural DSL code."""


class OllamaProvider(LLMProvider):
    """Ollama (local LLM) integration."""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434") -> None:
        """
        Initialize Ollama provider.
        
        Args:
            model: Ollama model name (llama2, mistral, etc.)
            base_url: Ollama server URL
        """
        self.model: str = model
        self.base_url: str = base_url
        self._available: bool = False
        
        # Check if Ollama is available
        try:
            import requests
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            self._available = response.status_code == 200
        except:
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._available
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using Ollama."""
        if not self.is_available():
            raise RuntimeError("Ollama not available. Make sure Ollama is running.")
        
        import requests
        
        full_prompt = f"{self._get_system_prompt()}\n\nUser: {prompt}\nAssistant:"
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                **kwargs
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}")
        
        return response.json().get('response', '')
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Neural DSL generation."""
        return """You are an expert in neural networks and the Neural DSL language.
Convert natural language descriptions into valid Neural DSL code.

Few-shot examples:

Example 1: Create a CNN for MNIST
network MNISTClassifier {
    input: (28, 28, 1)
    layers:
        Conv2D(32, (3, 3), "relu")
        MaxPooling2D((2, 2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}

Example 2: Text classifier with LSTM
network TextClassifier {
    input: (100,)
    layers:
        Embedding(vocab_size=10000, embedding_dim=128)
        LSTM(64)
        Dense(64, "relu")
        Output(5, "softmax")
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}

Respond only with the Neural DSL code."""


class LLMIntegration:
    """
    Unified LLM integration layer.
    
    Automatically selects available provider or allows manual selection.
    """
    
    def __init__(self, provider: Optional[str] = None) -> None:
        """
        Initialize LLM integration.
        
        Args:
            provider: Provider to use ('openai', 'anthropic', 'ollama', or None for auto)
        """
        self.providers: Dict[str, LLMProvider] = {
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider(),
            'ollama': OllamaProvider()
        }
        
        if provider:
            self.current_provider: Optional[LLMProvider] = self.providers.get(provider)
        else:
            # Auto-select first available provider
            self.current_provider = self._auto_select_provider()
    
    def _auto_select_provider(self) -> Optional[LLMProvider]:
        """Auto-select first available provider."""
        # Prefer local (Ollama) for privacy, then API providers
        priority = ['ollama', 'openai', 'anthropic']
        
        for provider_name in priority:
            provider = self.providers[provider_name]
            if provider.is_available():
                return provider
        
        return None
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        return self.current_provider is not None and self.current_provider.is_available()
    
    def generate_dsl(self, natural_language: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate Neural DSL from natural language.
        
        Args:
            natural_language: Natural language description
            context: Optional context (current model state, etc.)
            
        Returns:
            Generated Neural DSL code
        """
        if not self.is_available():
            raise RuntimeError("No LLM provider available. Install and configure a provider.")
        
        prompt = self._build_prompt(natural_language, context)
        response = self.current_provider.generate(prompt)
        
        # Extract DSL code from response (might include explanations)
        dsl_code = self._extract_dsl_code(response)
        
        return dsl_code
    
    def _build_prompt(self, natural_language: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for LLM."""
        prompt = f"Convert this natural language description into Neural DSL code:\n\n"
        prompt += f"{natural_language}\n\n"
        
        if context:
            prompt += f"Context: Current model has {len(context.get('layers', []))} layers.\n"
        
        prompt += "\nProvide only the Neural DSL code, no explanations."
        
        return prompt
    
    def _extract_dsl_code(self, response: str) -> str:
        """Extract DSL code from LLM response."""
        # Look for code blocks
        import re
        
        # Try to find code in markdown code blocks
        code_block = re.search(r'```(?:neural|python)?\n?(.*?)```', response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # Try to find network { ... } pattern
        network_match = re.search(r'network\s+\w+\s*\{.*?\}', response, re.DOTALL)
        if network_match:
            return network_match.group(0).strip()
        
        # Return as-is if no pattern found
        return response.strip()
