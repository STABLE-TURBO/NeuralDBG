"""
Natural Language Processor for Neural DSL

Converts natural language descriptions into Neural DSL code.
Supports multiple languages through translation layer.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class IntentType(Enum):
    """Types of user intents."""
    CREATE_MODEL = "create_model"
    ADD_LAYER = "add_layer"
    MODIFY_LAYER = "modify_layer"
    SET_OPTIMIZER = "set_optimizer"
    SET_LOSS = "set_loss"
    SET_TRAINING = "set_training"
    COMPILE = "compile"
    VISUALIZE = "visualize"
    UNKNOWN = "unknown"


class NaturalLanguageProcessor:
    """
    Processes natural language input and extracts intent for Neural DSL generation.
    
    This is the first step in the AI integration pipeline. It can work with:
    1. Rule-based pattern matching (current implementation)
    2. LLM-based intent extraction (future enhancement)
    """
    
    def __init__(self) -> None:
        """Initialize the natural language processor."""
        self.layer_keywords: Dict[str, List[str]] = {
            'conv2d': ['convolutional', 'conv', 'cnn', 'conv2d', 'convolution'],
            'dense': ['dense', 'fully connected', 'fc', 'linear'],
            'dropout': ['dropout', 'drop'],
            'maxpooling2d': ['max pooling', 'maxpool', 'max pooling2d'],
            'flatten': ['flatten', 'flattening'],
            'lstm': ['lstm', 'long short term memory'],
            'gru': ['gru', 'gated recurrent unit'],
            'batchnormalization': ['batch norm', 'batch normalization', 'batchnorm'],
            'output': ['output', 'final layer', 'classification']
        }
        
        self.activation_keywords: Dict[str, List[str]] = {
            'relu': ['relu', 'rectified linear unit'],
            'sigmoid': ['sigmoid'],
            'tanh': ['tanh', 'hyperbolic tangent'],
            'softmax': ['softmax'],
            'linear': ['linear', 'none']
        }
        
        self.optimizer_keywords: Dict[str, List[str]] = {
            'adam': ['adam'],
            'sgd': ['sgd', 'stochastic gradient descent'],
            'rmsprop': ['rmsprop', 'rms prop'],
            'adagrad': ['adagrad', 'ada grad'],
            'adamax': ['adamax', 'adam ax']
        }
        
        self.loss_keywords: Dict[str, List[str]] = {
            'categorical_crossentropy': ['categorical crossentropy', 'categorical cross entropy', 'crossentropy'],
            'binary_crossentropy': ['binary crossentropy', 'binary cross entropy'],
            'mse': ['mse', 'mean squared error', 'mean square error'],
            'mae': ['mae', 'mean absolute error']
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Uses langdetect library if available, falls back to heuristic detection.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code (e.g., 'en', 'fr', 'es', 'de', 'it')
        """
        try:
            from langdetect import detect, LangDetectException
            try:
                detected = detect(text)
                return detected
            except LangDetectException:
                pass
        except ImportError:
            pass
        
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii > len(text) * 0.3:
            return 'auto'
        
        return 'en'
    
    def extract_intent(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """
        Extract user intent from natural language text.
        
        Args:
            text: Natural language input
            
        Returns:
            Tuple of (intent_type, extracted_parameters)
        """
        text_lower = text.lower().strip()
        
        # Create model intent
        if any(phrase in text_lower for phrase in ['create', 'make', 'build', 'new model', 'new network']):
            return self._extract_create_model_intent(text)
        
        # Add layer intent
        if any(phrase in text_lower for phrase in ['add', 'insert', 'include', 'put']):
            return self._extract_add_layer_intent(text)
        
        # Modify layer intent
        if any(phrase in text_lower for phrase in ['change', 'modify', 'update', 'set']):
            return self._extract_modify_intent(text)
        
        # Compile intent
        if any(phrase in text_lower for phrase in ['compile', 'generate', 'build code']):
            return (IntentType.COMPILE, {})
        
        # Visualize intent
        if any(phrase in text_lower for phrase in ['visualize', 'show', 'display', 'plot']):
            return (IntentType.VISUALIZE, {})
        
        return (IntentType.UNKNOWN, {'text': text})
    
    def _extract_create_model_intent(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Extract parameters for creating a new model."""
        params: Dict[str, Any] = {}
        text_lower = text.lower()
        
        # Extract model name
        name_match = re.search(r'(?:named|name|called)\s+(\w+)', text_lower)
        if name_match:
            params['name'] = name_match.group(1)
        else:
            params['name'] = 'MyModel'
        
        # Extract input shape
        shape_match = re.search(r'(\d+)\s*[x×]\s*(\d+)', text_lower)
        if shape_match:
            params['input_shape'] = (int(shape_match.group(1)), int(shape_match.group(2)), 1)
        elif 'mnist' in text_lower:
            params['input_shape'] = (28, 28, 1)
        elif 'cifar' in text_lower:
            params['input_shape'] = (32, 32, 3)
        else:
            params['input_shape'] = (28, 28, 1)  # Default
        
        # Extract number of classes
        classes_match = re.search(r'(\d+)\s+classes?', text_lower)
        if classes_match:
            params['num_classes'] = int(classes_match.group(1))
        elif 'mnist' in text_lower:
            params['num_classes'] = 10
        else:
            params['num_classes'] = 10  # Default
        
        return (IntentType.CREATE_MODEL, params)
    
    def _extract_add_layer_intent(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Extract parameters for adding a layer."""
        params: Dict[str, Any] = {}
        text_lower = text.lower()
        
        # Detect layer type
        layer_type: Optional[str] = None
        for dsl_type, keywords in self.layer_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                layer_type = dsl_type
                break
        
        if not layer_type:
            return (IntentType.UNKNOWN, {'error': 'Layer type not recognized'})
        
        params['layer_type'] = layer_type
        
        # Extract layer-specific parameters
        if layer_type == 'conv2d':
            # Extract filters
            filters_match = re.search(r'(\d+)\s*(?:filters?|channels?)', text_lower)
            params['filters'] = int(filters_match.group(1)) if filters_match else 32
            
            # Extract kernel size
            kernel_match = re.search(r'kernel\s*(?:size|of)?\s*(\d+)', text_lower)
            if not kernel_match:
                kernel_match = re.search(r'(\d+)\s*[x×]\s*(\d+)', text_lower)
            params['kernel_size'] = int(kernel_match.group(1)) if kernel_match else 3
            
            # Extract activation
            params['activation'] = self._extract_activation(text_lower)
        
        elif layer_type == 'dense':
            # Extract units
            units_match = re.search(r'(\d+)\s*(?:units?|neurons?|nodes?)', text_lower)
            params['units'] = int(units_match.group(1)) if units_match else 128
            
            # Extract activation
            params['activation'] = self._extract_activation(text_lower)
        
        elif layer_type == 'dropout':
            # Extract rate
            rate_match = re.search(r'(\d*\.?\d+)\s*(?:rate|dropout)', text_lower)
            params['rate'] = float(rate_match.group(1)) if rate_match else 0.5
        
        elif layer_type == 'maxpooling2d':
            # Extract pool size
            pool_match = re.search(r'pool\s*(?:size|of)?\s*(\d+)', text_lower)
            params['pool_size'] = int(pool_match.group(1)) if pool_match else 2
        
        return (IntentType.ADD_LAYER, params)
    
    def _extract_modify_intent(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """Extract parameters for modifying model configuration."""
        params: Dict[str, Any] = {}
        text_lower = text.lower()
        
        # Check for optimizer
        for opt_name, keywords in self.optimizer_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                params['optimizer'] = opt_name
                # Extract learning rate
                lr_match = re.search(r'learning\s*rate\s*(?:of|:)?\s*(\d*\.?\d+)', text_lower)
                if lr_match:
                    params['learning_rate'] = float(lr_match.group(1))
                return (IntentType.SET_OPTIMIZER, params)
        
        # Check for loss
        for loss_name, keywords in self.loss_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                params['loss'] = loss_name
                return (IntentType.SET_LOSS, params)
        
        return (IntentType.UNKNOWN, {'error': 'Modification type not recognized'})
    
    def _extract_activation(self, text: str) -> str:
        """Extract activation function from text."""
        text_lower = text.lower()
        for act_name, keywords in self.activation_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return act_name
        return 'relu'  # Default


class DSLGenerator:
    """
    Generates Neural DSL code from extracted intents.
    """
    
    def __init__(self) -> None:
        """Initialize the DSL generator."""
        self.current_model: Dict[str, Any] = {
            'name': 'MyModel',
            'input_shape': (28, 28, 1),
            'layers': [],
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'num_classes': 10
        }
    
    def generate_from_intent(self, intent_type: IntentType, params: Dict[str, Any]) -> str:
        """
        Generate DSL code from intent and parameters.
        
        Args:
            intent_type: Type of intent
            params: Extracted parameters
            
        Returns:
            Generated DSL code snippet or full model
        """
        if intent_type == IntentType.CREATE_MODEL:
            return self._generate_model(params)
        elif intent_type == IntentType.ADD_LAYER:
            return self._generate_layer(params)
        elif intent_type == IntentType.SET_OPTIMIZER:
            return self._generate_optimizer(params)
        elif intent_type == IntentType.SET_LOSS:
            return self._generate_loss(params)
        else:
            return ""
    
    def _generate_model(self, params: Dict[str, Any]) -> str:
        """Generate a complete model DSL."""
        self.current_model.update(params)
        
        dsl = f"network {self.current_model['name']} {{\n"
        dsl += f"    input: {self.current_model['input_shape']}\n"
        dsl += "    layers:\n"
        
        # Add default layers if none specified
        if not self.current_model['layers']:
            dsl += "        Conv2D(32, (3, 3), \"relu\")\n"
            dsl += "        MaxPooling2D((2, 2))\n"
            dsl += "        Flatten()\n"
            dsl += f"        Dense(128, \"relu\")\n"
            dsl += f"        Output({self.current_model['num_classes']}, \"softmax\")\n"
        
        dsl += f"    loss: \"{self.current_model['loss']}\"\n"
        dsl += f"    optimizer: {self.current_model['optimizer'].title()}(learning_rate=0.001)\n"
        dsl += "}\n"
        
        return dsl
    
    def _generate_layer(self, params: Dict[str, Any]) -> str:
        """Generate a layer DSL snippet."""
        layer_type = params.get('layer_type')
        
        if layer_type == 'conv2d':
            filters = params.get('filters', 32)
            kernel_size = params.get('kernel_size', 3)
            activation = params.get('activation', 'relu')
            return f"        Conv2D({filters}, ({kernel_size}, {kernel_size}), \"{activation}\")\n"
        
        elif layer_type == 'dense':
            units = params.get('units', 128)
            activation = params.get('activation', 'relu')
            return f"        Dense({units}, \"{activation}\")\n"
        
        elif layer_type == 'dropout':
            rate = params.get('rate', 0.5)
            return f"        Dropout({rate})\n"
        
        elif layer_type == 'maxpooling2d':
            pool_size = params.get('pool_size', 2)
            return f"        MaxPooling2D(({pool_size}, {pool_size}))\n"
        
        elif layer_type == 'flatten':
            return "        Flatten()\n"
        
        elif layer_type == 'output':
            num_classes = params.get('num_classes', 10)
            return f"        Output({num_classes}, \"softmax\")\n"
        
        return ""
    
    def _generate_optimizer(self, params: Dict[str, Any]) -> str:
        """Generate optimizer DSL snippet."""
        opt_name = params.get('optimizer', 'adam')
        lr = params.get('learning_rate', 0.001)
        return f"    optimizer: {opt_name.title()}(learning_rate={lr})\n"
    
    def _generate_loss(self, params: Dict[str, Any]) -> str:
        """Generate loss DSL snippet."""
        loss = params.get('loss', 'categorical_crossentropy')
        return f"    loss: \"{loss}\"\n"
    
    def get_full_model(self) -> str:
        """Get the complete current model as DSL."""
        return self._generate_model(self.current_model)
