"""
AI Assistant for Neural DSL

Main interface for AI-powered features including natural language to DSL conversion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from .natural_language_processor import NaturalLanguageProcessor, IntentType, DSLGenerator

logger = logging.getLogger(__name__)


class NeuralAIAssistant:
    """
    AI Assistant for Neural DSL.
    
    Provides natural language to DSL conversion using rule-based processing.
    """
    
    def __init__(self) -> None:
        """Initialize AI Assistant."""
        self.nlp: NaturalLanguageProcessor = NaturalLanguageProcessor()
        self.dsl_generator: DSLGenerator = DSLGenerator()
    
    def chat(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process user input and generate response.
        
        Args:
            user_input: User's natural language input
            context: Optional context (current model state, conversation history, etc.)
            
        Returns:
            Dictionary with:
            - response: Text response to user
            - dsl_code: Generated DSL code (if any)
            - intent: Detected intent
            - success: Whether operation was successful
        """
        intent_type, params = self.nlp.extract_intent(user_input)
        
        if intent_type == IntentType.UNKNOWN:
            return {
                'response': "I didn't understand that. Try: 'Create a CNN for image classification' or 'Add a dense layer with 128 units'",
                'dsl_code': None,
                'intent': 'unknown',
                'success': False
            }
        
        dsl_snippet = self.dsl_generator.generate_from_intent(intent_type, params)
        
        if intent_type == IntentType.CREATE_MODEL:
            self.dsl_generator.current_model.update(params)
            full_dsl = self.dsl_generator.get_full_model()
            return {
                'response': f"Created model '{params.get('name', 'MyModel')}'. Here's the DSL:\n\n{full_dsl}",
                'dsl_code': full_dsl,
                'intent': 'create_model',
                'success': True
            }
        
        elif intent_type == IntentType.ADD_LAYER:
            layer_dsl = dsl_snippet
            self.dsl_generator.current_model['layers'].append(layer_dsl.strip())
            return {
                'response': f"Added layer: {layer_dsl.strip()}",
                'dsl_code': layer_dsl,
                'intent': 'add_layer',
                'success': True
            }
        
        else:
            return {
                'response': f"Processed: {intent_type.value}",
                'dsl_code': dsl_snippet,
                'intent': intent_type.value,
                'success': True
            }
    
    def get_current_model(self) -> str:
        """Get the current model as DSL."""
        return self.dsl_generator.get_full_model()
    
    def reset(self) -> None:
        """Reset the assistant state."""
        self.dsl_generator = DSLGenerator()
