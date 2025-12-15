"""
AI Integration Module for Neural DSL

This module provides AI-powered features for Neural:
- Natural language to DSL conversion
- Basic AI assistant for model creation
"""

from .ai_assistant import NeuralAIAssistant
from .natural_language_processor import NaturalLanguageProcessor, IntentType, DSLGenerator

__version__ = "0.2.0"

__all__ = [
    'NeuralAIAssistant',
    'NaturalLanguageProcessor',
    'DSLGenerator',
    'IntentType',
]
