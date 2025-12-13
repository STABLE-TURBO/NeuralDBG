"""
AI Integration Module for Neural DSL

This module provides AI-powered features for Neural:
- Natural language to DSL conversion
- Multi-language support
- AI code assistant
- Intelligent suggestions
- Model optimization recommendations
- Transfer learning advice
- Data augmentation strategies
- Debugging assistance
- Context-aware conversations
"""

from .ai_assistant import NeuralAIAssistant
from .enhanced_assistant import EnhancedAIAssistant
from .model_optimizer import ModelOptimizer, OptimizationCategory
from .transfer_learning import TransferLearningAdvisor, TaskType, DatasetSize
from .data_augmentation import DataAugmentationAdvisor, DataType, AugmentationLevel
from .debugging_assistant import DebuggingAssistant, IssueType
from .context_manager import ContextManager, SessionContext, ConversationMessage
from .llm_integration import LLMIntegration
from .natural_language_processor import NaturalLanguageProcessor, IntentType, DSLGenerator
from .multi_language import MultiLanguageSupport

__version__ = "0.2.0"

__all__ = [
    # Main assistants
    'NeuralAIAssistant',
    'EnhancedAIAssistant',
    
    # Specialized assistants
    'ModelOptimizer',
    'TransferLearningAdvisor',
    'DataAugmentationAdvisor',
    'DebuggingAssistant',
    
    # Context management
    'ContextManager',
    'SessionContext',
    'ConversationMessage',
    
    # Core components
    'LLMIntegration',
    'NaturalLanguageProcessor',
    'DSLGenerator',
    'MultiLanguageSupport',
    
    # Enums
    'OptimizationCategory',
    'TaskType',
    'DatasetSize',
    'DataType',
    'AugmentationLevel',
    'IssueType',
    'IntentType',
]

