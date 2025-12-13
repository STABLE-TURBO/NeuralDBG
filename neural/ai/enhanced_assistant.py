"""
Enhanced AI Assistant

Main interface combining all AI features: optimization suggestions, transfer learning,
data augmentation, debugging assistance, and context-aware conversations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from .model_optimizer import ModelOptimizer
from .transfer_learning import TransferLearningAdvisor, TaskType, DatasetSize
from .data_augmentation import DataAugmentationAdvisor, DataType, AugmentationLevel
from .debugging_assistant import DebuggingAssistant, IssueType
from .context_manager import ContextManager, SessionContext
from .llm_integration import LLMIntegration

logger = logging.getLogger(__name__)


class EnhancedAIAssistant:
    """
    Enhanced AI Assistant with comprehensive model optimization and assistance features.
    
    Features:
    - Conversational model optimization suggestions
    - Automatic architecture refinement
    - Transfer learning recommendations
    - Data augmentation suggestions
    - Debugging assistance
    - Context retention across sessions
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_provider: Optional[str] = None,
        persistence_dir: Optional[str] = None
    ) -> None:
        """
        Initialize enhanced AI assistant.
        
        Args:
            use_llm: Whether to use LLM for advanced processing
            llm_provider: Specific LLM provider ('openai', 'anthropic', 'ollama')
            persistence_dir: Directory for session persistence
        """
        # Initialize specialized assistants
        self.optimizer = ModelOptimizer()
        self.transfer_learning = TransferLearningAdvisor()
        self.augmentation = DataAugmentationAdvisor()
        self.debugger = DebuggingAssistant()
        self.context_manager = ContextManager(persistence_dir)
        
        # Initialize LLM if requested
        self.use_llm = use_llm
        self.llm: Optional[LLMIntegration] = None
        if use_llm:
            try:
                self.llm = LLMIntegration(provider=llm_provider)
            except Exception as e:
                logger.warning("LLM not available: %s", e)
                self.use_llm = False
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Session ID
        """
        session = self.context_manager.start_session(session_id)
        return session.session_id
    
    def chat(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input with context-aware assistance.
        
        Args:
            user_input: User's natural language input
            context: Optional context (metrics, model config, etc.)
            
        Returns:
            Response dictionary with answer and metadata
        """
        # Ensure session exists
        if not self.context_manager.current_session:
            self.start_session()
        
        # Add user message to history
        self.context_manager.current_session.add_message(
            'user', user_input, metadata=context
        )
        
        # Determine intent and route to appropriate assistant
        response = self._route_query(user_input, context)
        
        # Add assistant response to history
        self.context_manager.current_session.add_message(
            'assistant', response['response']
        )
        
        # Update context if model state changed
        if context and 'model_config' in context:
            self.context_manager.current_session.update_model_state(
                context['model_config']
            )
        
        return response
    
    def _route_query(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route query to appropriate assistant based on intent."""
        query_lower = user_input.lower()
        
        # Debugging queries
        if any(word in query_lower for word in ['debug', 'error', 'nan', 'stuck', 'memory']):
            response = self.debugger.get_conversational_response(user_input, context)
            return {
                'response': response,
                'category': 'debugging',
                'success': True
            }
        
        # Optimization queries
        if any(word in query_lower for word in ['optimize', 'improve', 'overfit', 'underfit']):
            response = self.optimizer.get_conversational_response(user_input, context)
            return {
                'response': response,
                'category': 'optimization',
                'success': True
            }
        
        # Transfer learning queries
        if any(word in query_lower for word in ['transfer', 'pretrained', 'fine-tune', 'finetune']):
            response = self.transfer_learning.get_conversational_response(user_input, context)
            return {
                'response': response,
                'category': 'transfer_learning',
                'success': True
            }
        
        # Data augmentation queries
        if any(word in query_lower for word in ['augment', 'augmentation', 'mixup', 'cutmix']):
            response = self.augmentation.get_conversational_response(user_input, context)
            return {
                'response': response,
                'category': 'data_augmentation',
                'success': True
            }
        
        # General help or model generation
        return self._general_response(user_input, context)
    
    def _general_response(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate general response using all available information."""
        response = """I'm your AI assistant for Neural DSL! I can help with:

**🔧 Optimization & Training**
- Analyze training metrics and suggest improvements
- Detect overfitting/underfitting
- Recommend learning rate schedules
- Architecture refinements

**🎯 Transfer Learning**
- Recommend pre-trained models
- Fine-tuning strategies
- Task similarity analysis

**📊 Data Augmentation**
- Suggest augmentation techniques
- Generate augmentation pipelines
- Dataset-specific strategies

**🐛 Debugging**
- Diagnose training issues
- NaN/Inf loss debugging
- Gradient problems
- Memory optimization

**💬 Ask me:**
- "Why is my model overfitting?"
- "Which pre-trained model should I use?"
- "What augmentations for images?"
- "My loss is NaN, help!"
- "How to improve convergence?"

Share your metrics or model config for personalized advice!
"""
        
        return {
            'response': response,
            'category': 'general',
            'success': True
        }
    
    def analyze_training_metrics(
        self,
        metrics: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze training metrics and provide comprehensive suggestions.
        
        Args:
            metrics: Training metrics (train_loss, val_loss, etc.)
            model_config: Optional model configuration
            
        Returns:
            Analysis with suggestions from multiple assistants
        """
        analysis = {
            'optimization_suggestions': [],
            'architecture_refinement': None,
            'overall_assessment': ''
        }
        
        # Get optimization suggestions
        opt_suggestions = self.optimizer.analyze_metrics(metrics, model_config)
        analysis['optimization_suggestions'] = opt_suggestions
        
        # Get architecture refinement if model config provided
        if model_config:
            refinement = self.optimizer.suggest_architecture_refinement(
                model_config, metrics
            )
            analysis['architecture_refinement'] = refinement
        
        # Generate overall assessment
        if opt_suggestions:
            primary_issue = opt_suggestions[0]['category']
            analysis['overall_assessment'] = (
                f"Primary issue detected: {primary_issue}. "
                f"I've identified {len(opt_suggestions)} improvement opportunities."
            )
        else:
            analysis['overall_assessment'] = (
                "Your model appears to be training well! "
                "Continue monitoring metrics for any emerging issues."
            )
        
        return analysis
    
    def recommend_transfer_learning(
        self,
        task_type: str,
        dataset_size: int,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend transfer learning approach.
        
        Args:
            task_type: Type of task (e.g., 'image_classification')
            dataset_size: Number of samples
            constraints: Optional constraints (max_params, etc.)
            
        Returns:
            List of transfer learning recommendations
        """
        # Convert string to TaskType enum
        task_type_map = {
            'image_classification': TaskType.IMAGE_CLASSIFICATION,
            'object_detection': TaskType.OBJECT_DETECTION,
            'text_classification': TaskType.TEXT_CLASSIFICATION
        }
        task = task_type_map.get(task_type.lower(), TaskType.IMAGE_CLASSIFICATION)
        
        # Determine dataset size category
        if dataset_size < 1000:
            size = DatasetSize.TINY
        elif dataset_size < 10000:
            size = DatasetSize.SMALL
        elif dataset_size < 100000:
            size = DatasetSize.MEDIUM
        else:
            size = DatasetSize.LARGE
        
        recommendations = self.transfer_learning.recommend_model(
            task, size, constraints=constraints
        )
        
        return recommendations
    
    def generate_augmentation_pipeline(
        self,
        data_type: str,
        dataset_size: int,
        level: str = 'moderate'
    ) -> Dict[str, Any]:
        """
        Generate data augmentation pipeline.
        
        Args:
            data_type: Type of data ('image', 'text', 'time_series')
            dataset_size: Number of samples
            level: Augmentation level ('light', 'moderate', 'aggressive')
            
        Returns:
            Augmentation pipeline configuration
        """
        # Convert string to enums
        data_type_map = {
            'image': DataType.IMAGE,
            'text': DataType.TEXT,
            'time_series': DataType.TIME_SERIES
        }
        dtype = data_type_map.get(data_type.lower(), DataType.IMAGE)
        
        level_map = {
            'light': AugmentationLevel.LIGHT,
            'moderate': AugmentationLevel.MODERATE,
            'aggressive': AugmentationLevel.AGGRESSIVE
        }
        aug_level = level_map.get(level.lower(), AugmentationLevel.MODERATE)
        
        pipeline = self.augmentation.generate_augmentation_pipeline(
            dtype, dataset_size, aug_level
        )
        
        return pipeline
    
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
            metrics: Training metrics
            error_message: Optional error message
            
        Returns:
            Diagnosis with recommended solutions
        """
        diagnosis = self.debugger.diagnose_issue(
            symptoms=symptoms,
            metrics=metrics,
            error_message=error_message
        )
        
        return diagnosis
    
    def get_debugging_code(self, issue_type: str) -> Dict[str, str]:
        """
        Get debugging code for specific issue type.
        
        Args:
            issue_type: Type of issue (e.g., 'loss_nan')
            
        Returns:
            Dictionary with debugging code and explanation
        """
        issue_type_map = {
            'loss_nan': IssueType.LOSS_NAN,
            'gradient_vanishing': IssueType.GRADIENT_VANISHING,
            'gradient_exploding': IssueType.GRADIENT_EXPLODING,
            'memory_error': IssueType.MEMORY_ERROR,
            'shape_mismatch': IssueType.SHAPE_MISMATCH
        }
        
        issue = issue_type_map.get(issue_type.lower())
        if not issue:
            return {'error': 'Unknown issue type'}
        
        return self.debugger.suggest_debugging_code(issue)
    
    def get_session_summary(self) -> str:
        """Get summary of current session."""
        return self.context_manager.get_context_summary()
    
    def save_session(self) -> bool:
        """Save current session to disk."""
        return self.context_manager.save_session()
    
    def resume_session(self, session_id: str) -> bool:
        """
        Resume a previous session.
        
        Args:
            session_id: Session ID to resume
            
        Returns:
            True if resumed successfully
        """
        session = self.context_manager.resume_session(session_id)
        return session is not None
    
    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        return self.context_manager.list_sessions()
    
    def get_comprehensive_advice(
        self,
        task_description: str,
        dataset_info: Dict[str, Any],
        current_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive advice for a task.
        
        Args:
            task_description: Description of the task
            dataset_info: Information about dataset (size, type, etc.)
            current_metrics: Optional current training metrics
            
        Returns:
            Comprehensive advice covering all aspects
        """
        advice = {
            'task': task_description,
            'transfer_learning': None,
            'data_augmentation': None,
            'architecture_tips': [],
            'training_tips': [],
            'metrics_analysis': None
        }
        
        # Transfer learning recommendations
        if dataset_info.get('type') and dataset_info.get('size'):
            task_type = dataset_info.get('task_type', 'image_classification')
            recommendations = self.recommend_transfer_learning(
                task_type,
                dataset_info['size']
            )
            advice['transfer_learning'] = recommendations
        
        # Data augmentation recommendations
        if dataset_info.get('type') and dataset_info.get('size'):
            pipeline = self.generate_augmentation_pipeline(
                dataset_info['type'],
                dataset_info['size']
            )
            advice['data_augmentation'] = pipeline
        
        # Architecture tips
        advice['architecture_tips'] = [
            "Start with a proven architecture (ResNet, EfficientNet)",
            "Use batch normalization after convolutional layers",
            "Add dropout before output layer for regularization",
            "Consider skip connections for deep networks"
        ]
        
        # Training tips
        advice['training_tips'] = [
            "Use learning rate schedule (cosine decay or step decay)",
            "Monitor validation metrics to detect overfitting early",
            "Save checkpoints regularly during training",
            "Use early stopping to prevent overfitting"
        ]
        
        # Metrics analysis if provided
        if current_metrics:
            advice['metrics_analysis'] = self.analyze_training_metrics(
                current_metrics,
                dataset_info.get('model_config')
            )
        
        return advice
