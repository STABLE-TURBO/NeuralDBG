"""
Transfer Learning Recommendations

Provides intelligent suggestions for transfer learning based on task type,
dataset size, and available pre-trained models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class TaskType(Enum):
    """Types of machine learning tasks."""
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    TEXT_CLASSIFICATION = "text_classification"
    SEQUENCE_LABELING = "sequence_labeling"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"


class DatasetSize(Enum):
    """Dataset size categories."""
    TINY = "tiny"  # < 1000 samples
    SMALL = "small"  # 1000-10000 samples
    MEDIUM = "medium"  # 10000-100000 samples
    LARGE = "large"  # > 100000 samples


class TransferLearningAdvisor:
    """
    Provides transfer learning recommendations based on task and dataset characteristics.
    """
    
    def __init__(self) -> None:
        """Initialize transfer learning advisor."""
        self.pretrained_models = self._initialize_model_catalog()
    
    def _initialize_model_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Initialize catalog of available pre-trained models."""
        return {
            # Image models
            'ResNet50': {
                'task': TaskType.IMAGE_CLASSIFICATION,
                'input_shape': (224, 224, 3),
                'params': 25.6e6,
                'dataset': 'ImageNet',
                'top1_accuracy': 0.759,
                'use_cases': ['general image classification', 'feature extraction', 'fine-tuning']
            },
            'EfficientNetB0': {
                'task': TaskType.IMAGE_CLASSIFICATION,
                'input_shape': (224, 224, 3),
                'params': 5.3e6,
                'dataset': 'ImageNet',
                'top1_accuracy': 0.772,
                'use_cases': ['efficient classification', 'mobile deployment', 'small datasets']
            },
            'VGG16': {
                'task': TaskType.IMAGE_CLASSIFICATION,
                'input_shape': (224, 224, 3),
                'params': 138e6,
                'dataset': 'ImageNet',
                'top1_accuracy': 0.713,
                'use_cases': ['feature extraction', 'legacy applications', 'transfer learning']
            },
            'MobileNetV2': {
                'task': TaskType.IMAGE_CLASSIFICATION,
                'input_shape': (224, 224, 3),
                'params': 3.5e6,
                'dataset': 'ImageNet',
                'top1_accuracy': 0.713,
                'use_cases': ['mobile apps', 'edge devices', 'fast inference']
            },
            'InceptionV3': {
                'task': TaskType.IMAGE_CLASSIFICATION,
                'input_shape': (299, 299, 3),
                'params': 23.9e6,
                'dataset': 'ImageNet',
                'top1_accuracy': 0.779,
                'use_cases': ['multi-scale features', 'complex patterns', 'fine-grained classification']
            },
            # Text models
            'BERT-base': {
                'task': TaskType.TEXT_CLASSIFICATION,
                'input_shape': (512,),
                'params': 110e6,
                'dataset': 'Wikipedia + BookCorpus',
                'use_cases': ['text classification', 'NER', 'question answering']
            },
            'DistilBERT': {
                'task': TaskType.TEXT_CLASSIFICATION,
                'input_shape': (512,),
                'params': 66e6,
                'dataset': 'Wikipedia + BookCorpus',
                'use_cases': ['fast text classification', 'resource-constrained', 'deployment']
            }
        }
    
    def recommend_model(
        self,
        task_type: TaskType,
        dataset_size: DatasetSize,
        input_shape: Optional[Tuple[int, ...]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend pre-trained models for transfer learning.
        
        Args:
            task_type: Type of ML task
            dataset_size: Size category of dataset
            input_shape: Expected input shape
            constraints: Optional constraints (max_params, max_latency, etc.)
            
        Returns:
            List of recommended models with details and usage instructions
        """
        recommendations = []
        
        # Filter models by task type
        candidate_models = {
            name: info for name, info in self.pretrained_models.items()
            if info['task'] == task_type
        }
        
        if not candidate_models:
            return [{
                'message': f'No pre-trained models available for {task_type.value}',
                'suggestion': 'Consider training from scratch or using a similar task model'
            }]
        
        # Apply constraints
        if constraints:
            max_params = constraints.get('max_params')
            if max_params:
                candidate_models = {
                    name: info for name, info in candidate_models.items()
                    if info['params'] <= max_params
                }
        
        # Rank models
        for name, info in candidate_models.items():
            strategy = self._determine_strategy(dataset_size, info)
            
            recommendation = {
                'model_name': name,
                'strategy': strategy,
                'expected_accuracy': self._estimate_accuracy(dataset_size, info),
                'training_time': self._estimate_training_time(dataset_size, info),
                'params': info['params'],
                'description': self._generate_description(name, info, strategy),
                'code_example': self._generate_code_example(name, strategy),
                'fine_tuning_tips': self._generate_fine_tuning_tips(strategy, dataset_size)
            }
            
            recommendations.append(recommendation)
        
        # Sort by expected accuracy (descending)
        recommendations.sort(key=lambda x: x['expected_accuracy'], reverse=True)
        
        return recommendations
    
    def _determine_strategy(
        self,
        dataset_size: DatasetSize,
        model_info: Dict[str, Any]
    ) -> str:
        """Determine the best transfer learning strategy."""
        if dataset_size == DatasetSize.TINY:
            return 'feature_extraction'
        elif dataset_size == DatasetSize.SMALL:
            return 'fine_tune_top_layers'
        elif dataset_size == DatasetSize.MEDIUM:
            return 'fine_tune_all_layers'
        else:
            return 'full_fine_tuning'
    
    def _estimate_accuracy(
        self,
        dataset_size: DatasetSize,
        model_info: Dict[str, Any]
    ) -> float:
        """Estimate expected accuracy based on transfer learning."""
        base_acc = model_info.get('top1_accuracy', 0.75)
        
        # Adjust based on dataset size
        if dataset_size == DatasetSize.TINY:
            return base_acc * 0.7  # Feature extraction only
        elif dataset_size == DatasetSize.SMALL:
            return base_acc * 0.85
        elif dataset_size == DatasetSize.MEDIUM:
            return base_acc * 0.95
        else:
            return base_acc * 1.0
    
    def _estimate_training_time(
        self,
        dataset_size: DatasetSize,
        model_info: Dict[str, Any]
    ) -> str:
        """Estimate training time."""
        size_factor = {
            DatasetSize.TINY: 'minutes',
            DatasetSize.SMALL: '1-2 hours',
            DatasetSize.MEDIUM: '4-8 hours',
            DatasetSize.LARGE: '1-2 days'
        }
        return size_factor.get(dataset_size, 'variable')
    
    def _generate_description(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        strategy: str
    ) -> str:
        """Generate human-readable description."""
        desc = f"{model_name} pre-trained on {model_info.get('dataset', 'large dataset')}. "
        desc += f"Best used for {', '.join(model_info.get('use_cases', [])[:2])}. "
        desc += f"Recommended strategy: {strategy.replace('_', ' ')}."
        return desc
    
    def _generate_code_example(self, model_name: str, strategy: str) -> str:
        """Generate code example for transfer learning."""
        examples = {
            'feature_extraction': f"""
# Load pre-trained {model_name} as feature extractor
network TransferModel {{
    input: (224, 224, 3)
    
    # Load pre-trained base (frozen)
    base_model: {model_name}(weights="imagenet", trainable=false)
    
    layers:
        # Add custom classification head
        GlobalAveragePooling2D()
        Dense(256, "relu")
        Dropout(0.5)
        Output(num_classes, "softmax")
    
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.001)
}}
""",
            'fine_tune_top_layers': f"""
# Fine-tune top layers of {model_name}
network TransferModel {{
    input: (224, 224, 3)
    
    # Load pre-trained base, unfreeze top 10 layers
    base_model: {model_name}(weights="imagenet", freeze_until=-10)
    
    layers:
        GlobalAveragePooling2D()
        Dense(512, "relu")
        BatchNormalization()
        Dropout(0.5)
        Output(num_classes, "softmax")
    
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.0001)  # Lower LR for fine-tuning
}}
""",
            'fine_tune_all_layers': f"""
# Fine-tune all layers of {model_name}
network TransferModel {{
    input: (224, 224, 3)
    
    # Load pre-trained base, all layers trainable
    base_model: {model_name}(weights="imagenet", trainable=true)
    
    layers:
        GlobalAveragePooling2D()
        Dense(512, "relu")
        BatchNormalization()
        Dropout(0.5)
        Dense(256, "relu")
        Dropout(0.3)
        Output(num_classes, "softmax")
    
    loss: "categorical_crossentropy"
    optimizer: Adam(learning_rate=0.00001)  # Very low LR
    
    # Two-stage training
    training: {{
        # Stage 1: Train head only (5 epochs)
        freeze_base: true
        epochs: 5
        
        # Stage 2: Fine-tune all (20 epochs)
        freeze_base: false
        epochs: 20
    }}
}}
""",
            'full_fine_tuning': f"""
# Full fine-tuning with {model_name}
network TransferModel {{
    input: (224, 224, 3)
    
    base_model: {model_name}(weights="imagenet", trainable=true)
    
    layers:
        GlobalAveragePooling2D()
        Dense(1024, "relu")
        BatchNormalization()
        Dropout(0.5)
        Dense(512, "relu")
        BatchNormalization()
        Dropout(0.3)
        Output(num_classes, "softmax")
    
    loss: "categorical_crossentropy"
    optimizer: AdamW(learning_rate=0.00001, weight_decay=0.01)
    
    # Learning rate schedule
    lr_schedule: CosineDecay(initial_lr=0.0001, decay_steps=10000)
}}
"""
        }
        
        return examples.get(strategy, examples['fine_tune_top_layers'])
    
    def _generate_fine_tuning_tips(
        self,
        strategy: str,
        dataset_size: DatasetSize
    ) -> List[str]:
        """Generate fine-tuning tips."""
        tips = []
        
        if strategy == 'feature_extraction':
            tips.extend([
                "Keep base model frozen to preserve learned features",
                "Use higher learning rate (0.001) for new layers",
                "Train for 10-20 epochs",
                "Monitor overfitting carefully with small dataset"
            ])
        elif strategy == 'fine_tune_top_layers':
            tips.extend([
                "Freeze early layers, unfreeze last 10-30 layers",
                "Use lower learning rate (0.0001) to avoid catastrophic forgetting",
                "Train for 20-30 epochs",
                "Use data augmentation to increase effective dataset size"
            ])
        elif strategy in ['fine_tune_all_layers', 'full_fine_tuning']:
            tips.extend([
                "Use very low learning rate (0.00001) to preserve pre-trained features",
                "Consider two-stage training: freeze base first, then unfreeze",
                "Train for 30-50 epochs with learning rate schedule",
                "Use strong data augmentation",
                "Monitor for catastrophic forgetting"
            ])
        
        # Dataset-specific tips
        if dataset_size == DatasetSize.TINY:
            tips.append("With tiny dataset, strongly prefer feature extraction")
        elif dataset_size == DatasetSize.SMALL:
            tips.append("Use aggressive data augmentation to prevent overfitting")
        
        return tips
    
    def analyze_task_similarity(
        self,
        source_task: str,
        target_task: str
    ) -> Dict[str, Any]:
        """
        Analyze similarity between source (pre-trained) and target tasks.
        
        Args:
            source_task: Task the pre-trained model was trained on
            target_task: Your target task description
            
        Returns:
            Similarity analysis and recommendations
        """
        # Simple keyword-based similarity (can be enhanced with embeddings)
        source_lower = source_task.lower()
        target_lower = target_task.lower()
        
        # Check for domain similarity
        image_keywords = ['image', 'photo', 'picture', 'visual', 'object', 'scene']
        text_keywords = ['text', 'language', 'sentence', 'document', 'nlp']
        
        source_is_image = any(kw in source_lower for kw in image_keywords)
        target_is_image = any(kw in target_lower for kw in image_keywords)
        
        source_is_text = any(kw in source_lower for kw in text_keywords)
        target_is_text = any(kw in target_lower for kw in text_keywords)
        
        if (source_is_image and target_is_image) or (source_is_text and target_is_text):
            similarity = 'high'
            recommendation = 'Excellent candidate for transfer learning'
        elif (source_is_image and not target_is_image) or (source_is_text and not target_is_text):
            similarity = 'low'
            recommendation = 'Transfer learning may not help; consider training from scratch'
        else:
            similarity = 'medium'
            recommendation = 'Transfer learning may help; experiment with different strategies'
        
        return {
            'similarity': similarity,
            'recommendation': recommendation,
            'explanation': self._generate_similarity_explanation(similarity)
        }
    
    def _generate_similarity_explanation(self, similarity: str) -> str:
        """Generate explanation for similarity level."""
        explanations = {
            'high': (
                "The source and target tasks are highly similar. "
                "Transfer learning should work very well. "
                "Pre-trained features will be directly applicable."
            ),
            'medium': (
                "The source and target tasks have moderate similarity. "
                "Transfer learning may help, but benefits are less certain. "
                "Consider experimenting with different fine-tuning strategies."
            ),
            'low': (
                "The source and target tasks are quite different. "
                "Transfer learning may not provide significant benefits. "
                "Consider training from scratch or finding a more similar source task."
            )
        }
        return explanations.get(similarity, '')
    
    def get_conversational_response(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate conversational response about transfer learning.
        
        Args:
            user_query: User's question
            context: Optional context (task_type, dataset_size, etc.)
            
        Returns:
            Natural language response
        """
        query_lower = user_query.lower()
        
        if 'which model' in query_lower or 'what model' in query_lower:
            return self._explain_model_selection()
        elif 'how to' in query_lower and 'fine' in query_lower:
            return self._explain_fine_tuning_process()
        elif 'feature extraction' in query_lower:
            return self._explain_feature_extraction()
        elif 'when to use' in query_lower or 'should i use' in query_lower:
            return self._explain_when_to_use_transfer_learning()
        elif context:
            # Provide specific recommendations
            task_type = context.get('task_type', TaskType.IMAGE_CLASSIFICATION)
            dataset_size = context.get('dataset_size', DatasetSize.SMALL)
            recommendations = self.recommend_model(task_type, dataset_size)
            
            if recommendations:
                response = f"For {task_type.value} with {dataset_size.value} dataset, I recommend:\n\n"
                for i, rec in enumerate(recommendations[:2], 1):
                    response += f"{i}. **{rec['model_name']}** - {rec['strategy']}\n"
                    response += f"   Expected accuracy: {rec['expected_accuracy']:.1%}\n"
                    response += f"   Training time: {rec['training_time']}\n\n"
                return response
            else:
                return "I need more information about your task to make recommendations."
        else:
            return (
                "I can help with transfer learning! Ask me:\n"
                "- Which model should I use?\n"
                "- How to fine-tune a model?\n"
                "- When to use transfer learning?\n"
                "- What's feature extraction?\n"
                "Or provide your task details for specific recommendations!"
            )
    
    def _explain_model_selection(self) -> str:
        """Explain model selection criteria."""
        return """
**How to choose a pre-trained model:**

Consider these factors:
1. **Task Similarity**: Choose models trained on similar tasks
   - ImageNet models for general images
   - BERT for text tasks
   
2. **Dataset Size**:
   - Tiny (<1K): Use lightweight models (MobileNet, DistilBERT)
   - Small-Medium: Standard models (ResNet50, BERT-base)
   - Large: Can use any model
   
3. **Computational Constraints**:
   - Mobile/Edge: MobileNet, EfficientNet
   - Server: ResNet, InceptionV3
   
4. **Accuracy Requirements**:
   - High accuracy: EfficientNet, InceptionV3
   - Balanced: ResNet50
   - Fast inference: MobileNet

Popular choices:
- **General images**: ResNet50 or EfficientNetB0
- **Text**: BERT-base or DistilBERT
- **Mobile**: MobileNetV2 or EfficientNetB0
"""
    
    def _explain_fine_tuning_process(self) -> str:
        """Explain fine-tuning process."""
        return """
**Fine-tuning process:**

Step-by-step:
1. **Load pre-trained model** with frozen weights
2. **Replace head** with task-specific layers
3. **Train head only** (5-10 epochs)
4. **Unfreeze top layers** (gradually)
5. **Fine-tune with low LR** (0.0001 or less)
6. **Monitor validation** to prevent overfitting

Key principles:
- Use **lower learning rate** than training from scratch
- Start with **frozen base**, gradually unfreeze
- Apply **data augmentation** to prevent overfitting
- Use **learning rate schedule** (cosine decay)
- Train in **stages** (head first, then full model)

Example strategy:
```
# Stage 1: Train new head (base frozen)
epochs: 5, lr: 0.001

# Stage 2: Fine-tune top layers
unfreeze top 10 layers, lr: 0.0001, epochs: 15

# Stage 3: Fine-tune all (optional)
unfreeze all, lr: 0.00001, epochs: 10
```
"""
    
    def _explain_feature_extraction(self) -> str:
        """Explain feature extraction approach."""
        return """
**Feature Extraction (frozen base model):**

What it is:
- Use pre-trained model as **fixed feature extractor**
- Only train new classification head
- Base model weights stay frozen

When to use:
- **Small datasets** (<1000 samples)
- **Limited compute** resources
- **Fast prototyping**
- Tasks **very similar** to pre-training

Advantages:
✓ Very fast training
✓ Less prone to overfitting
✓ Lower computational cost
✓ Works well with small datasets

Disadvantages:
✗ May not adapt to new domain
✗ Lower accuracy potential
✗ Less flexibility

Example:
```
base_model: ResNet50(weights="imagenet", trainable=false)

layers:
    GlobalAveragePooling2D()
    Dense(256, "relu")
    Dropout(0.5)
    Output(num_classes, "softmax")
```

Tip: Start with feature extraction, then try fine-tuning if needed!
"""
    
    def _explain_when_to_use_transfer_learning(self) -> str:
        """Explain when to use transfer learning."""
        return """
**When to use transfer learning:**

✅ **Use transfer learning when:**
- You have limited training data
- Task is similar to common tasks (ImageNet classification, text processing)
- You need faster training
- Limited computational resources
- Want better generalization

❌ **Don't use transfer learning when:**
- Very large dataset (>1M samples) available
- Task is completely novel/unique
- Pre-trained models not available for your domain
- Domain shift is too large (e.g., medical images from natural images)

**Dataset size guidelines:**
- **< 1,000 samples**: Definitely use transfer learning (feature extraction)
- **1,000 - 10,000**: Strongly recommended (fine-tune top layers)
- **10,000 - 100,000**: Beneficial (fine-tune all layers)
- **> 100,000**: Optional (can train from scratch, but transfer learning still helps)

**Rule of thumb:**
If pre-trained models exist for a related task, transfer learning almost always helps!
Even with large datasets, it provides:
- Faster convergence
- Better initialization
- Improved generalization
- Reduced training time
"""
