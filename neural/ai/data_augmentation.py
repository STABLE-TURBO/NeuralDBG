"""
Data Augmentation Suggestions

Provides intelligent recommendations for data augmentation strategies
based on data type, task, and dataset characteristics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class DataType(Enum):
    """Types of data."""
    IMAGE = "image"
    TEXT = "text"
    TIME_SERIES = "time_series"
    TABULAR = "tabular"
    AUDIO = "audio"


class AugmentationLevel(Enum):
    """Augmentation intensity levels."""
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class DataAugmentationAdvisor:
    """
    Provides data augmentation recommendations based on data characteristics.
    """
    
    def __init__(self) -> None:
        """Initialize data augmentation advisor."""
        self.augmentation_catalog = self._initialize_augmentation_catalog()
    
    def _initialize_augmentation_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Initialize catalog of augmentation techniques."""
        return {
            # Image augmentations
            'RandomFlip': {
                'data_type': DataType.IMAGE,
                'description': 'Randomly flip images horizontally or vertically',
                'use_cases': ['general images', 'object detection', 'classification'],
                'parameters': {'mode': ['horizontal', 'vertical', 'both']},
                'effectiveness': 0.9,
                'cost': 'low'
            },
            'RandomRotation': {
                'data_type': DataType.IMAGE,
                'description': 'Randomly rotate images by specified degrees',
                'use_cases': ['orientation-invariant tasks', 'general classification'],
                'parameters': {'degrees': [15, 30, 45]},
                'effectiveness': 0.85,
                'cost': 'low'
            },
            'RandomCrop': {
                'data_type': DataType.IMAGE,
                'description': 'Randomly crop and resize images',
                'use_cases': ['scale invariance', 'localization tasks'],
                'parameters': {'scale': [(0.8, 1.0)], 'ratio': [(0.75, 1.33)]},
                'effectiveness': 0.9,
                'cost': 'low'
            },
            'ColorJitter': {
                'data_type': DataType.IMAGE,
                'description': 'Randomly adjust brightness, contrast, saturation, hue',
                'use_cases': ['lighting variations', 'color robustness'],
                'parameters': {
                    'brightness': [0.2, 0.3],
                    'contrast': [0.2, 0.3],
                    'saturation': [0.2, 0.3]
                },
                'effectiveness': 0.8,
                'cost': 'low'
            },
            'RandomErasing': {
                'data_type': DataType.IMAGE,
                'description': 'Randomly erase rectangular regions',
                'use_cases': ['occlusion robustness', 'dropout-like regularization'],
                'parameters': {'probability': [0.5], 'area_ratio': [(0.02, 0.33)]},
                'effectiveness': 0.75,
                'cost': 'low'
            },
            'GaussianNoise': {
                'data_type': DataType.IMAGE,
                'description': 'Add Gaussian noise to images',
                'use_cases': ['noise robustness', 'medical imaging'],
                'parameters': {'mean': [0.0], 'std': [0.01, 0.05]},
                'effectiveness': 0.7,
                'cost': 'low'
            },
            'MixUp': {
                'data_type': DataType.IMAGE,
                'description': 'Mix two images and their labels',
                'use_cases': ['advanced regularization', 'limited data'],
                'parameters': {'alpha': [0.2, 0.4]},
                'effectiveness': 0.85,
                'cost': 'medium'
            },
            'CutMix': {
                'data_type': DataType.IMAGE,
                'description': 'Cut and paste patches between images',
                'use_cases': ['localization improvement', 'regularization'],
                'parameters': {'alpha': [1.0]},
                'effectiveness': 0.85,
                'cost': 'medium'
            },
            'AutoAugment': {
                'data_type': DataType.IMAGE,
                'description': 'Automatically learned augmentation policy',
                'use_cases': ['state-of-the-art performance', 'competitive models'],
                'parameters': {'policy': ['imagenet', 'cifar10', 'svhn']},
                'effectiveness': 0.95,
                'cost': 'high'
            },
            
            # Text augmentations
            'SynonymReplacement': {
                'data_type': DataType.TEXT,
                'description': 'Replace words with synonyms',
                'use_cases': ['text classification', 'sentiment analysis'],
                'parameters': {'num_words': [1, 2, 3]},
                'effectiveness': 0.8,
                'cost': 'low'
            },
            'RandomInsertion': {
                'data_type': DataType.TEXT,
                'description': 'Insert random synonyms into text',
                'use_cases': ['text robustness', 'length variation'],
                'parameters': {'num_words': [1, 2]},
                'effectiveness': 0.75,
                'cost': 'low'
            },
            'RandomSwap': {
                'data_type': DataType.TEXT,
                'description': 'Randomly swap word positions',
                'use_cases': ['word order robustness', 'text classification'],
                'parameters': {'num_swaps': [1, 2]},
                'effectiveness': 0.7,
                'cost': 'low'
            },
            'RandomDeletion': {
                'data_type': DataType.TEXT,
                'description': 'Randomly delete words',
                'use_cases': ['robustness to missing words', 'compression'],
                'parameters': {'probability': [0.1, 0.2]},
                'effectiveness': 0.75,
                'cost': 'low'
            },
            'BackTranslation': {
                'data_type': DataType.TEXT,
                'description': 'Translate to another language and back',
                'use_cases': ['paraphrasing', 'semantic preservation'],
                'parameters': {'target_lang': ['de', 'fr', 'es']},
                'effectiveness': 0.9,
                'cost': 'high'
            },
            
            # Time series augmentations
            'TimeJitter': {
                'data_type': DataType.TIME_SERIES,
                'description': 'Add jitter to time series values',
                'use_cases': ['noise robustness', 'sensor data'],
                'parameters': {'sigma': [0.03, 0.05]},
                'effectiveness': 0.8,
                'cost': 'low'
            },
            'TimeWarping': {
                'data_type': DataType.TIME_SERIES,
                'description': 'Warp time axis with smooth distortions',
                'use_cases': ['temporal variations', 'speed changes'],
                'parameters': {'sigma': [0.2], 'knot': [4]},
                'effectiveness': 0.85,
                'cost': 'medium'
            },
            'MagnitudeWarping': {
                'data_type': DataType.TIME_SERIES,
                'description': 'Warp magnitude values',
                'use_cases': ['amplitude variations', 'signal strength'],
                'parameters': {'sigma': [0.2], 'knot': [4]},
                'effectiveness': 0.85,
                'cost': 'medium'
            },
            'WindowSlicing': {
                'data_type': DataType.TIME_SERIES,
                'description': 'Extract random windows from series',
                'use_cases': ['variable length series', 'long sequences'],
                'parameters': {'reduce_ratio': [0.9]},
                'effectiveness': 0.8,
                'cost': 'low'
            }
        }
    
    def recommend_augmentations(
        self,
        data_type: DataType,
        dataset_size: int,
        task_description: Optional[str] = None,
        level: AugmentationLevel = AugmentationLevel.MODERATE
    ) -> List[Dict[str, Any]]:
        """
        Recommend augmentation strategies.
        
        Args:
            data_type: Type of data
            dataset_size: Number of samples in dataset
            task_description: Optional description of the task
            level: Desired augmentation intensity
            
        Returns:
            List of recommended augmentation techniques
        """
        recommendations = []
        
        # Filter by data type
        candidates = {
            name: info for name, info in self.augmentation_catalog.items()
            if info['data_type'] == data_type
        }
        
        # Determine augmentation intensity based on dataset size
        if dataset_size < 1000:
            target_level = AugmentationLevel.AGGRESSIVE
        elif dataset_size < 10000:
            target_level = AugmentationLevel.MODERATE
        else:
            target_level = level
        
        # Select augmentations based on level and effectiveness
        for name, info in candidates.items():
            if target_level == AugmentationLevel.LIGHT:
                if info['cost'] == 'low' and info['effectiveness'] >= 0.8:
                    recommendations.append(self._create_recommendation(name, info))
            elif target_level == AugmentationLevel.MODERATE:
                if info['cost'] in ['low', 'medium']:
                    recommendations.append(self._create_recommendation(name, info))
            else:  # AGGRESSIVE
                recommendations.append(self._create_recommendation(name, info))
        
        # Sort by effectiveness
        recommendations.sort(key=lambda x: x['effectiveness'], reverse=True)
        
        return recommendations
    
    def _create_recommendation(
        self,
        name: str,
        info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a recommendation dictionary."""
        return {
            'name': name,
            'description': info['description'],
            'use_cases': info['use_cases'],
            'parameters': info['parameters'],
            'effectiveness': info['effectiveness'],
            'cost': info['cost'],
            'code_example': self._generate_code_example(name, info),
            'tips': self._generate_usage_tips(name, info)
        }
    
    def _generate_code_example(
        self,
        name: str,
        info: Dict[str, Any]
    ) -> str:
        """Generate code example for augmentation."""
        data_type = info['data_type']
        params = info['parameters']
        
        if data_type == DataType.IMAGE:
            param_str = ', '.join(
                f'{k}={v[0] if isinstance(v, list) and v else v}'
                for k, v in params.items()
            )
            return f"""
data_augmentation: {{
    pipeline: [
        {name}({param_str}),
        # Combine with other augmentations
    ]
}}
"""
        elif data_type == DataType.TEXT:
            return f"""
text_augmentation: {{
    techniques: [
        {{name: "{name}", **{params}}}
    ]
    probability: 0.5
}}
"""
        else:
            return f"# Apply {name} with parameters: {params}"
    
    def _generate_usage_tips(self, name: str, info: Dict[str, Any]) -> List[str]:
        """Generate usage tips for augmentation."""
        tips = []
        
        if info['cost'] == 'high':
            tips.append("This augmentation is computationally expensive; use sparingly")
        
        if info['effectiveness'] >= 0.9:
            tips.append("Highly effective augmentation; strongly recommended")
        
        if name in ['MixUp', 'CutMix']:
            tips.append("Works best with small to medium datasets")
            tips.append("Can be combined with standard augmentations")
        
        if name == 'AutoAugment':
            tips.append("Automatically finds optimal augmentation policy")
            tips.append("Best for competitive performance but slower")
        
        if name in ['RandomFlip', 'RandomRotation']:
            tips.append("Essential baseline augmentation")
            tips.append("Very low overhead, always recommended")
        
        if info['data_type'] == DataType.TEXT:
            tips.append("Preserve semantic meaning when augmenting text")
            tips.append("Test augmented samples manually to ensure quality")
        
        return tips
    
    def generate_augmentation_pipeline(
        self,
        data_type: DataType,
        dataset_size: int,
        level: AugmentationLevel = AugmentationLevel.MODERATE
    ) -> Dict[str, Any]:
        """
        Generate a complete augmentation pipeline.
        
        Args:
            data_type: Type of data
            dataset_size: Number of samples
            level: Augmentation intensity
            
        Returns:
            Complete pipeline configuration
        """
        recommendations = self.recommend_augmentations(
            data_type, dataset_size, level=level
        )
        
        # Select top augmentations based on level
        if level == AugmentationLevel.LIGHT:
            selected = recommendations[:3]
        elif level == AugmentationLevel.MODERATE:
            selected = recommendations[:5]
        else:
            selected = recommendations[:8]
        
        pipeline = {
            'data_type': data_type.value,
            'level': level.value,
            'augmentations': [
                {
                    'name': rec['name'],
                    'parameters': rec['parameters']
                }
                for rec in selected
            ],
            'code': self._generate_pipeline_code(data_type, selected),
            'expected_improvement': self._estimate_improvement(dataset_size, len(selected))
        }
        
        return pipeline
    
    def _generate_pipeline_code(
        self,
        data_type: DataType,
        augmentations: List[Dict[str, Any]]
    ) -> str:
        """Generate complete pipeline code."""
        if data_type == DataType.IMAGE:
            code = "data_augmentation: {\n"
            code += "    pipeline: [\n"
            for aug in augmentations:
                params = aug['parameters']
                param_str = ', '.join(
                    f'{k}={v[0] if isinstance(v, list) and v else v}'
                    for k, v in params.items()
                )
                code += f"        {aug['name']}({param_str}),\n"
            code += "    ]\n"
            code += "    probability: 0.8  # Apply to 80% of samples\n"
            code += "}\n"
            return code
        elif data_type == DataType.TEXT:
            code = "text_augmentation: {\n"
            code += "    techniques: [\n"
            for aug in augmentations:
                code += f"        {aug['name']},\n"
            code += "    ]\n"
            code += "    probability: 0.5\n"
            code += "}\n"
            return code
        else:
            return f"# Augmentation pipeline for {data_type.value}"
    
    def _estimate_improvement(self, dataset_size: int, num_augmentations: int) -> str:
        """Estimate expected improvement from augmentation."""
        if dataset_size < 1000:
            base_improvement = 15  # 15% for tiny datasets
        elif dataset_size < 5000:
            base_improvement = 10
        elif dataset_size < 10000:
            base_improvement = 7
        else:
            base_improvement = 5
        
        # More augmentations = more improvement (with diminishing returns)
        aug_factor = min(1.0 + (num_augmentations - 3) * 0.1, 1.5)
        
        total_improvement = int(base_improvement * aug_factor)
        
        return f"{total_improvement}% improvement in validation accuracy expected"
    
    def get_conversational_response(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate conversational response about data augmentation.
        
        Args:
            user_query: User's question
            context: Optional context (data_type, dataset_size, etc.)
            
        Returns:
            Natural language response
        """
        query_lower = user_query.lower()
        
        if 'image' in query_lower:
            return self._explain_image_augmentation()
        elif 'text' in query_lower:
            return self._explain_text_augmentation()
        elif 'how much' in query_lower or 'when to' in query_lower:
            return self._explain_when_to_augment()
        elif 'mixup' in query_lower or 'cutmix' in query_lower:
            return self._explain_advanced_augmentation()
        elif context:
            data_type = context.get('data_type', DataType.IMAGE)
            dataset_size = context.get('dataset_size', 5000)
            
            recommendations = self.recommend_augmentations(
                data_type, dataset_size
            )
            
            response = f"For {data_type.value} data with {dataset_size} samples:\n\n"
            response += "Top recommendations:\n"
            for i, rec in enumerate(recommendations[:3], 1):
                response += f"\n{i}. **{rec['name']}** (effectiveness: {rec['effectiveness']:.0%})\n"
                response += f"   {rec['description']}\n"
                response += f"   Cost: {rec['cost']}\n"
            
            return response
        else:
            return (
                "I can help with data augmentation! Ask me:\n"
                "- Image augmentation techniques\n"
                "- Text augmentation strategies\n"
                "- When to use augmentation\n"
                "- Advanced techniques (MixUp, CutMix)\n"
                "Or provide your data details for specific recommendations!"
            )
    
    def _explain_image_augmentation(self) -> str:
        """Explain image augmentation."""
        return """
**Image Augmentation Techniques:**

Essential (always use):
1. **RandomFlip** - Horizontal/vertical flips
2. **RandomRotation** - Rotate by small angles (±15°)
3. **RandomCrop** - Crop and resize patches

Effective (recommended):
4. **ColorJitter** - Adjust brightness, contrast, saturation
5. **RandomErasing** - Simulate occlusions
6. **GaussianNoise** - Add noise for robustness

Advanced (for competitive performance):
7. **MixUp** - Blend two images and labels
8. **CutMix** - Cut and paste image patches
9. **AutoAugment** - Learned augmentation policies

Example pipeline:
```python
data_augmentation: {
    pipeline: [
        RandomFlip(mode='horizontal'),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2),
        RandomCrop(scale=(0.8, 1.0)),
        RandomErasing(probability=0.5)
    ]
    probability: 0.8
}
```

Expected improvement: 5-15% accuracy boost, depending on dataset size!
"""
    
    def _explain_text_augmentation(self) -> str:
        """Explain text augmentation."""
        return """
**Text Augmentation Techniques:**

Basic techniques:
1. **Synonym Replacement** - Replace words with synonyms
2. **Random Insertion** - Insert random synonyms
3. **Random Swap** - Swap word positions
4. **Random Deletion** - Delete words randomly

Advanced:
5. **Back Translation** - Translate to another language and back
6. **Contextual Augmentation** - Use language models for substitution

Example:
```python
text_augmentation: {
    techniques: [
        SynonymReplacement(num_words=2),
        RandomInsertion(num_words=1),
        RandomSwap(num_swaps=2),
        RandomDeletion(probability=0.1)
    ]
    probability: 0.5  # Apply to 50% of samples
}
```

Important notes:
- Preserve semantic meaning
- Don't over-augment (keep probability ≤ 0.5)
- Manually verify augmented samples
- Works best with small datasets (<10K samples)

Expected improvement: 3-8% accuracy on small text datasets
"""
    
    def _explain_when_to_augment(self) -> str:
        """Explain when to use augmentation."""
        return """
**When to use data augmentation:**

✅ **Definitely augment when:**
- Dataset is small (<10K samples)
- Model is overfitting
- Need better generalization
- Want to improve robustness

📊 **Dataset size guidelines:**
- **< 1,000**: Aggressive augmentation (8-10 techniques)
- **1,000 - 5,000**: Moderate augmentation (5-7 techniques)
- **5,000 - 50,000**: Light augmentation (3-5 techniques)
- **> 50,000**: Minimal augmentation (2-3 basic techniques)

⚠️ **Be careful with:**
- Tasks where transformations change meaning
  - Medical imaging (rotations might not be valid)
  - Text sentiment (synonyms might change sentiment)
  - Time series (temporal order matters)
  
- Over-augmentation (can hurt performance):
  - Keep probability ≤ 0.8 for images
  - Keep probability ≤ 0.5 for text
  - Monitor validation performance

💡 **Pro tip:**
Start with basic augmentations (flip, rotate, crop) and add more if validation
accuracy plateaus. Always validate that augmentations make sense for your task!

Expected benefits:
- 5-15% accuracy improvement (small datasets)
- 2-5% improvement (medium datasets)
- Better robustness to real-world variations
- Reduced overfitting
"""
    
    def _explain_advanced_augmentation(self) -> str:
        """Explain advanced augmentation techniques."""
        return """
**Advanced Augmentation: MixUp & CutMix**

MixUp:
- Blends two images and their labels
- `mixed_image = α * image1 + (1-α) * image2`
- `mixed_label = α * label1 + (1-α) * label2`
- α sampled from Beta(0.2, 0.2) or Beta(0.4, 0.4)

Benefits:
✓ Strong regularization effect
✓ Smoother decision boundaries
✓ Better calibration
✓ Works well with small datasets

CutMix:
- Cuts a patch from one image and pastes onto another
- Labels mixed proportionally to patch size
- More localization-aware than MixUp

Benefits:
✓ Improves localization ability
✓ Better than MixUp for detection tasks
✓ Encourages using full image
✓ Less information loss than MixUp

When to use:
- **MixUp**: Classification tasks, small datasets
- **CutMix**: When localization matters, object detection

Example:
```python
data_augmentation: {
    pipeline: [
        # Standard augmentations
        RandomFlip(mode='horizontal'),
        RandomCrop(scale=(0.8, 1.0)),
        
        # Advanced mixing
        MixUp(alpha=0.2, probability=0.5),
        # OR
        CutMix(alpha=1.0, probability=0.5)
    ]
}
```

Expected improvement:
- 2-5% accuracy boost on top of standard augmentation
- Better robustness to adversarial examples
- Improved model calibration

Note: Can't visualize mixed samples as easily, but trust the process! 🚀
"""
