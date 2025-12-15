"""
Feature importance ranking for neural networks.
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np
import logging

from neural.exceptions import DependencyError

logger = logging.getLogger(__name__)


class FeatureImportanceRanker:
    """
    Rank features by importance using various methods:
    - Permutation importance
    - Gradient-based importance
    - Integrated gradients importance
    - SHAP-based importance
    """
    
    def __init__(
        self,
        model: Any,
        backend: str = 'tensorflow',
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize feature importance ranker.
        
        Args:
            model: The model to analyze
            backend: ML framework ('tensorflow', 'pytorch', 'onnx')
            feature_names: Names of input features
        """
        self.model = model
        self.backend = backend.lower()
        self.feature_names = feature_names
        
        logger.info(f"Initialized FeatureImportanceRanker for {backend} model")
    
    def rank(
        self,
        input_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'permutation',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Rank features by importance.
        
        Args:
            input_data: Input dataset
            labels: True labels (required for permutation importance)
            method: Ranking method ('permutation', 'gradient', 'integrated_gradient', 'shap')
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary containing feature rankings and scores
        """
        if method == 'permutation':
            if labels is None:
                raise ValueError("Labels required for permutation importance")
            importance_scores = self._permutation_importance(input_data, labels, **kwargs)
        elif method == 'gradient':
            importance_scores = self._gradient_importance(input_data, **kwargs)
        elif method == 'integrated_gradient':
            importance_scores = self._integrated_gradient_importance(input_data, **kwargs)
        elif method == 'shap':
            importance_scores = self._shap_importance(input_data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        ranked_indices = np.argsort(importance_scores)[::-1]
        
        if self.feature_names is not None:
            ranked_features = [self.feature_names[i] for i in ranked_indices]
        else:
            ranked_features = [f"Feature_{i}" for i in ranked_indices]
        
        results = {
            'method': method,
            'importance_scores': importance_scores,
            'ranked_indices': ranked_indices,
            'ranked_features': ranked_features,
            'rankings': list(zip(ranked_features, importance_scores[ranked_indices]))
        }
        
        logger.info(f"Ranked {len(importance_scores)} features using {method}")
        
        return results
    
    def _get_prediction_function(self) -> Callable:
        """Create a prediction function for the model."""
        if self.backend == 'tensorflow':
            try:
                import tensorflow as tf
            except ImportError as e:
                raise DependencyError(
                    dependency="tensorflow",
                    feature="TensorFlow feature importance",
                    install_hint="pip install tensorflow"
                ) from e
            def predict_fn(x):
                return self.model(x).numpy()
        elif self.backend == 'pytorch':
            try:
                import torch
            except ImportError as e:
                raise DependencyError(
                    dependency="torch",
                    feature="PyTorch feature importance",
                    install_hint="pip install torch"
                ) from e
            def predict_fn(x):
                if not isinstance(x, torch.Tensor):
                    x = torch.FloatTensor(x)
                with torch.no_grad():
                    return self.model(x).numpy()
        else:
            def predict_fn(x):
                return self.model.predict(x)
        
        return predict_fn
    
    def _compute_metric(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metric: str = 'accuracy'
    ) -> float:
        """Compute evaluation metric."""
        if metric == 'accuracy':
            pred_classes = np.argmax(predictions, axis=1)
            return np.mean(pred_classes == labels)
        elif metric == 'mse':
            return np.mean((predictions - labels) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(predictions - labels))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _permutation_importance(
        self,
        input_data: np.ndarray,
        labels: np.ndarray,
        num_repeats: int = 10,
        metric: str = 'accuracy'
    ) -> np.ndarray:
        """
        Compute permutation importance.
        
        Args:
            input_data: Input dataset
            labels: True labels
            num_repeats: Number of permutation repeats
            metric: Evaluation metric
            
        Returns:
            Feature importance scores
        """
        predict_fn = self._get_prediction_function()
        
        baseline_predictions = predict_fn(input_data)
        baseline_score = self._compute_metric(baseline_predictions, labels, metric)
        
        num_features = input_data.shape[1] if len(input_data.shape) == 2 else np.prod(input_data.shape[1:])
        importance_scores = np.zeros(num_features)
        
        for feature_idx in range(num_features):
            feature_scores = []
            
            for _ in range(num_repeats):
                permuted_data = input_data.copy()
                
                if len(input_data.shape) == 2:
                    np.random.shuffle(permuted_data[:, feature_idx])
                else:
                    flat_idx = np.unravel_index(feature_idx, input_data.shape[1:])
                    feature_column = permuted_data[:, flat_idx[0], flat_idx[1] if len(flat_idx) > 1 else 0]
                    np.random.shuffle(feature_column)
                    permuted_data[:, flat_idx[0], flat_idx[1] if len(flat_idx) > 1 else 0] = feature_column
                
                permuted_predictions = predict_fn(permuted_data)
                permuted_score = self._compute_metric(permuted_predictions, labels, metric)
                
                feature_scores.append(baseline_score - permuted_score)
            
            importance_scores[feature_idx] = np.mean(feature_scores)
        
        return importance_scores
    
    def _gradient_importance(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute gradient-based importance.
        
        Args:
            input_data: Input dataset
            target_class: Target class for gradients
            
        Returns:
            Feature importance scores
        """
        from .saliency_maps import SaliencyMapGenerator
        
        saliency_generator = SaliencyMapGenerator(self.model, self.backend)
        
        all_gradients = []
        
        for sample in input_data:
            result = saliency_generator.generate(sample, target_class, method='vanilla')
            gradients = result['saliency_map']
            all_gradients.append(gradients.flatten())
        
        all_gradients = np.array(all_gradients)
        
        importance_scores = np.mean(np.abs(all_gradients), axis=0)
        
        return importance_scores
    
    def _integrated_gradient_importance(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        num_steps: int = 50
    ) -> np.ndarray:
        """
        Compute integrated gradient importance.
        
        Args:
            input_data: Input dataset
            target_class: Target class
            num_steps: Number of integration steps
            
        Returns:
            Feature importance scores
        """
        from .saliency_maps import SaliencyMapGenerator
        
        saliency_generator = SaliencyMapGenerator(self.model, self.backend)
        
        all_gradients = []
        
        for sample in input_data:
            result = saliency_generator.generate(
                sample,
                target_class,
                method='integrated',
                num_steps=num_steps
            )
            gradients = result['saliency_map']
            all_gradients.append(gradients.flatten())
        
        all_gradients = np.array(all_gradients)
        
        importance_scores = np.mean(np.abs(all_gradients), axis=0)
        
        return importance_scores
    
    def _shap_importance(
        self,
        input_data: np.ndarray,
        num_background_samples: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP-based importance.
        
        Args:
            input_data: Input dataset
            num_background_samples: Number of background samples
            
        Returns:
            Feature importance scores
        """
        from .shap_explainer import SHAPExplainer
        
        shap_explainer = SHAPExplainer(self.model, self.backend)
        
        result = shap_explainer.explain_dataset(
            input_data,
            num_background_samples=num_background_samples
        )
        
        importance_scores = result['mean_abs_shap'].flatten()
        
        return importance_scores
    
    def plot_importance(
        self,
        importance_scores: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        output_path: Optional[str] = None
    ) -> Any:
        """
        Plot feature importance.
        
        Args:
            importance_scores: Feature importance scores
            feature_names: Names of features
            top_k: Number of top features to display
            output_path: Path to save plot
            
        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise DependencyError(
                dependency="matplotlib",
                feature="feature importance visualization",
                install_hint="pip install matplotlib"
            ) from e
        
        if feature_names is None:
            feature_names = self.feature_names or [f"Feature_{i}" for i in range(len(importance_scores))]
        
        sorted_indices = np.argsort(importance_scores)[::-1][:top_k]
        sorted_scores = importance_scores[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_k} Feature Importance')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved feature importance plot to {output_path}")
        
        return fig
