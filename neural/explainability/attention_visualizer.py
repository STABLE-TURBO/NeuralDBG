"""
Attention visualization for transformer models with Neural DSL integration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualize attention weights in transformer models.
    
    Supports visualization of self-attention, cross-attention, and multi-head attention
    with seamless integration for Neural DSL compiled models.
    """
    
    def __init__(
        self,
        model: Any,
        backend: str = 'tensorflow'
    ):
        """
        Initialize attention visualizer.
        
        Args:
            model: The model with attention layers
            backend: ML framework ('tensorflow', 'pytorch')
        """
        self.model = model
        self.backend = backend.lower()
        
        logger.info("Initialized AttentionVisualizer for %s model", backend)
    
    def extract_attention_weights(
        self,
        input_data: np.ndarray,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from specified layers.
        
        Args:
            input_data: Input data
            layer_names: Names of attention layers to extract
            
        Returns:
            Dictionary mapping layer names to attention weights
        """
        attention_weights = {}
        
        if self.backend == 'tensorflow':
            attention_weights = self._extract_tf_attention(input_data, layer_names)
        elif self.backend == 'pytorch':
            attention_weights = self._extract_torch_attention(input_data, layer_names)
        else:
            raise ValueError(f"Backend {self.backend} not supported for attention visualization")
        
        logger.info("Extracted attention weights from %d layers", len(attention_weights))
        
        return attention_weights
    
    def _extract_tf_attention(
        self,
        input_data: np.ndarray,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Extract attention weights from TensorFlow model."""
        import tensorflow as tf
        
        attention_weights = {}
        
        if layer_names is None:
            layer_names = [layer.name for layer in self.model.layers 
                          if any(keyword in layer.name.lower() 
                                for keyword in ['attention', 'transformer', 'multihead'])]
        
        for layer_name in layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                
                attention_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
                
                outputs = attention_model(input_data)
                
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    attention_weights[layer_name] = outputs[1].numpy()
                elif hasattr(layer, 'attention_weights'):
                    attention_weights[layer_name] = layer.attention_weights.numpy()
                else:
                    attention_weights[layer_name] = outputs.numpy()
                    
            except Exception as e:
                logger.warning("Could not extract attention from layer %s: %s", layer_name, e)
        
        return attention_weights
    
    def _extract_torch_attention(
        self,
        input_data: np.ndarray,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Extract attention weights from PyTorch model."""
        import torch
        
        attention_weights = {}
        
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.FloatTensor(input_data)
        
        hooks = []
        attention_outputs = {}
        
        def attention_hook(name):
            def hook(module, input_tensor, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attention_outputs[name] = output[1].detach().cpu().numpy()
                elif hasattr(output, 'attention_weights'):
                    attention_outputs[name] = output.attention_weights.detach().cpu().numpy()
                else:
                    attention_outputs[name] = output.detach().cpu().numpy()
            return hook
        
        if layer_names is None:
            for name, module in self.model.named_modules():
                if any(keyword in name.lower() 
                      for keyword in ['attention', 'transformer', 'multihead']):
                    hooks.append(module.register_forward_hook(attention_hook(name)))
        else:
            for name, module in self.model.named_modules():
                if name in layer_names:
                    hooks.append(module.register_forward_hook(attention_hook(name)))
        
        with torch.no_grad():
            _ = self.model(input_data)
        
        for hook in hooks:
            hook.remove()
        
        attention_weights = attention_outputs
        
        return attention_weights
    
    def visualize(
        self,
        input_data: np.ndarray,
        layer_name: Optional[str] = None,
        head_index: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visualize attention patterns.
        
        Args:
            input_data: Input data
            layer_name: Specific layer to visualize
            head_index: Specific attention head to visualize
            tokens: Token labels for visualization
            output_path: Path to save visualization
            
        Returns:
            Dictionary containing attention visualizations
        """
        layer_names = [layer_name] if layer_name else None
        attention_weights = self.extract_attention_weights(input_data, layer_names)
        
        if not attention_weights:
            logger.warning("No attention weights extracted")
            return {'attention_weights': {}}
        
        results = {
            'attention_weights': attention_weights,
            'visualizations': {}
        }
        
        for name, weights in attention_weights.items():
            viz = self._create_attention_heatmap(
                weights,
                head_index=head_index,
                tokens=tokens,
                title=f"Attention: {name}",
                output_path=output_path
            )
            results['visualizations'][name] = viz
        
        return results
    
    def _create_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        head_index: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        title: str = "Attention Heatmap",
        output_path: Optional[str] = None
    ) -> Any:
        """Create attention heatmap visualization."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if len(attention_weights.shape) == 4:
                batch_idx = 0
                if head_index is not None:
                    weights = attention_weights[batch_idx, head_index]
                else:
                    weights = attention_weights[batch_idx].mean(axis=0)
            elif len(attention_weights.shape) == 3:
                if head_index is not None:
                    weights = attention_weights[head_index]
                else:
                    weights = attention_weights.mean(axis=0)
            else:
                weights = attention_weights
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(
                weights,
                cmap='viridis',
                cbar=True,
                square=True,
                ax=ax,
                xticklabels=tokens if tokens else False,
                yticklabels=tokens if tokens else False
            )
            
            ax.set_title(title)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                logger.info("Saved attention heatmap to %s", output_path)
            
            return fig
            
        except ImportError:
            logger.warning("matplotlib/seaborn not available for visualization")
            return None
    
    def plot_attention_heads(
        self,
        attention_weights: np.ndarray,
        tokens: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Any:
        """
        Plot all attention heads in a grid.
        
        Args:
            attention_weights: Attention weights with shape (num_heads, seq_len, seq_len)
            tokens: Token labels
            output_path: Path to save plot
            
        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            num_heads = attention_weights.shape[0] if len(attention_weights.shape) > 2 else 1
            
            cols = min(4, num_heads)
            rows = (num_heads + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            if num_heads == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for head_idx in range(num_heads):
                weights = attention_weights[head_idx] if len(attention_weights.shape) > 2 else attention_weights
                
                sns.heatmap(
                    weights,
                    cmap='viridis',
                    cbar=True,
                    square=True,
                    ax=axes[head_idx],
                    xticklabels=tokens if tokens else False,
                    yticklabels=tokens if tokens else False
                )
                
                axes[head_idx].set_title(f'Head {head_idx}')
            
            for idx in range(num_heads, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                logger.info("Saved attention heads plot to %s", output_path)
            
            return fig
            
        except ImportError:
            logger.warning("matplotlib/seaborn not available for visualization")
            return None
    
    def visualize_from_dsl(
        self,
        dsl_file: str,
        input_data: np.ndarray,
        backend: str = 'tensorflow',
        layer_name: Optional[str] = None,
        head_index: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        output_dir: str = 'attention_outputs'
    ) -> Dict[str, Any]:
        """
        Visualize attention from a Neural DSL model file.
        
        Args:
            dsl_file: Path to .neural or .nr file
            input_data: Input data for attention visualization
            backend: Backend to compile model with
            layer_name: Specific attention layer to visualize
            head_index: Specific attention head
            tokens: Token labels
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with attention weights and visualizations
        """
        import os
        from pathlib import Path
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            from neural.parser.parser import create_parser, ModelTransformer
            from neural.code_generation import generate_code
            
            parser = create_parser(start_rule='network')
            with open(dsl_file, 'r') as f:
                dsl_content = f.read()
            
            tree = parser.parse(dsl_content)
            model_data = ModelTransformer().transform(tree)
            
            logger.info(f"Parsed Neural DSL file: {dsl_file}")
            
            code = generate_code(model_data, backend)
            
            model_path = os.path.join(output_dir, f'compiled_model_{backend}.py')
            with open(model_path, 'w') as f:
                f.write(code)
            
            logger.info(f"Generated {backend} code for model")
            
            if backend == 'tensorflow':
                import tensorflow as tf
                import importlib.util
                
                spec = importlib.util.spec_from_file_location("compiled_model", model_path)
                compiled_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(compiled_module)
                
                model = compiled_module.create_model()
                logger.info("Loaded compiled TensorFlow model")
                
            elif backend == 'pytorch':
                import torch
                import importlib.util
                
                spec = importlib.util.spec_from_file_location("compiled_model", model_path)
                compiled_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(compiled_module)
                
                model = compiled_module.create_model()
                logger.info("Loaded compiled PyTorch model")
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            
            self.model = model
            self.backend = backend
            
            results = self.visualize(
                input_data,
                layer_name=layer_name,
                head_index=head_index,
                tokens=tokens,
                output_path=os.path.join(output_dir, 'attention_heatmap.png')
            )
            
            if results['attention_weights']:
                for layer_name, weights in results['attention_weights'].items():
                    if len(weights.shape) >= 3:
                        heads_path = os.path.join(output_dir, f'attention_heads_{layer_name}.png')
                        self.plot_attention_heads(
                            weights[0] if len(weights.shape) == 4 else weights,
                            tokens=tokens,
                            output_path=heads_path
                        )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to visualize attention from DSL: {e}")
            raise
    
    def create_interactive_visualization(
        self,
        attention_weights: Dict[str, np.ndarray],
        tokens: Optional[List[str]] = None,
        output_path: str = 'attention_interactive.html'
    ) -> str:
        """
        Create an interactive attention visualization using plotly.
        
        Args:
            attention_weights: Dictionary of attention weights by layer
            tokens: Token labels
            output_path: Path to save HTML file
            
        Returns:
            Path to generated HTML file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            num_layers = len(attention_weights)
            
            fig = make_subplots(
                rows=1, cols=num_layers,
                subplot_titles=list(attention_weights.keys()),
                horizontal_spacing=0.1
            )
            
            for idx, (layer_name, weights) in enumerate(attention_weights.items(), 1):
                if len(weights.shape) == 4:
                    weights = weights[0].mean(axis=0)
                elif len(weights.shape) == 3:
                    weights = weights.mean(axis=0)
                
                heatmap = go.Heatmap(
                    z=weights,
                    x=tokens if tokens else list(range(weights.shape[1])),
                    y=tokens if tokens else list(range(weights.shape[0])),
                    colorscale='Viridis',
                    showscale=True if idx == num_layers else False
                )
                
                fig.add_trace(heatmap, row=1, col=idx)
            
            fig.update_layout(
                title='Attention Weights Visualization',
                height=500,
                width=400 * num_layers
            )
            
            fig.write_html(output_path)
            logger.info(f"Saved interactive visualization to {output_path}")
            
            return output_path
            
        except ImportError:
            logger.warning("plotly not available for interactive visualization")
            return None
    
    def analyze_attention_patterns(
        self,
        attention_weights: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns for insights.
        
        Args:
            attention_weights: Attention weights to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        if len(attention_weights.shape) == 4:
            weights = attention_weights[0]
        elif len(attention_weights.shape) == 3:
            weights = attention_weights
        else:
            weights = attention_weights[np.newaxis, :]
        
        num_heads = weights.shape[0]
        
        analysis['num_heads'] = num_heads
        analysis['attention_entropy'] = []
        analysis['max_attention_per_head'] = []
        analysis['attention_distribution'] = []
        
        for head_idx in range(num_heads):
            head_weights = weights[head_idx]
            
            attention_probs = head_weights / (head_weights.sum(axis=-1, keepdims=True) + 1e-10)
            entropy = -(attention_probs * np.log(attention_probs + 1e-10)).sum(axis=-1).mean()
            analysis['attention_entropy'].append(float(entropy))
            
            max_attention = head_weights.max(axis=-1).mean()
            analysis['max_attention_per_head'].append(float(max_attention))
            
            analysis['attention_distribution'].append({
                'mean': float(head_weights.mean()),
                'std': float(head_weights.std()),
                'min': float(head_weights.min()),
                'max': float(head_weights.max())
            })
        
        analysis['avg_entropy'] = float(np.mean(analysis['attention_entropy']))
        analysis['avg_max_attention'] = float(np.mean(analysis['max_attention_per_head']))
        
        diagonal_attention = []
        for head_idx in range(num_heads):
            head_weights = weights[head_idx]
            diag_strength = np.diag(head_weights).mean() / (head_weights.mean() + 1e-10)
            diagonal_attention.append(float(diag_strength))
        
        analysis['diagonal_attention_strength'] = diagonal_attention
        analysis['avg_diagonal_strength'] = float(np.mean(diagonal_attention))
        
        return analysis
