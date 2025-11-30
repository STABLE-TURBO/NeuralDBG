"""
Pretrained Models Module for Neural DSL
Provides access to pretrained models and utilities for model optimization.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class PretrainedModelHub:
    """Hub for accessing pretrained neural network models."""
    
    def __init__(self):
        """Initialize the pretrained model hub."""
        self.available_models = {
            "resnet50": self._load_resnet50,
            "vgg16": self._load_vgg16,
            "mobilenet": self._load_mobilenet,
        }
    
    def load(self, model_name: str, pretrained: bool = True) -> Optional[nn.Module]:
        """
        Load a pretrained model.
        
        Args:
            model_name: Name of the model to load
            pretrained: Whether to load pretrained weights
            
        Returns:
            Loaded model
            
        Raises:
            ValueError: If model is not found
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.available_models.keys())}")
        
        return self.available_models[model_name](pretrained)
    
    def _load_resnet50(self, pretrained: bool) -> nn.Module:
        """Load ResNet50 model."""
        try:
            import torchvision.models as models
            return models.resnet50(pretrained=pretrained)
        except ImportError:
            # Fallback: create a simple dummy model
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 1000)
            )
    
    def _load_vgg16(self, pretrained: bool) -> nn.Module:
        """Load VGG16 model."""
        try:
            import torchvision.models as models
            return models.vgg16(pretrained=pretrained)
        except ImportError:
            # Fallback: create a simple dummy model
            return nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 1000)
            )
    
    def _load_mobilenet(self, pretrained: bool) -> nn.Module:
        """Load MobileNet model."""
        try:
            import torchvision.models as models
            return models.mobilenet_v2(pretrained=pretrained)
        except ImportError:
            # Fallback: create a simple dummy model
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, 1000)
            )


def fuse_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: Optional[torch.Tensor],
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
    eps: float = 1e-5
) -> tuple:
    """
    Fuse convolutional and batch normalization weights.
    
    This optimization combines Conv and BatchNorm layers into a single Conv layer,
    which is useful for inference optimization.
    
    Args:
        conv_w: Convolutional weights [out_channels, in_channels, kH, kW]
        conv_b: Convolutional bias [out_channels] (optional)
        bn_rm: BatchNorm running mean [out_channels]
        bn_rv: BatchNorm running variance [out_channels]
        bn_w: BatchNorm weight (gamma) [out_channels]
        bn_b: BatchNorm bias (beta) [out_channels]
        eps: BatchNorm epsilon
        
    Returns:
        Tuple of (fused_weight, fused_bias)
    """
    # Compute the scaling factor from BatchNorm
    # scale = gamma / sqrt(var + eps)
    bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
    scale = bn_w * bn_var_rsqrt
    
    # Reshape scale to broadcast correctly with conv weights
    # [out_channels] -> [out_channels, 1, 1, 1]
    scale_reshaped = scale.reshape(-1, 1, 1, 1)
    
    # Fuse weights: w_fused = w_conv * scale
    fused_w = conv_w * scale_reshaped
    
    # Fuse bias: b_fused = (b_conv - running_mean) * scale + beta
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    
    fused_b = (conv_b - bn_rm) * scale + bn_b
    
    return fused_w, fused_b
