"""
Abstract base class for segmentation backbones.
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class SegmentationBackbone(nn.Module, ABC):
    """
    Abstract base class for segmentation backbones.
    
    All segmentation backbones should inherit from this class and implement
    the required methods for feature extraction and full segmentation.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def extract_features(self, images, detach=True):
        """
        Extract features from input images.
        
        Args:
            images: Input images [B, C, H, W]
            detach: Whether to detach gradients from backbone
            
        Returns:
            features: Feature maps [B, feature_dim, H', W']
        """
        pass
    
    @abstractmethod
    def forward(self, images):
        """
        Full forward pass for segmentation.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            output: Segmentation logits [B, num_classes, H, W]
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self):
        """Return the feature dimension."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self):
        """Return the number of output classes."""
        pass
