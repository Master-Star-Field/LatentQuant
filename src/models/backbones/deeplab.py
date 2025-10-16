"""
DeepLabV3-ResNet50 segmentation backbone implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from .base import SegmentationBackbone


class DeepLabV3SegmentationBackbone(SegmentationBackbone):
    """
    DeepLabV3-ResNet50 segmentation backbone.
    
    Uses pre-trained DeepLabV3-ResNet50 for segmentation.
    """
    
    def __init__(
        self,
        weights_name='COCO_WITH_VOC_LABELS_V1',
        num_classes=21,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        weights = getattr(DeepLabV3_ResNet50_Weights, weights_name)
        model = deeplabv3_resnet50(weights=weights)
        
        self.backbone = model.backbone
        self.classifier = model.classifier
        
        self._num_classes = num_classes
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def extract_features(self, images, detach=True):
        """
        Extract DeepLabV3 backbone features.
        
        Args:
            images: Input images [B, C, H, W]
            detach: Whether to detach gradients from backbone
            
        Returns:
            features: Feature maps [B, 2048, H/8, W/8]
        """
        with torch.set_grad_enabled(not detach):
            features = self.backbone(images)['out']
        return features

    def forward(self, images):
        """
        Full DeepLabV3 segmentation forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            output: Segmentation logits [B, num_classes, H, W]
        """
        features = self.backbone(images)['out']
        output = self.classifier(features)
        output = F.interpolate(output, size=images.shape[-2:], mode='bilinear')
        return {'out': output}

    @property
    def feature_dim(self):
        """Return DeepLabV3 feature dimension."""
        return 2048

    @property
    def num_classes(self):
        """Return number of segmentation classes."""
        return self._num_classes
