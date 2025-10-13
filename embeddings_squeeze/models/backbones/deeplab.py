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
        
        # Split classifier into feature extractor (ASPP) and final conv
        # Original classifier structure: [0]=ASPP, [1]=Conv, [2]=BN, [3]=ReLU, [4]=Final Conv
        classifier_modules = list(model.classifier.children())
        
        # Backbone now includes: ResNet + ASPP + intermediate convs (all frozen)
        self.backbone = nn.ModuleDict({
            'resnet': model.backbone,
            'aspp_and_convs': nn.Sequential(*classifier_modules[:-1])  # Everything except last layer
        })
        
        # Classifier is only the final 1x1 convolution (trainable)
        self.classifier = classifier_modules[-1]  # Final Conv2d layer
        
        self._num_classes = num_classes
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def extract_features(self, images, detach=True):
        """
        Extract DeepLabV3 backbone features (ResNet + ASPP).
        
        Args:
            images: Input images [B, C, H, W]
            detach: Whether to detach gradients from backbone
            
        Returns:
            features: Feature maps after ASPP [B, 256, H/8, W/8]
        """
        with torch.set_grad_enabled(not detach):
            # Pass through ResNet backbone
            resnet_features = self.backbone['resnet'](images)['out']
            # Pass through ASPP and intermediate convs
            features = self.backbone['aspp_and_convs'](resnet_features)
        return features

    def forward(self, images):
        """
        Full DeepLabV3 segmentation forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            output: Segmentation logits [B, num_classes, H, W]
        """
        # Pass through ResNet + ASPP (frozen backbone)
        resnet_features = self.backbone['resnet'](images)['out']
        features = self.backbone['aspp_and_convs'](resnet_features)
        
        # Pass through final classifier (trainable)
        output = self.classifier(features)
        output = F.interpolate(output, size=images.shape[-2:], mode='bilinear')
        return {'out': output}

    @property
    def feature_dim(self):
        """Return DeepLabV3 feature dimension (output of ASPP)."""
        return 256  # ASPP output dimension

    @property
    def num_classes(self):
        """Return number of segmentation classes."""
        return self._num_classes
