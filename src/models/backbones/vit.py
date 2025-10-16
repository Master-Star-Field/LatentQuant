"""
ViT-based segmentation backbone implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_32, ViT_B_32_Weights

from .base import SegmentationBackbone


class _ViTBackboneWrapper(nn.Module):
    """Wrapper for ViT backbone to extract spatial features."""
    
    def __init__(self, vit_model: nn.Module, freeze: bool = True):
        super().__init__()
        self.vit = vit_model
        self.hidden_dim = vit_model.hidden_dim
        self.patch_size = vit_model.conv_proj.kernel_size[0]

        # Remove classifier head - we only need encoder features
        self.vit.heads = nn.Identity()

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

    def forward(self, x):
        b, _, h, w = x.shape
        tokens = self.vit._process_input(x)                    # (B, N, D)
        if self.vit.class_token is not None:
            cls = self.vit.class_token.expand(b, -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)           # (B, 1+N, D)

        encoded = self.vit.encoder(tokens)                     # (B, 1+N, D)
        patch_tokens = encoded[:, 1:]                          # drop class token
        features = patch_tokens.transpose(1, 2)                # (B, D, N)

        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        features = features.reshape(b, self.hidden_dim, grid_h, grid_w)
        return {'out': features}


class _ViTSegmentationHead(nn.Sequential):
    """Segmentation head for ViT features."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )


class ViTSegmentationBackbone(SegmentationBackbone):
    """
    ViT-based segmentation backbone.
    
    Uses ViT-B/32 as backbone with custom segmentation head.
    """
    
    def __init__(
        self,
        model_fn=vit_b_32,
        weights=ViT_B_32_Weights.IMAGENET1K_V1,
        num_classes=21,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        base_vit = model_fn(weights=weights)
        self.backbone = _ViTBackboneWrapper(base_vit, freeze=freeze_backbone)
        self.classifier = _ViTSegmentationHead(self.backbone.hidden_dim, num_classes)
        
        self._num_classes = num_classes

    def extract_features(self, images, detach=True):
        """
        Extract ViT backbone feature maps.
        
        Args:
            images: Input images [B, C, H, W]
            detach: Whether to detach gradients from backbone
            
        Returns:
            features: Feature maps [B, hidden_dim, H/patch, W/patch]
        """
        feats = self.backbone(images)['out']
        return feats.detach() if detach else feats

    def forward(self, images):
        """
        Full ViT segmentation forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            output: Segmentation logits [B, num_classes, H, W]
        """
        features = self.backbone(images)['out']
        logits = self.classifier(features)
        logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
        return {'out': logits}

    @property
    def feature_dim(self):
        """Return ViT hidden dimension."""
        return self.backbone.hidden_dim

    @property
    def num_classes(self):
        """Return number of segmentation classes."""
        return self._num_classes
