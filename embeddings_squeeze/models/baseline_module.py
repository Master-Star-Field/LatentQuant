"""
PyTorch Lightning module for baseline segmentation training without VQ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional

from .backbones.base import SegmentationBackbone


class BaselineSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for baseline segmentation training.
    
    Wraps segmentation backbone without Vector Quantization for comparison.
    """
    
    def __init__(
        self,
        backbone: SegmentationBackbone,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = backbone
        self.learning_rate = learning_rate
        
        # Segmentation loss
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, images):
        """
        Forward pass through backbone.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            output: Segmentation logits [B, num_classes, H, W]
        """
        output = self.backbone(images)
        # Handle both dict and tensor returns
        if isinstance(output, dict):
            return output['out']
        return output

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        # Forward pass
        output = self(images)
        
        # Compute loss
        seg_loss = self.seg_criterion(output, masks)
        
        # Log metrics
        self.log('train/seg_loss', seg_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return seg_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        # Forward pass
        output = self(images)
        
        # Compute loss
        seg_loss = self.seg_criterion(output, masks)
        
        # Compute IoU (simplified pixel accuracy)
        preds = output.argmax(dim=1)
        valid = masks != 255
        pixel_acc = (preds[valid] == masks[valid]).float().mean()
        
        # Log metrics
        self.log('val/seg_loss', seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/pixel_acc', pixel_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return seg_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        # Optimize only trainable params
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.learning_rate)

    def predict(self, images):
        """
        Predict segmentation masks.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            predictions: Segmentation predictions [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            output = self(images)
            predictions = output.argmax(dim=1)
        return predictions

    def predict_logits(self, images):
        """
        Predict segmentation logits.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            logits: Segmentation logits [B, num_classes, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self(images)
        return logits
