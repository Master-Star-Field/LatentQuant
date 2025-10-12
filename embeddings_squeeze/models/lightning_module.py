"""
PyTorch Lightning module for VQ compression training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional

from .vq.quantizer import VectorQuantizer
from .backbones.base import SegmentationBackbone


class VQSqueezeModule(pl.LightningModule):
    """
    PyTorch Lightning module for VQ compression training.
    
    Wraps any segmentation backbone with Vector Quantization for compression.
    """
    
    def __init__(
        self,
        backbone: SegmentationBackbone,
        num_vectors: int = 128,
        commitment_cost: float = 0.25,
        metric_type: str = 'euclidean',
        learning_rate: float = 1e-4,
        vq_loss_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.vq_loss_weight = vq_loss_weight
        
        # Initialize VQ with backbone's feature dimension
        self.vq = VectorQuantizer(
            num_vectors=num_vectors,
            vector_dim=backbone.feature_dim,
            commitment_cost=commitment_cost,
            metric_type=metric_type
        )
        
        # Segmentation loss
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, images):
        """
        Forward pass through backbone + VQ.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            output: Segmentation logits [B, num_classes, H, W]
            vq_loss: VQ loss
        """
        # Extract features
        features = self.backbone.extract_features(images, detach=False)
        
        # Quantize features
        B, C, H, W = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        quant_flat, vq_loss = self.vq(feat_flat)
        quantized = quant_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Segmentation forward pass
        output = self.backbone.classifier(quantized)
        output = F.interpolate(output, size=images.shape[-2:], mode='bilinear')
        
        return output, vq_loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        # Forward pass
        output, vq_loss = self(images)
        
        # Compute losses
        seg_loss = self.seg_criterion(output, masks)
        total_loss = seg_loss + self.vq_loss_weight * vq_loss
        
        # Log metrics
        self.log('train/seg_loss', seg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/vq_loss', vq_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        # Forward pass
        output, vq_loss = self(images)
        
        # Compute losses
        seg_loss = self.seg_criterion(output, masks)
        total_loss = seg_loss + self.vq_loss_weight * vq_loss
        
        # Compute IoU (simplified pixel accuracy)
        preds = output.argmax(dim=1)
        valid = masks != 255
        pixel_acc = (preds[valid] == masks[valid]).float().mean()
        
        # Log metrics
        self.log('val/seg_loss', seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/vq_loss', vq_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/pixel_acc', pixel_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        params = list(self.backbone.parameters()) + list(self.vq.parameters())
        return torch.optim.Adam(params, lr=self.learning_rate)

    def predict_with_vq(self, images):
        """
        Predict with VQ compression applied.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            predictions: Segmentation predictions [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            output, _ = self(images)
            predictions = output.argmax(dim=1)
        return predictions

    def predict_without_vq(self, images):
        """
        Predict without VQ compression (original backbone).
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            predictions: Segmentation predictions [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            output = self.backbone(images)
            predictions = output['out'].argmax(dim=1)
        return predictions
