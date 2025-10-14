"""
PyTorch Lightning module for VQ compression training with advanced features.
Supports multiple quantizers (VQ, FSQ, LFQ, RVQ), adapters, and loss functions.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from torchmetrics import JaccardIndex, Accuracy

from .backbones.base import SegmentationBackbone
from .losses import DiceLoss, FocalLoss, CombinedLoss


class VQSqueezeModule(pl.LightningModule):
    """
    PyTorch Lightning module for VQ compression training.
    
    Features:
    - Multiple quantizer support (VQ, FSQ, LFQ, RVQ)
    - Adapter layers for fine-tuning frozen backbones
    - Advanced loss functions (CE, Dice, Focal, Combined)
    - Embedding extraction and saving
    """
    
    def __init__(
        self,
        backbone: SegmentationBackbone,
        quantizer: Optional[nn.Module] = None,
        num_classes: int = 21,
        learning_rate: float = 1e-4,
        vq_loss_weight: float = 0.1,
        loss_type: str = 'ce',
        class_weights: Optional[list] = None,
        add_adapter: bool = False,
        feature_dim: int = 2048,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'quantizer'])
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.vq_loss_weight = vq_loss_weight
        self.loss_type = loss_type
        self.add_adapter = add_adapter
        self.feature_dim = feature_dim
        
        # Setup backbone with optional adapters
        self.backbone = backbone
        self._setup_backbone_with_adapters(feature_dim, add_adapter)
        
        # Quantizer (optional)
        self.quantizer = quantizer
        
        # Loss function
        self.criterion = self._init_loss(loss_type, class_weights)
        
        # Metrics
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Embedding storage
        self.embedding_dir = "embeddings"
        os.makedirs(self.embedding_dir, exist_ok=True)
        self.all_val_features = []
    
    def _setup_backbone_with_adapters(self, feature_dim: int, add_adapter: bool):
        """Setup backbone with optional adapter layers."""
        if add_adapter:
            # Freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Add adapter after feature extraction
            self.feature_adapter = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(),
                nn.Conv2d(feature_dim, feature_dim, 1)
            )
            # Zero initialization for residual connection
            nn.init.zeros_(self.feature_adapter[3].weight)
            nn.init.zeros_(self.feature_adapter[3].bias)
        else:
            self.feature_adapter = None
    
    def _init_loss(self, loss_type: str, class_weights: Optional[list]):
        """Initialize loss function based on type."""
        if loss_type == 'ce':
            weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
            return nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        elif loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'focal':
            return FocalLoss()
        elif loss_type == 'combined':
            return CombinedLoss(
                ce_weight=1.0, 
                dice_weight=1.0, 
                focal_weight=0.5, 
                class_weights=class_weights
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, images):
        """
        Forward pass through backbone + optional quantizer + decoder.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            output: Segmentation logits [B, num_classes, H, W]
            quant_loss: Quantization loss (0 if no quantizer)
            features: Extracted features (before quantization)
        """
        # Extract features
        features = self.backbone.extract_features(images, detach=self.feature_adapter is not None)
        
        # Apply adapter if present
        if self.feature_adapter is not None:
            features = features + self.feature_adapter(features)
        
        # Store original features for embedding extraction
        original_features = features
        
        # Quantize if quantizer is present
        quant_loss = torch.tensor(0.0, device=images.device)
        if self.quantizer is not None:
            features, quant_loss = self.quantizer.quantize_spatial(features)
        
        # Decode to segmentation logits
        output = self.backbone.classifier(features)
        output = F.interpolate(output, size=images.shape[-2:], mode='bilinear', align_corners=False)
        
        return output, quant_loss, original_features
    
    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor, quant_loss: torch.Tensor):
        """Compute total loss including segmentation and quantization losses."""
        if self.loss_type == 'combined':
            total, ce, dice, focal = self.criterion(pred, target)
            seg_loss = total
            # Log component losses
            self.log('loss/ce', ce, on_step=False, on_epoch=True)
            self.log('loss/dice', dice, on_step=False, on_epoch=True)
            self.log('loss/focal', focal, on_step=False, on_epoch=True)
        else:
            seg_loss = self.criterion(pred, target)
        
        # Add quantization loss if present
        if isinstance(quant_loss, torch.Tensor) and quant_loss.item() != 0:
            total_loss = seg_loss + self.vq_loss_weight * quant_loss
            self.log('loss/vq', quant_loss, on_step=False, on_epoch=True)
        else:
            total_loss = seg_loss
        
        self.log('loss/seg', seg_loss, on_step=False, on_epoch=True)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, masks = batch
        
        # Handle mask dimensions
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        masks = masks.long()
        
        # Forward pass
        output, quant_loss, _ = self(images)
        
        # Compute loss
        loss = self._compute_loss(output, masks, quant_loss)
        
        # Compute metrics
        iou = self.train_iou(output, masks)
        acc = self.train_acc(output, masks)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch
        
        # Handle mask dimensions
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        masks = masks.long()
        
        # Forward pass
        output, quant_loss, features = self(images)
        
        # Compute loss
        loss = self._compute_loss(output, masks, quant_loss)
        
        # Compute metrics
        iou = self.val_iou(output, masks)
        acc = self.val_acc(output, masks)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Save features for embedding extraction
        self.all_val_features.append(features.detach().cpu())
        
        return loss
    
    def on_train_end(self):
        """Save final embeddings and model weights."""
        # Save model weights
        save_path = os.path.join(self.embedding_dir, "final_model.pth")
        try:
            torch.save(self.state_dict(), save_path)
            print(f"Model weights saved: {save_path}")
        except Exception as e:
            print(f"Failed to save model weights: {e}")
        
        # Save validation embeddings
        if self.all_val_features:
            try:
                all_features = torch.cat(self.all_val_features, dim=0)
                emb_path = os.path.join(self.embedding_dir, "val_embeddings_final.pt")
                torch.save(all_features, emb_path)
                print(f"Saved validation embeddings: {emb_path}")
            except Exception as e:
                print(f"Failed to save embeddings: {e}")

    def configure_optimizers(self):
        """Configure optimizer - only trainable parameters."""
        params = []
        
        # Add adapter parameters if present
        if self.feature_adapter is not None:
            params += list(self.feature_adapter.parameters())
        
        # Add quantizer parameters if present
        if self.quantizer is not None:
            params += list(self.quantizer.parameters())
        
        # Add backbone parameters if not frozen
        if self.feature_adapter is None:
            params += [p for p in self.backbone.parameters() if p.requires_grad]
        
        # Remove duplicates
        params = list({id(p): p for p in params}.values())
        
        if not params:
            raise ValueError("No trainable parameters found!")
        
        return torch.optim.AdamW(params, lr=self.learning_rate)
    
    def on_train_start(self):
        """Ensure frozen backbone stays in eval mode."""
        if self.feature_adapter is not None:
            self.backbone.eval()
