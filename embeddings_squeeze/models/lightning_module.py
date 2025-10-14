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
from torchmetrics import JaccardIndex, Accuracy, Precision, Recall, F1Score

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
        clearml_logger: Optional[Any] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'quantizer', 'clearml_logger'])
        
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
        self.train_prec = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_prec = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.train_rec = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_rec = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        # Epoch-wise stats tracking for Plotly
        self.epoch_stats: Dict[str, list] = {
            "train_loss": [], "val_loss": [], 
            "train_iou": [], "val_iou": [],
            "train_precision": [], "val_precision": [], 
            "train_recall": [], "val_recall": [],
            "train_f1": [], "val_f1": []
        }
        
        # ClearML logger
        self.clearml_logger = clearml_logger
        
        # Embedding storage (per-epoch, first batch only)
        self.embedding_dir = "embeddings"
        os.makedirs(self.embedding_dir, exist_ok=True)
        self._first_val_batch_features = None
    
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
        prec = self.train_prec(output, masks)
        rec = self.train_rec(output, masks)
        f1 = self.train_f1(output, masks)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True)
        
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
        prec = self.val_prec(output, masks)
        rec = self.val_rec(output, masks)
        f1 = self.val_f1(output, masks)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', prec, on_step=False, on_epoch=True)
        self.log('val/recall', rec, on_step=False, on_epoch=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True)
        
        # Save only first batch features for this epoch
        if batch_idx == 0:
            self._first_val_batch_features = features.detach().cpu()
        
        return loss
    
    def on_validation_epoch_end(self):
        """Called after validation epoch ends - log Plotly visualizations and save embeddings."""
        # Collect epoch stats from trainer callback metrics
        cm = self.trainer.callback_metrics
        
        def push_if_exists(k_from, k_to):
            """Helper to extract metrics from callback_metrics."""
            if k_from in cm:
                val = cm[k_from]
                try:
                    v = float(val)
                except Exception:
                    v = val.item()
                self.epoch_stats[k_to].append(v)
        
        # Push metrics to epoch_stats
        keys = [
            "train/loss", "val/loss", "train/iou", "val/iou",
            "train/precision", "val/precision", "train/recall", "val/recall",
            "train/f1", "val/f1"
        ]
        key_mapping = {
            "train/loss": "train_loss", "val/loss": "val_loss",
            "train/iou": "train_iou", "val/iou": "val_iou",
            "train/precision": "train_precision", "val/precision": "val_precision",
            "train/recall": "train_recall", "val/recall": "val_recall",
            "train/f1": "train_f1", "val/f1": "val_f1"
        }
        for k_from, k_to in key_mapping.items():
            push_if_exists(k_from, k_to)
        
        # Generate Plotly visualizations
        try:
            import plotly.graph_objects as go
            
            epoch = self.current_epoch
            epochs = list(range(len(self.epoch_stats["val_loss"])))
            
            # Loss plot
            fig_loss = go.Figure()
            if len(self.epoch_stats["train_loss"]) > 0:
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=self.epoch_stats["train_loss"],
                    mode="lines+markers", name="train_loss"
                ))
            if len(self.epoch_stats["val_loss"]) > 0:
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=self.epoch_stats["val_loss"],
                    mode="lines+markers", name="val_loss"
                ))
            fig_loss.update_layout(title="Loss", xaxis_title="epoch", yaxis_title="loss")
            
            if self.clearml_logger:
                self.clearml_logger.report_plotly(
                    title="Loss", series="loss", iteration=epoch, figure=fig_loss
                )
            
            # Metrics plot
            fig_m = go.Figure()
            metrics_to_plot = [
                ("train_iou", "val_iou"),
                ("train_precision", "val_precision"),
                ("train_recall", "val_recall"),
                ("train_f1", "val_f1")
            ]
            for train_k, val_k in metrics_to_plot:
                if len(self.epoch_stats[train_k]) > 0:
                    fig_m.add_trace(go.Scatter(
                        x=epochs, y=self.epoch_stats[train_k],
                        mode="lines+markers", name=train_k
                    ))
                if len(self.epoch_stats[val_k]) > 0:
                    fig_m.add_trace(go.Scatter(
                        x=epochs, y=self.epoch_stats[val_k],
                        mode="lines+markers", name=val_k
                    ))
            fig_m.update_layout(title="Metrics", xaxis_title="epoch", yaxis_title="value")
            
            if self.clearml_logger:
                self.clearml_logger.report_plotly(
                    title="Metrics", series="metrics", iteration=epoch, figure=fig_m
                )
        except Exception as e:
            if self.clearml_logger:
                self.clearml_logger.report_text(
                    f"Plotly reporting failed at epoch {self.current_epoch}: {e}"
                )
        
        # Save per-epoch embedding (first validation batch only)
        try:
            if self._first_val_batch_features is not None:
                emb_path = os.path.join(
                    self.embedding_dir,
                    f"val_embedding_epoch{self.current_epoch}.pt"
                )
                torch.save(self._first_val_batch_features, emb_path)
                if self.clearml_logger:
                    self.clearml_logger.report_text(f"Saved small embedding: {emb_path}")
                # Reset for next epoch
                self._first_val_batch_features = None
        except Exception as e:
            if self.clearml_logger:
                self.clearml_logger.report_text(f"Failed saving epoch embedding: {e}")

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
