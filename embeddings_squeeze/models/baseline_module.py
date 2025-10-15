"""
PyTorch Lightning module for baseline segmentation training without VQ.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from torchmetrics import JaccardIndex, Accuracy, Precision, Recall, F1Score

from .backbones.base import SegmentationBackbone


class BaselineSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for baseline segmentation training.
    
    Wraps segmentation backbone without Vector Quantization for comparison.
    """
    
    def __init__(
        self,
        backbone: SegmentationBackbone,
        num_classes: int = 21,
        learning_rate: float = 1e-4,
        loss_type: str = 'ce',
        class_weights: Optional[list] = None,
        clearml_logger: Optional[Any] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'clearml_logger'])
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Segmentation loss
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
            self.seg_criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        else:
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
        
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
        
        # Compute metrics
        iou = self.train_iou(output, masks)
        acc = self.train_acc(output, masks)
        prec = self.train_prec(output, masks)
        rec = self.train_rec(output, masks)
        f1 = self.train_f1(output, masks)
        
        # Log metrics
        # Step-wise loss goes to a separate ClearML plot
        self.log('train_step/loss', seg_loss, on_step=True, on_epoch=False, prog_bar=False)
        # Epoch loss stays with other train metrics
        self.log('train/loss', seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True)
        
        return seg_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        # Forward pass
        output = self(images)
        
        # Compute loss
        seg_loss = self.seg_criterion(output, masks)
        
        # Compute metrics
        iou = self.val_iou(output, masks)
        acc = self.val_acc(output, masks)
        prec = self.val_prec(output, masks)
        rec = self.val_rec(output, masks)
        f1 = self.val_f1(output, masks)
        
        # Log metrics
        self.log('val/loss', seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', prec, on_step=False, on_epoch=True)
        self.log('val/recall', rec, on_step=False, on_epoch=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True)
        
        return seg_loss
    
    def on_validation_epoch_end(self):
        """Called after validation epoch ends - log Plotly visualizations."""
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
