"""
PyTorch Lightning module for VQ compression training with advanced features.
Supports multiple quantizers (VQ, FSQ, LFQ, RVQ), adapters, and loss functions.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Optional
from torchmetrics import JaccardIndex, Accuracy, Precision, Recall, F1Score

from .backbones.base import SegmentationBackbone
from .losses import DiceLoss, FocalLoss, CombinedLoss


class VQSqueezeModule(pl.LightningModule):
    """
    PyTorch Lightning module for VQ compression training.
    
    Features:
    - Multiple quantizer support (VQ, FSQ, LFQ, RVQ)
    - Fine-tuning support with last layer unfreezing option
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
        unfreeze_last_layer: bool = False,
        clearml_logger: Optional[Any] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'quantizer', 'clearml_logger'])
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.vq_loss_weight = vq_loss_weight
        self.loss_type = loss_type
        self.unfreeze_last_layer = unfreeze_last_layer
        
        # Setup backbone with optional last layer unfreezing
        self.backbone = backbone
        self._setup_backbone_freezing(unfreeze_last_layer)
        
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
        
        # UMAP visualization storage
        self._val_backbone_embeddings = []
        self._val_quantized_embeddings = []
        
        # Perplexity tracking
        self._train_indices = []
        self._val_indices = []
    
    def _setup_backbone_freezing(self, unfreeze_last_layer: bool):
        """Setup backbone freezing with optional last layer unfreezing."""
        if unfreeze_last_layer:
            # Freeze entire backbone first
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze the last layer based on backbone type
            # Check if backbone has a 'backbone' attribute (DeepLabV3)
            if hasattr(self.backbone, 'backbone'):
                # DeepLabV3 with ResNet backbone
                if hasattr(self.backbone.backbone, 'layer4'):
                    # Unfreeze last ResNet layer (layer4)
                    for param in self.backbone.backbone.layer4.parameters():
                        param.requires_grad = True
                    trainable_params = sum(p.numel() for p in self.backbone.backbone.layer4.parameters())
                    print(f"Unfroze last layer (layer4) with {trainable_params:,} parameters")
                else:
                    print("Warning: Could not identify last layer in backbone")
            # Check if it's a ViT backbone
            elif hasattr(self.backbone, 'backbone') and hasattr(self.backbone.backbone, 'vit'):
                # ViT backbone - unfreeze last encoder block
                if hasattr(self.backbone.backbone.vit.encoder, 'layers'):
                    last_block = self.backbone.backbone.vit.encoder.layers[-1]
                    for param in last_block.parameters():
                        param.requires_grad = True
                    trainable_params = sum(p.numel() for p in last_block.parameters())
                    print(f"Unfroze last transformer block with {trainable_params:,} parameters")
                else:
                    print("Warning: Could not identify last layer in ViT backbone")
            else:
                print("Warning: Unknown backbone type, could not unfreeze last layer")
    
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
            original_features: Extracted features (before quantization)
            quantized_features: Features after quantization (same as original if no quantizer)
            indices: Quantization indices for perplexity calculation (None if no quantizer)
        """
        # Extract features (detach only if backbone is fully frozen)
        features = self.backbone.extract_features(images, detach=False)
        
        # Store original features for embedding extraction
        original_features = features
        
        # Quantize if quantizer is present
        quant_loss = torch.tensor(0.0, device=images.device)
        quantized_features = original_features  # Default to original if no quantizer
        indices = None
        if self.quantizer is not None:
            features, quant_loss, indices = self.quantizer.quantize_spatial(features)
            quantized_features = features
        
        # Decode to segmentation logits
        output = self.backbone.classifier(features)
        output = F.interpolate(output, size=images.shape[-2:], mode='bilinear', align_corners=False)
        
        return output, quant_loss, original_features, quantized_features, indices
    
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
        output, quant_loss, _, _, indices = self(images)
        
        # Store indices for perplexity calculation
        if indices is not None:
            self._train_indices.append(indices.detach().cpu())
        
        # Compute loss
        loss = self._compute_loss(output, masks, quant_loss)
        
        # Compute metrics
        iou = self.train_iou(output, masks)
        acc = self.train_acc(output, masks)
        prec = self.train_prec(output, masks)
        rec = self.train_rec(output, masks)
        f1 = self.train_f1(output, masks)
        
        # Log metrics
        self.log('train_step/loss', loss, on_step=True, on_epoch=False, prog_bar=False)

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
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
        output, quant_loss, backbone_features, quantized_features, indices = self(images)
        
        # Store indices for perplexity calculation
        if indices is not None:
            self._val_indices.append(indices.detach().cpu())
        
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
        
        # Accumulate embeddings for UMAP visualization
        self._val_backbone_embeddings.append(backbone_features.detach().cpu())
        self._val_quantized_embeddings.append(quantized_features.detach().cpu())
        
        # Save only first batch features for this epoch
        if batch_idx == 0:
            self._first_val_batch_features = backbone_features.detach().cpu()
        
        return loss
    
    def on_train_epoch_end(self):
        """Calculate and log perplexity at end of training epoch."""
        if self.quantizer is not None and len(self._train_indices) > 0:
            self._calculate_and_log_perplexity('train')
            self._train_indices.clear()
    
    def on_validation_epoch_start(self):
        """Clear accumulated embeddings at the start of each validation epoch."""
        self._val_backbone_embeddings.clear()
        self._val_quantized_embeddings.clear()
    
    def on_validation_epoch_end(self):
        """Called after validation epoch ends - log Plotly visualizations and save embeddings."""
        # Calculate and log perplexity if quantizer is present
        if self.quantizer is not None and len(self._val_indices) > 0:
            self._calculate_and_log_perplexity('val')
            self._val_indices.clear()
        
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
        
        # Generate UMAP visualizations on even epochs
        if self.current_epoch % 2 == 0:
            try:
                import umap.umap_ as umap_module
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Only proceed if we have embeddings
                if len(self._val_backbone_embeddings) > 0 and len(self._val_quantized_embeddings) > 0:
                    # Concatenate all accumulated embeddings
                    backbone_emb_flat = torch.cat(self._val_backbone_embeddings, dim=0)
                    quantized_emb_flat = torch.cat(self._val_quantized_embeddings, dim=0)
                    
                    # Flatten spatial dimensions: [B, C, H, W] -> [B*H*W, C]
                    backbone_emb_flat = backbone_emb_flat.permute(0, 2, 3, 1).reshape(-1, backbone_emb_flat.shape[1])
                    quantized_emb_flat = quantized_emb_flat.permute(0, 2, 3, 1).reshape(-1, quantized_emb_flat.shape[1])
                    
                    # Convert to numpy
                    backbone_emb_np = backbone_emb_flat.numpy()
                    quantized_emb_np = quantized_emb_flat.numpy()
                    
                    # Limit samples for performance (take subset if too large)
                    max_samples = 10000
                    if len(backbone_emb_np) > max_samples:
                        indices = np.random.choice(len(backbone_emb_np), max_samples, replace=False)
                        backbone_emb_np = backbone_emb_np[indices]
                        quantized_emb_np = quantized_emb_np[indices]
                    
                    # Generate 2D UMAP projections
                    proj_2d_backbone = umap_module.UMAP(n_neighbors=3, min_dist=0.1, metric='cosine').fit_transform(backbone_emb_np)
                    proj_2d_quantized = umap_module.UMAP(n_neighbors=3, min_dist=0.1, metric='cosine').fit_transform(quantized_emb_np)
                    
                    # Create 2D Plotly figure with subplots
                    fig_2d = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('2D UMAP: Backbone Embeddings', '2D UMAP: Quantized Embeddings')
                    )
                    
                    fig_2d.add_trace(
                        go.Scatter(
                            x=proj_2d_backbone[:, 0], 
                            y=proj_2d_backbone[:, 1],
                            mode='markers',
                            marker=dict(opacity=0.3, size=3),
                            name='Backbone'
                        ),
                        row=1, col=1
                    )
                    
                    fig_2d.add_trace(
                        go.Scatter(
                            x=proj_2d_quantized[:, 0],
                            y=proj_2d_quantized[:, 1],
                            mode='markers',
                            marker=dict(opacity=0.3, size=3),
                            name='Quantized'
                        ),
                        row=1, col=2
                    )
                    
                    fig_2d.update_layout(
                        title_text=f"2D UMAP Embeddings (Epoch {self.current_epoch})",
                        showlegend=False,
                        height=500
                    )
                    
                    if self.clearml_logger:
                        self.clearml_logger.report_plotly(
                            title="UMAP_2D",
                            series=f"epoch_{self.current_epoch}",
                            iteration=self.current_epoch,
                            figure=fig_2d
                        )
                    
                    # Generate 3D UMAP projections
                    proj_3d_backbone = umap_module.UMAP(n_neighbors=3, min_dist=0.1, metric='cosine', n_components=3).fit_transform(backbone_emb_np)
                    proj_3d_quantized = umap_module.UMAP(n_neighbors=3, min_dist=0.1, metric='cosine', n_components=3).fit_transform(quantized_emb_np)
                    
                    # Create 3D Plotly figure with subplots
                    fig_3d = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('3D UMAP: Backbone Embeddings', '3D UMAP: Quantized Embeddings'),
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
                    )
                    
                    fig_3d.add_trace(
                        go.Scatter3d(
                            x=proj_3d_backbone[:, 0],
                            y=proj_3d_backbone[:, 1],
                            z=proj_3d_backbone[:, 2],
                            mode='markers',
                            marker=dict(opacity=0.3, size=2),
                            name='Backbone'
                        ),
                        row=1, col=1
                    )
                    
                    fig_3d.add_trace(
                        go.Scatter3d(
                            x=proj_3d_quantized[:, 0],
                            y=proj_3d_quantized[:, 1],
                            z=proj_3d_quantized[:, 2],
                            mode='markers',
                            marker=dict(opacity=0.3, size=2),
                            name='Quantized'
                        ),
                        row=1, col=2
                    )
                    
                    fig_3d.update_layout(
                        title_text=f"3D UMAP Embeddings (Epoch {self.current_epoch})",
                        showlegend=False,
                        height=500
                    )
                    
                    if self.clearml_logger:
                        self.clearml_logger.report_plotly(
                            title="UMAP_3D",
                            series=f"epoch_{self.current_epoch}",
                            iteration=self.current_epoch,
                            figure=fig_3d
                        )
                
                # Clear accumulated embeddings after logging
                self._val_backbone_embeddings.clear()
                self._val_quantized_embeddings.clear()
                
            except Exception as e:
                if self.clearml_logger:
                    self.clearml_logger.report_text(
                        f"UMAP visualization failed at epoch {self.current_epoch}: {e}"
                    )
                # Clear embeddings even if visualization failed
                self._val_backbone_embeddings.clear()
                self._val_quantized_embeddings.clear()
        
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
    
    def _calculate_and_log_perplexity(self, split: str):
        """Calculate perplexity and norm from accumulated indices."""
        # Get indices list
        indices_list = self._train_indices if split == 'train' else self._val_indices
        if not indices_list:
            return
        
        # Concatenate all indices
        all_indices = torch.cat(indices_list, dim=0)  # [total_tokens]
        
        # Get codebook size
        if hasattr(self.quantizer, 'vq'):
            codebook_size = self.quantizer.vq.codebook_size
        else:
            # For other quantizer types, might need different logic
            return
        
        # Calculate usage distribution
        encodings = F.one_hot(all_indices, num_classes=codebook_size).float().mean(dim=0)
        encodings = encodings / (encodings.sum() + 1e-10)  # Normalize
        
        # Calculate perplexity
        perplexity = torch.exp(-torch.sum(encodings * torch.log(encodings + 1e-10)))
        
        # Calculate norm
        norm = codebook_size / perplexity
        
        # Log to metrics
        self.log(f'{split}/perplexity', perplexity, on_epoch=True)
        self.log(f'{split}/norm', norm, on_epoch=True)
        
        # Log to ClearML if available
        if self.clearml_logger:
            self.clearml_logger.report_scalar(
                title="Codebook Metrics",
                series=f"{split}_perplexity",
                value=perplexity.item(),
                iteration=self.current_epoch
            )
            self.clearml_logger.report_scalar(
                title="Codebook Metrics",
                series=f"{split}_norm",
                value=norm.item(),
                iteration=self.current_epoch
            )

    def configure_optimizers(self):
        """Configure optimizer - only trainable parameters."""
        params = []
        
        # Add backbone parameters that are trainable
        params += [p for p in self.backbone.parameters() if p.requires_grad]
        
        # Add quantizer parameters if present
        if self.quantizer is not None:
            params += list(self.quantizer.parameters())
        
        # Remove duplicates
        params = list({id(p): p for p in params}.values())
        
        if not params:
            raise ValueError("No trainable parameters found!")
        
        return torch.optim.AdamW(params, lr=self.learning_rate)
    
    def on_train_start(self):
        """Set backbone to appropriate mode based on freezing state."""
        if self.unfreeze_last_layer:
            # Set backbone to train mode so unfrozen last layer can be trained
            # But frozen parts will still not update due to requires_grad=False
            self.backbone.train()
