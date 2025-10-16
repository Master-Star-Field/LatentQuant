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
        # Get device from the first tensor in images dict
        if isinstance(images, dict):
            first_tensor = next(iter(images.values()))
            if isinstance(first_tensor, torch.Tensor):
                device = first_tensor.device
            else:
                device = torch.device('cpu')
        else:
            device = images.device
        quant_loss = torch.tensor(0.0, device=device)
        if self.quantizer is not None:
            features, quant_loss = self.quantizer.quantize_spatial(features)
        
        # Decode to segmentation logits
        if hasattr(self.backbone, 'classifier'):
            output = self.backbone.classifier(features)
            output = F.interpolate(output, size=images.shape[-2:], mode='bilinear', align_corners=False)
        else:
            # For ML3D backbones, use the forward method directly
            output = self.backbone(images)
            if isinstance(output, dict):
                output = output.get("out", output.get("logits", features))
        
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
            # Reshape for loss computation
            if pred.dim() == 4:  # [B, C, H, W]
                pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1, pred.size(1))
                target = target.reshape(-1)
            elif pred.dim() == 3:  # [B, N, C] for point clouds
                pred = pred.reshape(-1, pred.size(-1))  # [B*N, C]
                target = target.reshape(-1)  # [B*N]
            elif pred.dim() == 2:  # [B, C] - already flattened
                pred = pred.reshape(-1, pred.size(-1))
                target = target.reshape(-1)
            
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
        # Handle different batch formats
        if isinstance(batch, dict):
            # S3DIS format: {'point': ..., 'color': ..., 'label': ...}
            if 'point' in batch and 'label' in batch:
                # Keep the full dict for ML3D backbones
                images = {'point': batch['point'], 'color': batch['color']}
                masks = batch['label']   # segmentation labels
            else:
                # Other dict formats
                images = batch.get('image', batch.get('data', batch.get('input')))
                masks = batch.get('mask', batch.get('label', batch.get('target')))
        else:
            # Tuple format: (images, masks)
            images, masks = batch
        
        # Handle mask dimensions
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        masks = masks.long()
        
        # Forward pass
        output, quant_loss, _ = self(images)
        
        # Compute loss
        loss = self._compute_loss(output, masks, quant_loss)
        
        # Compute metrics - reshape for point cloud data
        if output.dim() == 3:  # [B, N, C] for point clouds
            output_flat = output.reshape(-1, output.size(-1))  # [B*N, C]
            masks_flat = masks.reshape(-1)  # [B*N]
        else:
            output_flat = output
            masks_flat = masks
            
        iou = self.train_iou(output_flat, masks_flat)
        acc = self.train_acc(output_flat, masks_flat)
        prec = self.train_prec(output_flat, masks_flat)
        rec = self.train_rec(output_flat, masks_flat)
        f1 = self.train_f1(output_flat, masks_flat)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/precision', prec, on_step=False, on_epoch=True)
        self.log('train/recall', rec, on_step=False, on_epoch=True)
        self.log('train/f1', f1, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        print("STEP")
        """
        Универсальная валидация:
        - Images: (images, masks) или {'image':..., 'mask':...}
        - S3DIS:  {'point':[B,N,3], 'color':[B,N,3], 'label':[B,N]}
        Ожидаемый forward:
        self(...) -> либо (logits, vq_loss[, features]), либо {'out', 'vq_loss'[, 'features']}
        """

        # ---------- 1) Распаковка входа и GT ----------
        if isinstance(batch, tuple) or isinstance(batch, list):
            # классический image case: (images, masks)
            images, masks = batch
            labels = masks
            # приведение mask к [B,H,W]
            if labels.dim() == 4 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            labels = labels.long()
            fwd_input = images
        elif isinstance(batch, dict):
            if "label" in batch and ("point" in batch or "color" in batch):
                # S3DIS point-cloud case
                labels = batch["label"].long()            # [B,N]
                # Keep only point and color for ML3D backbones
                fwd_input = {'point': batch['point'], 'color': batch['color']}
            elif "mask" in batch:
                # возможно dict для картинок
                labels = batch["mask"].long()
                if labels.dim() == 4 and labels.size(1) == 1:
                    labels = labels.squeeze(1)
                fwd_input = batch.get("image", batch)      # поддержка {'image','mask'} или совместимость
            else:
                raise RuntimeError("Unknown batch format for validation_step(dict).")
        else:
            raise RuntimeError(f"Unsupported batch type: {type(batch)}")

        B = labels.size(0)

        # ---------- 2) Forward ----------
        out = self(fwd_input)
        features = None
        quant_loss = torch.tensor(0.0, device=labels.device)

        if isinstance(out, dict):
            raw_logits = out.get("out")
            quant_loss = out.get("vq_loss", quant_loss)
            features   = out.get("features", None)
        else:
            if not isinstance(out, (tuple, list)):
                raw_logits = out
            elif len(out) == 2:
                raw_logits, quant_loss = out
            else:
                raw_logits, quant_loss, features = out

        if not isinstance(raw_logits, torch.Tensor):
            raise RuntimeError("Model forward must return logits tensor (or dict with key 'out').")

        # ---------- 3) Loss (используем «сырые» формы, как в твоём _compute_loss) ----------
        loss = self._compute_loss(raw_logits, labels, quant_loss)

        # ---------- 4) Приводим logits для метрик к [B, M, K] ----------
        logits = raw_logits
        # если logits channel-first (класс в dim=1), перенесём класс в конец
        num_classes = getattr(self, "num_classes", None) or getattr(self.hparams, "num_classes", None)
        if logits.dim() >= 3:
            if num_classes is not None and logits.size(-1) == num_classes:
                pass  # уже класс-канал последний
            elif num_classes is not None and logits.size(1) == num_classes:
                logits = logits.movedim(1, -1)  # [B, ..., K]
            else:
                # эвристика: если второй dim маленький (<=256) и остальное «пространство», считаем что это K
                if logits.size(1) <= 256 and (logits.dim() > 3 or logits.size(-1) > logits.size(1)):
                    logits = logits.movedim(1, -1)
                # иначе предполагаем, что класс уже последний

        # теперь приведём к [B, M, K]
        if logits.dim() == 3:
            # уже [B, *, K]
            M, K = logits.size(1), logits.size(2)
            flat_logits = logits
            flat_labels = labels
            # для картинок labels могут быть [B,H,W] — расплющим
            if labels.dim() == 3:
                flat_labels = labels.reshape(B, -1)
                flat_logits = logits.reshape(B, -1, K)
        elif logits.dim() >= 4:
            # [B, C?, H, W] или [B, H, W, K] или 3D-варианты — расплющим всё, кроме B и K
            K = logits.size(-1)
            flat_logits = logits.reshape(B, -1, K)
            flat_labels = labels.reshape(B, -1)
        else:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        # ---------- 5) Метрики на валидных позициях ----------
        ignore_index = getattr(self, "ignore_index", -100)
        valid = flat_labels.ne(ignore_index) if flat_labels.dim() == 2 else torch.ones_like(flat_labels, dtype=torch.bool)

        with torch.no_grad():
            pred = flat_logits.argmax(dim=-1)  # [B,M]
            if valid.any():
                acc = (pred[valid] == flat_labels[valid]).float().mean()
            else:
                acc = torch.tensor(0.0, device=flat_logits.device)

            # IoU / Precision / Recall / F1 (macro)
            K = flat_logits.size(-1)
            ious, precs, recs, f1s = [], [], [], []
            for c in range(K):
                p_c = (pred == c) & valid
                t_c = (flat_labels == c) & valid
                inter = (p_c & t_c).sum().float()
                union = p_c.sum().float() + t_c.sum().float() - inter
                if union > 0:
                    ious.append(inter / union)

                tp = inter
                fp = (p_c & (~t_c)).sum().float()
                fn = ((~p_c) & t_c).sum().float()
                prec_c = tp / (tp + fp) if (tp + fp) > 0 else None
                rec_c  = tp / (tp + fn) if (tp + fn) > 0 else None
                if prec_c is not None and rec_c is not None and (prec_c + rec_c) > 0:
                    f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c)
                    precs.append(prec_c); recs.append(rec_c); f1s.append(f1_c)

            iou = (torch.stack(ious).mean() if ious else torch.tensor(0.0, device=flat_logits.device))
            precision = (torch.stack(precs).mean() if precs else torch.tensor(0.0, device=flat_logits.device))
            recall    = (torch.stack(recs).mean()  if recs  else torch.tensor(0.0, device=flat_logits.device))
            f1        = (torch.stack(f1s).mean()   if f1s   else torch.tensor(0.0, device=flat_logits.device))

        # ---------- 6) Логгинг ----------
        self.log('val/loss',      loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/iou',       iou,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc',       acc,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', precision, on_step=False, on_epoch=True)
        self.log('val/recall',    recall,    on_step=False, on_epoch=True)
        self.log('val/f1',        f1,        on_step=False, on_epoch=True)

        # ---------- 7) Сохранить features с первого батча (если есть) ----------
        if batch_idx == 0:
            to_store = features if features is not None else flat_logits.detach()
            self._first_val_batch_features = to_store.detach().cpu()

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
