import os
import glob
from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchmetrics import JaccardIndex, Accuracy, Precision, Recall, F1Score
import clearml
from clearml import Task
import plotly.graph_objects as go

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(pred, dim=1)
        C = prob.shape[1]
        t_onehot = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()
        dice = 0.0
        for c in range(C):
            p = prob[:, c]
            t = t_onehot[:, c]
            inter = (p * t).sum()
            denom = p.sum() + t.sum()
            dice += 1.0 - ((2.0 * inter + self.smooth) / (denom + self.smooth))
        return dice / float(C)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target.long(), reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=0.5, class_weights=None):
        super().__init__()
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        self.focal_w = focal_weight
        self._class_weights = class_weights
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if self._class_weights is not None:
            if not isinstance(self._class_weights, torch.Tensor):
                w = torch.tensor(self._class_weights, dtype=torch.float32, device=pred.device)
            else:
                w = self._class_weights.to(pred.device)
            ce = F.cross_entropy(pred, target.long(), weight=w)
        else:
            ce = F.cross_entropy(pred, target.long())
        dice_val = self.dice(pred, target)
        focal_val = self.focal(pred, target)
        total = self.ce_w * ce + self.dice_w * dice_val + self.focal_w * focal_val
        return total, ce, dice_val, focal_val

class DeepLabV3_ResNet50(LightningModule):
    def __init__(self, lr: float = 1e-3, num_classes: int = 3, pretrained: bool = True, quantizer: Optional[Any] = None,
                 add_adapter: bool = True, feature_dim: int = 2048, loss_type: str = "combined", class_weights: Optional[Any] = None,
                 clearml_logger=None):
        super().__init__()
        self.save_hyperparameters(ignore=['quantizer', 'clearml_logger'])
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
        seg_model = deeplabv3_resnet50(weights=weights)
        self.encoder = self._make_encoder(seg_model.backbone, feature_dim, add_adapter)
        self.decoder = self._make_decoder(seg_model.classifier, feature_dim, add_adapter)
        self.decoder.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.quantizer = quantizer
        self.loss_type = loss_type
        self._class_weights = class_weights
        if loss_type == "ce":
            w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
            self.criterion = nn.CrossEntropyLoss(weight=w)
        elif loss_type == "dice":
            self.criterion = DiceLoss()
        elif loss_type == "focal":
            self.criterion = FocalLoss()
        else:
            self.criterion = CombinedLoss(class_weights=class_weights)
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
        self.epoch_stats: Dict[str, list] = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
                                             "train_precision": [], "val_precision": [], "train_recall": [], "val_recall": [],
                                             "train_f1": [], "val_f1": []}
        self.clearml_logger = clearml_logger
        self._first_val_batch_features = None
        self.embedding_dir = "embeddings"
        os.makedirs(self.embedding_dir, exist_ok=True)

    def _make_encoder(self, backbone, feature_dim, add_adapter: bool):
        for p in backbone.parameters():
            p.requires_grad = False
        class Encoder(nn.Module):
            def __init__(self, backbone, feature_dim, add_adapter):
                super().__init__()
                self.backbone = backbone
                self.add_adapter = add_adapter
                if add_adapter:
                    self.adapter = nn.Sequential(
                        nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(feature_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
                    )
                    try:
                        nn.init.zeros_(self.adapter[3].weight)
                        nn.init.zeros_(self.adapter[3].bias)
                    except Exception:
                        pass
                else:
                    self.adapter = None
            def forward(self, x):
                feats = self.backbone(x)["out"]
                if self.adapter is not None:
                    feats = feats + self.adapter(feats)
                return feats
        return Encoder(backbone, feature_dim, add_adapter)

    def _make_decoder(self, classifier, feature_dim, add_adapter: bool):
        for p in classifier.parameters():
            p.requires_grad = False
        class Decoder(nn.Module):
            def __init__(self, classifier, feature_dim, add_adapter):
                super().__init__()
                self.classifier = classifier
                self.add_adapter = add_adapter
                if add_adapter:
                    self.adapter = nn.Sequential(
                        nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(feature_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
                    )
                    try:
                        nn.init.zeros_(self.adapter[3].weight)
                        nn.init.zeros_(self.adapter[3].bias)
                    except Exception:
                        pass
                else:
                    self.adapter = None
            def forward(self, latents, size):
                if self.adapter is not None:
                    latents = latents + self.adapter(latents)
                out = self.classifier(latents)
                return F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return Decoder(classifier, feature_dim, add_adapter)

    def forward(self, x):
        feats = self.encoder(x)
        quant_loss = 0.0
        if self.quantizer is not None:
            feats, quant_loss = self.quantizer.quantize_spatial(feats)
        logits = self.decoder(feats, x.shape[-2:])
        return logits, quant_loss, feats

    def _compute_seg_loss(self, logits, target, quant_loss):
        if isinstance(self.criterion, CombinedLoss):
            total, ce, dice, focal = self.criterion(logits, target)
            seg_loss = total
            return seg_loss + (quant_loss if isinstance(quant_loss, torch.Tensor) else 0.0), {"total": total, "ce": ce, "dice": dice, "focal": focal}
        else:
            seg = self.criterion(logits, target)
            return seg + (quant_loss if isinstance(quant_loss, torch.Tensor) else 0.0), {"total": seg}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)
        logits, qloss, _ = self(x)
        loss, _ = self._compute_seg_loss(logits, y, qloss)
        iou = self.train_iou(logits, y)
        prec = self.train_prec(logits, y)
        rec = self.train_rec(logits, y)
        f1 = self.train_f1(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_iou", iou, on_step=False, on_epoch=True)
        self.log("train_precision", prec, on_step=False, on_epoch=True)
        self.log("train_recall", rec, on_step=False, on_epoch=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)
        logits, qloss, feats = self(x)
        loss, _ = self._compute_seg_loss(logits, y, qloss)
        iou = self.val_iou(logits, y)
        prec = self.val_prec(logits, y)
        rec = self.val_rec(logits, y)
        f1 = self.val_f1(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True)
        self.log("val_precision", prec, on_step=False, on_epoch=True)
        self.log("val_recall", rec, on_step=False, on_epoch=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._first_val_batch_features = feats.detach().cpu()
        return loss
    
    def on_validation_epoch_end(self):
        cm = self.trainer.callback_metrics
        def push_if_exists(k_from, k_to):
            if k_from in cm:
                val = cm[k_from]
                try:
                    v = float(val)
                except Exception:
                    v = val.item()
                self.epoch_stats[k_to].append(v)
        keys = ["train_loss","val_loss","train_iou","val_iou","train_precision","val_precision","train_recall","val_recall","train_f1","val_f1"]
        for k in keys:
            push_if_exists(k, k)
        try:
            epoch = self.current_epoch
            epochs = list(range(len(self.epoch_stats["val_loss"])))
            fig_loss = go.Figure()
            if len(self.epoch_stats["train_loss"]) > 0:
                fig_loss.add_trace(go.Scatter(x=epochs, y=self.epoch_stats["train_loss"], mode="lines+markers", name="train_loss"))
            if len(self.epoch_stats["val_loss"]) > 0:
                fig_loss.add_trace(go.Scatter(x=epochs, y=self.epoch_stats["val_loss"], mode="lines+markers", name="val_loss"))
            fig_loss.update_layout(title="Loss", xaxis_title="epoch", yaxis_title="loss")
            if self.clearml_logger:
                self.clearml_logger.report_plotly(title="Loss", series="loss", iteration=epoch, figure=fig_loss)
            fig_m = go.Figure()
            metrics_to_plot = [("train_iou", "val_iou"), ("train_precision", "val_precision"), ("train_recall", "val_recall"), ("train_f1", "val_f1")]
            for train_k, val_k in metrics_to_plot:
                if len(self.epoch_stats[train_k]) > 0:
                    fig_m.add_trace(go.Scatter(x=epochs, y=self.epoch_stats[train_k], mode="lines+markers", name=train_k))
                if len(self.epoch_stats[val_k]) > 0:
                    fig_m.add_trace(go.Scatter(x=epochs, y=self.epoch_stats[val_k], mode="lines+markers", name=val_k))
            fig_m.update_layout(title="Metrics", xaxis_title="epoch", yaxis_title="value")
            if self.clearml_logger:
                self.clearml_logger.report_plotly(title="Metrics", series="metrics", iteration=epoch, figure=fig_m)
        except Exception as e:
            if self.clearml_logger:
                self.clearml_logger.report_text(f"Plotly reporting failed at epoch {self.current_epoch}: {e}")
        try:
            if self._first_val_batch_features is not None:
                emb_path = os.path.join(self.embedding_dir, f"val_embedding_epoch{self.current_epoch}.pt")
                torch.save(self._first_val_batch_features, emb_path)
                if self.clearml_logger:
                    self.clearml_logger.report_text(f"Saved small embedding: {emb_path}")
                self._first_val_batch_features = None
        except Exception as e:
            if self.clearml_logger:
                self.clearml_logger.report_text(f"Failed saving epoch embedding: {e}")

    def configure_optimizers(self):
        for p in self.parameters():
            p.requires_grad = False

        trainable_params = []

        if hasattr(self.encoder, "adapter") and self.encoder.adapter is not None:
            for p in self.encoder.adapter.parameters():
                p.requires_grad = True
                trainable_params.append(p)

        if hasattr(self.decoder, "adapter") and self.decoder.adapter is not None:
            for p in self.decoder.adapter.parameters():
                p.requires_grad = True
                trainable_params.append(p)

        if hasattr(self.decoder, "classifier"):
            for name, p in self.decoder.classifier.named_parameters():
                if name == "4.weight" or name == "4.bias":  
                    p.requires_grad = True
                    trainable_params.append(p)

        # Отключаем квантизатор, если нужно
        # if self.quantizer is not None:
        #     for p in self.quantizer.parameters():
        #         p.requires_grad = True
        #         trainable_params.append(p)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
        print(f"Frozen params: {frozen:,} ({frozen/total*100:.2f}%)")

        return torch.optim.AdamW(trainable_params, lr=self.hparams.lr)


class ClearMLUploadCallback(Callback):
    def __init__(self, task: Task, checkpoint_dir: str = "tb_logs", ckpt_glob: str = "**/*.ckpt"):
        super().__init__()
        self.task = task
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_glob = ckpt_glob

    def _find_latest_ckpt(self) -> Optional[str]:
        matches = glob.glob(os.path.join(self.checkpoint_dir, "**", "*.ckpt"), recursive=True)
        if not matches:
            return None
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        try:
            ckpt = self._find_latest_ckpt()
            if ckpt:
                self.task.upload_artifact(name=f"checkpoint_epoch{pl_module.current_epoch}", artifact_object=ckpt)
                pl_module.clearml_logger.report_text(f"Uploaded checkpoint: {ckpt}")
        except Exception as e:
            try:
                pl_module.clearml_logger.report_text(f"Failed uploading checkpoint: {e}")
            except Exception:
                pass
        try:
            emb_path = os.path.join(pl_module.embedding_dir, f"val_embedding_epoch{pl_module.current_epoch}.pt")
            if os.path.exists(emb_path):
                self.task.upload_artifact(name=f"val_embedding_epoch{pl_module.current_epoch}", artifact_object=emb_path)
                pl_module.clearml_logger.report_text(f"Uploaded embedding: {emb_path}")
                try:
                    os.remove(emb_path)
                except Exception:
                    pass
        except Exception as e:
            try:
                pl_module.clearml_logger.report_text(f"Failed uploading embedding: {e}")
            except Exception:
                pass
