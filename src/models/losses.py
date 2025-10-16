"""
Loss functions for segmentation tasks.
Includes: Cross Entropy, Dice Loss, Focal Loss, and Combined Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, C, H, W] logits
        # target: [B, H, W] long
        prob = F.softmax(pred, dim=1)
        num_classes = prob.shape[1]
        
        # one-hot: [B, H, W] -> [B, H, W, C] -> permute -> [B, C, H, W]
        target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        for c in range(num_classes):
            p = prob[:, c]
            t = target_one_hot[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice_score)
        
        return dice_loss / float(num_classes)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (multi-class via CE per-pixel)"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: logits [B,C,H,W], target: [B,H,W]
        ce = F.cross_entropy(pred, target.long(), reduction='none')  # shape [B,H,W]
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal  # per-pixel


class CombinedLoss(nn.Module):
    """Combined loss: CE + Dice + Focal. Returns (total, ce, dice, focal)."""
    def __init__(
        self, 
        ce_weight: float = 1.0, 
        dice_weight: float = 1.0, 
        focal_weight: float = 0.5, 
        class_weights=None
    ):
        super().__init__()
        # class_weights can be None or a tensor/list
        if class_weights is not None:
            # Leave tensor creation to forward (to place on correct device) but store raw
            self._class_weights = class_weights
        else:
            self._class_weights = None

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # Instantiate component losses
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # Ensure CE has weights on same device
        if self._class_weights is not None:
            if not isinstance(self._class_weights, torch.Tensor):
                w = torch.tensor(self._class_weights, dtype=torch.float32, device=pred.device)
            else:
                w = self._class_weights.to(pred.device)
            ce = F.cross_entropy(pred, target.long(), weight=w)
        else:
            ce = F.cross_entropy(pred, target.long())

        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        total = self.ce_weight * ce + self.dice_weight * dice + self.focal_weight * focal
        return total, ce, dice, focal

