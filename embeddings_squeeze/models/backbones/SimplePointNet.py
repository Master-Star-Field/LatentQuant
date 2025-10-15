import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class SimplePointNet(nn.Module):
    """
    Simple PointNet-like model for point cloud segmentation.
    Works with S3DIS data format: {'point': [B, N, 3], 'color': [B, N, 3], 'label': [B, N]}
    """
    
    def __init__(self, num_classes: int = 19, feature_dim: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Input features: 3 (xyz) + 3 (rgb) = 6
        self.input_dim = 6
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(self.input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Dict with keys 'point' [B, N, 3] and 'color' [B, N, 3]
            
        Returns:
            Dict with key 'out' containing logits [B, N, num_classes]
        """
        if isinstance(x, dict) and 'point' in x and 'color' in x:
            points = x['point']  # [B, N, 3]
            colors = x['color']  # [B, N, 3]
            features = torch.cat([points, colors], dim=-1)  # [B, N, 6]
        else:
            # Fallback: assume x is already features
            features = x if isinstance(x, torch.Tensor) else x['features']
        
        # Transpose to [B, C, N] for conv1d
        x = features.transpose(1, 2)  # [B, 6, N]
        
        # Shared MLP layers
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, N]
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 512, N]
        
        # Global max pooling
        x = torch.max(x, dim=2, keepdim=True)[0]  # [B, 512, 1]
        x = x.squeeze(-1)  # [B, 512]
        
        # Classification
        logits = self.classifier(x)  # [B, num_classes]
        
        # Expand to per-point predictions (broadcast)
        logits = logits.unsqueeze(1).expand(-1, features.shape[1], -1)  # [B, N, num_classes]
        
        return {"out": logits}
    
    def extract_features(self, x: Dict[str, torch.Tensor], detach: bool = False) -> torch.Tensor:
        """
        Extract features for quantization.
        
        Args:
            x: Input data
            detach: Whether to detach gradients
            
        Returns:
            Features tensor [B, N, feature_dim]
        """
        if isinstance(x, dict) and 'point' in x and 'color' in x:
            points = x['point']  # [B, N, 3]
            colors = x['color']  # [B, N, 3]
            features = torch.cat([points, colors], dim=-1)  # [B, N, 6]
        else:
            features = x if isinstance(x, torch.Tensor) else x['features']
        
        # Transpose to [B, C, N] for conv1d
        x = features.transpose(1, 2)  # [B, 6, N]
        
        # Extract features up to the last conv layer
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, N]
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 512, N]
        
        # Transpose back to [B, N, C]
        x = x.transpose(1, 2)  # [B, N, 512]
        
        if detach:
            x = x.detach()
            
        return x


class SimplePointNetSegmentationBackbone(nn.Module):
    """
    Wrapper for SimplePointNet to match the interface expected by the lightning module.
    """
    
    def __init__(self, num_classes: int = 19, feature_dim: int = 512):
        super().__init__()
        self.backbone = SimplePointNet(num_classes, feature_dim)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.backbone(x)
    
    def extract_features(self, x: Dict[str, torch.Tensor], detach: bool = False) -> torch.Tensor:
        return self.backbone.extract_features(x, detach)
