
# ======= ML3D modular structure, Vit-style: classes and separation =======
import torch
import torch.nn as nn
import open3d
from .base import SegmentationBackbone
from ..preprocessing.randlanet_preprocessing import preprocess_for_randlanet

class ML3DBackboneWrapper(nn.Module):
    """
    Thin wrapper for Opend3/Open3D-ML backbone models to standardize feature extraction.
    Supports lazy feature_dim inference.
    """
    def __init__(self, backbone, lazy_feature_dim: bool = True):
        super().__init__()
        self.backbone = backbone
        # Get feature_dim if possible; fallback to -1 if lazy
        self.feature_dim = getattr(backbone, "hidden_dim", -1)
        self._detect_feature_dim = lazy_feature_dim or self.feature_dim < 1

    def forward(self, x):
        # Handle different input formats
        if isinstance(x, dict):
            # S3DIS format: {'point': ..., 'color': ..., 'label': ...}
            if 'point' in x and 'color' in x:
                # Convert to ML3D format
                points = x['point']  # [B, N, 3]
                colors = x['color']  # [B, N, 3]
                
                # Preprocess for RandLANet
                x = preprocess_for_randlanet(points, colors, num_layers=4, k=16)
            elif 'features' in x:
                # Keep as dict with 'features' key
                x = x
            else:
                # Try to find features in other keys
                for key in ['point', 'color', 'data', 'input']:
                    if key in x:
                        x = {'features': x[key], 'coords': x[key][:, :, :3]}  # Use first 3 dims as coords
                        break
        
        # ML3D models take dict with 'features' key
        out = self.backbone(x)
        if isinstance(out, dict):
            feats = out.get("out") or out.get("features") or out.get("feat") or None
            if feats is None:
                # fallback: get first tensor
                feats = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
        else:
            feats = out
        if feats is None or not isinstance(feats, torch.Tensor):
            raise RuntimeError("ML3D backbone should return feature tensor or dict with features/out.")
        if self._detect_feature_dim and self.feature_dim == -1:
            self.feature_dim = feats.shape[1]  # channel dimension
        return {"out": feats}

class ML3DSegmentationHead(nn.Sequential):
    """
    Point cloud segmentation head for ML3D features. Accepts feature_dim, outputs num_classes.
    """
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes)
        )

class ML3DSegmentationBackbone(SegmentationBackbone):
    """
    Top-level segmentation backbone module for ML3D.
    """
    def __init__(
        self,
        config_path: str = None,                # path to yaml config (optional)
        backbone_name: str = None,              # class name in opend3 (will be read from config if None)
        pretrained: bool = False,
        num_classes: int = 21,
        in_channels: int = 1,
        freeze_backbone: bool = True,
        backbone_config: dict = None,           # config from file or dict
        **backbone_kwargs
    ):
        # RandLA-Net
        import open3d.ml as _ml3d
        import open3d.ml.torch as ml3d
        from open3d.ml.torch import models as M

        super().__init__()
        if open3d is None:
            raise ImportError("opend3 must be installed to use ML3DSegmentationBackbone.")
        if config_path is None:
            raise ValueError("config_path must be provided for ML3DSegmentationBackbone.")
        cfg = _ml3d.utils.Config.load_from_file(config_path)
        if not hasattr(cfg, "model"):
            raise RuntimeError("Loaded config missing required 'model' attribute.")
        
        # Get backbone name from config if not provided
        if backbone_name is None:
            backbone_name = cfg.model.get("name", "RandLANet")
        
        # Update num_classes from config if available
        if hasattr(cfg.model, "num_classes"):
            num_classes = cfg.model.num_classes
        
        # Update in_channels from config if available
        if hasattr(cfg.model, "in_channels"):
            in_channels = cfg.model.in_channels
        
        # Get backbone class
        backbone_cls = getattr(M, backbone_name, None)
        print(f"Looking for backbone: {backbone_name}")
        if backbone_cls is None:
            # Try alternative names
            alt_names = [backbone_name, backbone_name.replace("-", "_"), backbone_name.replace("_", "-")]
            for alt_name in alt_names:
                backbone_cls = getattr(M, alt_name, None)
                if backbone_cls is not None:
                    print(f"Found backbone with alternative name: {alt_name}")
                    break
            if backbone_cls is None:
                available = [attr for attr in dir(M) if not attr.startswith('_')]
                raise ValueError(f"Open3D backbone class '{backbone_name}' not found in opend3. Available: {available}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create the backbone as in: backbone = M.RandLANet(**cfg.model).to(device)
        backbone = backbone_cls(**cfg.model)
        backbone = backbone.to(device)
        # Set device attribute for RandLANet
        if hasattr(backbone, 'device'):
            backbone.device = device
        else:
            # Add device attribute if it doesn't exist
            backbone.device = device
        
        # Ensure all parameters are on the correct device
        for param in backbone.parameters():
            param.data = param.data.to(device)
        for buffer in backbone.buffers():
            buffer.data = buffer.data.to(device)
        self.backbone = ML3DBackboneWrapper(backbone)
        feature_dim = self.backbone.feature_dim
        self._feature_dim = feature_dim
        # Head - lazy if feature_dim undetermined
        if feature_dim == -1:
            self.head = None
        else:
            self.head = ML3DSegmentationHead(feature_dim, num_classes)
        self._num_classes = num_classes
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            
    def extract_features(self, x, detach=True):
        feats = self.backbone(x)["out"]
        # If head not yet initialized, do it now
        if (self.head is None) and feats is not None:
            # Use the last dimension as feature dimension
            self._feature_dim = feats.shape[-1]
            # Only create head if we don't already have logits
            if feats.shape[-1] != self._num_classes:
                self.head = ML3DSegmentationHead(self._feature_dim, self._num_classes)
        return feats.detach() if detach else feats

    def forward(self, x):
        feats = self.extract_features(x, detach=False)
        if self.head is None:
            # Check if feats are already logits (num_classes dimension)
            if feats.shape[-1] == self._num_classes:
                # Already logits, return as is
                return feats
            else:
                # Need to create head for features
                self.head = ML3DSegmentationHead(feats.shape[-1], self._num_classes)
        
        if self.head is not None:
            logits = self.head(feats)
        else:
            # Already logits
            logits = feats
            
        return logits

    @property
    def feature_dim(self):
        """Return feature dimension."""
        return self._feature_dim

    @property
    def num_classes(self):
        """Return number of segmentation classes."""
        return self._num_classes


