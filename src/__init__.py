"""
embeddings_squeeze: Vector Quantization for Segmentation Model Compression
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .models.quantizers import (
    VQWithProjection,
    FSQWithProjection,
    LFQWithProjection,
    ResidualVQWithProjection,
    BaseQuantizer,
)
from .models.backbones.base import SegmentationBackbone
from .models.lightning_module import VQSqueezeModule
from .models.baseline_module import BaselineSegmentationModule
from .data.base import BaseDataModule
from .loggers import setup_clearml, ClearMLLogger

__all__ = [
    "VQWithProjection",
    "FSQWithProjection",
    "LFQWithProjection",
    "ResidualVQWithProjection",
    "BaseQuantizer",
    "SegmentationBackbone", 
    "VQSqueezeModule",
    "BaselineSegmentationModule",
    "BaseDataModule",
    "setup_clearml",
    "ClearMLLogger",
]
