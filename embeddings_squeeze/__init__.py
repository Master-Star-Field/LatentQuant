"""
embeddings_squeeze: Vector Quantization for Segmentation Model Compression
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .models.vq.quantizer import VectorQuantizer
from .models.backbones.base import SegmentationBackbone
from .models.lightning_module import VQSqueezeModule
from .data.base import BaseDataModule

__all__ = [
    "VectorQuantizer",
    "SegmentationBackbone", 
    "VQSqueezeModule",
    "BaseDataModule",
]
