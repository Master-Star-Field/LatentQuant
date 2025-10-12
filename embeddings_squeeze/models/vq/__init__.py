"""Vector Quantization components."""

from .codebook import Codebook, DistanceMetric
from .quantizer import VectorQuantizer, VectorEncoder, QuantizationLayer, StraightThroughEstimator
from .losses import VQLoss

__all__ = [
    "Codebook",
    "DistanceMetric", 
    "VectorQuantizer",
    "VectorEncoder",
    "QuantizationLayer", 
    "StraightThroughEstimator",
    "VQLoss",
]
