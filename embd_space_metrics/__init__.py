"""
Embedding Space Metrics Evaluation Module

This module provides tools for evaluating similarity between original and quantized
feature representations using various metrics (CKA, PWCCA, Geometry Score, RSA, etc.)
"""

__version__ = "0.1.0"

from .metrics import METRIC_REGISTRY
from .evaluation import MetricsEvaluator

__all__ = [
    "METRIC_REGISTRY",
    "MetricsEvaluator",
]

