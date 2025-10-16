"""Similarity metrics for comparing feature representations."""

from .registry import METRIC_REGISTRY, create_metric, list_available_metrics
from .base import SimilarityMetric
from .cka import CKAMetric
from .pwcca import PWCCAMetric
from .geometry import GeometryScoreMetric
from .rsa import RSAMetric
from .relational import RelationalKnowledgeLossMetric
from .jaccard import JaccardKNNMetric

__all__ = [
    "METRIC_REGISTRY",
    "create_metric",
    "list_available_metrics",
    "SimilarityMetric",
    "CKAMetric",
    "PWCCAMetric",
    "GeometryScoreMetric",
    "RSAMetric",
    "RelationalKnowledgeLossMetric",
    "JaccardKNNMetric",
]

