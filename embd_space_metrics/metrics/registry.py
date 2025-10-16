"""Metric registry for easy metric creation."""

from .cka import CKAMetric
from .pwcca import PWCCAMetric
from .geometry import GeometryScoreMetric
from .rsa import RSAMetric
from .relational import RelationalKnowledgeLossMetric
from .jaccard import JaccardKNNMetric


# Global registry of available metrics
METRIC_REGISTRY = {
    'cka': CKAMetric,
    'pwcca': PWCCAMetric,
    'geometry': GeometryScoreMetric,
    'rsa': RSAMetric,
    'relational': RelationalKnowledgeLossMetric,
    'jaccard_knn': JaccardKNNMetric,
}


def create_metric(metric_name, device=None, **kwargs):
    """
    Create a metric by name.
    
    Args:
        metric_name: Name of the metric (e.g., 'cka', 'pwcca', etc.)
        device: Torch device for computation
        **kwargs: Additional metric-specific arguments
        
    Returns:
        SimilarityMetric: Instance of the requested metric
        
    Raises:
        ValueError: If metric_name is not recognized
    """
    if metric_name not in METRIC_REGISTRY:
        available = ', '.join(METRIC_REGISTRY.keys())
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Available metrics: {available}"
        )
    
    metric_class = METRIC_REGISTRY[metric_name]
    return metric_class(device=device, **kwargs)


def list_available_metrics():
    """
    List all available metrics.
    
    Returns:
        list: Names of available metrics
    """
    return list(METRIC_REGISTRY.keys())

