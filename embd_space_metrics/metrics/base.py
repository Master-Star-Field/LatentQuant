"""Base class for similarity metrics."""

import torch


class SimilarityMetric:
    """
    Base class for similarity metrics.
    
    All metrics should inherit from this class and implement the compute method.
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
    
    def compute(self, features_1, features_2):
        """
        Compute similarity between two sets of features.
        
        Args:
            features_1: First feature set, shape [N, D] or [N, C, H, W]
            features_2: Second feature set, same shape as features_1
            
        Returns:
            float: Similarity score
        """
        raise NotImplementedError("Subclasses must implement compute()")
    
    @property
    def name(self):
        """Return metric name."""
        return self.__class__.__name__.replace('Metric', '').lower()

