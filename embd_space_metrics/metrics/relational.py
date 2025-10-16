"""Relational Knowledge Loss metric implementation."""

import torch
import numpy as np
from scipy.stats import pearsonr
from .base import SimilarityMetric
from .helpers import compute_pairwise_distances, pool_spatial_features


class RelationalKnowledgeLossMetric(SimilarityMetric):
    """
    Relational Knowledge Loss metric.
    
    Measures how well relational structure (pairwise relationships) is preserved.
    Uses Pearson correlation of pairwise distances.
    
    Reference: Park et al. "Relational Knowledge Distillation" (CVPR 2019)
    """
    
    def compute(self, features_1, features_2):
        """
        Compute Relational Knowledge Loss between two feature sets.
        
        Args:
            features_1: First feature set [N, D] or [N, C, H, W]
            features_2: Second feature set [N, D] or [N, C, H, W]
            
        Returns:
            float: Relational similarity score (Pearson correlation, -1 to 1)
        """
        # Pool spatial dimensions if needed
        X = pool_spatial_features(features_1)
        Y = pool_spatial_features(features_2)
        
        # Convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()
        
        # Compute pairwise Euclidean distances
        dist_X = compute_pairwise_distances(X, metric='euclidean')
        dist_Y = compute_pairwise_distances(Y, metric='euclidean')
        
        # Flatten upper triangular part (excluding diagonal)
        n = dist_X.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        
        dist_X_flat = dist_X[triu_indices]
        dist_Y_flat = dist_Y[triu_indices]
        
        # Compute Pearson correlation (linear relationship)
        correlation, _ = pearsonr(dist_X_flat, dist_Y_flat)
        
        return float(correlation)

