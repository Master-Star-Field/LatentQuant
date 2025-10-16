"""Geometry Score metric implementation."""

import torch
import numpy as np
from scipy.stats import spearmanr
from .base import SimilarityMetric
from .helpers import compute_pairwise_distances, pool_spatial_features


class GeometryScoreMetric(SimilarityMetric):
    """
    Geometry Score metric.
    
    Measures how well the pairwise distance structure is preserved
    between original and quantized representations using Spearman correlation.
    
    Reference: Shahbazi et al. "Geometry Score" (NeurIPS 2021)
    """
    
    def compute(self, features_1, features_2):
        """
        Compute Geometry Score between two feature sets.
        
        Args:
            features_1: First feature set [N, D] or [N, C, H, W]
            features_2: Second feature set [N, D] or [N, C, H, W]
            
        Returns:
            float: Geometry score (Spearman correlation, -1 to 1)
        """
        # Pool spatial dimensions if needed
        X = pool_spatial_features(features_1)
        Y = pool_spatial_features(features_2)
        
        # Convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()
        
        # Compute pairwise distances
        dist_X = compute_pairwise_distances(X, metric='euclidean')
        dist_Y = compute_pairwise_distances(Y, metric='euclidean')
        
        # Flatten upper triangular part (excluding diagonal)
        n = dist_X.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        
        dist_X_flat = dist_X[triu_indices]
        dist_Y_flat = dist_Y[triu_indices]
        
        # Compute Spearman correlation
        correlation, _ = spearmanr(dist_X_flat, dist_Y_flat)
        
        return float(correlation)

