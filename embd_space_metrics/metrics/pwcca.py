"""PWCCA (Projection Weighted CCA) metric implementation."""

import torch
import numpy as np
from .base import SimilarityMetric
from .helpers import compute_cca_svd, pool_spatial_features


class PWCCAMetric(SimilarityMetric):
    """
    PWCCA (Projection Weighted Canonical Correlation Analysis) metric.
    
    PWCCA weights CCA dimensions by their importance (variance explained).
    
    Reference: Morcos et al. "Insights on representational similarity in 
    neural networks with canonical correlation" (NeurIPS 2018)
    """
    
    def compute(self, features_1, features_2):
        """
        Compute PWCCA between two feature sets.
        
        Args:
            features_1: First feature set [N, D] or [N, C, H, W]
            features_2: Second feature set [N, D] or [N, C, H, W]
            
        Returns:
            float: PWCCA similarity score (0 to 1)
        """
        # Pool spatial dimensions if needed
        X = pool_spatial_features(features_1)
        Y = pool_spatial_features(features_2)
        
        # Convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()
        
        # Compute CCA correlations
        rho = compute_cca_svd(X, Y)
        
        # Compute variance explained by each CCA dimension in X
        X_centered = X - X.mean(axis=0)
        cov_X = X_centered.T @ X_centered / (X.shape[0] - 1)
        
        try:
            # SVD of X to get variance explained
            _, s_X, _ = np.linalg.svd(X_centered, full_matrices=False)
            variance_explained = (s_X ** 2) / (X.shape[0] - 1)
            
            # Normalize to get weights
            weights = variance_explained / variance_explained.sum()
            
            # Weighted average of CCA correlations
            # Use min length to avoid indexing errors
            min_len = min(len(rho), len(weights))
            pwcca = np.sum(rho[:min_len] * weights[:min_len])
            
        except np.linalg.LinAlgError:
            # Fallback: unweighted mean CCA
            pwcca = np.mean(rho)
        
        return float(pwcca)

