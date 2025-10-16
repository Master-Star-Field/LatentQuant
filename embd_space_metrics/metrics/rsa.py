"""RSA (Representational Similarity Analysis) metric implementation."""

import torch
import numpy as np
from scipy.stats import spearmanr
from .base import SimilarityMetric
from .helpers import compute_pairwise_distances, pool_spatial_features


class RSAMetric(SimilarityMetric):
    """
    RSA (Representational Similarity Analysis) metric.
    
    Compares representational dissimilarity matrices (RDMs) using correlation.
    Uses correlation distance for computing RDMs.
    
    Reference: Kriegeskorte et al. "Representational similarity analysis" (2008)
    """
    
    def compute(self, features_1, features_2):
        """
        Compute RSA between two feature sets.
        
        Args:
            features_1: First feature set [N, D] or [N, C, H, W]
            features_2: Second feature set [N, D] or [N, C, H, W]
            
        Returns:
            float: RSA score (Spearman correlation of RDMs, -1 to 1)
        """
        # Pool spatial dimensions if needed
        X = pool_spatial_features(features_1)
        Y = pool_spatial_features(features_2)
        
        # Convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()
        
        # Compute RDMs using correlation distance
        rdm_X = compute_pairwise_distances(X, metric='correlation')
        rdm_Y = compute_pairwise_distances(Y, metric='correlation')
        
        # Flatten upper triangular part (excluding diagonal)
        n = rdm_X.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        
        rdm_X_flat = rdm_X[triu_indices]
        rdm_Y_flat = rdm_Y[triu_indices]
        
        # Compute Spearman correlation between RDMs
        correlation, _ = spearmanr(rdm_X_flat, rdm_Y_flat)
        
        return float(correlation)

