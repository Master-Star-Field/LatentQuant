"""CKA (Centered Kernel Alignment) metric implementation."""

import torch
import numpy as np
from .base import SimilarityMetric
from .helpers import pool_spatial_features


class CKAMetric(SimilarityMetric):
    """
    Linear CKA (Centered Kernel Alignment) metric.
    
    CKA measures similarity between two sets of representations by comparing
    their centered Gram matrices.
    
    Reference: Kornblith et al. "Similarity of Neural Network Representations 
    Revisited" (ICML 2019)
    """
    
    def compute(self, features_1, features_2):
        """
        Compute linear CKA between two feature sets.
        
        Args:
            features_1: First feature set [N, D] or [N, C, H, W]
            features_2: Second feature set [N, D] or [N, C, H, W]
            
        Returns:
            float: CKA similarity score (0 to 1)
        """
        # Pool spatial dimensions if needed
        X = pool_spatial_features(features_1)
        Y = pool_spatial_features(features_2)
        
        # Move to device for computation
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Center the features
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices (linear kernel: K = X @ X.T)
        K_X = X @ X.T  # [N, N]
        K_Y = Y @ Y.T  # [N, N]
        
        # Compute HSIC (Hilbert-Schmidt Independence Criterion)
        # HSIC(K_X, K_Y) = tr(K_X @ K_Y) / (n-1)^2
        hsic_xy = torch.sum(K_X * K_Y)
        hsic_xx = torch.sum(K_X * K_X)
        hsic_yy = torch.sum(K_Y * K_Y)
        
        # CKA = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy + 1e-10)
        
        return cka.item()

