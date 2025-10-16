"""Jaccard k-NN metric implementation."""

import torch
import numpy as np
from .base import SimilarityMetric
from .helpers import compute_knn_indices, jaccard_similarity, pool_spatial_features


class JaccardKNNMetric(SimilarityMetric):
    """
    Jaccard k-NN metric.
    
    Measures how well k-nearest neighbor structure is preserved by computing
    average Jaccard similarity between k-NN sets.
    """
    
    def __init__(self, device=None, k=5):
        """
        Initialize Jaccard k-NN metric.
        
        Args:
            device: Torch device
            k: Number of nearest neighbors to consider
        """
        super().__init__(device)
        self.k = k
    
    def compute(self, features_1, features_2):
        """
        Compute Jaccard k-NN similarity between two feature sets.
        
        Args:
            features_1: First feature set [N, D] or [N, C, H, W]
            features_2: Second feature set [N, D] or [N, C, H, W]
            
        Returns:
            float: Average Jaccard similarity (0 to 1)
        """
        # Pool spatial dimensions if needed
        X = pool_spatial_features(features_1)
        Y = pool_spatial_features(features_2)
        
        # Convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()
        
        # Compute k-NN for both feature sets
        knn_X = compute_knn_indices(X, k=self.k)
        knn_Y = compute_knn_indices(Y, k=self.k)
        
        # Compute Jaccard similarity for each sample
        jaccard_scores = []
        for i in range(len(X)):
            score = jaccard_similarity(knn_X[i], knn_Y[i])
            jaccard_scores.append(score)
        
        # Return average Jaccard similarity
        return float(np.mean(jaccard_scores))

