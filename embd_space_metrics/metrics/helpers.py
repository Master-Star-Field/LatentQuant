"""Helper functions for computing metrics."""

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_cca_svd(X, Y):
    """
    Compute CCA using SVD.
    
    Args:
        X: Features [N, D1]
        Y: Features [N, D2]
        
    Returns:
        np.ndarray: CCA correlation coefficients
    """
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute covariance matrices
    Cxx = X.T @ X / (X.shape[0] - 1)
    Cyy = Y.T @ Y / (Y.shape[0] - 1)
    Cxy = X.T @ Y / (X.shape[0] - 1)
    
    # Add small regularization for numerical stability
    eps = 1e-5
    Cxx = Cxx + eps * np.eye(Cxx.shape[0])
    Cyy = Cyy + eps * np.eye(Cyy.shape[0])
    
    # Compute CCA via SVD
    try:
        Cxx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cxx))
        Cyy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cyy))
        
        M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt.T
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        
        return S  # Canonical correlations
    except np.linalg.LinAlgError:
        # Fallback to eigendecomposition if Cholesky fails
        Cxx_inv = np.linalg.pinv(Cxx)
        Cyy_inv = np.linalg.pinv(Cyy)
        
        M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
        eigenvalues = np.linalg.eigvalsh(M)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        
        return np.sqrt(eigenvalues[::-1])  # Return in descending order


def compute_pairwise_distances(X, metric='euclidean'):
    """
    Compute pairwise distances between samples.
    
    Args:
        X: Features [N, D]
        metric: Distance metric ('euclidean', 'cosine', 'correlation')
        
    Returns:
        np.ndarray: Distance matrix [N, N]
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    # Use scipy's pdist for efficiency
    distances = pdist(X, metric=metric)
    return squareform(distances)


def compute_knn_indices(X, k=5):
    """
    Compute k-nearest neighbors for each sample.
    
    Args:
        X: Features [N, D]
        k: Number of neighbors
        
    Returns:
        np.ndarray: Indices of k-nearest neighbors [N, k]
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    # Compute pairwise distances
    distances = compute_pairwise_distances(X, metric='euclidean')
    
    # For each sample, find k+1 nearest (including itself), then exclude itself
    knn_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    
    return knn_indices


def jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set (or array of indices)
        set2: Second set (or array of indices)
        
    Returns:
        float: Jaccard similarity (intersection over union)
    """
    set1 = set(set1)
    set2 = set(set2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def pool_spatial_features(features):
    """
    Pool spatial dimensions if features are 4D.
    
    Args:
        features: Tensor of shape [N, C] or [N, C, H, W]
        
    Returns:
        Tensor of shape [N, C]
    """
    if features.dim() == 4:
        # Spatial pooling: [N, C, H, W] -> [N, C]
        return features.mean(dim=(2, 3))
    return features

