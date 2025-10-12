"""
Codebook and distance metric implementations for Vector Quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DistanceMetric(nn.Module):
    """Distance metric for finding closest codebook vectors."""
    
    def __init__(self, metric_type='euclidean'):
        super(DistanceMetric, self).__init__()
        self.metric_type = metric_type

    def forward(self, inputs, codebook):
        """
        Compute distances between inputs and codebook vectors.
        
        Args:
            inputs: Input vectors [N, D]
            codebook: Codebook vectors [K, D]
            
        Returns:
            distances: Distance matrix [N, K]
        """
        if self.metric_type == 'euclidean':
            distances = (
                torch.sum(inputs**2, dim=1, keepdim=True) +
                torch.sum(codebook**2, dim=1) -
                2 * torch.matmul(inputs, codebook.t())
            )
        elif self.metric_type == 'cosine':
            inputs_norm = F.normalize(inputs, dim=1)
            codebook_norm = F.normalize(codebook, dim=1)
            distances = 1 - torch.matmul(inputs_norm, codebook_norm.t())
        else:
            raise ValueError(f"Unknown metric: {self.metric_type}")

        return distances


class Codebook(nn.Module):
    """
    Codebook for storing vector embeddings.
    Manages initialization and retrieval of codebook vectors.
    """
    
    def __init__(self, num_vectors, vector_dim):
        super(Codebook, self).__init__()
        self.num_vectors = num_vectors
        self.vector_dim = vector_dim

        # Initialize with random vectors scaled by 1/sqrt(dim)
        self.embeddings = nn.Parameter(
            torch.randn(num_vectors, vector_dim) / np.sqrt(vector_dim)
        )

    def forward(self):
        """Return current codebook vectors."""
        return self.embeddings

    def get_vector(self, indices):
        """
        Get vectors by indices.
        
        Args:
            indices: Tensor of indices [N]
            
        Returns:
            vectors: Selected vectors [N, vector_dim]
        """
        return self.embeddings[indices]
