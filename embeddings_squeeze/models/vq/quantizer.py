"""
Vector Quantization components: encoder, quantizer, and STE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import Codebook, DistanceMetric
from .losses import VQLoss


class VectorEncoder(nn.Module):
    """
    Encodes vectors by finding closest codebook vectors.
    Handles the non-differentiable encoding operation.
    """
    
    def __init__(self, distance_metric):
        super(VectorEncoder, self).__init__()
        self.distance_metric = distance_metric

    def forward(self, inputs, codebook):
        """
        Find indices of closest codebook vectors.
        
        Args:
            inputs: Input vectors [N, D]
            codebook: Codebook vectors [K, D]
            
        Returns:
            encoding_indices: Indices of closest vectors [N]
            encodings: One-hot encoding [N, K]
        """
        # Compute distances
        distances = self.distance_metric(inputs, codebook)

        # Find minimum distance indices (non-differentiable!)
        encoding_indices = torch.argmin(distances, dim=1)

        # One-hot encoding for quantization
        encodings = torch.zeros(
            encoding_indices.shape[0],
            codebook.shape[0],
            device=inputs.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        return encoding_indices, encodings


class QuantizationLayer(nn.Module):
    """
    Performs quantization by selecting vectors from codebook.
    """
    
    def __init__(self):
        super(QuantizationLayer, self).__init__()

    def forward(self, encodings, codebook):
        """
        Select vectors from codebook based on one-hot encoding.
        
        Args:
            encodings: One-hot encoding [N, K]
            codebook: Codebook vectors [K, D]
            
        Returns:
            quantized: Quantized vectors [N, D]
        """
        quantized = torch.matmul(encodings, codebook)
        return quantized


class StraightThroughEstimator(nn.Module):
    """
    Straight-Through Estimator for gradient flow through discrete operations.
    """
    
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, quantized, inputs):
        """
        Apply STE for gradient flow.
        
        Forward pass: output = quantized
        Backward pass: gradients flow through inputs
        
        Args:
            quantized: Quantized vectors
            inputs: Original input vectors
            
        Returns:
            output: STE output
        """
        # Copy gradient from inputs to quantized
        # Forward: output = inputs + (quantized - inputs) = quantized
        # Backward: grad_inputs = grad_output (due to detach on difference)
        output = inputs + (quantized - inputs).detach()
        return output


class VectorQuantizer(nn.Module):
    """
    Complete Vector Quantizer combining all components.
    """
    
    def __init__(self, num_vectors, vector_dim, commitment_cost=0.25, metric_type='euclidean'):
        super(VectorQuantizer, self).__init__()
        self.vector_dim = vector_dim
        self.num_vectors = num_vectors

        # Initialize modular components
        self.codebook = Codebook(num_vectors, vector_dim)
        self.distance_metric = DistanceMetric(metric_type)
        self.encoder = VectorEncoder(self.distance_metric)
        self.quantizer = QuantizationLayer()
        self.loss_fn = VQLoss(commitment_cost)
        self.ste = StraightThroughEstimator()

    def forward(self, inputs):
        """
        Complete vector quantization process.
        
        Args:
            inputs: Input tensor [B, vector_dim] or [B, C, H, W]
            
        Returns:
            quantized_ste: Quantized vectors with STE applied
            loss: VQ loss
        """
        # Remember original shape for reshaping
        input_shape = inputs.shape

        # Flatten to [N, vector_dim] where N = B * H * W
        flat_inputs = inputs.view(-1, self.vector_dim)

        # Get codebook vectors
        codebook_vectors = self.codebook()

        # 1. Encoding: find closest vectors
        encoding_indices, encodings = self.encoder(flat_inputs, codebook_vectors)

        # 2. Quantization: select vectors from codebook
        quantized = self.quantizer(encodings, codebook_vectors)

        # Restore original shape
        quantized = quantized.view(input_shape)

        # 3. Compute losses
        loss, loss_dict = self.loss_fn(quantized, inputs)

        # 4. Apply STE
        quantized_ste = self.ste(quantized, inputs)

        return quantized_ste, loss
