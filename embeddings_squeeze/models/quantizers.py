"""
Vector Quantization implementations using vector_quantize_pytorch library.
Supports: VQ-VAE, FSQ, LFQ, and Residual VQ.
"""

import torch
import torch.nn as nn
import math
from vector_quantize_pytorch import VectorQuantize, FSQ, ResidualVQ, LFQ


class BaseQuantizer(nn.Module):
    """Base class for all quantizers"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        raise NotImplementedError

    def quantize_spatial(self, features: torch.Tensor):
        """
        Quantize spatial features [B, C, H, W] or [B, N, C] for point clouds
        
        Args:
            features: Tensor of shape [B, C, H, W] or [B, N, C]
        
        Returns:
            quantized: Quantized features [B, C, H, W] or [B, N, C]
            loss: Quantization loss (scalar)
        """
        if features.dim() == 4:
            # 2D case: [B, C, H, W]
            B, C, H, W = features.shape
            # Transform [B, C, H, W] -> [B, H*W, C]
            seq = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            # Quantize
            quantized, indices, loss = self.forward(seq)
            
            # Transform back [B, H*W, C] -> [B, C, H, W]
            quantized = quantized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif features.dim() == 3:
            # 3D case: [B, N, C] for point clouds
            B, N, C = features.shape
            # Quantize directly
            quantized, indices, loss = self.forward(features)
        else:
            raise ValueError(f"Unsupported feature dimension: {features.dim()}")
        
        # Handle loss (may be tensor with multiple elements)
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()
        
        return quantized, loss


class VQWithProjection(BaseQuantizer):
    """
    Vector Quantization (VQ-VAE) with projections
    
    Uses EMA for codebook updates (no gradients needed for codebook)
    ~9 bits per vector at codebook_size=512
    """
    def __init__(
        self, 
        input_dim: int, 
        codebook_size: int = 512, 
        bottleneck_dim: int = 64,
        decay: float = 0.99, 
        commitment_weight: float = 0.25
    ):
        super().__init__(input_dim)
        self.bottleneck_dim = bottleneck_dim
        
        # Down projection (e.g., 2048 -> 64)
        self.project_in = nn.Linear(input_dim, bottleneck_dim)
        
        # Vector Quantization
        self.vq = VectorQuantize(
            dim=bottleneck_dim,
            codebook_size=codebook_size,
            decay=decay,  # EMA decay for codebook
            commitment_weight=commitment_weight  # Commitment loss weight
        )
        
        # Up projection (64 -> 2048)
        self.project_out = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        x_proj = self.project_in(x)
        quantized, indices, commit_loss = self.vq(x_proj)
        x_out = self.project_out(quantized)
        return x_out, indices, commit_loss


class FSQWithProjection(BaseQuantizer):
    """
    Finite Scalar Quantization (FSQ)
    
    Quantization without codebook - each dimension quantized independently
    ~10 bits per vector at levels=[8,5,5,5]
    """
    def __init__(self, input_dim: int, levels: list = None):
        super().__init__(input_dim)
        if levels is None:
            levels = [8, 5, 5, 5]  # 8*5*5*5 = 1000 codes ≈ 2^10
        
        self.num_levels = len(levels)
        
        # Projection to quantization space
        self.project_in = nn.Linear(input_dim, self.num_levels)
        
        # FSQ quantization
        self.fsq = FSQ(levels=levels, dim=self.num_levels)
        
        # Projection back
        self.project_out = nn.Linear(self.num_levels, input_dim)

    def forward(self, x):
        x_proj = self.project_in(x)
        quantized, indices = self.fsq(x_proj)
        x_out = self.project_out(quantized)
        
        # FSQ has no explicit loss
        loss = torch.tensor(0.0, device=x.device)
        return x_out, indices, loss


class LFQWithProjection(BaseQuantizer):
    """
    Lookup-Free Quantization (LFQ)
    
    Uses entropy loss for code diversity
    ~9 bits per vector at codebook_size=512
    """
    def __init__(
        self, 
        input_dim: int, 
        codebook_size: int = 512,
        entropy_loss_weight: float = 0.1, 
        diversity_gamma: float = 0.1, 
        spherical: bool = False
    ):
        super().__init__(input_dim)
        # Quantization dimension = log2(codebook_size)
        self.quant_dim = int(math.log2(codebook_size))
        
        # Projection with normalization
        self.project_in = nn.Sequential(
            nn.Linear(input_dim, self.quant_dim),
            nn.LayerNorm(self.quant_dim)
        )
        
        # LFQ quantization
        self.lfq = LFQ(
            dim=self.quant_dim,
            codebook_size=codebook_size,
            entropy_loss_weight=entropy_loss_weight,
            diversity_gamma=diversity_gamma,
            spherical=spherical
        )
        
        # Projection back
        self.project_out = nn.Linear(self.quant_dim, input_dim)

    def forward(self, x):
        x_proj = self.project_in(x)
        quantized, indices, entropy_loss = self.lfq(x_proj)
        x_out = self.project_out(quantized)
        return x_out, indices, entropy_loss


class ResidualVQWithProjection(BaseQuantizer):
    """
    Residual Vector Quantization (RVQ)
    
    Multi-level quantization - each level quantizes the residual of the previous
    32 bits per vector at num_quantizers=4, codebook_size=256 (4*8 bits)
    """
    def __init__(
        self, 
        input_dim: int, 
        num_quantizers: int = 4,
        codebook_size: int = 256, 
        bottleneck_dim: int = 64,
        decay: float = 0.99, 
        commitment_weight: float = 0.25
    ):
        super().__init__(input_dim)
        self.bottleneck_dim = bottleneck_dim
        
        # Down projection
        self.project_in = nn.Linear(input_dim, bottleneck_dim)
        
        # Residual VQ
        self.residual_vq = ResidualVQ(
            dim=bottleneck_dim,
            num_quantizers=num_quantizers,  # Number of levels
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight
        )
        
        # Up projection
        self.project_out = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        x_proj = self.project_in(x)
        quantized, indices, loss = self.residual_vq(x_proj)
        x_out = self.project_out(quantized)
        return x_out, indices, loss

