"""
Vector Quantization implementations using vector_quantize_pytorch library.
Supports: VQ-VAE, FSQ, LFQ, and Residual VQ.
"""

import code
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
        Quantize spatial features [B, C, H, W]
        
        Args:
            features: Tensor of shape [B, C, H, W]
        
        Returns:
            quantized: Quantized features [B, C, H, W]
            loss: Quantization loss (scalar)
            indices_flat: Flattened indices [B*H*W] for perplexity calculation
        """
        B, C, H, W = features.shape
        # Transform [B, C, H, W] -> [B, H*W, C]
        seq = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Quantize
        quantized, indices, loss = self.forward(seq)
        
        # Transform back [B, H*W, C] -> [B, C, H, W]
        quantized = quantized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Flatten indices for perplexity calculation: [B, H*W] -> [B*H*W]
        indices_flat = indices.reshape(-1) if indices is not None else None
        
        # Handle loss (may be tensor with multiple elements)
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()
        
        return quantized, loss, indices_flat


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
        self.codebook_size = codebook_size
        
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
    def __init__(self,
        input_dim: int,
        levels: list = None,
        bottleneck_dim=256,
        codebook_size=512,
    ):
        super().__init__(input_dim)
        if levels is None:
            levels = [8, 5, 5, 5]

        self.codebook_size = codebook_size
        self.bottleneck_dim = codebook_size

        self.project_in = nn.Linear(input_dim, bottleneck_dim)
        self.fsq = FSQ(levels=levels, dim=bottleneck_dim)
        self.project_out = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        x_proj = self.project_in(x)
        quantized, indices = self.fsq(x_proj)
        x_out = self.project_out(quantized)
        loss = torch.tensor(0.0, device=x.device)
        return x_out, indices, loss


class LFQWithProjection(BaseQuantizer):
    """
    Lookup-Free Quantization (LFQ)

    • Квантайзер без lookup-таблицы, с энтропийным и диверсификационным лоссами.  
    • Подходит для встраивания в encoder bottleneck.
    • Кодирует ~log2(codebook_size) бит информации на вектор.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,        
        codebook_size: int = 512,
        entropy_loss_weight: float = 0.1,
        diversity_gamma: float = 0.1,
        spherical: bool = False,
    ):
        super().__init__(input_dim)

        self.codebook_size = codebook_size
        self.bottleneck_dim = codebook_size

        self.project_in = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )

        self.lfq = LFQ(
            dim=bottleneck_dim,
            codebook_size=512,
            entropy_loss_weight=entropy_loss_weight,
            diversity_gamma=diversity_gamma,
            spherical=spherical,
        )

        self.project_out = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor (..., input_dim)
        Returns:
            x_out: реконструированный вектор
            indices: индексы активированных кодов
            entropy_loss: вспомогательный лосс LFQ
        """
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
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
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

