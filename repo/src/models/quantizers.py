import torch
import torch.nn as nn
import math
from vector_quantize_pytorch import VectorQuantize, FSQ, ResidualVQ, LFQ


class BaseQuantizer(nn.Module):
    """Базовый класс для всех квантизаторов"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        raise NotImplementedError

    def quantize_spatial(self, features: torch.Tensor):
        """
        Квантизация пространственных фич [B, C, H, W]
        
        Args:
            features: Тензор размера [B, C, H, W]
        
        Returns:
            quantized: Квантованные фичи [B, C, H, W]
            loss: Квантизационный loss (scalar)
        """
        B, C, H, W = features.shape
        # Преобразуем [B, C, H, W] -> [B, H*W, C]
        seq = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Квантизация
        quantized, indices, loss = self.forward(seq)
        
        # Преобразуем обратно [B, H*W, C] -> [B, C, H, W]
        quantized = quantized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Обработка loss (может быть тензором с несколькими элементами)
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()
        
        return quantized, loss


class VQWithProjection(BaseQuantizer):
    """
    Vector Quantization (VQ-VAE) с проекциями
    
    Использует EMA для обновления codebook (не требует градиентов для codebook)
    ~9 бит на вектор при codebook_size=512
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
        
        # Проекция вниз (2048 -> 64)
        self.project_in = nn.Linear(input_dim, bottleneck_dim)
        
        # Vector Quantization
        self.vq = VectorQuantize(
            dim=bottleneck_dim,
            codebook_size=codebook_size,
            decay=decay,  # EMA decay для codebook
            commitment_weight=commitment_weight  # Вес commitment loss
        )
        
        # Проекция вверх (64 -> 2048)
        self.project_out = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        x_proj = self.project_in(x)
        quantized, indices, commit_loss = self.vq(x_proj)
        x_out = self.project_out(quantized)
        return x_out, indices, commit_loss


class FSQWithProjection(BaseQuantizer):
    def __init__(self, input_dim: int, levels: list = None, bottleneck_dim=256):
        super().__init__(input_dim)
        if levels is None:
            levels = [8, 5, 5, 5]

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

        self.project_in = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )

        self.lfq = LFQ(
            dim=bottleneck_dim,
            codebook_size=codebook_size,
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
    
    Многоуровневая квантизация - каждый уровень квантует остаток предыдущего
    32 бита на вектор при num_quantizers=4, codebook_size=256 (4*8 бит)
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
        
        # Проекция вниз
        self.project_in = nn.Linear(input_dim, bottleneck_dim)
        
        # Residual VQ
        self.residual_vq = ResidualVQ(
            dim=bottleneck_dim,
            num_quantizers=num_quantizers,  # Количество уровней
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight
        )
        
        # Проекция вверх
        self.project_out = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        x_proj = self.project_in(x)
        quantized, indices, loss = self.residual_vq(x_proj)
        x_out = self.project_out(quantized)
        return x_out, indices, loss