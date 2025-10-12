"""
Loss functions for Vector Quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLoss(nn.Module):
    """
    Vector Quantization loss function.
    Computes codebook loss and commitment loss.
    """
    
    def __init__(self, commitment_cost=0.25):
        super(VQLoss, self).__init__()
        self.commitment_cost = commitment_cost

    def forward(self, quantized, inputs):
        """
        Compute VQ losses.
        
        Args:
            quantized: Quantized vectors (from codebook)
            inputs: Original input vectors (from encoder)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Codebook loss: ||sg[z_e] - e||^2
        # Updates codebook embeddings to move closer to encoder output
        codebook_loss = F.mse_loss(quantized, inputs.detach())

        # Commitment loss: ||z_e - sg[e]||^2  
        # Forces encoder to generate vectors close to codebook
        commitment_loss = F.mse_loss(quantized.detach(), inputs)

        # Total loss
        total_loss = codebook_loss + self.commitment_cost * commitment_loss

        loss_dict = {
            'total': total_loss,
            'codebook': codebook_loss,
            'commitment': commitment_loss
        }

        return total_loss, loss_dict
