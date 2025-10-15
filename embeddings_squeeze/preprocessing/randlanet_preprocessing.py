import torch
import numpy as np
from typing import List, Tuple, Dict
import torch.nn.functional as F


def compute_neighbor_indices(points: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    Compute neighbor indices for each point using simple distance-based approach.
    
    Args:
        points: Point cloud [B, N, 3]
        k: Number of neighbors
        
    Returns:
        Neighbor indices [B, N, k]
    """
    batch_size, num_points, _ = points.shape
    device = points.device
    
    # Compute pairwise distances
    points_expanded = points.unsqueeze(2)  # [B, N, 1, 3]
    points_transposed = points.unsqueeze(1)  # [B, 1, N, 3]
    
    distances = torch.norm(points_expanded - points_transposed, dim=3)  # [B, N, N]
    
    # Get k nearest neighbors (including self)
    _, neighbor_indices = torch.topk(distances, k, dim=2, largest=False)  # [B, N, k]
    
    return neighbor_indices


def compute_subsample_indices(points: torch.Tensor, ratio: float = 0.25) -> torch.Tensor:
    """
    Compute subsample indices by randomly selecting points.
    
    Args:
        points: Point cloud [B, N, 3]
        ratio: Subsampling ratio
        
    Returns:
        Subsample indices [B, M] where M = int(N * ratio)
    """
    batch_size, num_points, _ = points.shape
    device = points.device
    
    num_subsample = int(num_points * ratio)
    
    # Random sampling
    indices = torch.randperm(num_points, device=device)[:num_subsample]
    indices = indices.unsqueeze(0).expand(batch_size, -1)  # [B, M]
    
    return indices


def compute_interpolation_indices(points: torch.Tensor, subsample_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute interpolation indices by finding nearest neighbors.
    
    Args:
        points: Original point cloud [B, N, 3]
        subsample_indices: Subsampled point indices [B, M]
        
    Returns:
        Interpolation indices [B, N] - for each original point, index of nearest subsampled point
    """
    batch_size, num_points, _ = points.shape
    device = points.device
    
    # Get subsampled points
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
    subsampled_points = points[batch_indices, subsample_indices]  # [B, M, 3]
    
    # Compute distances from each original point to each subsampled point
    points_expanded = points.unsqueeze(2)  # [B, N, 1, 3]
    subsampled_expanded = subsampled_points.unsqueeze(1)  # [B, 1, M, 3]
    
    distances = torch.norm(points_expanded - subsampled_expanded, dim=3)  # [B, N, M]
    
    # Find nearest subsampled point for each original point
    _, interpolation_indices = torch.min(distances, dim=2)  # [B, N]
    
    return interpolation_indices


def preprocess_for_randlanet(points: torch.Tensor, colors: torch.Tensor, 
                           num_layers: int = 4, k: int = 16) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess point cloud data for RandLANet.
    
    Args:
        points: Point coordinates [B, N, 3]
        colors: Point colors [B, N, 3]
        num_layers: Number of layers in RandLANet
        k: Number of neighbors
        
    Returns:
        Dictionary with preprocessed data for each layer
    """
    batch_size, num_points, _ = points.shape
    device = points.device
    
    # Concatenate points and colors as features
    features = torch.cat([points, colors], dim=-1)  # [B, N, 6]
    
    # Initialize lists for each layer
    coords_list = []
    neighbor_indices_list = []
    sub_idx_list = []
    interp_idx_list = []
    
    current_points = points
    current_features = features
    
    for layer in range(num_layers):
        # Compute neighbor indices for current layer
        neighbor_indices = compute_neighbor_indices(current_points, k)
        neighbor_indices_list.append(neighbor_indices)
        
        # Store current coordinates
        coords_list.append(current_points)
        
        # Compute subsample indices (reduce points by factor of 4 each layer)
        subsample_ratio = 0.25 if layer < num_layers - 1 else 1.0  # Last layer keeps all points
        sub_idx = compute_subsample_indices(current_points, subsample_ratio)
        sub_idx_list.append(sub_idx)
        
        # Compute interpolation indices
        interp_idx = compute_interpolation_indices(current_points, sub_idx)
        interp_idx_list.append(interp_idx)
        
        # Update for next layer (subsample points)
        if layer < num_layers - 1:
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            current_points = current_points[batch_indices, sub_idx]
            current_features = current_features[batch_indices, sub_idx]
    
    return {
        'features': features,
        'coords': coords_list,
        'neighbor_indices': neighbor_indices_list,
        'sub_idx': sub_idx_list,
        'interp_idx': interp_idx_list
    }
