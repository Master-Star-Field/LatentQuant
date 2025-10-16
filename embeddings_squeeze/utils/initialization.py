"""
Codebook initialization utilities.
"""

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Optional


def initialize_codebook_from_data(
    vq_model, 
    backbone, 
    train_loader, 
    device, 
    max_samples: int = 50_000
):
    """
    Initialize codebook using k-means clustering on real data.
    
    Args:
        vq_model: VectorQuantizer model (e.g., VQWithProjection)
        backbone: Segmentation backbone
        train_loader: Training data loader
        device: Device to run on
        max_samples: Maximum number of samples for k-means
    """
    print("Collecting features for k-means initialization...")
    all_features = []
    
    # Move projection layer to device and set to eval
    vq_model.project_in.to(device)
    vq_model.project_in.eval()
    
    backbone.eval()
    
    num_collected = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i % 10 == 0:
                print(f"Processing batch {i}, collected {num_collected} samples...")
            
            features = backbone.extract_features(images.to(device))
            
            B, C, H, W = features.shape
            # Transform to sequence format [B, H*W, C]
            feat_seq = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            # Project features through the projection layer to get bottleneck dims
            feat_proj = vq_model.project_in(feat_seq)  # [B, H*W, bottleneck_dim]
            
            # Flatten and move to CPU immediately to save GPU memory
            feat_flat = feat_proj.reshape(-1, feat_proj.shape[-1]).cpu()
            all_features.append(feat_flat)
            
            num_collected += feat_flat.shape[0]
            if num_collected >= max_samples:
                print(f"Reached {num_collected} samples, stopping collection.")
                break
    
    all_features = torch.cat(all_features[:int(max_samples / feat_flat.shape[0]) + 1])
    # Subsample if we collected too many
    if all_features.shape[0] > max_samples:
        indices = torch.randperm(all_features.shape[0])[:max_samples]
        all_features = all_features[indices]
    
    all_features = all_features.numpy()
    
    print(f"Running k-means on {all_features.shape[0]} features (dim={all_features.shape[1]})...")
    print(f"Target codebook_size: {vq_model.codebook_size}")
    
    kmeans = MiniBatchKMeans(
        n_clusters=vq_model.codebook_size,
        random_state=0,
        batch_size=1000,
        max_iter=100,
        verbose=0
    )
    kmeans.fit(all_features)
    
    # Get cluster centers and convert to tensor
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    
    # Update codebook with cluster centers
    # The VectorQuantize library may use different internal structures
    # Try to detect the right way to access the codebook
    if hasattr(vq_model, 'vq'):
        quantizer = vq_model.vq
    elif hasattr(vq_model, 'residual_vq'):
        quantizer = vq_model.residual_vq
    else:
        return
    
    # VectorQuantize uses shape [num_codebooks, codebook_size, dim]
    # For single codebook: [1, codebook_size, dim]
    if quantizer._codebook.embed.ndim == 3:
        # 3D tensor for multi-codebook support
        for i in range(quantizer._codebook.embed.shape[0]):
            quantizer._codebook.embed.data[i] = cluster_centers
    else:
        # 2D tensor for single codebook
        quantizer._codebook.embed.data = cluster_centers
