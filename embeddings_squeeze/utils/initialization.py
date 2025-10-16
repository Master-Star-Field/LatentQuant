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
    kmeans = MiniBatchKMeans(
        n_clusters=vq_model.codebook_size,
        random_state=0,
        batch_size=1000,
        max_iter=100,
        verbose=0
    )
    kmeans.fit(all_features)
    
    # Update codebook with cluster centers
    # Access the underlying VectorQuantize codebook
    vq_model.vq._codebook.embed.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    
    print(f"Codebook initialized:")
    print(f"  Shape: {vq_model.vq._codebook.embed.shape}")
    print(f"  Mean: {vq_model.vq._codebook.embed.mean():.4f}")
    print(f"  Std: {vq_model.vq._codebook.embed.std():.4f}")
    print(f"  Norm: {vq_model.vq._codebook.embed.norm(dim=1).mean():.4f}")
