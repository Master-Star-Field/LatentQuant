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
        vq_model: VectorQuantizer model
        backbone: Segmentation backbone
        train_loader: Training data loader
        device: Device to run on
        max_samples: Maximum number of samples for k-means
    """
    print("Collecting features for k-means initialization...")
    all_features = []
    
    backbone.eval()
    with torch.no_grad():
        i = 0
        for images, _ in train_loader:
            print(f"Processing batch {i}")
            features = backbone.extract_features(images.to(device))
            i += 1
            
            B, C, H, W = features.shape
            feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
            all_features.append(feat_flat.cpu())
            
            if len(all_features) * feat_flat.shape[0] > max_samples:
                break
    
    all_features = torch.cat(all_features).numpy()
    
    print(f"Running k-means on {all_features.shape[0]} features...")
    kmeans = MiniBatchKMeans(
        n_clusters=vq_model.num_vectors,
        random_state=0,
        batch_size=1000,
        max_iter=100
    )
    kmeans.fit(all_features)
    
    # Update codebook with cluster centers
    vq_model.codebook.embeddings.data = torch.tensor(kmeans.cluster_centers_).to(device).float()
    
    print(f"Codebook initialized:")
    print(f"  Mean: {vq_model.codebook.embeddings.mean():.2f}")
    print(f"  Std: {vq_model.codebook.embeddings.std():.2f}")
    print(f"  Norm: {vq_model.codebook.embeddings.norm(dim=1).mean():.2f}")
