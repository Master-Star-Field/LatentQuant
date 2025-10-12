"""
Compression analysis and metrics utilities.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def measure_compression(vq_model, backbone, test_loader, device):
    """
    Measure compression ratio achieved by VQ.
    
    Args:
        vq_model: VectorQuantizer model
        backbone: Segmentation backbone
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        compression_ratio: Compression ratio achieved
    """
    vq_model.eval()
    
    total_original_bits = 0
    total_compressed_bits = 0
    num_features = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            features = backbone.extract_features(images)
            
            B, C, H, W = features.shape
            feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
            
            # Original size (float32 = 32 bits)
            original_bits = feat_flat.numel() * 32
            
            # Compressed size (only indices)
            num_codes = vq_model.num_vectors
            bits_per_index = np.ceil(np.log2(num_codes))
            compressed_bits = feat_flat.shape[0] * bits_per_index
            
            total_original_bits += original_bits
            total_compressed_bits += compressed_bits
            num_features += feat_flat.shape[0]
    
    compression_ratio = total_original_bits / total_compressed_bits
    
    print("="*70)
    print("COMPRESSION ANALYSIS")
    print("="*70)
    print(f"Total features processed: {num_features}")
    print(f"Feature dimension: {vq_model.vector_dim}")
    print(f"Codebook size: {vq_model.num_vectors}")
    print()
    print(f"Original storage:")
    print(f"  Per feature: {vq_model.vector_dim} × 32 bits = {vq_model.vector_dim * 32} bits = {vq_model.vector_dim * 4} bytes")
    print(f"  Total: {total_original_bits / 8 / 1024 / 1024:.2f} MB")
    print()
    print(f"Compressed storage:")
    print(f"  Per feature: {bits_per_index:.0f} bits (index)")
    print(f"  Total: {total_compressed_bits / 8 / 1024:.2f} KB")
    print(f"  + Codebook: {vq_model.num_vectors * vq_model.vector_dim * 4 / 1024:.2f} KB")
    print()
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Space savings: {(1 - 1/compression_ratio)*100:.1f}%")
    print("="*70)
    
    return compression_ratio


def compute_iou_metrics(predictions, targets, num_classes, ignore_index=255):
    """
    Compute IoU metrics for segmentation.
    
    Args:
        predictions: Predicted masks [B, H, W]
        targets: Ground truth masks [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        metrics: Dictionary with IoU metrics
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Remove ignored pixels
    valid_mask = target_flat != ignore_index
    pred_flat = pred_flat[valid_mask]
    target_flat = target_flat[valid_mask]
    
    # Compute IoU for each class
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
            
        ious.append(iou)
    
    # Compute mean IoU
    mean_iou = np.mean(ious)
    
    # Compute pixel accuracy
    pixel_acc = (pred_flat == target_flat).mean()
    
    metrics = {
        'mean_iou': mean_iou,
        'pixel_accuracy': pixel_acc,
        'class_ious': ious
    }
    
    return metrics
