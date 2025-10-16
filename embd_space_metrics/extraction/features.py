"""Feature extraction from segmentation backbones."""

import torch
from tqdm import tqdm


def extract_features_from_backbone(
    backbone,
    data_loader,
    device,
    pool_spatial=True,
    desc="Extracting features"
):
    """
    Extract features from a segmentation backbone.
    
    This function is compatible with any backbone that implements the
    SegmentationBackbone interface (e.g., DeepLabV3, ViT).
    
    Args:
        backbone: SegmentationBackbone instance
        data_loader: DataLoader for images
        device: Torch device (cuda/cpu)
        pool_spatial: Whether to spatially pool features [B,C,H,W] -> [B,C]
        desc: Description for progress bar
        
    Returns:
        torch.Tensor: Extracted features [N, C] if pool_spatial=True, 
                     otherwise [N, C, H, W]
    """
    backbone.eval()
    all_features = []
    
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc=desc):
            images = images.to(device)
            
            # Extract features using backbone's method
            features = backbone.extract_features(images, detach=True)
            
            # Spatial pooling if requested (memory efficient)
            if pool_spatial and features.dim() == 4:
                # Pool on GPU, then move to CPU
                features = features.mean(dim=(2, 3))
            
            all_features.append(features.cpu())
    
    # Concatenate all features
    return torch.cat(all_features, dim=0)


def extract_quantized_features(
    backbone,
    quantizer,
    data_loader,
    device,
    pool_spatial=True,
    desc="Extracting quantized features"
):
    """
    Extract features after quantization.
    
    Args:
        backbone: SegmentationBackbone instance
        quantizer: Quantizer module (VQ, FSQ, LFQ, ResidualVQ)
        data_loader: DataLoader for images
        device: Torch device
        pool_spatial: Whether to spatially pool features
        desc: Description for progress bar
        
    Returns:
        torch.Tensor: Quantized features [N, C] if pool_spatial=True,
                     otherwise [N, C, H, W]
    """
    backbone.eval()
    if quantizer is not None:
        quantizer.eval()
    
    all_features = []
    
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc=desc):
            images = images.to(device)
            
            # Extract backbone features
            features = backbone.extract_features(images, detach=True)
            
            # Apply quantization if quantizer exists
            if quantizer is not None:
                # Use quantize_spatial method for spatial features
                if hasattr(quantizer, 'quantize_spatial'):
                    features, _ = quantizer.quantize_spatial(features)
                else:
                    # Fallback for simple quantizers
                    features, _, _ = quantizer(features)
            
            # Spatial pooling if requested
            if pool_spatial and features.dim() == 4:
                features = features.mean(dim=(2, 3))
            
            all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)

