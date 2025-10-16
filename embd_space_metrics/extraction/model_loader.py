"""Model loading utilities for checkpoints."""

import torch
from pathlib import Path
import sys
import os

# Add parent directory to path to import from embeddings_squeeze
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from embeddings_squeeze.models.backbones import (
    DeepLabV3SegmentationBackbone,
    ViTSegmentationBackbone
)
from embeddings_squeeze.models.quantizers import (
    VQWithProjection,
    FSQWithProjection,
    LFQWithProjection,
    ResidualVQWithProjection
)


def detect_checkpoint_type(checkpoint_path):
    """
    Detect checkpoint format (Lightning vs raw PyTorch).
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        str: 'lightning' or 'raw'
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Lightning checkpoints have 'state_dict', 'epoch', 'hyper_parameters'
    if 'state_dict' in ckpt and 'epoch' in ckpt:
        return 'lightning'
    else:
        return 'raw'


def detect_backbone_type(checkpoint_path, checkpoint_type='raw'):
    """
    Auto-detect backbone type from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        checkpoint_type: 'lightning' or 'raw'
        
    Returns:
        str: 'deeplab' or 'vit'
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    if checkpoint_type == 'lightning':
        keys = list(ckpt.get('state_dict', {}).keys())
    else:
        # Check all keys in the checkpoint
        all_keys = []
        for key in ckpt.keys():
            if isinstance(ckpt[key], dict):
                all_keys.extend(ckpt[key].keys())
        keys = all_keys
    
    # Check for ViT-specific keys
    if any('vit' in k.lower() or 'vision_transformer' in k.lower() or 'patch_embed' in k.lower() for k in keys):
        return 'vit'
    
    # Check for DeepLab/ResNet keys
    if any('resnet' in k.lower() or 'deeplab' in k.lower() or 'layer1' in k.lower() for k in keys):
        return 'deeplab'
    
    # Default to deeplab (most common in this project)
    return 'deeplab'


def load_original_backbone(backbone_type='deeplab', num_classes=21, device='cuda'):
    """
    Load original (non-quantized) backbone.
    
    Args:
        backbone_type: 'deeplab' or 'vit'
        num_classes: Number of segmentation classes
        device: Target device
        
    Returns:
        SegmentationBackbone: Original backbone model
    """
    if backbone_type == 'deeplab':
        backbone = DeepLabV3SegmentationBackbone(
            weights_name='COCO_WITH_VOC_LABELS_V1',
            num_classes=num_classes,
            freeze_backbone=True
        )
    elif backbone_type == 'vit':
        backbone = ViTSegmentationBackbone(
            num_classes=num_classes,
            freeze_backbone=True
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    backbone.to(device)
    backbone.eval()
    
    return backbone


def load_checkpoint(checkpoint_path, device='cuda', backbone_type=None):
    """
    Load model checkpoint (supports both Lightning and raw PyTorch formats).
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth or .ckpt)
        device: Target device
        backbone_type: Optional backbone type ('deeplab' or 'vit'). 
                      If None, auto-detect.
        
    Returns:
        tuple: (backbone, quantizer, checkpoint_dict)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Detect checkpoint type
    ckpt_type = detect_checkpoint_type(checkpoint_path)
    
    # Auto-detect backbone if not specified
    if backbone_type is None:
        backbone_type = detect_backbone_type(checkpoint_path, ckpt_type)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load backbone
    backbone = load_original_backbone(backbone_type, num_classes=21, device=device)
    
    # Load quantizer from checkpoint
    quantizer = None
    
    if ckpt_type == 'lightning':
        # Lightning checkpoint format
        state_dict = checkpoint.get('state_dict', {})
        
        # Extract quantizer state
        quantizer_keys = [k for k in state_dict.keys() if 'quantizer' in k]
        if quantizer_keys:
            # Determine quantizer type from hyperparameters or keys
            hyper_params = checkpoint.get('hyper_parameters', {})
            quantizer_type = hyper_params.get('quantizer_type', 'vq')
            
            quantizer = _create_quantizer(
                quantizer_type=quantizer_type,
                feature_dim=backbone.feature_dim,
                device=device
            )
            
            # Load quantizer weights
            quantizer_state = {
                k.replace('quantizer.', ''): v 
                for k, v in state_dict.items() 
                if 'quantizer' in k
            }
            quantizer.load_state_dict(quantizer_state)
    
    else:
        # Raw PyTorch checkpoint format (from notebook)
        if 'quantizer_state_dict' in checkpoint:
            # Detect quantizer type from checkpoint keys
            quantizer_keys = list(checkpoint['quantizer_state_dict'].keys())
            
            if any('fsq' in k.lower() for k in quantizer_keys):
                quantizer_type = 'fsq'
            elif any('lfq' in k.lower() for k in quantizer_keys):
                quantizer_type = 'lfq'
            elif 'residual_vq' in quantizer_keys[0] if quantizer_keys else False:
                quantizer_type = 'residualvq'
            elif any('project_in' in k for k in quantizer_keys):
                # Has projection layers - check if ResidualVQ
                if any('residual' in k.lower() for k in quantizer_keys):
                    quantizer_type = 'residualvq'
                else:
                    quantizer_type = 'vq'
            else:
                # Check for VQ layers without projection
                if any('layers' in k for k in quantizer_keys):
                    quantizer_type = 'residualvq_no_proj'
                else:
                    quantizer_type = 'vq'
            
            quantizer = _create_quantizer(
                quantizer_type=quantizer_type,
                feature_dim=backbone.feature_dim,
                device=device
            )
            
            quantizer.load_state_dict(checkpoint['quantizer_state_dict'])
    
    if quantizer is not None:
        quantizer.eval()
    
    return backbone, quantizer, checkpoint


def _create_quantizer(quantizer_type, feature_dim=2048, device='cuda'):
    """Create quantizer based on type."""
    
    if quantizer_type == 'vq':
        quantizer = VQWithProjection(
            input_dim=feature_dim,
            codebook_size=512,
            bottleneck_dim=64
        )
    elif quantizer_type == 'fsq':
        quantizer = FSQWithProjection(
            input_dim=feature_dim,
            levels=[8, 5, 5, 5],
            bottleneck_dim=256
        )
    elif quantizer_type == 'lfq':
        quantizer = LFQWithProjection(
            input_dim=feature_dim,
            bottleneck_dim=64,
            codebook_size=512
        )
    elif quantizer_type in ['residualvq', 'residualvq_no_proj']:
        quantizer = ResidualVQWithProjection(
            input_dim=feature_dim,
            num_quantizers=4,
            codebook_size=256,
            bottleneck_dim=64
        )
    else:
        raise ValueError(f"Unknown quantizer type: {quantizer_type}")
    
    quantizer.to(device)
    return quantizer


def find_checkpoints(checkpoints_dir, extensions=['.pth', '.ckpt']):
    """
    Find all checkpoint files in a directory.
    
    Args:
        checkpoints_dir: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        list: List of checkpoint paths
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints = []
    
    for ext in extensions:
        checkpoints.extend(checkpoints_dir.glob(f'*{ext}'))
    
    return sorted(checkpoints)

