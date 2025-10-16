#!/usr/bin/env python3
"""
Visualization script for comparing VQ vs baseline segmentation results.

Usage:
    python visualize.py --vq_checkpoint ./outputs/vit_vq_256/version_0/last.ckpt --baseline_checkpoint ./outputs/vit_baseline/version_0/last.ckpt
    
Examples:
    # Compare ViT VQ vs ViT baseline
    python visualize.py --vq_checkpoint ./outputs/vit_vq_256/version_0/last.ckpt --baseline_checkpoint ./outputs/vit_baseline/version_0/last.ckpt --model vit
    
    # Compare DeepLab VQ vs DeepLab baseline  
    python visualize.py --vq_checkpoint ./outputs/deeplab_vq_512/version_0/last.ckpt --baseline_checkpoint ./outputs/deeplab_baseline/version_0/last.ckpt --model deeplab
"""

import argparse
import os
import torch
import pytorch_lightning as pl
from pathlib import Path

from models.backbones import ViTSegmentationBackbone, DeepLabV3SegmentationBackbone
from models.lightning_module import VQSqueezeModule
from models.baseline_module import BaselineSegmentationModule
from data import OxfordPetDataModule
from utils.comparison import prepare_visualization_data, visualize_comparison
from configs.default import get_default_config, update_config_from_args


def create_backbone(config):
    """Create segmentation backbone based on config."""
    if config.model.backbone.lower() == "vit":
        backbone = ViTSegmentationBackbone(
            num_classes=config.model.num_classes,
            freeze_backbone=True  # Always freeze backbone for visualization
        )
    elif config.model.backbone.lower() == "deeplab":
        backbone = DeepLabV3SegmentationBackbone(
            weights_name=config.model.deeplab_weights,
            num_classes=config.model.num_classes,
            freeze_backbone=True  # Always freeze backbone for visualization
        )
    else:
        raise ValueError(f"Unknown backbone: {config.model.backbone}")
    
    return backbone


def create_data_module(config):
    """Create data module based on config."""
    if config.data.dataset.lower() == "oxford_pet":
        data_module = OxfordPetDataModule(
            data_path=config.data.data_path,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            image_size=config.data.image_size,
            subset_size=config.data.subset_size
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")
    
    return data_module


def load_models(vq_checkpoint_path: str, baseline_checkpoint_path: str, config, device):
    """
    Load VQ and baseline models from checkpoints.
    
    Args:
        vq_checkpoint_path: Path to VQ model checkpoint
        baseline_checkpoint_path: Path to baseline model checkpoint
        config: Configuration object
        device: Device to load models on
        
    Returns:
        vq_model: Loaded VQ model
        baseline_model: Loaded baseline model
    """
    # Check if checkpoint files exist
    if not os.path.exists(vq_checkpoint_path):
        raise FileNotFoundError(f"VQ checkpoint not found: {vq_checkpoint_path}")
    if not os.path.exists(baseline_checkpoint_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint_path}")
    
    print(f"Loading VQ model from: {vq_checkpoint_path}")
    try:
        vq_model = VQSqueezeModule.load_from_checkpoint(vq_checkpoint_path)
        vq_model.to(device)
        vq_model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load VQ model: {e}")
    
    print(f"Loading baseline model from: {baseline_checkpoint_path}")
    try:
        baseline_model = BaselineSegmentationModule.load_from_checkpoint(baseline_checkpoint_path)
        baseline_model.to(device)
        baseline_model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load baseline model: {e}")
    
    return vq_model, baseline_model


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="VQ vs Baseline Segmentation Visualization")
    
    # Required checkpoint paths
    parser.add_argument("--vq_checkpoint", type=str, required=True,
                       help="Path to VQ model checkpoint")
    parser.add_argument("--baseline_checkpoint", type=str, required=True,
                       help="Path to baseline model checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset_split", type=str, default="test",
                       choices=["test", "val"], help="Dataset split to use")
    parser.add_argument("--dataset", type=str, default="oxford_pet",
                       choices=["oxford_pet"], help="Dataset name")
    parser.add_argument("--data_path", type=str, default="./data",
                       help="Path to dataset")
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Subset size for quick testing")
    
    # Visualization arguments
    parser.add_argument("--n_best", type=int, default=5,
                       help="Number of best results to show")
    parser.add_argument("--n_worst", type=int, default=5,
                       help="Number of worst results to show")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Output directory for visualization figures")
    
    # Model arguments (should match training config)
    parser.add_argument("--model", type=str, default="vit",
                       choices=["vit", "deeplab"], help="Backbone model")
    parser.add_argument("--num_classes", type=int, default=21,
                       help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    # Additional model arguments
    parser.add_argument("--deeplab_weights", type=str, default="COCO_WITH_VOC_LABELS_V1",
                       help="DeepLab weights name")
    parser.add_argument("--vit_weights", type=str, default="IMAGENET1K_V1",
                       help="ViT weights name")
    
    args = parser.parse_args()
    
    # Create configuration
    config = get_default_config()
    config = update_config_from_args(config, vars(args))
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    vq_model, baseline_model = load_models(
        args.vq_checkpoint, 
        args.baseline_checkpoint, 
        config, 
        device
    )
    
    # Create data module
    data_module = create_data_module(config)
    if args.dataset_split == "test":
        data_module.setup("test")
    else:
        data_module.setup("fit")
    
    # Get appropriate dataloader
    if args.dataset_split == "test":
        dataloader = data_module.test_dataloader()
    else:
        dataloader = data_module.val_dataloader()
    
    print(f"Evaluating on {args.dataset_split} split with {len(dataloader.dataset)} samples")
    
    # Prepare visualization data
    best_samples, worst_samples = prepare_visualization_data(
        vq_model=vq_model,
        baseline_model=baseline_model,
        dataloader=dataloader,
        device=device,
        num_classes=config.model.num_classes,
        n_best=args.n_best,
        n_worst=args.n_worst
    )
    
    # Create visualizations
    print(f"Creating visualization for {len(best_samples)} best samples...")
    best_output_path = os.path.join(args.output_dir, f"best_results_{args.dataset_split}.png")
    visualize_comparison(
        samples=best_samples,
        title=f"Best VQ Results ({args.dataset_split} split)",
        output_path=best_output_path,
        num_classes=config.model.num_classes
    )
    
    print(f"Creating visualization for {len(worst_samples)} worst samples...")
    worst_output_path = os.path.join(args.output_dir, f"worst_results_{args.dataset_split}.png")
    visualize_comparison(
        samples=worst_samples,
        title=f"Worst VQ Results ({args.dataset_split} split)",
        output_path=worst_output_path,
        num_classes=config.model.num_classes
    )
    
    # Print summary statistics
    best_ious = [sample[1] for sample in best_samples]
    worst_ious = [sample[1] for sample in worst_samples]
    
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Dataset split: {args.dataset_split}")
    print(f"Model backbone: {config.model.backbone}")
    print(f"Number of classes: {config.model.num_classes}")
    print(f"VQ checkpoint: {args.vq_checkpoint}")
    print(f"Baseline checkpoint: {args.baseline_checkpoint}")
    print()
    print(f"Best VQ IoU scores: {[f'{iou:.3f}' for iou in best_ious]}")
    print(f"Worst VQ IoU scores: {[f'{iou:.3f}' for iou in worst_ious]}")
    print(f"Best IoU range: {min(best_ious):.3f} - {max(best_ious):.3f}")
    print(f"Worst IoU range: {min(worst_ious):.3f} - {max(worst_ious):.3f}")
    print()
    print(f"Visualizations saved to:")
    print(f"  Best results: {best_output_path}")
    print(f"  Worst results: {worst_output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
