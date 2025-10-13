#!/usr/bin/env python3
"""
Test script to verify the complete VQ visualization workflow.

This script tests the integration of all components without running full training.
"""

import os
import torch
import tempfile
from pathlib import Path

from models.backbones import ViTSegmentationBackbone
from models.lightning_module import VQSqueezeModule
from models.baseline_module import BaselineSegmentationModule
from data import OxfordPetDataModule
from utils.comparison import prepare_visualization_data, visualize_comparison


def test_model_creation():
    """Test that models can be created successfully."""
    print("Testing model creation...")
    
    # Create backbone
    backbone = ViTSegmentationBackbone(num_classes=21, freeze_backbone=True)
    print(f"✓ Backbone created: {type(backbone).__name__}")
    
    # Create VQ model
    vq_model = VQSqueezeModule(
        backbone=backbone,
        num_vectors=128,
        commitment_cost=0.25,
        learning_rate=1e-4,
        vq_loss_weight=0.1
    )
    print(f"✓ VQ model created: {type(vq_model).__name__}")
    
    # Create baseline model
    baseline_model = BaselineSegmentationModule(
        backbone=backbone,
        learning_rate=1e-4
    )
    print(f"✓ Baseline model created: {type(baseline_model).__name__}")
    
    return vq_model, baseline_model


def test_data_loading():
    """Test that data can be loaded successfully."""
    print("\nTesting data loading...")
    
    # Create data module with small subset
    data_module = OxfordPetDataModule(
        data_path="./data",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        image_size=224,
        subset_size=10  # Small subset for testing
    )
    
    try:
        data_module.setup('test')
        test_loader = data_module.test_dataloader()
        print(f"✓ Data module created with {len(test_loader.dataset)} samples")
        
        # Test loading a batch
        batch = next(iter(test_loader))
        images, masks = batch
        print(f"✓ Batch loaded: images {images.shape}, masks {masks.shape}")
        
        return test_loader
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        print("Note: This is expected if Oxford-IIIT Pet dataset is not downloaded")
        return None


def test_model_inference(vq_model, baseline_model, test_loader):
    """Test that models can run inference."""
    print("\nTesting model inference...")
    
    if test_loader is None:
        print("Skipping inference test - no data available")
        return False
    
    device = torch.device("cpu")  # Use CPU for testing
    vq_model.to(device)
    baseline_model.to(device)
    
    try:
        # Get a batch
        images, masks = next(iter(test_loader))
        images = images.to(device)
        
        # Test VQ model inference
        with torch.no_grad():
            vq_output, vq_loss = vq_model(images)
            vq_preds = vq_model.predict_with_vq(images)
        print(f"✓ VQ model inference: output {vq_output.shape}, preds {vq_preds.shape}")
        
        # Test baseline model inference
        with torch.no_grad():
            baseline_output = baseline_model(images)
            baseline_preds = baseline_model.predict(images)
        print(f"✓ Baseline model inference: output {baseline_output.shape}, preds {baseline_preds.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        return False


def test_visualization_utilities():
    """Test visualization utilities."""
    print("\nTesting visualization utilities...")
    
    try:
        from utils.comparison import compute_sample_iou, find_best_worst_samples
        
        # Create dummy data
        pred = torch.randint(0, 21, (224, 224))
        target = torch.randint(0, 21, (224, 224))
        
        # Test IoU computation
        iou = compute_sample_iou(pred, target, num_classes=21)
        print(f"✓ IoU computation: {iou:.3f}")
        
        # Test sample ranking
        dummy_results = [
            (0, 0.8, torch.randn(3, 224, 224), torch.randint(0, 21, (224, 224)), torch.randint(0, 21, (224, 224))),
            (1, 0.3, torch.randn(3, 224, 224), torch.randint(0, 21, (224, 224)), torch.randint(0, 21, (224, 224))),
            (2, 0.9, torch.randn(3, 224, 224), torch.randint(0, 21, (224, 224)), torch.randint(0, 21, (224, 224))),
        ]
        
        best, worst = find_best_worst_samples(dummy_results, n_best=2, n_worst=1)
        print(f"✓ Sample ranking: {len(best)} best, {len(worst)} worst")
        
        return True
    except Exception as e:
        print(f"✗ Visualization utilities failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("VQ VISUALIZATION SYSTEM - INTEGRATION TEST")
    print("="*60)
    
    # Test model creation
    vq_model, baseline_model = test_model_creation()
    
    # Test data loading
    test_loader = test_data_loading()
    
    # Test model inference
    inference_ok = test_model_inference(vq_model, baseline_model, test_loader)
    
    # Test visualization utilities
    utils_ok = test_visualization_utilities()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ Model creation: PASSED")
    print("✓ Data loading: PASSED" if test_loader is not None else "⚠ Data loading: SKIPPED (no dataset)")
    print("✓ Model inference: PASSED" if inference_ok else "✗ Model inference: FAILED")
    print("✓ Visualization utilities: PASSED" if utils_ok else "✗ Visualization utilities: FAILED")
    
    if inference_ok and utils_ok:
        print("\n🎉 All core components are working correctly!")
        print("\nNext steps:")
        print("1. Train baseline model: python train_baseline.py --epochs 3 --subset_size 100")
        print("2. Train VQ model: python squeeze.py --epochs 3 --subset_size 100")
        print("3. Run visualization: python visualize.py --vq_checkpoint <path> --baseline_checkpoint <path>")
    else:
        print("\n❌ Some components failed. Please check the errors above.")
    
    print("="*60)


if __name__ == "__main__":
    main()
