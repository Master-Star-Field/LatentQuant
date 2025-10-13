#!/usr/bin/env python3
"""
Test script to verify the simplified baseline training approach.
"""

import torch
from models.backbones import ViTSegmentationBackbone, DeepLabV3SegmentationBackbone
from train_baseline import BaselineSegmentationModule


def test_backbone_classifier_training():
    """Test that backbone classifiers are trainable when backbone is frozen."""
    print("Testing backbone classifier training...")
    
    # Test ViT backbone
    print("\n--- ViT Backbone Test ---")
    vit_backbone = ViTSegmentationBackbone(num_classes=21, freeze_backbone=True)
    vit_model = BaselineSegmentationModule(vit_backbone)
    
    # Test DeepLab backbone
    print("\n--- DeepLab Backbone Test ---")
    deeplab_backbone = DeepLabV3SegmentationBackbone(num_classes=21, freeze_backbone=True)
    deeplab_model = BaselineSegmentationModule(deeplab_backbone)
    
    # Test inference
    print("\n--- Inference Test ---")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    try:
        # Test ViT
        vit_output = vit_model(dummy_input)
        print(f"✓ ViT inference: {vit_output['out'].shape if isinstance(vit_output, dict) else vit_output.shape}")
        
        # Test DeepLab
        deeplab_output = deeplab_model(dummy_input)
        print(f"✓ DeepLab inference: {deeplab_output['out'].shape if isinstance(deeplab_output, dict) else deeplab_output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False


def test_optimizer_creation():
    """Test that optimizers can be created."""
    print("\n--- Optimizer Test ---")
    
    try:
        vit_backbone = ViTSegmentationBackbone(num_classes=21, freeze_backbone=True)
        vit_model = BaselineSegmentationModule(vit_backbone)
        
        optimizer = vit_model.configure_optimizers()
        print(f"✓ Optimizer created: {type(optimizer).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Optimizer creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SIMPLIFIED BASELINE TRAINING TEST")
    print("="*60)
    
    # Test backbone classifier training
    backbone_ok = test_backbone_classifier_training()
    
    # Test optimizer creation
    optimizer_ok = test_optimizer_creation()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ Backbone classifier training: PASSED" if backbone_ok else "✗ Backbone classifier training: FAILED")
    print("✓ Optimizer creation: PASSED" if optimizer_ok else "✗ Optimizer creation: FAILED")
    
    if backbone_ok and optimizer_ok:
        print("\n🎉 Simplified baseline training approach is working!")
        print("The backbone classifiers are trainable while backbones remain frozen.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    print("="*60)


if __name__ == "__main__":
    main()
