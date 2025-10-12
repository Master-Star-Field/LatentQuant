#!/usr/bin/env python3
"""
Test script to verify the embeddings_squeeze package works correctly.
"""

import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from embeddings_squeeze.models.vq.quantizer import VectorQuantizer
        from embeddings_squeeze.models.vq.codebook import Codebook, DistanceMetric
        from embeddings_squeeze.models.backbones import ViTSegmentationBackbone, DeepLabV3SegmentationBackbone
        from embeddings_squeeze.models.lightning_module import VQSqueezeModule
        from embeddings_squeeze.data import OxfordPetDataModule
        from embeddings_squeeze.utils.compression import measure_compression
        from embeddings_squeeze.utils.initialization import initialize_codebook_from_data
        from embeddings_squeeze.configs import get_default_config
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_vq_components():
    """Test VQ components work correctly."""
    print("Testing VQ components...")
    
    try:
        from embeddings_squeeze.models.vq.quantizer import VectorQuantizer
        
        # Create VQ
        vq = VectorQuantizer(num_vectors=64, vector_dim=512)
        
        # Test forward pass
        batch_size = 2
        features = torch.randn(batch_size, 512)
        
        quantized, loss = vq(features)
        
        assert quantized.shape == features.shape, f"Shape mismatch: {quantized.shape} vs {features.shape}"
        assert loss.item() > 0, "Loss should be positive"
        
        print("✓ VQ components work correctly")
        return True
    except Exception as e:
        print(f"✗ VQ test failed: {e}")
        return False

def test_backbone():
    """Test backbone creation."""
    print("Testing backbone creation...")
    
    try:
        from embeddings_squeeze.models.backbones import ViTSegmentationBackbone
        
        # Create backbone
        backbone = ViTSegmentationBackbone(num_classes=21)
        
        # Test feature extraction
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        features = backbone.extract_features(images)
        assert features.shape[0] == batch_size, f"Batch size mismatch: {features.shape[0]} vs {batch_size}"
        assert features.shape[1] == backbone.feature_dim, f"Feature dim mismatch: {features.shape[1]} vs {backbone.feature_dim}"
        
        print("✓ Backbone works correctly")
        return True
    except Exception as e:
        print(f"✗ Backbone test failed: {e}")
        return False

def test_lightning_module():
    """Test Lightning module creation."""
    print("Testing Lightning module...")
    
    try:
        from embeddings_squeeze.models.backbones import ViTSegmentationBackbone
        from embeddings_squeeze.models.lightning_module import VQSqueezeModule
        
        # Create backbone and model
        backbone = ViTSegmentationBackbone(num_classes=21)
        model = VQSqueezeModule(
            backbone=backbone,
            num_vectors=64,
            commitment_cost=0.25
        )
        
        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        masks = torch.randint(0, 21, (batch_size, 1, 224, 224))
        
        output, vq_loss = model(images)
        
        assert output.shape == (batch_size, 21, 224, 224), f"Output shape mismatch: {output.shape}"
        assert vq_loss.item() > 0, "VQ loss should be positive"
        
        print("✓ Lightning module works correctly")
        return True
    except Exception as e:
        print(f"✗ Lightning module test failed: {e}")
        return False

def test_config():
    """Test configuration system."""
    print("Testing configuration...")
    
    try:
        from embeddings_squeeze.configs import get_default_config
        
        config = get_default_config()
        
        assert config.model.backbone == "vit", f"Default backbone should be 'vit', got {config.model.backbone}"
        assert config.model.num_vectors == 128, f"Default num_vectors should be 128, got {config.model.num_vectors}"
        assert config.training.epochs == 10, f"Default epochs should be 10, got {config.training.epochs}"
        
        print("✓ Configuration works correctly")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running embeddings_squeeze package tests...\n")
    
    tests = [
        test_imports,
        test_vq_components,
        test_backbone,
        test_lightning_module,
        test_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready to use.")
        print("\nTo install the package:")
        print("pip install -e .")
        print("\nTo run training:")
        print("python squeeze.py --model vit --dataset oxford_pet --num_vectors 128 --epochs 3")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
