#!/usr/bin/env python
"""
Quick test script to verify the metrics module works correctly.
"""

import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embd_space_metrics.metrics import create_metric, list_available_metrics


def test_metrics():
    """Test all metrics with dummy data."""
    print("\n" + "="*80)
    print("TESTING EMBEDDING SPACE METRICS MODULE")
    print("="*80 + "\n")
    
    # Create dummy features [N=100, D=512]
    torch.manual_seed(42)
    features_1 = torch.randn(100, 512)
    
    # Create similar features (with noise)
    features_2 = features_1 + 0.1 * torch.randn(100, 512)
    
    print(f"Feature shapes: {features_1.shape}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    
    # Test each metric
    available_metrics = list_available_metrics()
    print(f"Testing {len(available_metrics)} metrics:\n")
    
    results = {}
    
    for metric_name in available_metrics:
        print(f"  Testing {metric_name}...", end=" ")
        
        try:
            metric = create_metric(metric_name)
            score = metric.compute(features_1, features_2)
            results[metric_name] = score
            print(f"✓ Score: {score:.4f}")
        except Exception as e:
            print(f"✗ Error: {e}")
            results[metric_name] = None
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for metric_name, score in results.items():
        status = "✓" if score is not None else "✗"
        score_str = f"{score:.4f}" if score is not None else "FAILED"
        print(f"  {status} {metric_name:15s}: {score_str}")
    
    print("\n" + "="*80)
    
    # Check if all passed
    failed = [m for m, s in results.items() if s is None]
    if failed:
        print(f"\n⚠️  {len(failed)} metric(s) failed: {', '.join(failed)}")
        return 1
    else:
        print("\n✅ All metrics passed!")
        return 0


if __name__ == '__main__':
    sys.exit(test_metrics())

