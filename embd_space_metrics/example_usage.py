#!/usr/bin/env python
"""
Example usage of the embedding space metrics module.

This script demonstrates how to use the module programmatically.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embd_space_metrics.evaluation import MetricsEvaluator
from embd_space_metrics.evaluation import visualize_results, save_results, print_results_table


def main():
    """Run example evaluation."""
    
    # Configuration
    CHECKPOINTS_DIR = './models'  # Directory with .pth or .ckpt files
    DATA_DIR = './data'
    SPLIT = 'trainval'  # 'train', 'test', or 'trainval'
    METRICS = ['cka', 'pwcca', 'rsa']  # Or None for all metrics
    BATCH_SIZE = 8
    DEVICE = 'cuda'  # or 'cpu'
    
    print("\n" + "="*80)
    print("EXAMPLE: Embedding Space Metrics Evaluation")
    print("="*80 + "\n")
    
    # Create evaluator
    print("Creating evaluator...")
    evaluator = MetricsEvaluator(
        checkpoints_dir=CHECKPOINTS_DIR,
        data_dir=DATA_DIR,
        split=SPLIT,
        metrics=METRICS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        backbone_type='deeplab'
    )
    
    # Run evaluation
    print("\nRunning evaluation...\n")
    results = evaluator.evaluate(verbose=True)
    
    # Print results as table
    print("\nResults:")
    print_results_table(results)
    
    # Save results to JSON
    output_path = 'evaluation_results.json'
    save_results(results, output_path)
    
    # Create visualization
    plot_path = 'evaluation_plots.png'
    visualize_results(results, output_path=plot_path)
    
    print("\n" + "="*80)
    print("DONE!")
    print(f"Results saved to: {output_path}")
    print(f"Plots saved to: {plot_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

