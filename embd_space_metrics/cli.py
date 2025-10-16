#!/usr/bin/env python
"""
Command-line interface for embedding space metrics evaluation.

This script provides a CLI for evaluating similarity between original and
quantized model representations using various metrics.
"""

import argparse
import sys
from pathlib import Path

from .evaluation import MetricsEvaluator, visualize_results, save_results, print_results_table
from .metrics import list_available_metrics


def create_parser():
    """Create argument parser with all commands."""
    parser = argparse.ArgumentParser(
        description='Embedding Space Metrics Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models with all metrics
  python -m embd_space_metrics.cli evaluate --checkpoints-dir ./models --split trainval

  # Evaluate with specific metrics
  python -m embd_space_metrics.cli evaluate \\
      --checkpoints-dir ./models \\
      --metrics cka pwcca rsa \\
      --output results.json

  # Evaluate on test split
  python -m embd_space_metrics.cli evaluate \\
      --checkpoints-dir ./lightning_logs \\
      --split test \\
      --backbone vit

  # List available metrics
  python -m embd_space_metrics.cli list-metrics

  # Visualize results
  python -m embd_space_metrics.cli visualize \\
      --results results.json \\
      --output plots.png
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # ============================================================================
    # EVALUATE command
    # ============================================================================
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate models using similarity metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    evaluate_parser.add_argument(
        '--checkpoints-dir',
        type=str,
        required=True,
        help='Directory containing model checkpoints (.pth or .ckpt files)'
    )
    
    # Data arguments
    data_group = evaluate_parser.add_argument_group('data arguments')
    data_group.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Path to dataset directory'
    )
    data_group.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'trainval'],
        default='trainval',
        help='Dataset split to use for evaluation'
    )
    data_group.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for data loading'
    )
    data_group.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Model arguments
    model_group = evaluate_parser.add_argument_group('model arguments')
    model_group.add_argument(
        '--backbone',
        type=str,
        choices=['deeplab', 'vit'],
        default='deeplab',
        help='Backbone architecture type'
    )
    
    # Metric arguments
    metric_group = evaluate_parser.add_argument_group('metric arguments')
    metric_group.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=None,
        help=f'Metrics to compute (default: all). Available: {", ".join(list_available_metrics())}'
    )
    
    # Output arguments
    output_group = evaluate_parser.add_argument_group('output arguments')
    output_group.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output path for results JSON file'
    )
    output_group.add_argument(
        '--visualize',
        type=str,
        default=None,
        help='Optional path to save visualization plot'
    )
    output_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    # Device arguments
    device_group = evaluate_parser.add_argument_group('device arguments')
    device_group.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    
    # ============================================================================
    # LIST-METRICS command
    # ============================================================================
    list_parser = subparsers.add_parser(
        'list-metrics',
        help='List all available metrics'
    )
    
    # ============================================================================
    # VISUALIZE command
    # ============================================================================
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Visualize evaluation results from JSON file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    visualize_parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to results JSON file'
    )
    visualize_parser.add_argument(
        '--output',
        type=str,
        default='evaluation_plot.png',
        help='Output path for visualization'
    )
    visualize_parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[12, 6],
        help='Figure size (width height)'
    )
    
    return parser


def cmd_evaluate(args):
    """Execute evaluate command."""
    print("\n" + "="*80)
    print("EMBEDDING SPACE METRICS EVALUATION")
    print("="*80)
    
    # Validate metrics
    if args.metrics:
        available = list_available_metrics()
        invalid = [m for m in args.metrics if m not in available]
        if invalid:
            print(f"\nError: Unknown metrics: {', '.join(invalid)}")
            print(f"Available metrics: {', '.join(available)}")
            return 1
    
    # Create evaluator
    evaluator = MetricsEvaluator(
        checkpoints_dir=args.checkpoints_dir,
        data_dir=args.data_dir,
        split=args.split,
        metrics=args.metrics,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        backbone_type=args.backbone
    )
    
    # Run evaluation
    results = evaluator.evaluate(verbose=not args.quiet)
    
    # Print results table
    if not args.quiet:
        print_results_table(results)
    
    # Save results
    save_results(results, args.output)
    
    # Visualize if requested
    if args.visualize:
        visualize_results(results, output_path=args.visualize, figsize=tuple(args.figsize))
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")
    
    return 0


def cmd_list_metrics(args):
    """Execute list-metrics command."""
    metrics = list_available_metrics()
    
    print("\nAvailable Metrics:")
    print("="*50)
    for metric in metrics:
        print(f"  - {metric}")
    print("="*50)
    print(f"\nTotal: {len(metrics)} metrics\n")
    
    return 0


def cmd_visualize(args):
    """Execute visualize command."""
    from .evaluation.visualizer import load_results
    
    print(f"\nLoading results from: {args.results}")
    results = load_results(args.results)
    
    print("Creating visualization...")
    visualize_results(results, output_path=args.output, figsize=tuple(args.figsize))
    
    print(f"Done!\n")
    
    return 0


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no command specified
    if args.command is None:
        parser.print_help()
        return 0
    
    # Execute command
    if args.command == 'evaluate':
        return cmd_evaluate(args)
    elif args.command == 'list-metrics':
        return cmd_list_metrics(args)
    elif args.command == 'visualize':
        return cmd_visualize(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

