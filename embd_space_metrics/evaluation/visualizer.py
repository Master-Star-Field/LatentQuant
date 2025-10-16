"""Visualization and result saving utilities."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


def save_results(results, output_path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary from MetricsEvaluator
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def load_results(results_path):
    """
    Load evaluation results from JSON file.
    
    Args:
        results_path: Path to JSON file
        
    Returns:
        dict: Results dictionary
    """
    with open(results_path, 'r') as f:
        return json.load(f)


def visualize_results(results, output_path=None, figsize=(12, 6)):
    """
    Visualize evaluation results as bar plots.
    
    Args:
        results: Results dictionary from MetricsEvaluator
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    # Extract model names and metrics
    model_names = list(results.keys())
    metric_names = list(results[model_names[0]].keys())
    
    # Create subplots (one per metric)
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols//3, figsize[1]*n_rows))
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        
        scores = [results[model][metric_name] for model in model_names]
        
        # Create bar plot
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, scores, alpha=0.7)
        
        # Color bars based on score
        colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize plot
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(f'{metric_name.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (white text with black outline)
        for i, (x, score) in enumerate(zip(x_pos, scores)):
            # Add white text with black stroke for better visibility
            ax.text(x, score + 0.02*max(scores), f'{score:.3f}', 
                   ha='center', va='bottom', fontsize=8,
                   color='white', weight='bold',
                   path_effects=[
                       path_effects.Stroke(linewidth=2, foreground='black'),
                       path_effects.Normal()
                   ])
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_results_table(results):
    """
    Print results as a formatted table.
    
    Args:
        results: Results dictionary from MetricsEvaluator
    """
    # Extract model names and metrics
    model_names = list(results.keys())
    metric_names = list(results[model_names[0]].keys())
    
    # Calculate column widths
    model_col_width = max(len('Model'), max(len(name) for name in model_names))
    metric_col_widths = {
        metric: max(len(metric), 8)  # Min width 8 for "0.XXXX"
        for metric in metric_names
    }
    
    # Print header
    header = f"{'Model':<{model_col_width}}"
    for metric in metric_names:
        header += f" | {metric:>{metric_col_widths[metric]}}"
    
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))
    
    # Print rows
    for model_name in model_names:
        row = f"{model_name:<{model_col_width}}"
        for metric in metric_names:
            score = results[model_name][metric]
            row += f" | {score:>{metric_col_widths[metric]}.4f}"
        print(row)
    
    print("="*len(header) + "\n")


def find_best_models(results, metric_name):
    """
    Find best performing models for a given metric.
    
    Args:
        results: Results dictionary
        metric_name: Metric to rank by
        
    Returns:
        list: Sorted list of (model_name, score) tuples
    """
    scores = [
        (model_name, model_results[metric_name])
        for model_name, model_results in results.items()
    ]
    
    # Sort by score (descending)
    return sorted(scores, key=lambda x: x[1], reverse=True)


def compare_metrics(results, model_name):
    """
    Compare all metrics for a specific model.
    
    Args:
        results: Results dictionary
        model_name: Name of model to analyze
        
    Returns:
        dict: Metric scores for the model
    """
    if model_name not in results:
        raise ValueError(f"Model '{model_name}' not found in results")
    
    return results[model_name]

