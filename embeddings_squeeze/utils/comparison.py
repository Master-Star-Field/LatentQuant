"""
Comparison utilities for VQ vs baseline segmentation evaluation.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def compute_sample_iou(prediction: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """
    Compute IoU for a single sample.
    
    Args:
        prediction: Predicted mask [H, W]
        target: Ground truth mask [H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        iou: Mean IoU across all classes
    """
    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()
    
    # Flatten arrays
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    
    # Remove ignored pixels
    valid_mask = target_flat != ignore_index
    pred_flat = pred_flat[valid_mask]
    target_flat = target_flat[valid_mask]
    
    # Compute IoU for each class
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
            
        ious.append(iou)
    
    # Return mean IoU
    return np.mean(ious)


def evaluate_model(model, dataloader, device, num_classes: int = 21) -> List[Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Evaluate model on dataset and collect results.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to run on
        num_classes: Number of classes
        
    Returns:
        results: List of (sample_idx, iou, image, mask, prediction) tuples
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)
            
            # Get predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(images)
            else:
                # For VQ model
                predictions = model.predict_with_vq(images)
            
            # Process each sample in batch
            for i in range(images.shape[0]):
                image = images[i]
                mask = masks[i]
                pred = predictions[i]
                
                # Compute IoU
                iou = compute_sample_iou(pred, mask, num_classes)
                
                # Store results
                sample_idx = batch_idx * dataloader.batch_size + i
                results.append((sample_idx, iou, image, mask, pred))
    
    return results


def find_best_worst_samples(results: List[Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor]], 
                           n_best: int = 5, n_worst: int = 5) -> Tuple[List, List]:
    """
    Find best and worst samples based on IoU.
    
    Args:
        results: List of (sample_idx, iou, image, mask, prediction) tuples
        n_best: Number of best samples to return
        n_worst: Number of worst samples to return
        
    Returns:
        best_samples: List of best sample tuples
        worst_samples: List of worst sample tuples
    """
    # Sort by IoU (descending)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Get best and worst
    best_samples = sorted_results[:n_best]
    worst_samples = sorted_results[-n_worst:]
    
    return best_samples, worst_samples


def prepare_visualization_data(vq_model, baseline_model, dataloader, device, 
                              num_classes: int = 21, n_best: int = 5, n_worst: int = 5):
    """
    Prepare data for visualization by running both models and ranking results.
    
    Args:
        vq_model: VQ model
        baseline_model: Baseline model
        dataloader: Data loader
        device: Device to run on
        num_classes: Number of classes
        n_best: Number of best samples
        n_worst: Number of worst samples
        
    Returns:
        best_samples: List of best sample tuples with both predictions
        worst_samples: List of worst sample tuples with both predictions
    """
    # Evaluate VQ model
    print("Evaluating VQ model...")
    vq_results = evaluate_model(vq_model, dataloader, device, num_classes)
    
    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_results = evaluate_model(baseline_model, dataloader, device, num_classes)
    
    # Combine results (assuming same order)
    combined_results = []
    for (idx1, iou1, img1, mask1, pred_vq), (idx2, iou2, img2, mask2, pred_baseline) in zip(vq_results, baseline_results):
        assert idx1 == idx2, "Sample indices don't match"
        assert torch.equal(img1, img2), "Images don't match"
        assert torch.equal(mask1, mask2), "Masks don't match"
        
        combined_results.append((idx1, iou1, img1, mask1, pred_baseline, pred_vq))
    
    # Find best and worst based on VQ IoU
    best_samples, worst_samples = find_best_worst_samples(combined_results, n_best, n_worst)
    
    return best_samples, worst_samples


def denormalize_image(image: torch.Tensor, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Denormalize image for visualization.
    
    Args:
        image: Normalized image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        denormalized: Denormalized image tensor
    """
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(image, 0, 1)


def create_segmentation_colormap(num_classes: int = 21) -> ListedColormap:
    """
    Create a colormap for segmentation visualization.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        colormap: Matplotlib colormap
    """
    # Generate distinct colors
    colors = sns.color_palette("husl", num_classes)
    colors = [(0, 0, 0)] + colors  # Add black for background
    return ListedColormap(colors)


def visualize_comparison(samples: List[Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], 
                        title: str, output_path: str, num_classes: int = 21):
    """
    Create visualization comparing baseline and VQ predictions.
    
    Args:
        samples: List of (idx, iou, image, mask, pred_baseline, pred_vq) tuples
        title: Figure title
        output_path: Path to save figure
        num_classes: Number of classes
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    colormap = create_segmentation_colormap(num_classes)
    
    for i, (idx, iou, image, mask, pred_baseline, pred_vq) in enumerate(samples):
        # Denormalize image
        image_vis = denormalize_image(image)
        
        # Convert to numpy for plotting
        image_np = image_vis.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.cpu().numpy()
        pred_baseline_np = pred_baseline.cpu().numpy()
        pred_vq_np = pred_vq.cpu().numpy()
        
        # Plot original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Original Image\nSample {idx}")
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(mask_np, cmap=colormap, vmin=0, vmax=num_classes-1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Plot baseline prediction
        axes[i, 2].imshow(pred_baseline_np, cmap=colormap, vmin=0, vmax=num_classes-1)
        axes[i, 2].set_title("Baseline Prediction")
        axes[i, 2].axis('off')
        
        # Plot VQ prediction
        axes[i, 3].imshow(pred_vq_np, cmap=colormap, vmin=0, vmax=num_classes-1)
        axes[i, 3].set_title(f"VQ Prediction\nIoU: {iou:.3f}")
        axes[i, 3].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
