"""Main evaluation logic for metrics."""

import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from embeddings_squeeze.data import OxfordPetDataModule

from ..metrics import create_metric
from ..extraction import (
    load_checkpoint,
    load_original_backbone,
    find_checkpoints,
    extract_features_from_backbone,
    extract_quantized_features
)


class MetricsEvaluator:
    """
    Main evaluator for comparing original and quantized features.
    
    This class handles:
    - Loading models from checkpoints
    - Extracting features from original and quantized models
    - Computing similarity metrics
    - Organizing results
    """
    
    def __init__(
        self,
        checkpoints_dir,
        data_dir='./data',
        split='trainval',
        metrics=None,
        batch_size=8,
        num_workers=4,
        device=None,
        backbone_type='deeplab'
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoints_dir: Directory containing model checkpoints
            data_dir: Path to dataset
            split: Dataset split to use ('train', 'test', 'trainval')
            metrics: List of metric names (default: all available)
            batch_size: Batch size for data loading
            num_workers: Number of data loading workers
            device: Torch device (auto-detect if None)
            backbone_type: Backbone architecture ('deeplab' or 'vit')
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.backbone_type = backbone_type
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Setup metrics
        if metrics is None:
            from ..metrics import list_available_metrics
            self.metrics = list_available_metrics()
        else:
            self.metrics = metrics
        
        # Setup data
        self._setup_data()
    
    def _setup_data(self):
        """Setup data module and loader."""
        self.data_module = OxfordPetDataModule(
            data_path=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        # Setup datasets
        if self.split in ['train', 'trainval']:
            self.data_module.setup(stage='fit')
        if self.split == 'test':
            self.data_module.setup(stage='test')
        
        # Get appropriate dataloader
        if self.split == 'train':
            self.data_loader = self.data_module.train_dataloader()
        elif self.split == 'test':
            self.data_loader = self.data_module.test_dataloader()
        elif self.split == 'trainval':
            # Use validation split for full trainval evaluation
            self.data_loader = self.data_module.val_dataloader()
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def evaluate(self, verbose=True):
        """
        Run evaluation on all checkpoints.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            dict: Results dictionary with structure:
                  {model_name: {metric_name: score, ...}, ...}
        """
        # Find all checkpoints
        checkpoints = find_checkpoints(self.checkpoints_dir)
        
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {self.checkpoints_dir}")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"EVALUATION SUMMARY")
            print(f"{'='*80}")
            print(f"Checkpoints directory: {self.checkpoints_dir}")
            print(f"Found {len(checkpoints)} checkpoint(s)")
            print(f"Dataset split: {self.split}")
            print(f"Metrics: {', '.join(self.metrics)}")
            print(f"Device: {self.device}")
            print(f"{'='*80}\n")
        
        # Load original backbone
        if verbose:
            print("Loading original backbone...")
        
        original_backbone = load_original_backbone(
            backbone_type=self.backbone_type,
            device=self.device
        )
        
        # Extract original features once
        if verbose:
            print("Extracting original features...")
        
        original_features = extract_features_from_backbone(
            backbone=original_backbone,
            data_loader=self.data_loader,
            device=self.device,
            pool_spatial=True,
            desc="Original features"
        )
        
        # Evaluate each checkpoint
        results = {}
        
        for checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints", disable=not verbose):
            model_name = checkpoint_path.stem
            
            if verbose:
                print(f"\n{'-'*80}")
                print(f"Evaluating: {model_name}")
                print(f"{'-'*80}")
            
            # Load checkpoint
            backbone, quantizer, _ = load_checkpoint(
                checkpoint_path,
                device=self.device,
                backbone_type=self.backbone_type
            )
            
            # Extract quantized features
            quantized_features = extract_quantized_features(
                backbone=backbone,
                quantizer=quantizer,
                data_loader=self.data_loader,
                device=self.device,
                pool_spatial=True,
                desc=f"Quantized features ({model_name})"
            )
            
            # Compute metrics
            model_results = {}
            
            for metric_name in tqdm(self.metrics, desc="Computing metrics", leave=False, disable=not verbose):
                metric = create_metric(metric_name, device=self.device)
                score = metric.compute(original_features, quantized_features)
                model_results[metric_name] = float(score)
                
                if verbose:
                    print(f"  {metric_name}: {score:.4f}")
            
            results[model_name] = model_results
        
        if verbose:
            print(f"\n{'='*80}")
            print("EVALUATION COMPLETE")
            print(f"{'='*80}\n")
        
        return results
    
    def evaluate_single_metric(self, metric_name, verbose=True):
        """
        Evaluate a single metric on all checkpoints.
        
        Args:
            metric_name: Name of the metric to compute
            verbose: Whether to print progress
            
        Returns:
            dict: Results {model_name: score, ...}
        """
        # Temporarily set metrics to single metric
        original_metrics = self.metrics
        self.metrics = [metric_name]
        
        # Run evaluation
        results = self.evaluate(verbose=verbose)
        
        # Restore original metrics
        self.metrics = original_metrics
        
        # Flatten results to just scores
        return {
            model_name: model_results[metric_name]
            for model_name, model_results in results.items()
        }

