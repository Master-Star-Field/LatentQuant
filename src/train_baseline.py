#!/usr/bin/env python3
"""
CLI script for baseline segmentation training without VQ.

This script directly uses the existing backbone's trainable classifier head
while keeping the backbone frozen.

Usage:
    python train_baseline.py --model vit --dataset oxford_pet --epochs 3
"""

import argparse
import os
import random
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.backbones import ViTSegmentationBackbone, DeepLabV3SegmentationBackbone
from models.baseline_module import BaselineSegmentationModule
from data import OxfordPetDataModule
from configs.default import get_default_config, update_config_from_args
from loggers import setup_clearml, ClearMLLogger, ClearMLUploadCallback


# Use imported BaselineSegmentationModule instead of defining it here
_UNUSED_BaselineSegmentationModule = BaselineSegmentationModule


class _DeprecatedBaselineSegmentationModule(pl.LightningModule):
    """
    Simple Lightning module that wraps existing backbone directly.
    
    The backbone already has a trainable classifier when freeze_backbone=True.
    """
    
    def __init__(
        self,
        backbone,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = backbone
        self.learning_rate = learning_rate
        
        # Segmentation loss
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        
        # Verify backbone setup
        self._verify_backbone_setup()

    def _verify_backbone_setup(self):
        """Verify that backbone is properly configured."""
        total_params = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Backbone parameter summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Freeze ratio: {frozen_params/total_params*100:.1f}%")
        
        if trainable_params == 0:
            raise ValueError("No trainable parameters found! Check backbone configuration.")

    def forward(self, images):
        """Forward pass through backbone (backbone + classifier)."""
        return self.backbone(images)

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        # Forward pass through backbone (frozen backbone + trainable classifier)
        output = self(images)
        
        # Extract logits from dict if needed
        if isinstance(output, dict):
            logits = output['out']
        else:
            logits = output
        
        # Compute loss
        seg_loss = self.seg_criterion(logits, masks)
        
        # Log metrics
        self.log('train/seg_loss', seg_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return seg_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        # Forward pass
        output = self(images)
        
        # Extract logits from dict if needed
        if isinstance(output, dict):
            logits = output['out']
        else:
            logits = output
        
        # Compute loss
        seg_loss = self.seg_criterion(logits, masks)
        
        # Compute IoU (simplified pixel accuracy)
        preds = logits.argmax(dim=1)
        valid = masks != 255
        pixel_acc = (preds[valid] == masks[valid]).float().mean()
        
        # Log metrics
        self.log('val/seg_loss', seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/pixel_acc', pixel_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return seg_loss

    def configure_optimizers(self):
        """Configure optimizer - only trainable parameters."""
        # Only optimize trainable params (classifier)
        params = [p for p in self.parameters() if p.requires_grad]
        
        print(f"Optimizing {len(params)} parameter groups")
        print(f"Total trainable parameters: {sum(p.numel() for p in params):,}")
        
        return torch.optim.Adam(params, lr=self.learning_rate)

    def predict(self, images):
        """Predict segmentation masks."""
        self.eval()
        with torch.no_grad():
            output = self(images)
            if isinstance(output, dict):
                logits = output['out']
            else:
                logits = output
            predictions = logits.argmax(dim=1)
        return predictions


def create_backbone(config):
    """Create segmentation backbone based on config."""
    # Auto-detect feature_dim based on backbone if not set or invalid
    if config.model.backbone.lower() == "vit":
        # ViT uses 768-dim features
        if config.model.feature_dim is None or config.model.feature_dim == 2048:
            config.model.feature_dim = 768
        backbone = ViTSegmentationBackbone(
            num_classes=config.model.num_classes,
            freeze_backbone=True  # Always freeze backbone, train only classifier
        )
    elif config.model.backbone.lower() == "deeplab":
        # DeepLab uses 2048-dim features
        if config.model.feature_dim is None:
            config.model.feature_dim = 2048
        backbone = DeepLabV3SegmentationBackbone(
            weights_name=config.model.deeplab_weights,
            num_classes=config.model.num_classes,
            freeze_backbone=True  # Always freeze backbone, train only classifier
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


def setup_logging_and_callbacks(config):
    """Setup logging and callbacks."""
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup ClearML
    if config.logger.use_clearml:
        clearml_task = setup_clearml(
            project_name=config.logger.project_name,
            task_name=config.logger.task_name
        )
    else:
        clearml_task = None
    
    # Setup ClearML
    clearml_task = setup_clearml(
        project_name=config.logger.project_name,
        task_name=config.logger.task_name
    )
    
    # Create ClearML logger wrapper
    clearml_logger = ClearMLLogger(clearml_task) if clearml_task else None
       
    # TensorBoard logger for ClearML auto-logging
    pl_logger = TensorBoardLogger(
        save_dir=config.output_dir,
        name=config.experiment_name,
        version=None
    )
    
    # Callbacks
    monitor_metric = 'val/loss'  # Unified metric name
    checkpoint_dir = os.path.join(config.output_dir, config.experiment_name)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val/loss:.2f}',
        monitor=monitor_metric,
        mode='min',  # Minimize loss
        save_top_k=config.training.save_top_k,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        mode='min',  # Minimize loss
        patience=5,
        verbose=True
    )
    
    callbacks = [checkpoint_callback, early_stop_callback]
    
    # Add ClearML upload callback if task exists
    if clearml_task:
        clearml_upload_callback = ClearMLUploadCallback(
            task=clearml_task,
            clearml_logger=clearml_logger,
            checkpoint_dir=checkpoint_dir,
            embedding_dir="embeddings"
        )
        callbacks.append(clearml_upload_callback)
        print("ClearML upload callback enabled")
    
    return pl_logger, clearml_logger, callbacks


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Baseline Segmentation Training")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="vit", 
                       choices=["vit", "deeplab"], help="Backbone model")
    parser.add_argument("--num_classes", type=int, default=21,
                       help="Number of classes")
    parser.add_argument("--loss_type", type=str, default="ce",
                       choices=["ce", "dice", "focal", "combined"], help="Loss function type")
    
    # Logger arguments
    parser.add_argument("--use_clearml", action="store_true",
                       help="Use ClearML for logging")
    parser.add_argument("--project_name", type=str, default="embeddings_squeeze",
                       help="Project name for logging")
    parser.add_argument("--task_name", type=str, default=None,
                       help="Task name for logging (defaults to experiment_name)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Limit number of batches per epoch for train/val/test")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="oxford_pet",
                       choices=["oxford_pet"], help="Dataset name")
    parser.add_argument("--data_path", type=str, default="./data",
                       help="Path to dataset")
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Subset size for quick testing")
    
    # Experiment arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="segmentation_baseline",
                       help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    
    # Create configuration
    config = get_default_config()
    
    # Set task_name from experiment_name if not provided
    args_dict = vars(args)
    if args_dict.get('task_name') is None:
        args_dict['task_name'] = args_dict.get('experiment_name', 'segmentation_baseline')
    
    config = update_config_from_args(config, args_dict)
    
    # Override experiment name to indicate baseline
    config.experiment_name = f"{config.experiment_name}"
    
    print(f"Starting baseline experiment: {config.experiment_name}")
    print(f"Model: {config.model.backbone}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Loss type: {config.model.loss_type}")
    print(f"Epochs: {config.training.epochs}")
    print("Training strategy: Frozen backbone + trainable classifier")
    
    # Create components
    backbone = create_backbone(config)
    data_module = create_data_module(config)
    
    # Setup logging and callbacks (do this before creating model to get clearml_logger)
    pl_logger, clearml_logger, callbacks = setup_logging_and_callbacks(config)
    
    model = BaselineSegmentationModule(
        backbone=backbone,
        num_classes=config.model.num_classes,
        learning_rate=config.training.learning_rate,
        loss_type=config.model.loss_type,
        class_weights=config.model.class_weights,
        clearml_logger=clearml_logger
    )

    # Setup data
    data_module.setup('fit')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        logger=pl_logger,
        callbacks=callbacks,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        accelerator='auto', devices='auto',
        precision=16 if torch.cuda.is_available() else 32,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Finalize ClearML logging
    if clearml_logger:
        print("Finalizing ClearML task...")
        clearml_logger.finalize()
    
    print(f"Baseline training completed!")
    print(f"Results saved to: {config.output_dir}/{config.experiment_name}")


if __name__ == "__main__":
    main()
