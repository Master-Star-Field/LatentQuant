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
from data import OxfordPetDataModule
from configs.default import get_default_config, update_config_from_args


class BaselineSegmentationModule(pl.LightningModule):
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
    if config.model.backbone.lower() == "vit":
        backbone = ViTSegmentationBackbone(
            num_classes=config.model.num_classes,
            freeze_backbone=True  # Always freeze backbone, train only classifier
        )
    elif config.model.backbone.lower() == "deeplab":
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
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.output_dir,
        name=config.experiment_name,
        version=None
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_dir, config.experiment_name),
        filename='{epoch:02d}-{val/seg_loss:.2f}',
        monitor=config.training.monitor,
        mode=config.training.mode,
        save_top_k=config.training.save_top_k,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config.training.monitor,
        mode=config.training.mode,
        patience=5,
        verbose=True
    )
    
    callbacks = [checkpoint_callback, early_stop_callback]
    
    return logger, callbacks


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Baseline Segmentation Training")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="vit", 
                       choices=["vit", "deeplab"], help="Backbone model")
    
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
    parser.add_argument("--experiment_name", type=str, default="baseline_segmentation",
                       help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    
    # Create configuration
    config = get_default_config()
    config = update_config_from_args(config, vars(args))
    
    # Override experiment name to indicate baseline
    config.experiment_name = f"{config.experiment_name}_baseline"
    
    print(f"Starting baseline experiment: {config.experiment_name}")
    print(f"Model: {config.model.backbone}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Epochs: {config.training.epochs}")
    print("Training strategy: Frozen backbone + trainable classifier")
    
    # Create components
    backbone = create_backbone(config)
    data_module = create_data_module(config)
    model = BaselineSegmentationModule(
        backbone=backbone,
        learning_rate=config.training.learning_rate
    )

    # Setup data
    data_module.setup('fit')
    
    # Setup logging and callbacks
    logger, callbacks = setup_logging_and_callbacks(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        accelerator="gpu", devices=1,
        num_sanity_val_steps=0,
        profiler="simple",
        precision=16 if torch.cuda.is_available() else 32,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, data_module)
    
    print(f"Baseline training completed!")
    print(f"Results saved to: {config.output_dir}/{config.experiment_name}")


if __name__ == "__main__":
    main()
