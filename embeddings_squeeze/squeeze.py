#!/usr/bin/env python3
"""
CLI script for VQ compression training.

Usage:
    python squeeze.py --model vit --dataset oxford_pet --num_vectors 128 --epochs 3
"""

import argparse
import os
import random
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.backbones import ViTSegmentationBackbone, DeepLabV3SegmentationBackbone
from models.lightning_module import VQSqueezeModule
from data import OxfordPetDataModule
from utils.initialization import initialize_codebook_from_data
from utils.compression import measure_compression
from configs.default import get_default_config, update_config_from_args


def create_backbone(config):
    """Create segmentation backbone based on config."""
    if config.model.backbone.lower() == "vit":
        backbone = ViTSegmentationBackbone(
            num_classes=config.model.num_classes,
            freeze_backbone=config.model.freeze_backbone
        )
    elif config.model.backbone.lower() == "deeplab":
        backbone = DeepLabV3SegmentationBackbone(
            weights_name=config.model.deeplab_weights,
            num_classes=config.model.num_classes,
            freeze_backbone=config.model.freeze_backbone
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
        filename='{epoch:02d}-{val/total_loss:.2f}',
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
    parser = argparse.ArgumentParser(description="VQ Compression Training")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="vit", 
                       choices=["vit", "deeplab"], help="Backbone model")
    parser.add_argument("--num_vectors", type=int, default=128, 
                       help="Number of codebook vectors")
    parser.add_argument("--commitment_cost", type=float, default=0.25,
                       help="VQ commitment cost")
    parser.add_argument("--metric_type", type=str, default="euclidean",
                       choices=["euclidean", "cosine"], help="Distance metric")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--vq_loss_weight", type=float, default=0.1,
                       help="VQ loss weight")
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
    parser.add_argument("--experiment_name", type=str, default="vq_squeeze",
                       help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Other arguments
    parser.add_argument("--initialize_codebook", action="store_true",
                       help="Initialize codebook with k-means")
    parser.add_argument("--max_init_samples", type=int, default=50000,
                       help="Max samples for codebook initialization")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    
    # Create configuration
    config = get_default_config()
    config = update_config_from_args(config, vars(args))
    
    print(f"Starting experiment: {config.experiment_name}")
    print(f"Model: {config.model.backbone}")
    print(f"Dataset: {config.data.dataset}")
    print(f"VQ vectors: {config.model.num_vectors}")
    print(f"Epochs: {config.training.epochs}")
    
    # Create components
    backbone = create_backbone(config)
    data_module = create_data_module(config)
    model = VQSqueezeModule(
        backbone=backbone,
        num_vectors=config.model.num_vectors,
        commitment_cost=config.model.commitment_cost,
        metric_type=config.model.metric_type,
        learning_rate=config.training.learning_rate,
        vq_loss_weight=config.training.vq_loss_weight
    )

    # Setup data
    data_module.setup('fit')
    
    # Initialize codebook if requested
    if config.initialize_codebook:
        print("Initializing codebook with k-means...")
        initialize_codebook_from_data(
            model.vq, 
            backbone, 
            data_module.train_dataloader(max_batches=config.training.max_batches), 
            model.device,
            max_samples=config.max_init_samples
        )
    
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
    
    # Test compression
    # print("Measuring compression...")
    # data_module.setup('test')
    # compression_ratio = measure_compression(
    #     model.vq,
    #     backbone, 
    #     data_module.test_dataloader(max_batches=config.training.max_batches),
    #     model.device
    # )
    # print(f"Training completed!")
    # print(f"Compression ratio: {compression_ratio:.1f}x")
    # print(f"Results saved to: {config.output_dir}/{config.experiment_name}")
    
    # print(f"Training completed!")
    # print(f"Compression ratio: {compression_ratio:.1f}x")
    # print(f"Results saved to: {config.output_dir}/{config.experiment_name}")


if __name__ == "__main__":
    main()
