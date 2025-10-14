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
from models.quantizers import VQWithProjection, FSQWithProjection, LFQWithProjection, ResidualVQWithProjection
from data import OxfordPetDataModule
from utils.initialization import initialize_codebook_from_data
from utils.compression import measure_compression
from configs.default import get_default_config, update_config_from_args
from loggers import setup_clearml


def create_quantizer(config):
    """Create quantizer based on config."""
    if not config.quantizer.enabled:
        return None
    
    qtype = config.quantizer.type.lower()
    feature_dim = config.model.feature_dim
    
    if qtype == 'vq':
        return VQWithProjection(
            input_dim=feature_dim,
            codebook_size=config.quantizer.codebook_size,
            bottleneck_dim=config.quantizer.bottleneck_dim,
            decay=config.quantizer.decay,
            commitment_weight=config.quantizer.commitment_weight
        )
    elif qtype == 'fsq':
        return FSQWithProjection(
            input_dim=feature_dim,
            levels=config.quantizer.levels
        )
    elif qtype == 'lfq':
        return LFQWithProjection(
            input_dim=feature_dim,
            codebook_size=config.quantizer.codebook_size,
            entropy_loss_weight=config.quantizer.entropy_loss_weight,
            diversity_gamma=config.quantizer.diversity_gamma,
            spherical=config.quantizer.spherical
        )
    elif qtype == 'rvq':
        return ResidualVQWithProjection(
            input_dim=feature_dim,
            num_quantizers=config.quantizer.num_quantizers,
            codebook_size=config.quantizer.codebook_size,
            bottleneck_dim=config.quantizer.bottleneck_dim,
            decay=config.quantizer.decay,
            commitment_weight=config.quantizer.commitment_weight
        )
    elif qtype == 'none':
        return None
    else:
        raise ValueError(f"Unknown quantizer type: {qtype}")


def create_backbone(config):
    """Create segmentation backbone based on config."""
    # Set feature_dim based on backbone if not explicitly set
    if config.model.backbone.lower() == "vit":
        if config.model.feature_dim == 2048:  # Default value, override for ViT
            config.model.feature_dim = 768
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
    
    # Logger - ClearML or TensorBoard
    if config.logger.use_clearml:
        clearml_task = setup_clearml(
            project_name=config.logger.project_name,
            task_name=config.logger.task_name
        )
        if clearml_task:
            logger = None  # ClearML handles logging automatically
            print("Using ClearML for logging")
        else:
            logger = TensorBoardLogger(
                save_dir=config.output_dir,
                name=config.experiment_name,
                version=None
            )
            print("ClearML setup failed, falling back to TensorBoard")
    else:
        logger = TensorBoardLogger(
            save_dir=config.output_dir,
            name=config.experiment_name,
            version=None
        )
        print("Using TensorBoard for logging")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_dir, config.experiment_name),
        filename='{epoch:02d}-{val/loss:.2f}',
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
    parser.add_argument("--num_classes", type=int, default=21,
                       help="Number of classes")
    parser.add_argument("--add_adapter", action="store_true",
                       help="Add adapter layers to frozen backbone")
    parser.add_argument("--feature_dim", type=int, default=None,
                       help="Feature dimension (auto-detected if not set)")
    parser.add_argument("--loss_type", type=str, default="ce",
                       choices=["ce", "dice", "focal", "combined"], help="Loss function type")
    
    # Quantizer arguments
    parser.add_argument("--quantizer_type", type=str, default="vq",
                       choices=["vq", "fsq", "lfq", "rvq", "none"], help="Quantizer type")
    parser.add_argument("--quantizer_enabled", action="store_true", default=True,
                       help="Enable quantization")
    parser.add_argument("--codebook_size", type=int, default=512,
                       help="Codebook size for VQ/LFQ/RVQ")
    parser.add_argument("--bottleneck_dim", type=int, default=64,
                       help="Bottleneck dimension for VQ/RVQ")
    parser.add_argument("--num_quantizers", type=int, default=4,
                       help="Number of quantizers for RVQ")
    
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
    
    # Set task_name from experiment_name if not provided
    args_dict = vars(args)
    if args_dict.get('task_name') is None:
        args_dict['task_name'] = args_dict.get('experiment_name', 'vq_squeeze')
    
    config = update_config_from_args(config, args_dict)
    
    print(f"Starting experiment: {config.experiment_name}")
    print(f"Model: {config.model.backbone}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Quantizer: {config.quantizer.type if config.quantizer.enabled else 'None'}")
    print(f"Loss type: {config.model.loss_type}")
    print(f"Epochs: {config.training.epochs}")
    
    # Create components
    backbone = create_backbone(config)
    quantizer = create_quantizer(config)
    data_module = create_data_module(config)
    
    model = VQSqueezeModule(
        backbone=backbone,
        quantizer=quantizer,
        num_classes=config.model.num_classes,
        learning_rate=config.training.learning_rate,
        vq_loss_weight=config.training.vq_loss_weight,
        loss_type=config.model.loss_type,
        class_weights=config.model.class_weights,
        add_adapter=config.model.add_adapter,
        feature_dim=config.model.feature_dim
    )

    # Setup data
    data_module.setup('fit')
    
    # Initialize codebook if requested (only for VQ-based quantizers)
    # Note: Codebook initialization is currently disabled in this version
    # if config.initialize_codebook and quantizer is not None:
    #     print("Initializing codebook with k-means...")
    #     initialize_codebook_from_data(
    #         quantizer,
    #         backbone,
    #         data_module.train_dataloader(max_batches=config.training.max_batches),
    #         model.device,
    #         max_samples=config.max_init_samples
    #     )
    
    # Setup logging and callbacks
    logger, callbacks = setup_logging_and_callbacks(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        accelerator='auto', devices='auto',
        precision=16 if torch.cuda.is_available() else 32,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
