"""
Default configuration classes and settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "vit"  # "vit" or "deeplab"
    num_vectors: int = 128
    commitment_cost: float = 0.25
    metric_type: str = "euclidean"  # "euclidean" or "cosine"
    num_classes: int = 21
    freeze_backbone: bool = True
    
    # ViT specific
    vit_weights: str = "IMAGENET1K_V1"
    
    # DeepLab specific  
    deeplab_weights: str = "COCO_WITH_VOC_LABELS_V1"


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 4
    max_batches: int = None
    learning_rate: float = 1e-4
    vq_loss_weight: float = 0.1
    num_workers: int = 0
    pin_memory: bool = True
    
    # Optimization
    optimizer: str = "adam"
    weight_decay: float = 0.0
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    
    # Checkpointing
    save_top_k: int = 3
    monitor: str = "val/total_loss"
    mode: str = "min"


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "oxford_pet"
    data_path: str = "./data"
    image_size: int = 224
    subset_size: Optional[int] = None  # None for full dataset
    
    # Transforms
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    experiment_name: str = "vq_squeeze"
    output_dir: str = "./outputs"
    seed: int = 42
    
    # Initialization
    initialize_codebook: bool = False
    max_init_samples: int = 50000


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def update_config_from_args(config: ExperimentConfig, args: Dict[str, Any]) -> ExperimentConfig:
    """
    Update configuration from command line arguments.
    
    Args:
        config: Base configuration
        args: Command line arguments
        
    Returns:
        Updated configuration
    """
    # Model config
    if "model" in args:
        config.model.backbone = args["model"]
    if "num_vectors" in args:
        config.model.num_vectors = args["num_vectors"]
    if "commitment_cost" in args:
        config.model.commitment_cost = args["commitment_cost"]
    if "metric_type" in args:
        config.model.metric_type = args["metric_type"]
    
    # Training config
    if "epochs" in args:
        config.training.epochs = args["epochs"]
    if "batch_size" in args:
        config.training.batch_size = args["batch_size"]
    if "max_batches" in args:
        config.training.max_batches = args["max_batches"]
    if "lr" in args:
        config.training.learning_rate = args["lr"]
    if "vq_loss_weight" in args:
        config.training.vq_loss_weight = args["vq_loss_weight"]
    
    # Data config
    if "dataset" in args:
        config.data.dataset = args["dataset"]
    if "data_path" in args:
        config.data.data_path = args["data_path"]
    if "subset_size" in args:
        config.data.subset_size = args["subset_size"]
    
    # Experiment config
    if "output_dir" in args:
        config.output_dir = args["output_dir"]
    if "experiment_name" in args:
        config.experiment_name = args["experiment_name"]
    if "seed" in args:
        config.seed = args["seed"]
    if "initialize_codebook" in args:
        # argparse provides this key with a boolean even if the flag is not passed
        config.initialize_codebook = args["initialize_codebook"]
    if "max_init_samples" in args and args["max_init_samples"] is not None:
        config.max_init_samples = args["max_init_samples"]
    
    return config
