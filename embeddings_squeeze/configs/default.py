"""
Default configuration classes and settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class QuantizerConfig:
    """Quantizer configuration."""
    enabled: bool = True
    type: str = "vq"  # "vq", "fsq", "lfq", "rvq", "none"
    
    # VQ-specific
    codebook_size: int = 512
    bottleneck_dim: int = 64
    decay: float = 0.99
    commitment_weight: float = 0.25
    
    # FSQ-specific
    levels: list = field(default_factory=lambda: [8, 5, 5, 5])
    
    # LFQ-specific
    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 0.1
    spherical: bool = False
    
    # RVQ-specific
    num_quantizers: int = 4


@dataclass
class LoggerConfig:
    """Logger configuration."""
    use_clearml: bool = True
    use_tensorboard: bool = False
    project_name: str = "embeddings_squeeze"
    task_name: str = "vq_compression"
    credentials_file: str = "clearml_credentials.yaml"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "vit"  # "vit" or "deeplab"
    num_classes: int = 21
    freeze_backbone: bool = True
    
    # ViT specific
    vit_weights: str = "IMAGENET1K_V1"
    
    # DeepLab specific  
    deeplab_weights: str = "COCO_WITH_VOC_LABELS_V1"
    
    # Adapter configuration
    add_adapter: bool = False
    feature_dim: int = 768  # Default for ViT
    
    # Loss configuration
    loss_type: str = "ce"  # "ce", "dice", "focal", "combined"
    class_weights: Optional[list] = None  # e.g., [0.5, 1.5, 3.0] for class imbalance


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
    monitor: str = "val/loss"
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
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    
    # Experiment metadata
    experiment_name: str = "vq_squeeze"
    output_dir: str = "./outputs"
    seed: int = 42
    
    # Initialization
    initialize_codebook: bool = True
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
    if "num_classes" in args:
        config.model.num_classes = args["num_classes"]
    if "add_adapter" in args:
        config.model.add_adapter = args["add_adapter"]
    if "feature_dim" in args:
        config.model.feature_dim = args["feature_dim"]
    if "loss_type" in args:
        config.model.loss_type = args["loss_type"]
    if "class_weights" in args:
        config.model.class_weights = args["class_weights"]
    
    # Quantizer config
    if "quantizer_type" in args:
        config.quantizer.type = args["quantizer_type"]
    if "quantizer_enabled" in args:
        config.quantizer.enabled = args["quantizer_enabled"]
    if "codebook_size" in args:
        config.quantizer.codebook_size = args["codebook_size"]
    if "bottleneck_dim" in args:
        config.quantizer.bottleneck_dim = args["bottleneck_dim"]
    if "num_quantizers" in args:
        config.quantizer.num_quantizers = args["num_quantizers"]
    
    # Logger config
    if "use_clearml" in args:
        config.logger.use_clearml = args["use_clearml"]
    if "project_name" in args:
        config.logger.project_name = args["project_name"]
    if "task_name" in args:
        config.logger.task_name = args["task_name"]
    
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
