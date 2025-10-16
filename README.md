# Embeddings Squeeze

Vector Quantization for Segmentation Model Compression using PyTorch Lightning.

This package provides a modular framework for applying Vector Quantization (VQ) to segmentation models for compression, with support for multiple backbones (ViT, DeepLabV3) and datasets.

## Features

- **Modular Architecture**: Easy to swap models and datasets
- **PyTorch Lightning**: Professional training framework with logging and checkpointing
- **Multiple Backbones**: Support for ViT and DeepLabV3-ResNet50
- **VQ Compression**: Vector Quantization with configurable codebook sizes
- **Research-Friendly**: Quick experimentation via CLI arguments
- **Extensible**: Abstract base classes for adding new models/datasets

## Installation

### From Source

```bash
git clone https://github.com/yourusername/embeddings_squeeze.git
cd embeddings_squeeze
pip install -e .
```

After installation, you can run the CLI script using either:
```bash
# Using Python module syntax
python -m embeddings_squeeze.squeeze --help

# Or using the installed console script
embeddings-squeeze --help
```

### Dependencies

The package requires:
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- torchvision >= 0.15.0
- scikit-learn >= 1.0.0
- Other dependencies listed in `requirements.txt`

## Quick Start

### Basic Usage

Train VQ compression on Oxford-IIIT Pet dataset with ViT backbone:

```bash
# After pip install -e .
embeddings-squeeze --model vit --dataset oxford_pet --codebook_size 512 --epochs 3

# Or without installation
python -m embeddings_squeeze.squeeze --model vit --dataset oxford_pet --codebook_size 512 --epochs 3
```

### Advanced Usage

```bash
embeddings-squeeze \
    --model deeplab \
    --dataset oxford_pet \
    --quantizer_type vq \
    --codebook_size 512 \
    --bottleneck_dim 64 \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --data_path ./data \
    --output_dir ./outputs \
    --experiment_name deeplab_vq_512
```

### Command Line Arguments

#### Model Arguments
- `--model`: Backbone model (`vit` or `deeplab`)
- `--quantizer_type`: Quantizer type (`vq`, `fsq`, `lfq`, `rvq`, or `none`)
- `--codebook_size`: Number of codebook vectors (default: 512)
- `--bottleneck_dim`: Bottleneck dimension (default: 64)
- `--loss_type`: Loss function (`ce`, `dice`, `focal`, or `combined`)

#### Training Arguments
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--vq_loss_weight`: Weight for VQ loss (default: 0.1)

#### Data Arguments
- `--dataset`: Dataset name (`oxford_pet`)
- `--data_path`: Path to dataset (default: `./data`)
- `--subset_size`: Subset size for quick testing (default: None)

#### Experiment Arguments
- `--output_dir`: Output directory (default: `./outputs`)
- `--experiment_name`: Experiment name (default: `vq_squeeze`)
- `--seed`: Random seed (default: 42)
- `--use_clearml`: Enable ClearML logging
- `--project_name`: Project name for ClearML (default: `embeddings_squeeze`)
- `--task_name`: Task name for ClearML

## Package Structure

```
project/
├── src/                    # Source code (mapped to embeddings_squeeze package)
│   ├── models/            # Model architectures
│   │   └── backbones/     # Segmentation backbones
│   ├── data/              # Data modules
│   ├── loggers/           # Logging utilities (ClearML integration)
│   ├── utils/             # Utility functions
│   ├── configs/           # Configuration management
│   └── squeeze.py         # Main training CLI script
├── outputs/               # Training outputs (gitignored)
├── setup.py               # Package setup
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Usage as Python Package

```python
from embeddings_squeeze.models.backbones import ViTSegmentationBackbone
from embeddings_squeeze.models.quantizers import VQWithProjection
from embeddings_squeeze.models.lightning_module import VQSqueezeModule
from embeddings_squeeze.data import OxfordPetDataModule

# Create backbone
backbone = ViTSegmentationBackbone(num_classes=21)

# Create quantizer
quantizer = VQWithProjection(
    input_dim=768,  # ViT feature dimension
    codebook_size=512,
    bottleneck_dim=64
)

# Create model
model = VQSqueezeModule(
    backbone=backbone,
    quantizer=quantizer,
    num_classes=21
)

# Create data module
data_module = OxfordPetDataModule(
    data_path="./data",
    batch_size=4
)

# Train with PyTorch Lightning
import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

## Adding New Models/Datasets

### Adding a New Backbone

1. Inherit from `SegmentationBackbone`:

```python
from embeddings_squeeze.models.backbones.base import SegmentationBackbone

class MyBackbone(SegmentationBackbone):
    def extract_features(self, images, detach=True):
        # Implementation
        pass
    
    def forward(self, images):
        # Implementation
        pass
    
    @property
    def feature_dim(self):
        return 512
    
    @property
    def num_classes(self):
        return 21
```

2. Add to `models/backbones/__init__.py`
3. Update CLI script to support new backbone

### Adding a New Dataset

1. Inherit from `BaseDataModule`:

```python
from embeddings_squeeze.data.base import BaseDataModule

class MyDataModule(BaseDataModule):
    def setup(self, stage=None):
        # Implementation
        pass
    
    def train_dataloader(self):
        # Implementation
        pass
    
    def val_dataloader(self):
        # Implementation
        pass
    
    def test_dataloader(self):
        # Implementation
        pass
```

2. Add to `data/__init__.py`
3. Update CLI script to support new dataset

## Results

The package achieves significant compression ratios:

- **ViT-B/32**: ~2700x compression ratio
- **DeepLabV3-ResNet50**: ~2000x compression ratio

With minimal impact on segmentation quality (IoU > 0.84).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{embeddings_squeeze,
  title={Embeddings Squeeze: Vector Quantization for Segmentation Model Compression},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/embeddings_squeeze}
}
```
