# Embeddings Squeeze

Vector Quantization for Segmentation Model Compression using PyTorch Lightning.

This package provides a modular framework for applying Vector Quantization (VQ) to segmentation models for compression, with support for multiple backbones (ViT, DeepLabV3) and datasets.

## Features

- **Multiple Quantization Methods**: VQ-VAE, FSQ, LFQ, and Residual VQ
- **Modular Architecture**: Easy to swap models and datasets
- **PyTorch Lightning**: Professional training framework with logging and checkpointing
- **Multiple Backbones**: Support for ViT and DeepLabV3-ResNet50
- **Research-Friendly**: Quick experimentation via CLI arguments
- **Extensible**: Abstract base classes for adding new models/datasets
- **Comprehensive Visualization**: Compare VQ vs baseline results

## Installation

### From Source

```bash
git clone https://github.com/yourusername/embeddings_squeeze.git
cd embeddings_squeeze
pip install -e .
```

After installation, you can run the CLI scripts using either:
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
- vector_quantize_pytorch >= 1.0.0
- scikit-learn >= 1.0.0
- Other dependencies listed in `requirements.txt`

## CLI Scripts

The package provides three main CLI scripts for training and evaluation:

### 1. Training with VQ Compression (`squeeze.py`)

The main script for training segmentation models with vector quantization compression.

**Basic Usage:**

```bash
# Train ViT with VQ compression
python src/squeeze.py --model vit --dataset oxford_pet --quantizer_type vq --codebook_size 512 --epochs 10

# Train DeepLab with FSQ compression
python src/squeeze.py --model deeplab --dataset oxford_pet --quantizer_type fsq --codebook_size 512 --epochs 10
```

**Advanced Examples:**

```bash
# ViT with VQ-VAE (512 codebook, 64 bottleneck)
python src/squeeze.py \
    --model vit \
    --dataset oxford_pet \
    --quantizer_type vq \
    --codebook_size 512 \
    --bottleneck_dim 64 \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --vq_loss_weight 0.1 \
    --experiment_name vit_vq_512

# DeepLab with Residual VQ 256 codebook
python src/squeeze.py \
    --model deeplab \
    --dataset oxford_pet \
    --quantizer_type rvq \
    --codebook_size 256 \
    --bottleneck_dim 64 \
    --epochs 10 \
    --experiment_name deeplab_rvq_256

# ViT with LFQ (entropy-based quantization)
python src/squeeze.py \
    --model vit \
    --dataset oxford_pet \
    --quantizer_type lfq \
    --codebook_size 512 \
    --epochs 10 \
    --experiment_name vit_lfq_512

# Quick test on subset with ClearML logging
python src/squeeze.py \
    --model vit \
    --dataset oxford_pet \
    --quantizer_type vq \
    --codebook_size 256 \
    --epochs 3 \
    --subset_size 500 \
    --use_clearml \
    --project_name my_project \
    --task_name test_run
```

**Key Arguments:**

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model` | Backbone architecture | `vit` | `vit`, `deeplab` |
| `--quantizer_type` | Quantization method | `vq` | `vq`, `fsq`, `lfq`, `rvq`, `none` |
| `--codebook_size` | Number of codebook vectors | `512` | Any int |
| `--bottleneck_dim` | Bottleneck dimension | `64` | Any int |
| `--num_quantizers` | Number of RVQ levels | `4` | Any int |
| `--epochs` | Training epochs | `10` | Any int |
| `--batch_size` | Batch size | `4` | Any int |
| `--lr` | Learning rate | `1e-4` | Any float |
| `--vq_loss_weight` | VQ loss weight | `0.1` | Any float |
| `--loss_type` | Segmentation loss | `ce` | `ce`, `dice`, `focal`, `combined` |
| `--dataset` | Dataset name | `oxford_pet` | `oxford_pet` |
| `--data_path` | Path to dataset | `./data` | Any path |
| `--output_dir` | Output directory | `./outputs` | Any path |
| `--experiment_name` | Experiment identifier | `vq_squeeze` | Any string |
| `--use_clearml` | Enable ClearML logging | `False` | Flag |
| `--subset_size` | Use data subset | `None` | Any int |

**Outputs:**
- Model checkpoints: `./outputs/{experiment_name}/`
- TensorBoard logs: `./outputs/{experiment_name}/`
- Best model: `{experiment_name}/last.ckpt`

---

### 2. Training Baseline Models (`train_baseline.py`)

Train baseline segmentation models without quantization for comparison purposes.

**Basic Usage:**

```bash
# Train ViT baseline (frozen backbone + trainable classifier)
python src/train_baseline.py --model vit --dataset oxford_pet --epochs 10

# Train DeepLab baseline
python src/train_baseline.py --model deeplab --dataset oxford_pet --epochs 10
```

**Advanced Examples:**

```bash
# ViT baseline with custom settings
python src/train_baseline.py \
    --model vit \
    --dataset oxford_pet \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --loss_type combined \
    --experiment_name vit_baseline

# DeepLab baseline with ClearML
python src/train_baseline.py \
    --model deeplab \
    --dataset oxford_pet \
    --epochs 10 \
    --use_clearml \
    --project_name baselines \
    --task_name deeplab_baseline_v1

# Quick baseline test on subset
python src/train_baseline.py \
    --model vit \
    --dataset oxford_pet \
    --epochs 3 \
    --subset_size 500 \
    --experiment_name vit_baseline_quick
```

**Key Arguments:**

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--model` | Backbone architecture | `vit` | `vit`, `deeplab` |
| `--epochs` | Training epochs | `10` | Any int |
| `--batch_size` | Batch size | `4` | Any int |
| `--lr` | Learning rate | `1e-4` | Any float |
| `--loss_type` | Segmentation loss | `ce` | `ce`, `dice`, `focal`, `combined` |
| `--dataset` | Dataset name | `oxford_pet` | `oxford_pet` |
| `--data_path` | Path to dataset | `./data` | Any path |
| `--output_dir` | Output directory | `./outputs` | Any path |
| `--experiment_name` | Experiment identifier | `segmentation_baseline` | Any string |
| `--use_clearml` | Enable ClearML logging | `False` | Flag |
| `--subset_size` | Use data subset | `None` | Any int |

**Training Strategy:**
- Backbone is frozen (using pretrained weights)
- Only the classifier head is trained
- Provides upper-bound performance for comparison with VQ models

**Outputs:**
- Model checkpoints: `./outputs/{experiment_name}/`
- TensorBoard logs: `./outputs/{experiment_name}/`
- Best model: `{experiment_name}/last.ckpt`

---

### 3. Visualizing Results (`visualize.py`)

Compare VQ model predictions against baseline models with visual outputs.

**Basic Usage:**

```bash
# Compare ViT VQ vs ViT baseline
python src/visualize.py \
    --vq_checkpoint ./outputs/vit_vq_512/last.ckpt \
    --baseline_checkpoint ./outputs/vit_baseline/last.ckpt \
    --model vit

# Compare DeepLab VQ vs DeepLab baseline
python src/visualize.py \
    --vq_checkpoint ./outputs/deeplab_vq_512/last.ckpt \
    --baseline_checkpoint ./outputs/deeplab_baseline/last.ckpt \
    --model deeplab
```

**Advanced Examples:**

```bash
# Visualize with more samples
python src/visualize.py \
    --vq_checkpoint ./outputs/vit_vq_512/last.ckpt \
    --baseline_checkpoint ./outputs/vit_baseline/last.ckpt \
    --model vit \
    --n_best 10 \
    --n_worst 10 \
    --output_dir ./visualizations

# Visualize on validation set
python src/visualize.py \
    --vq_checkpoint ./outputs/vit_vq_512/last.ckpt \
    --baseline_checkpoint ./outputs/vit_baseline/last.ckpt \
    --model vit \
    --dataset_split val

# Custom output location
python src/visualize.py \
    --vq_checkpoint ./outputs/vit_lfq_512/last.ckpt \
    --baseline_checkpoint ./outputs/vit_baseline/last.ckpt \
    --model vit \
    --output_dir ./paper_figures \
    --n_best 5 \
    --n_worst 5
```

**Key Arguments:**

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--vq_checkpoint` | Path to VQ model checkpoint | - | Yes |
| `--baseline_checkpoint` | Path to baseline checkpoint | - | Yes |
| `--model` | Backbone architecture | `vit` | No |
| `--dataset_split` | Dataset split to visualize | `test` | No |
| `--n_best` | Number of best results | `5` | No |
| `--n_worst` | Number of worst results | `5` | No |
| `--output_dir` | Output directory | `./visualizations` | No |
| `--dataset` | Dataset name | `oxford_pet` | No |
| `--data_path` | Path to dataset | `./data` | No |
| `--batch_size` | Batch size | `4` | No |

**Outputs:**
- Best results visualization: `{output_dir}/{model_name}/best_results_{split}.png`
- Worst results visualization: `{output_dir}/{model_name}/worst_results_{split}.png`
- Console summary with IoU statistics

**Output Format:**
Each visualization shows a grid with columns:
1. **Input Image**: Original RGB image
2. **Ground Truth**: True segmentation mask
3. **Baseline Prediction**: Prediction from baseline model
4. **VQ Prediction**: Prediction from VQ model
5. **IoU Score**: Intersection over Union for VQ model

---

## Quantization Methods

This package implements four state-of-the-art vector quantization methods:

### 1. VQ (Vector Quantization / VQ-VAE)

**Description:**  
Classic vector quantization using learned codebooks with Exponential Moving Average (EMA) updates. No gradients needed for codebook learning.

**Key Parameters:**
- `codebook_size`: Number of codebook vectors (default: 512)
- `bottleneck_dim`: Compressed dimension (default: 64)
- `decay`: EMA decay rate (default: 0.99)
- `commitment_weight`: Commitment loss weight (default: 0.25)

**Characteristics:**
- ~9 bits per vector at `codebook_size=512`
- Stable training with EMA updates
- Well-established baseline method

**Reference:**  
[Paper](https://arxiv.org/abs/1711.00937)

**Usage:**
```bash
python src/squeeze.py --quantizer_type vq --codebook_size 512 --bottleneck_dim 64
```

---

### 2. FSQ (Finite Scalar Quantization)

**Description:**  
Quantization without codebook learning. Uses fixed quantization levels per dimension, avoiding codebook collapse issues.

**Key Parameters:**
- `levels`: Quantization levels per dimension (default: [8, 5, 5, 5])
- `bottleneck_dim`: Number of quantized dimensions (default: 256)

**Characteristics:**
- No codebook to learn (deterministic quantization)
- Effective codebook size: product of levels (e.g., 8×5×5×5 = 1000)
- No commitment loss needed
- More stable training

**Reference:**  
[Paper](https://arxiv.org/abs/2309.15505)

**Usage:**
```bash
python src/squeeze.py --quantizer_type fsq --codebook_size 512
```

---

### 3. LFQ (Lookup-Free Quantization)

**Description:**  
Quantization without explicit codebook lookup tables. Uses entropy and diversity losses to encourage uniform code usage and information preservation.

**Key Parameters:**
- `codebook_size`: Effective codebook size (default: 512)
- `bottleneck_dim`: Compressed dimension (default: 64)
- `entropy_loss_weight`: Weight for entropy loss (default: 0.1)
- `diversity_gamma`: Diversity loss weight (default: 0.1)
- `spherical`: Use spherical quantization (default: False)

**Characteristics:**
- ~log2(codebook_size) bits per vector
- No lookup table required
- Entropy regularization prevents collapse
- Good for embedding bottlenecks

**Reference:**  
[Paper](https://arxiv.org/abs/2310.05737)

**Usage:**
```bash
python src/squeeze.py --quantizer_type lfq --codebook_size 512 --bottleneck_dim 64
```

---

### 4. RVQ (Residual Vector Quantization)

**Description:**  
Multi-level quantization where each level quantizes the residual error from previous levels. Achieves higher bit rates and better reconstruction quality.

**Key Parameters:**
- `num_quantizers`: Number of quantization levels (default: 4)
- `codebook_size`: Codebook size per level (default: 256)
- `bottleneck_dim`: Compressed dimension (default: 64)
- `decay`: EMA decay rate (default: 0.99)
- `commitment_weight`: Commitment loss weight (default: 0.25)

**Characteristics:**
- Higher bit rate: `num_quantizers × log2(codebook_size)` bits per vector
- Example: 4 levels × 8 bits = 32 bits per vector
- Better reconstruction quality than single-level VQ
- Cascaded refinement of representations

**Reference:**  
[Paper](https://arxiv.org/pdf/2203.01941)

**Usage:**
```bash
python src/squeeze.py --quantizer_type rvq --num_quantizers 4 --codebook_size 256 --bottleneck_dim 64
```

---

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

## Package Structure

```
project/
├── src/                    # Source code (mapped to embeddings_squeeze package)
│   ├── models/            # Model architectures
│   │   ├── backbones/     # Segmentation backbones (ViT, DeepLab)
│   │   ├── quantizers.py  # VQ implementations (VQ, FSQ, LFQ, RVQ)
│   │   ├── lightning_module.py  # Main training module
│   │   └── baseline_module.py   # Baseline training module
│   ├── data/              # Data modules (Oxford Pet)
│   ├── loggers/           # Logging utilities (ClearML integration)
│   ├── utils/             # Utility functions (compression, visualization)
│   ├── configs/           # Configuration management
│   ├── squeeze.py         # Main VQ training CLI script
│   ├── train_baseline.py  # Baseline training CLI script
│   └── visualize.py       # Visualization CLI script
├── outputs/               # Training outputs (gitignored)
├── setup.py               # Package setup
├── pyproject.toml         # Project configuration
└── README.md              # This file
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
3. Update CLI scripts to support new backbone

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
3. Update CLI scripts to support new dataset

## Results

The package achieves significant compression ratios with minimal quality loss:

### Compression Ratios
- **ViT-B/32**: ~2700x compression ratio
- **DeepLabV3-ResNet50**: ~2000x compression ratio

### Segmentation Quality
- IoU > 0.84 on Oxford-IIIT Pet dataset
- Minimal degradation compared to baseline models

### Model Comparisons
Use `visualize.py` to generate comparison plots showing:
- Side-by-side predictions (baseline vs VQ)
- Per-sample IoU scores
- Best and worst performing examples

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
