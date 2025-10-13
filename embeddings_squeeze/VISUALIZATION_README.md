# VQ Visualization System

This document explains how to use the VQ visualization system to compare segmentation results with and without Vector Quantization.

## Overview

The visualization system consists of three main components:

1. **Baseline Training Script** (`train_baseline.py`) - Trains segmentation models without VQ
2. **VQ Training Script** (`squeeze.py`) - Trains segmentation models with VQ (existing)
3. **Visualization Script** (`visualize.py`) - Compares and visualizes results

## Training Approach

Both baseline and VQ models use the same training strategy:

- **Backbone**: Pre-trained weights (ViT-B/32 or DeepLabV3-ResNet50) remain **frozen**
- **Segmentation Head**: Only the final classification layer is trained
- **Purpose**: This ensures fair comparison between VQ and baseline, as both use identical frozen backbones

This approach is ideal for:
- Quick training (only small segmentation head)
- Fair comparison (same backbone features)
- Resource efficiency (minimal trainable parameters)

## Quick Start

### 1. Train Baseline Model

First, train a baseline segmentation model without VQ. **Note: Only the segmentation head is trained while the backbone remains frozen.**

```bash
cd embeddings_squeeze
python train_baseline.py --model vit --epochs 10 --batch_size 4 --experiment_name my_baseline
```

This will save checkpoints to `./outputs/my_baseline_baseline/version_X/`

### 2. Train VQ Model

Train a VQ-compressed model:

```bash
python squeeze.py --model vit --epochs 10 --batch_size 4 --experiment_name my_vq
```

This will save checkpoints to `./outputs/my_vq/version_X/`

### 3. Visualize Results

Compare the models and visualize best/worst results:

```bash
python visualize.py \
    --vq_checkpoint ./outputs/my_vq/version_0/last.ckpt \
    --baseline_checkpoint ./outputs/my_baseline_baseline/version_0/last.ckpt \
    --dataset_split test \
    --n_best 5 \
    --n_worst 5 \
    --output_dir ./visualizations
```

## Detailed Usage

### Baseline Training Script

```bash
python train_baseline.py [OPTIONS]
```

**Key Options:**
- `--model`: Backbone model (`vit` or `deeplab`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--experiment_name`: Name for the experiment
- `--subset_size`: Limit dataset size for quick testing

**Example:**
```bash
python train_baseline.py \
    --model deeplab \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --experiment_name deeplab_baseline
```

### VQ Training Script

```bash
python squeeze.py [OPTIONS]
```

**Key Options:**
- `--model`: Backbone model (`vit` or `deeplab`)
- `--num_vectors`: Number of VQ codebook vectors
- `--epochs`: Number of training epochs
- `--experiment_name`: Name for the experiment

**Example:**
```bash
python squeeze.py \
    --model deeplab \
    --num_vectors 256 \
    --epochs 20 \
    --experiment_name deeplab_vq_256
```

### Visualization Script

```bash
python visualize.py [OPTIONS]
```

**Required Options:**
- `--vq_checkpoint`: Path to VQ model checkpoint
- `--baseline_checkpoint`: Path to baseline model checkpoint

**Optional Options:**
- `--dataset_split`: Which split to use (`test` or `val`), default: `test`
- `--n_best`: Number of best results to show, default: 5
- `--n_worst`: Number of worst results to show, default: 5
- `--output_dir`: Output directory for visualizations, default: `./visualizations`
- `--model`: Backbone model (must match training), default: `vit`
- `--batch_size`: Batch size for inference, default: 4

**Example:**
```bash
python visualize.py \
    --vq_checkpoint ./outputs/deeplab_vq_256/version_0/last.ckpt \
    --baseline_checkpoint ./outputs/deeplab_baseline_baseline/version_0/last.ckpt \
    --dataset_split test \
    --n_best 10 \
    --n_worst 10 \
    --model deeplab \
    --output_dir ./results
```

## Output

The visualization script generates two PNG files:

1. **`best_results_{split}.png`** - Shows the N best VQ results
2. **`worst_results_{split}.png`** - Shows the 5 worst VQ results

Each figure contains rows with 4 columns:
- **Original Image**: Input image
- **Ground Truth**: True segmentation mask
- **Baseline Prediction**: Segmentation without VQ
- **VQ Prediction**: Segmentation with VQ (with IoU score)

## Testing

Run the integration test to verify everything works:

```bash
python test_simplified_baseline.py
```

This will test:
- Backbone classifier training
- Optimizer creation
- Model inference

## File Structure

```
embeddings_squeeze/
├── train_baseline.py          # Baseline training script
├── squeeze.py                 # VQ training script (existing)
├── visualize.py               # Visualization script
├── test_simplified_baseline.py # Integration test
├── models/
│   ├── lightning_module.py    # VQ Lightning module (existing)
│   └── ...
├── utils/
│   ├── comparison.py         # Comparison utilities
│   └── ...
└── ...
```

## Key Implementation Details

### Simplified Baseline Training

The baseline training uses a much simpler approach:

1. **Direct Backbone Usage**: Uses `ViTSegmentationBackbone` or `DeepLabV3SegmentationBackbone` directly
2. **Automatic Classifier Training**: The backbone's classifier is automatically trainable when `freeze_backbone=True`
3. **Minimal Wrapper**: `BaselineSegmentationModule` is just a thin Lightning wrapper around the backbone
4. **Same Interface**: Compatible with existing visualization and comparison utilities

### Backbone Architecture

- **ViT**: `_ViTSegmentationHead` (trainable) + frozen ViT encoder
- **DeepLab**: `model.classifier` (trainable) + frozen ResNet backbone
- **Both**: Return `{'out': logits}` for consistent interface

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Ensure checkpoint paths are correct and models were trained with matching configurations.

2. **CUDA out of memory**: Reduce batch size or use CPU:
   ```bash
   CUDA_VISIBLE_DEVICES="" python visualize.py ...
   ```

3. **Dataset not found**: Download the Oxford-IIIT Pet dataset or use `--subset_size` for testing.

4. **Model mismatch**: Ensure the `--model` argument matches the backbone used during training.

### Performance Tips

- Use `--subset_size` for quick testing during development
- Use smaller `--n_best` and `--n_worst` values for faster visualization
- Use `--batch_size 1` for memory-constrained environments

## Advanced Usage

### Custom Datasets

To use with custom datasets, ensure your data module follows the same interface as `OxfordPetDataModule` and update the visualization script accordingly.

### Different Backbones

The system supports any backbone that inherits from `SegmentationBackbone`. Add new backbones to `models/backbones/` and update the CLI scripts.

### Custom Metrics

Modify `utils/comparison.py` to add custom evaluation metrics beyond IoU.