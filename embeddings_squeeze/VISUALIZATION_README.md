# VQ Visualization System

This document explains how to use the new VQ visualization system to compare segmentation results with and without Vector Quantization.

## Overview

The visualization system consists of three main components:

1. **Baseline Training Script** (`train_baseline.py`) - Trains segmentation models without VQ
2. **VQ Training Script** (`squeeze.py`) - Trains segmentation models with VQ (existing)
3. **Visualization Script** (`visualize.py`) - Compares and visualizes results

## Quick Start

### 1. Train Baseline Model

First, train a baseline segmentation model without VQ:

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
python test_integration.py
```

This will test:
- Model creation
- Data loading (if dataset is available)
- Model inference
- Visualization utilities

## File Structure

```
embeddings_squeeze/
├── train_baseline.py          # Baseline training script
├── squeeze.py                 # VQ training script (existing)
├── visualize.py               # Visualization script
├── test_integration.py       # Integration test
├── models/
│   ├── baseline_module.py    # Baseline Lightning module
│   ├── lightning_module.py    # VQ Lightning module (existing)
│   └── ...
├── utils/
│   ├── comparison.py         # Comparison utilities
│   └── ...
└── ...
```

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
