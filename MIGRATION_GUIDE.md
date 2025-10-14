# Migration Guide: Fork Integration

This document describes the changes made to integrate the fork repository's features into the main `embeddings_squeeze` project.

## Summary of Changes

### 1. Configuration System (Dataclass-based)

The existing dataclass configuration system has been extended with new options:

**New Configurations Added:**
- `QuantizerConfig`: Configuration for multiple quantizer types (VQ, FSQ, LFQ, RVQ)
- `LoggerConfig`: Configuration for ClearML and TensorBoard logging
- `ModelConfig` extended with: `add_adapter`, `feature_dim`, `loss_type`, `class_weights`

**Location:** `embeddings_squeeze/configs/default.py`

### 2. Quantization System

Replaced custom VQ implementation with `vector_quantize_pytorch` library:

**New Quantizers:**
- `VQWithProjection`: Vector Quantization (VQ-VAE) with EMA ~9 bits/vector
- `FSQWithProjection`: Finite Scalar Quantization ~10 bits/vector
- `LFQWithProjection`: Lookup-Free Quantization ~9 bits/vector
- `ResidualVQWithProjection`: Residual VQ ~32 bits/vector

**Location:** `embeddings_squeeze/models/quantizers.py`

**Removed:** `embeddings_squeeze/models/vq/` directory (old custom VQ)

### 3. Loss Functions

Added advanced loss functions for segmentation:

- `DiceLoss`: Multi-class Dice loss
- `FocalLoss`: Focal loss for class imbalance
- `CombinedLoss`: Weighted combination of CE + Dice + Focal

**Location:** `embeddings_squeeze/models/losses.py`

### 4. Model Architecture

Updated Lightning modules with new features:

**VQSqueezeModule:**
- Support for all quantizer types via factory pattern
- Optional adapter layers for fine-tuning frozen backbones
- Advanced loss functions (ce/dice/focal/combined)
- Automatic feature extraction and embedding saving
- TorchMetrics integration (IoU, Accuracy)

**BaselineSegmentationModule:**
- Advanced loss functions support
- TorchMetrics integration
- Compatible with new configuration system

**Locations:**
- `embeddings_squeeze/models/lightning_module.py`
- `embeddings_squeeze/models/baseline_module.py`

### 5. ClearML Integration

Added ClearML logging with fallback to TensorBoard:

- Credentials loaded from YAML file
- Automatic framework connection
- Graceful fallback to TensorBoard if ClearML fails

**Locations:**
- `embeddings_squeeze/loggers/clearml_logger.py`
- `embeddings_squeeze/configs/clearml_credentials.yaml.example` (template)

**Setup:**
1. Copy `clearml_credentials.yaml.example` to `clearml_credentials.yaml`
2. Fill in your ClearML credentials
3. Add `clearml_credentials.yaml` to `.gitignore`

### 6. Data Preprocessing

Fixed Oxford Pet dataset mask preprocessing:
- Masks shifted from {1,2,3} to {0,1,2}
- Proper clamping to valid range
- Returns masks as [B, H, W] instead of [B, 1, H, W]

**Location:** `embeddings_squeeze/data/oxford_pet.py`

### 7. Training Scripts

Updated training scripts with new features:

**squeeze.py:**
- Quantizer factory function
- ClearML integration
- New CLI arguments for quantizers, adapters, loss types
- Auto-detection of feature dimensions

**train_baseline.py:**
- ClearML integration
- Loss type selection
- Updated metrics

## Usage Examples

### Basic Training with VQ

```bash
python embeddings_squeeze/squeeze.py \
    --model deeplab \
    --quantizer_type vq \
    --epochs 10 \
    --batch_size 4 \
    --loss_type combined
```

### Training with Different Quantizers

**FSQ (Finite Scalar Quantization):**
```bash
python embeddings_squeeze/squeeze.py \
    --quantizer_type fsq \
    --model vit
```

**LFQ (Lookup-Free Quantization):**
```bash
python embeddings_squeeze/squeeze.py \
    --quantizer_type lfq \
    --codebook_size 512
```

**RVQ (Residual VQ):**
```bash
python embeddings_squeeze/squeeze.py \
    --quantizer_type rvq \
    --num_quantizers 4 \
    --codebook_size 256
```

**No Quantization (Baseline):**
```bash
python embeddings_squeeze/squeeze.py \
    --quantizer_type none
```

### Training with Adapters

For frozen backbones with trainable adapter layers:

```bash
python embeddings_squeeze/squeeze.py \
    --model deeplab \
    --add_adapter \
    --quantizer_type vq
```

### Training with Different Loss Functions

**Dice Loss:**
```bash
python embeddings_squeeze/squeeze.py --loss_type dice
```

**Focal Loss:**
```bash
python embeddings_squeeze/squeeze.py --loss_type focal
```

**Combined Loss (CE + Dice + Focal):**
```bash
python embeddings_squeeze/squeeze.py --loss_type combined
```

### Using ClearML for Logging

```bash
python embeddings_squeeze/squeeze.py \
    --use_clearml \
    --project_name "my_project" \
    --task_name "vq_experiment_1"
```

### Baseline Training

```bash
python embeddings_squeeze/train_baseline.py \
    --model deeplab \
    --loss_type combined \
    --epochs 10
```

## Configuration Options

### Quantizer Types

| Type | Description | Bits/Vector | Parameters |
|------|-------------|-------------|------------|
| `vq` | Vector Quantization | ~9 | `codebook_size`, `bottleneck_dim` |
| `fsq` | Finite Scalar Quantization | ~10 | `levels` (default: [8,5,5,5]) |
| `lfq` | Lookup-Free Quantization | ~9 | `codebook_size`, `entropy_loss_weight` |
| `rvq` | Residual VQ | ~32 | `num_quantizers`, `codebook_size` |
| `none` | No quantization | N/A | - |

### Loss Types

| Type | Description | Best For |
|------|-------------|----------|
| `ce` | Cross Entropy | Balanced datasets |
| `dice` | Dice Loss | Segmentation tasks |
| `focal` | Focal Loss | Class imbalance |
| `combined` | CE + Dice + Focal | General purpose |

### Model Backbones

| Backbone | Feature Dim | Classes | Weights |
|----------|-------------|---------|---------|
| `vit` | 768 | 21 | IMAGENET1K_V1 |
| `deeplab` | 2048 | 3 | COCO_WITH_VOC_LABELS_V1 |

## Dependencies

New dependencies added to `requirements.txt`:
- `vector_quantize_pytorch>=1.0.0` - Quantization library
- `clearml>=1.0.0` - Experiment tracking
- `omegaconf>=2.3.0` - Config file parsing
- `torchmetrics>=0.11.0` - Metrics computation

Install with:
```bash
pip install -r requirements.txt
```

## Backward Compatibility

The migration maintains backward compatibility:
- Old training commands still work (with default quantizer=vq)
- TensorBoard logging is still the default
- Existing configs are automatically extended with defaults

## Testing

To verify the migration:

1. **Test VQ quantizer:**
```bash
python embeddings_squeeze/squeeze.py --epochs 1 --subset_size 100
```

2. **Test FSQ quantizer:**
```bash
python embeddings_squeeze/squeeze.py --quantizer_type fsq --epochs 1 --subset_size 100
```

3. **Test baseline without quantization:**
```bash
python embeddings_squeeze/train_baseline.py --epochs 1 --subset_size 100
```

4. **Test ClearML (if credentials configured):**
```bash
python embeddings_squeeze/squeeze.py --use_clearml --epochs 1 --subset_size 100
```

## Troubleshooting

### ClearML Credentials Error

If you see: `ClearML credentials file not found`

**Solution:** Copy the example file and fill in your credentials:
```bash
cp embeddings_squeeze/configs/clearml_credentials.yaml.example \
   embeddings_squeeze/configs/clearml_credentials.yaml
```

### Import Errors

If you see: `ModuleNotFoundError: No module named 'vector_quantize_pytorch'`

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Feature Dimension Mismatch

The feature dimension is auto-detected based on backbone:
- ViT: 768
- DeepLab: 2048

Override if needed: `--feature_dim 1024`

## Migration Notes

- The old `embeddings_squeeze/models/vq/` directory has been removed
- Custom VQ implementation replaced with `vector_quantize_pytorch`
- Default quantizer is VQ (maintains compatibility)
- Mask preprocessing now correctly handles Oxford Pet labels
- Metric names unified: `val/loss` instead of `val/seg_loss` or `val/total_loss`

## Next Steps

1. Set up ClearML credentials (optional)
2. Test different quantizers on your dataset
3. Experiment with loss functions for your task
4. Try adapter training for faster fine-tuning
5. Compare results with baseline (no quantization)

