import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.OxfordIIITPet_dataset import OxfordIIITPetDataModule
from models.DeepLabV3_ResNet50_model import DeepLabV3_ResNet50
import clearml
import os
import torch

clearml.Task.set_credentials(
    api_host="https://api.clear.ml",
    web_host="https://app.clear.ml",
    files_host="https://files.clear.ml",
    key="YZHY0DIXHI0QZ5JWBXR0ZBDKV6KM60",
    secret="BRqAEdqAR1rIkaOfFo5m_o8q1nAz9oxttZSH7YMsGJgNU8J9FG3M7M-4VOEQr9J-MRQ"
)

# 2. Создаём или подключаем задачу
task = clearml.Task.init(
    project_name="embedding_test",
    task_name="deeplabv3_training",
    tags=["resnet", "deeplabv3"],
    auto_connect_frameworks=True
)

# 3. Берём логгер
logger = task.get_logger()


from models.quantizers import VQWithProjection, FSQWithProjection, LFQWithProjection, ResidualVQWithProjection

CONFIG_DIR = "/home/jupyter/project/repo/repo/configs"


def create_quantizer(cfg):
    if not cfg.quantizer.enabled:
        return None

    qtype = cfg.quantizer.type
    feature_dim = cfg.model.feature_dim

    if qtype == 'vq':
        return VQWithProjection(
            input_dim=feature_dim,
            codebook_size=cfg.quantizer.get('codebook_size', 512),
            bottleneck_dim=cfg.quantizer.get('bottleneck_dim', 64),
            decay=cfg.quantizer.get('decay', 0.99),
            commitment_weight=cfg.quantizer.get('commitment_weight', 0.25)
        )
    elif qtype == 'fsq':
        return FSQWithProjection(
            input_dim=feature_dim,
            levels=cfg.quantizer.get('levels', [8, 5, 5, 5])
        )
    elif qtype == 'lfq':
        return LFQWithProjection(
            input_dim=feature_dim,
            codebook_size=cfg.quantizer.get('codebook_size', 512),
            entropy_loss_weight=cfg.quantizer.get('entropy_loss_weight', 0.1),
            diversity_gamma=cfg.quantizer.get('diversity_gamma', 0.1),
            spherical=cfg.quantizer.get('spherical', False)
        )
    elif qtype == 'rvq':
        return ResidualVQWithProjection(
            input_dim=feature_dim,
            num_quantizers=cfg.quantizer.get('num_quantizers', 4),
            codebook_size=cfg.quantizer.get('codebook_size', 256),
            bottleneck_dim=cfg.quantizer.get('bottleneck_dim', 64),
            decay=cfg.quantizer.get('decay', 0.99),
            commitment_weight=cfg.quantizer.get('commitment_weight', 0.25)
        )
    else:
        raise ValueError(f"Unknown quantizer type: {qtype}")


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def train(cfg: DictConfig):
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # 1. DataModule
    datamodule = OxfordIIITPetDataModule(
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        data_dir=cfg.dataset.get('data_dir', './data'),
        image_size=cfg.dataset.get('image_size', 256)
    )

    # 2. Quantizer
    quantizer = create_quantizer(cfg)
    if quantizer is None:
        print("  └─ Quantization disabled")

    # 3. Model
    model = DeepLabV3_ResNet50(
        lr=cfg.model.lr,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        quantizer=quantizer,
        add_adapter=cfg.model.get('add_adapter', True),
        feature_dim=cfg.model.get('feature_dim', 2048),
        loss_type=cfg.model.get('loss_type', 'combined'),
        class_weights=cfg.model.get('class_weights', [0.5, 1.5, 3.0]),
        clearml_logger=logger
    )


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ├─ Total params: {total_params:,}")
    print(f"  ├─ Trainable params: {trainable_params:,}")
    print(f"  └─ Frozen params: {total_params - trainable_params:,}")

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "tb_logs"),
        name="deeplabv3_clearml",
        default_hp_metric=False
    )

    # Checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="deeplabv3-{epoch:02d}-{val_loss:.4f}",
        save_weights_only=True
    )

    # GPU/CPU auto
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=tb_logger,
        callbacks=[checkpoint_cb],
    )

    # Train & Validate
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)

    # Upload best checkpoint to ClearML as artifact
    if checkpoint_cb is not None and checkpoint_cb.best_model_path:
        task.upload_artifact(name="best_model", artifact_object=checkpoint_cb.best_model_path)


    print("\n✓ Training complete!")


if __name__ == "__main__":
    train()