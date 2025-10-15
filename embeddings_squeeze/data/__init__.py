"""Data modules for different datasets."""

from .base import BaseDataModule
from .oxford_pet import OxfordPetDataModule
from .S3DIS import S3DISDataModule
__all__ = [
    "BaseDataModule",
    "OxfordPetDataModule",
    "S3DISDataModule"
]
