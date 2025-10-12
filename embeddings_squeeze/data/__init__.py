"""Data modules for different datasets."""

from .base import BaseDataModule
from .oxford_pet import OxfordPetDataModule

__all__ = [
    "BaseDataModule",
    "OxfordPetDataModule",
]
