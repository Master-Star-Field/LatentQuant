"""
Base data module for PyTorch Lightning.
"""

from abc import ABC, abstractmethod
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule, ABC):
    """
    Abstract base class for data modules.
    
    All dataset-specific data modules should inherit from this class.
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    @abstractmethod
    def setup(self, stage: str = None):
        """
        Setup datasets for training/validation/testing.
        
        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        pass
    
    @abstractmethod
    def train_dataloader(self):
        """Return training dataloader."""
        pass
    
    @abstractmethod
    def val_dataloader(self):
        """Return validation dataloader."""
        pass
    
    @abstractmethod
    def test_dataloader(self):
        """Return test dataloader."""
        pass
