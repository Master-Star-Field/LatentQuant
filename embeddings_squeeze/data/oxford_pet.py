"""
Oxford-IIIT Pet dataset data module.
"""

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

from .base import BaseDataModule


class PetSegmentationDataset:
    """Wrapper for Oxford-IIIT Pet dataset with proper transforms."""
    
    def __init__(self, pet_dataset, transform_image, transform_mask):
        self.dataset = pet_dataset
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        return image, mask


class OxfordPetDataModule(BaseDataModule):
    """
    Data module for Oxford-IIIT Pet segmentation dataset.
    """
    
    def __init__(
        self,
        data_path: str = './data',
        batch_size: int = 4,
        num_workers: int = 6,
        pin_memory: bool = True,
        image_size: int = 224,
        subset_size: int = None,
        **kwargs
    ):
        super().__init__(data_path, batch_size, num_workers, pin_memory, **kwargs)
        
        self.image_size = image_size
        self.subset_size = subset_size
        
        # Define transforms
        self.transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor()
        ])
        
        # Dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            # Check if dataset exists
            pet_path = os.path.join(self.data_path, 'oxford-iiit-pet')
            need_download = not os.path.exists(pet_path)
            
            # Load full dataset
            pet_dataset = OxfordIIITPet(
                root=self.data_path,
                split='trainval',
                target_types='segmentation',
                download=need_download
            )
            
            # Wrap with transforms
            wrapped_dataset = PetSegmentationDataset(
                pet_dataset, self.transform_image, self.transform_mask
            )
            
            # Create subset if specified
            if self.subset_size is not None:
                wrapped_dataset = Subset(wrapped_dataset, range(self.subset_size))
            
            # Split into train/val (80/20)
            total_size = len(wrapped_dataset)
            train_size = int(0.8 * total_size)
            
            self.train_dataset = Subset(wrapped_dataset, range(train_size))
            self.val_dataset = Subset(wrapped_dataset, range(train_size, total_size))
            
        if stage == 'test' or stage is None:
            # Load test dataset
            pet_dataset = OxfordIIITPet(
                root=self.data_path,
                split='test',
                target_types='segmentation',
                download=False
            )
            
            wrapped_dataset = PetSegmentationDataset(
                pet_dataset, self.transform_image, self.transform_mask
            )
            
            if self.subset_size is not None:
                wrapped_dataset = Subset(wrapped_dataset, range(min(self.subset_size, len(wrapped_dataset))))
            
            self.test_dataset = wrapped_dataset

    def train_dataloader(self, max_batches: int = None):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=max_batches is not None
        )

    def val_dataloader(self, max_batches: int = None):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=max_batches is not None
        )

    def test_dataloader(self, max_batches: int = None):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=max_batches is not None
        )
