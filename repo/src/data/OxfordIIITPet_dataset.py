import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from pytorch_lightning import LightningDataModule


class OxfordIIITPetDataModule(LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64, num_workers=4, image_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
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
    
    def prepare_data(self):
        # Загрузка датасета (автоматически скачает, если нужно)
        OxfordIIITPet(self.data_dir, split='trainval', target_types='segmentation', download=True)
        OxfordIIITPet(self.data_dir, split='test', target_types='segmentation', download=True)

    def setup(self, stage=None):
        # Wrapper для применения трансформаций
        class TransformedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform_image, transform_mask):
                self.dataset = dataset
                self.transform_image = transform_image
                self.transform_mask = transform_mask

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, mask = self.dataset[idx]
                img = self.transform_image(img)
                mask = self.transform_mask(mask)

                # === 🔧 Исправление ключевое ===
                mask = mask.squeeze(0)  # (1, H, W) → (H, W)
                mask = mask - 1         # OxfordPet имеет метки {1, 2, 3} → {0, 1, 2}

                # ограничим значения, если есть аномальные пиксели (например, фон=255)
                mask = torch.clamp(mask, min=0, max=2)

                return img, mask.long()
        
        if stage == 'fit' or stage is None:
            train_data = OxfordIIITPet(self.data_dir, split='trainval', target_types='segmentation', download=False)
            val_data = OxfordIIITPet(self.data_dir, split='test', target_types='segmentation', download=False)
            
            self.train_dataset = TransformedDataset(train_data, self.transform_image, self.transform_mask)
            self.val_dataset = TransformedDataset(val_data, self.transform_image, self.transform_mask)
            
        if stage == 'test' or stage is None:
            test_data = OxfordIIITPet(self.data_dir, split='test', target_types='segmentation', download=False)
            self.test_dataset = TransformedDataset(test_data, self.transform_image, self.transform_mask)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )