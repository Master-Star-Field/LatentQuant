import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning import LightningDataModule

class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def prepare_data(self):
        # Загрузка датасета (автоматически скачает, если нужно)
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.val_dataset = CIFAR10(self.data_dir, train=False, transform=self.transform)  # Используем test как val для простоты
        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)