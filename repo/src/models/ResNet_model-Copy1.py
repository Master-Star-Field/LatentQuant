import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import MulticlassAccuracy
from pytorch_lightning import LightningModule
from torchvision.models import resnet18, ResNet18_Weights

class ResNetClassifier(LightningModule):
    def __init__(self, lr=0.001, num_classes=10, pretrained=True):
        super().__init__()
        self.save_hyperparameters()

        # Загрузка предобученной модели
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)

        # Замена финального слоя для 10 классов CIFAR-10
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Метрики (multi-class, 10 классов)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_top5_acc = MulticlassAccuracy(num_classes=num_classes, top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.val_acc(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_top5_acc(y_hat, y)
        return loss

    def on_validation_epoch_end(self):
        # Логируем метрики в конце эпохи
        self.log('val_acc', self.val_acc.compute())
        self.log('val_precision', self.val_precision.compute())
        self.log('val_recall', self.val_recall.compute())
        self.log('val_f1', self.val_f1.compute())
        self.log('val_top5_acc', self.val_top5_acc.compute())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)