import pytorch_lightning as pl
import torch
from torch import nn
import torchvision.models as models
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix

class LitResNet50(pl.LightningModule):
    def __init__(self, learning_rate=5e-4, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Replace the final fully connected layer for 10 classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')

        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')

        self.val_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        # The input x is from MelSpectrogram: (batch, 1, n_mels, time)
        # ResNet50 expects 3-channel input (batch, 3, H, W)
        # We duplicate the 1 channel into 3 channels
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_acc(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.val_conf_matrix(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.test_acc(logits, y)
        self.test_precision(logits, y)
        self.test_recall(logits, y)
        self.test_conf_matrix(logits, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Use AdamW with weight decay for better regularization
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # Slower learning rate reduction on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }
