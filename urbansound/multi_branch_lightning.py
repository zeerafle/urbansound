import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix
from urbansound.multi_branch_fusion import MultiBranchAttentionFusion

class LitMultiBranchFusion(pl.LightningModule):
    def __init__(self, learning_rate=0.001, num_classes=10, cnn_out=256, gru_in_features=40, gru_out=128, eng_in_features=4, eng_out=64, hidden_dim=128):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Instantiate our multi-branch fusion model
        self.model = MultiBranchAttentionFusion(
            num_classes=num_classes,
            cnn_out=cnn_out,
            gru_in_features=gru_in_features,
            gru_out=gru_out,
            eng_in_features=eng_in_features,
            eng_out=eng_out,
            hidden_dim=hidden_dim
        )
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

        # GPU Accelerated Feature Extractors
        sample_rate = 22050
        n_mels = 64
        n_mfcc = 40
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=n_mels
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={"n_fft": 1024, "n_mels": n_mels, "hop_length": 512}
        )
        self.spec_centroid = torchaudio.transforms.SpectralCentroid(
            sample_rate=sample_rate, n_fft=1024, hop_length=512
        )

    def extract_features(self, waveforms):
        # 1. CNN Feature: Mel-Spectrogram
        mel_spec = self.mel_transform(waveforms)
        mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=10.0, top_db=80.0)

        # 2. GRU Feature: MFCCs
        mfcc = self.mfcc_transform(waveforms)
        # Input waveforms shape: (Batch, Channels=1, Time)
        # MFCC output: (Batch, Channels=1, n_mfcc, frames) -> We want (Batch, frames, n_mfcc)
        mfcc = mfcc.squeeze(1).transpose(1, 2)

        # 3. Engineered Features: Global Statistics
        centroid = self.spec_centroid(waveforms)

        cent_variance = centroid.var(dim=-1)
        cent_variance = torch.clamp(cent_variance, min=1e-8)

        cent_mean = torch.log1p(centroid.mean(dim=-1).squeeze(1))
        cent_var = torch.log1p(cent_variance.squeeze(1))

        energy = torch.mean(waveforms ** 2, dim=-1)
        energy = torch.sqrt(torch.clamp(energy, min=1e-8)).squeeze(1)

        zcr = (torch.diff(torch.sign(waveforms)) != 0).float().mean(dim=-1).squeeze(1)

        global_stats = torch.stack([cent_mean, cent_var, energy, zcr], dim=1)

        # Purge NaNs/Infs
        mel_spec = torch.nan_to_num(mel_spec, nan=0.0, posinf=0.0, neginf=0.0)
        mfcc = torch.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
        global_stats = torch.nan_to_num(global_stats, nan=0.0, posinf=0.0, neginf=0.0)

        return mel_spec, mfcc, global_stats

    def forward(self, mel_spec, mfcc, global_stats):
        # We unpack the three features here
        return self.model(mel_spec, mfcc, global_stats)

    def training_step(self, batch, batch_idx):
        waveforms, y = batch
        mel_spec, mfcc, global_stats = self.extract_features(waveforms)

        logits = self(mel_spec, mfcc, global_stats)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, y = batch
        mel_spec, mfcc, global_stats = self.extract_features(waveforms)

        logits = self(mel_spec, mfcc, global_stats)
        loss = self.loss_fn(logits, y)

        self.val_acc(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.val_conf_matrix(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        waveforms, y = batch
        mel_spec, mfcc, global_stats = self.extract_features(waveforms)

        logits = self(mel_spec, mfcc, global_stats)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }
