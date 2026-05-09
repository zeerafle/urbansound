import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from urbansound.multi_branch_lightning import LitMultiBranchFusion
from urbansound.urbansounddataset import (
    ANNOTATIONS_FILE,
    AUDIO_DIR,
    NUM_SAMPLES,
    SAMPLE_RATE,
    UrbanSoundDataset,
)

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
FAST_DEV_RUN = False


def get_dataloaders(val_fold, all_folds, transform, device):
    train_folds = [f for f in all_folds if f != val_fold]

    train_dataset = UrbanSoundDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        transform,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        target_folds=train_folds,
    )

    val_dataset = UrbanSoundDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        transform,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        target_folds=[val_fold],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    return train_dataloader, val_dataloader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    all_test_metrics = []
    accumulated_conf_matrix = None

    for val_fold in folds:
        print(f"\n{'=' * 30}")
        print(f"Multi-Branch CV: Fold {val_fold} as Validation")
        print(f"{'=' * 30}")

        train_loader, val_loader = get_dataloaders(val_fold, folds, lambda x: x, "cpu")

        model = LitMultiBranchFusion(
            learning_rate=LEARNING_RATE,
            num_classes=10,
            cnn_out=256,
            gru_in_features=40,  # Must match n_mfcc default
            eng_in_features=4,  # 4 engineered features (centroid mean, centroid var, energy, zcr)
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=7, mode="min", verbose=True
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="multi_branch_checkpoints",
            filename=f"mb-fold-{val_fold}-best",
            save_top_k=1,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            devices=1,
            callbacks=[early_stopping, checkpoint_callback, lr_monitor],
            log_every_n_steps=10,
            fast_dev_run=FAST_DEV_RUN,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Test on the validation set using the best model
        print(f"Evaluating Fold {val_fold} using best model...")
        if not FAST_DEV_RUN:
            test_results = trainer.test(dataloaders=val_loader, ckpt_path="best")[0]
        else:
            test_results = trainer.test(dataloaders=val_loader)[0]

        # Extract conf matrix from model state
        current_conf_matrix = model.test_conf_matrix.compute()
        if accumulated_conf_matrix is None:
            accumulated_conf_matrix = current_conf_matrix.clone()
        else:
            accumulated_conf_matrix += current_conf_matrix

        all_test_metrics.append(
            {
                "fold": val_fold,
                "accuracy": test_results.get("test_acc", 0),
                "precision": test_results.get("test_precision", 0),
                "recall": test_results.get("test_recall", 0),
            }
        )

        model.test_conf_matrix.reset()

    print("\n\n" + "=" * 40)
    print("5-FOLD CV COMPLETED - MULTI-BRANCH MDOEL")
    print("=" * 40)

    avg_acc = sum([m["accuracy"] for m in all_test_metrics]) / len(folds)
    avg_prec = sum([m["precision"] for m in all_test_metrics]) / len(folds)
    avg_rec = sum([m["recall"] for m in all_test_metrics]) / len(folds)

    print(f"Average Accuracy:  {avg_acc:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print(f"Average Recall:    {avg_rec:.4f}")

    print("\nAccumulated Confusion Matrix:")
    print(accumulated_conf_matrix)


if __name__ == "__main__":
    main()
