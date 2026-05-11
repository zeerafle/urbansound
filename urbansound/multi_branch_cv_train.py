import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from urbansound.multi_branch_lightning import LitMultiBranchFusion
from urbansound.urbansounddataset import (
    ANNOTATIONS_FILE,
    AUDIO_DIR,
    NUM_SAMPLES,
    SAMPLE_RATE,
    UrbanSoundDataset,
)

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 5e-4
FAST_DEV_RUN = False


def get_dataloaders(test_fold, val_fold, all_folds, transform, device):
    train_folds = [f for f in all_folds if f not in (val_fold, test_fold)]

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

    test_dataset = UrbanSoundDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        transform,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device,
        target_folds=[test_fold],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    all_test_metrics = []
    accumulated_conf_matrix = None

    for test_fold in folds:
        val_fold = test_fold + 1 if test_fold < 10 else 1
        print(f"\n{'=' * 30}")
        print(
            f"Multi-Branch CV: Fold {test_fold} as Test; Fold {val_fold} as Validation"
        )
        print(f"{'=' * 30}")

        train_loader, val_loader, test_loader = get_dataloaders(
            test_fold, val_fold, folds, lambda x: x, "cpu"
        )

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
            filename=f"mb-fold-{test_fold}-best",
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

        # Test on the test set using the best model
        print(f"Evaluating Fold {test_fold} using best model...")
        if not FAST_DEV_RUN:
            test_results = trainer.test(dataloaders=test_loader, ckpt_path="best")[0]
        else:
            test_results = trainer.test(dataloaders=test_loader)[0]

        # Extract conf matrix from model state
        current_conf_matrix = model.test_conf_matrix.compute()
        if accumulated_conf_matrix is None:
            accumulated_conf_matrix = current_conf_matrix.clone()
        else:
            accumulated_conf_matrix += current_conf_matrix

        all_test_metrics.append(
            {
                "fold": test_fold,
                "accuracy": test_results.get("test_acc", 0),
                "precision": test_results.get("test_precision", 0),
                "recall": test_results.get("test_recall", 0),
            }
        )

        model.test_conf_matrix.reset()

    avg_acc = sum([m["accuracy"] for m in all_test_metrics]) / len(folds)
    avg_prec = sum([m["precision"] for m in all_test_metrics]) / len(folds)
    avg_rec = sum([m["recall"] for m in all_test_metrics]) / len(folds)

    best_fold = max(all_test_metrics, key=lambda x: x["accuracy"])

    summary = []
    summary.append("\n" + "=" * 40)
    summary.append("10-FOLD CV COMPLETED - MULTI-BRANCH MODEL")
    summary.append("=" * 40)

    summary.append("\nPer-Fold Metrics:")
    for metrics in all_test_metrics:
        summary.append(
            f"Fold {metrics['fold']}: Accuracy = {metrics['accuracy']:.4f}, Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}"
        )

    summary.append(
        f"\nBest Results: Fold {best_fold['fold']} (Accuracy: {best_fold['accuracy']:.4f})"
    )

    summary.append("\nOverall Averages:")
    summary.append(f"Average Accuracy:  {avg_acc:.4f}")
    summary.append(f"Average Precision: {avg_prec:.4f}")
    summary.append(f"Average Recall:    {avg_rec:.4f}")

    summary.append("\nAccumulated Confusion Matrix:")
    # convert the confusion matrix tensor to properly formatted string
    conf_matrix_str = str(
        accumulated_conf_matrix.cpu().numpy()
        if hasattr(accumulated_conf_matrix, "cpu")
        else accumulated_conf_matrix
    )
    summary.append(conf_matrix_str)

    summary_text = "\n".join(summary)
    print(summary_text)

    # 1. Create a dedicated logger for the CV summary
    summary_logger = CSVLogger(save_dir="lightning_logs", name="cv_summary")

    # 2. Compile metrics dictionary
    summary_metrics = {
        "cv_avg_accuracy": avg_acc,
        "cv_avg_precision": avg_prec,
        "cv_avg_recall": avg_rec,
        "best_fold_id": best_fold["fold"],
        "best_fold_accuracy": best_fold["accuracy"],
    }

    # Add individual fold metrics dynamically
    for m in all_test_metrics:
        summary_metrics[f"fold_{m['fold']}_accuracy"] = m["accuracy"]

    # 3. Log the metrics
    summary_logger.log_metrics(summary_metrics)
    summary_logger.save()

    import os

    # 4. Save the text summary & confusion matrix into the assigned log_dir
    os.makedirs(summary_logger.log_dir, exist_ok=True)
    with open(
        os.path.join(summary_logger.log_dir, "summary_and_confusion_matrix.txt"), "w"
    ) as f:
        f.write(summary_text)

    print(
        f"\nResults and metrics properly saved to PyTorch Lightning logs at {summary_logger.log_dir}"
    )

