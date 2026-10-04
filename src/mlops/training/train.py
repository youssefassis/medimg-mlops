"""Fine-tunes the classifier on one batch and tracks the run in MLflow."""

import copy
import logging
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlops.config import TrainingConfig
from mlops.data.batches import batch_path, read_batch, split_off_patients
from mlops.data.chest_xray import LABELS, Sample
from mlops.data.dataset import ChestXrayDataset
from mlops.data.transforms import IMAGE_SIZE, build_transform
from mlops.evaluation.evaluate import evaluate_model, with_prefix
from mlops.training.model import build_model

logger = logging.getLogger(__name__)


def train(config: TrainingConfig, data_dir: Path, pretrained: bool = True) -> str:
    """Trains on batch number `config.batch` and returns the MLflow run ID.

    Keeps the epoch with the best validation ROC AUC. Scoring on test/ is done separately,
    by `mlops.evaluation.evaluate.evaluate_run`.
    """
    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    samples = read_batch(batch_path(data_dir, config.batch))
    train_samples, val_samples = split_off_patients(samples, config.val_fraction, config.seed)

    train_loader = make_loader(train_samples, data_dir, config, shuffle=True)
    val_loader = make_loader(val_samples, data_dir, config)

    model = build_model(pretrained).to(device)
    weights = class_weights(train_samples)
    weight_params = {
        f"class_weight_{label}": round(weight, 4)
        for label, weight in zip(LABELS, weights.tolist(), strict=True)
    }
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    mlflow.set_experiment(config.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                **asdict(config),
                "device": device.type,
                "pretrained": pretrained,
                "n_train": len(train_samples),
                "n_val": len(val_samples),
                **weight_params,
            }
        )

        best_auc, best_epoch, best_state = -1.0, 0, None
        for epoch in range(1, config.epochs + 1):
            start = time.monotonic()
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            model.eval()
            val_metrics = evaluate_model(model, val_loader, device)
            mlflow.log_metrics(
                {"train_loss": train_loss, **with_prefix("val_", val_metrics)}, step=epoch
            )
            logger.info(
                "epoch %d/%d: train_loss %.4f, val_roc_auc %.4f (%.0fs)",
                epoch,
                config.epochs,
                train_loss,
                val_metrics["roc_auc"],
                time.monotonic() - start,
            )
            if val_metrics["roc_auc"] > best_auc:
                best_auc, best_epoch = val_metrics["roc_auc"], epoch
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        mlflow.log_param("best_epoch", best_epoch)
        logger.info("best epoch %d, val_roc_auc %.4f", best_epoch, best_auc)

        log_model(model.cpu())
        return run.info.run_id


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    samples: list[Sample], data_dir: Path, config: TrainingConfig, shuffle: bool = False
) -> DataLoader:
    dataset = ChestXrayDataset(samples, data_dir, build_transform())
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, generator=generator)


def class_weights(samples: list[Sample]) -> torch.Tensor:
    """Weights each class by the inverse of its share, so both count the same in the loss."""
    counts = Counter(sample.label for sample in samples)
    missing = [label for label in LABELS if counts[label] == 0]
    if missing:
        raise ValueError(f"No training images for {missing}")
    return torch.tensor([len(samples) / (len(LABELS) * counts[label]) for label in LABELS])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Returns the mean training loss over the epoch."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def log_model(model: nn.Module) -> None:
    """Logs the model in MLflow's pt2 format, which is traced from an example input.

    The example has 2 images: tracing with 1 would fix the batch size to 1.
    """
    example = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    mlflow.pytorch.log_model(model.eval(), name="model", input_example=example.numpy())
