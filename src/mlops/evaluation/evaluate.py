"""Scores models: during training on the validation split, and afterwards on the test set."""

import logging
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlops.data.chest_xray import LABELS, list_samples
from mlops.data.dataset import ChestXrayDataset
from mlops.data.transforms import build_transform
from mlops.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """The model must already be in eval mode (models loaded from MLflow always are)."""
    labels: list[int] = []
    scores: list[float] = []
    for images, batch_labels in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1)
        scores += probabilities[:, LABELS.index("PNEUMONIA")].tolist()
        labels += batch_labels.tolist()
    return compute_metrics(labels, scores)


def evaluate_run(run_id: str, data_dir: Path, batch_size: int = 32) -> dict[str, float]:
    """Scores the run's logged model on test/ and logs the metrics to the same run."""
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    samples = list_samples(data_dir, "test")
    loader = DataLoader(ChestXrayDataset(samples, data_dir, build_transform()), batch_size)

    metrics = with_prefix("test_", evaluate_model(model, loader, torch.device("cpu")))
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("n_test", len(samples))
        mlflow.log_metrics(metrics)
    logger.info("run %s: %s", run_id, metrics)
    return metrics


def with_prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {prefix + name: value for name, value in metrics.items()}
