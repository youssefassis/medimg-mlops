"""Scores models: during training on the validation split, and afterwards on the test set."""

import logging
import math
from dataclasses import asdict
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlops.data.batches import heldout_path
from mlops.data.chest_xray import LABELS, Sample
from mlops.data.dataset import ChestXrayDataset
from mlops.data.lineage import log_dataset
from mlops.data.rsna import read_rows
from mlops.data.transforms import build_transform
from mlops.evaluation.metrics import bootstrap_intervals, compute_metrics
from mlops.evaluation.subgroups import SubgroupResult, subgroup_results

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list, list]:
    """Returns (labels, pneumonia probabilities). The model must already be in eval mode
    (models loaded from MLflow always are)."""
    labels: list[int] = []
    scores: list[float] = []
    for images, batch_labels in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1)
        scores += probabilities[:, LABELS.index("PNEUMONIA")].tolist()
        labels += batch_labels.tolist()
    return labels, scores


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    return compute_metrics(*predict(model, loader, device))


def evaluate_run(run_id: str, data_dir: Path, batch_size: int = 32) -> dict[str, float]:
    """Scores the run's model on the held-out test set, at the threshold chosen in training.

    Logs to the same run: test metrics with 95% confidence intervals, recall and specificity
    per subgroup, the full subgroup table as validation/subgroups.json, and the test set as a
    dataset input.
    """
    threshold = float(mlflow.get_run(run_id).data.params["threshold"])
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    rows = read_rows(heldout_path(data_dir))
    samples = [Sample(Path(row.path), row.label, row.patient_id) for row in rows]
    loader = DataLoader(ChestXrayDataset(samples, data_dir, build_transform()), batch_size)
    labels, scores = predict(model, loader, torch.device("cpu"))

    intervals = bootstrap_intervals(labels, scores, threshold)
    subgroups = subgroup_results(rows, labels, scores, threshold)
    metrics = with_prefix(
        "test_",
        compute_metrics(labels, scores, threshold)
        | {f"{name}_ci_low": low for name, (low, _) in intervals.items()}
        | {f"{name}_ci_high": high for name, (_, high) in intervals.items()}
        | subgroup_metrics(subgroups),
    )
    finite = {name: value for name, value in metrics.items() if not math.isnan(value)}
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("n_test", len(samples))
        log_dataset(heldout_path(data_dir), context="testing")
        mlflow.log_metrics(finite)
        mlflow.log_dict(
            {"threshold": threshold, "subgroups": [_json_row(result) for result in subgroups]},
            "validation/subgroups.json",
        )
    logger.info("run %s at threshold %.3f: %s", run_id, threshold, finite)
    return finite


def subgroup_metrics(results: list[SubgroupResult]) -> dict[str, float]:
    """Recall and specificity of each group, named like "recall_view_AP"."""
    return {
        result.metric_name(metric): getattr(result, metric)
        for result in results
        for metric in ("recall", "specificity")
    }


def _json_row(result: SubgroupResult) -> dict:
    """NaN isn't valid JSON: a metric that can't be computed is written as null."""
    return {
        name: None if isinstance(value, float) and math.isnan(value) else value
        for name, value in asdict(result).items()
    }


def with_prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {prefix + name: value for name, value in metrics.items()}
