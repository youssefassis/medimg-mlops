import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from torch import nn

from mlops.config import PromotionConfig
from mlops.data.lineage import log_dataset
from mlops.registry.promotion import promote
from mlops.reporting.report import log_report, render_summary

CONFIG = PromotionConfig(
    model_name="test-model", min_recall=0.9, min_specificity=0.7, max_drop=0.01
)


def metrics_with_intervals(recall: float) -> dict[str, float]:
    """Test metrics with intervals, as evaluation logs them, without the "test_" prefix."""
    metrics = {"recall": recall, "specificity": 0.8, "roc_auc": 0.9}
    low = {f"{name}_ci_low": value - 0.05 for name, value in metrics.items()}
    high = {f"{name}_ci_high": value + 0.05 for name, value in metrics.items()}
    return metrics | low | high


REPORT = {
    "run_id": "abc",
    "datasets": [
        {"context": "training", "name": "batch_1", "digest": "1"},
        {"context": "training", "name": "batch_2", "digest": "2"},
        {"context": "testing", "name": "test", "digest": "3"},
    ],
    "threshold": 0.276,
    "decision": {"promoted": False, "reasons": ["recall 0.850 < 0.9"]},
    "metrics": metrics_with_intervals(recall=0.85),
    "production": {"version": "1", "metrics": metrics_with_intervals(recall=0.95)},
}


def test_summary_shows_the_decision_against_production():
    summary = render_summary(REPORT)

    assert summary.startswith("**Rejected**:\n- recall 0.850 < 0.9\n")
    assert "| Metric | This model [95% CI] | Production (v1) |" in summary
    assert "| recall | 0.850 [0.800–0.900] | 0.950 |" in summary
    assert "Trained on batch_1, batch_2, threshold 0.276." in summary


def test_summary_without_production_model():
    summary = render_summary(
        REPORT | {"decision": {"promoted": True, "reasons": []}, "production": None}
    )

    assert summary.startswith("**Promoted** to production.\n")
    assert "| recall | 0.850 [0.800–0.900] | – |" in summary


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts under the working directory
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")


def log_run(tmp_path, recall: float) -> str:
    """Logs what training and evaluation log: params, datasets, model and test metrics."""
    (tmp_path / "batch_1.csv").write_text("path,label,patient_id\n")
    with mlflow.start_run() as run:
        mlflow.log_params({"n_train": 8, "n_val": 2, "n_test": 4, "threshold": 0.3})
        log_dataset(tmp_path / "batch_1.csv", context="training")
        mlflow.pytorch.log_model(
            nn.Linear(2, 2), name="model", input_example=np.zeros((2, 2), dtype=np.float32)
        )
        metrics = metrics_with_intervals(recall)
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
    return run.info.run_id


def test_report_records_the_decision_in_the_run_and_the_registry(mlflow_store, tmp_path):
    production_run = log_run(tmp_path, recall=0.95)
    log_report(production_run, promote(production_run, CONFIG), CONFIG)
    candidate_run = log_run(tmp_path, recall=0.85)

    log_report(candidate_run, promote(candidate_run, CONFIG), CONFIG)

    report = mlflow.artifacts.load_dict(f"runs:/{candidate_run}/report.json")
    assert report["schema_version"] == 1
    assert report["model"] == {"name": "test-model", "version": "2"}
    assert report["datasets"][0]["name"] == "batch_1"
    assert report["images"] == {"train": 8, "val": 2, "test": 4}
    assert report["decision"]["promoted"] is False
    assert report["production"]["version"] == "1"
    assert report["production"]["metrics"]["recall"] == 0.95
    description = MlflowClient().get_model_version("test-model", "2").description
    assert description.startswith("**Rejected**:\n- recall 0.850 < 0.9")
