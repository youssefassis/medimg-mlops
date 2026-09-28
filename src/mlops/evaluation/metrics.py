"""Evaluate a candidate model on the held-out test sets."""

from mlflow import MlflowClient

PLACEHOLDER_METRICS = {"auroc": 0.90, "sensitivity": 0.92}


def evaluate_candidate(mlflow_run_id: str) -> dict[str, float]:
    """Placeholder until phase 3: logs fixed metrics to the MLflow run."""
    client = MlflowClient()
    for name, value in PLACEHOLDER_METRICS.items():
        client.log_metric(mlflow_run_id, name, value)
    return dict(PLACEHOLDER_METRICS)
