"""Runs the training pipeline outside Airflow, on the batch set in config/training.yaml."""

import os
from pathlib import Path

import mlflow

from mlops import pipeline
from mlops.config import TrainingConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    batch = load_config(ROOT / "config" / "training.yaml", TrainingConfig).batch
    pipeline.validate(batch)
    run_id = pipeline.train(batch)
    pipeline.evaluate(run_id)


if __name__ == "__main__":
    main()
