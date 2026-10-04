"""Trains the classifier on one batch and logs the run to the MLflow server."""

import logging
import os
from pathlib import Path

import mlflow

from mlops.config import TrainingConfig, load_config
from mlops.training.train import train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(ROOT / "config" / "training.yaml", TrainingConfig)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    train(config, ROOT / config.data_dir)


if __name__ == "__main__":
    main()
