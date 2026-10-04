"""Trains the classifier on one batch, scores it on test/, and logs both to MLflow."""

import logging
import os
from pathlib import Path

import mlflow

from mlops.config import TrainingConfig, load_config
from mlops.evaluation.evaluate import evaluate_run
from mlops.training.train import train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(ROOT / "config" / "training.yaml", TrainingConfig)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    data_dir = ROOT / config.data_dir
    run_id = train(config, data_dir)
    evaluate_run(run_id, data_dir, config.batch_size)


if __name__ == "__main__":
    main()
