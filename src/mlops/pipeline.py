"""The training pipeline: one function per step, called by the Airflow DAG and scripts/train.py.

Settings come from config/training.yaml; only the batch number changes between runs.
"""

import logging
from dataclasses import replace
from pathlib import Path

from mlops.config import TrainingConfig, load_config
from mlops.data.batches import batch_path, read_batch
from mlops.data.chest_xray import list_samples
from mlops.data.validation import validate_batch
from mlops.evaluation.evaluate import evaluate_run
from mlops.training.train import train as train_model

ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def validate(batch: int) -> None:
    config = _load_config(batch)
    data_dir = ROOT / config.data_dir
    samples = read_batch(batch_path(data_dir, batch))
    test_patients = {sample.patient_id for sample in list_samples(data_dir, "test")}
    validate_batch(samples, data_dir, test_patients)
    logger.info("batch %d is valid: %d images", batch, len(samples))


def train(batch: int) -> str:
    """Returns the MLflow run ID."""
    config = _load_config(batch)
    return train_model(config, ROOT / config.data_dir)


def evaluate(run_id: str) -> dict[str, float]:
    config = _load_config()
    return evaluate_run(run_id, ROOT / config.data_dir, config.batch_size)


def _load_config(batch: int | None = None) -> TrainingConfig:
    # Steps run in fresh processes (Airflow tasks), so each one sets up its own logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(ROOT / "config" / "training.yaml", TrainingConfig)
    return config if batch is None else replace(config, batch=batch)
