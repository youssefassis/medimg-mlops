"""The training pipeline: one function per step, called by the Airflow DAG and scripts/train.py.

Settings come from config/training.yaml, config/promotion.yaml and config/ingestion.yaml; only
the batch number changes between runs. Each run uses batches 1 to that number: all the data
released so far.
"""

import logging
from dataclasses import replace
from pathlib import Path

from mlops.config import IngestionConfig, PromotionConfig, TrainingConfig, load_config
from mlops.data.batches import heldout_path, read_batch, read_batches
from mlops.data.ingestion import header_checks
from mlops.data.validation import validate_batch
from mlops.evaluation.evaluate import evaluate_run
from mlops.registry.promotion import promote as promote_model
from mlops.reporting.report import log_report
from mlops.training.train import train as train_model

ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def validate(batch: int) -> None:
    _setup_logging()
    config = _training_config(batch)
    data_dir = ROOT / config.data_dir
    samples = read_batches(data_dir, batch)
    test_patients = {sample.patient_id for sample in read_batch(heldout_path(data_dir))}
    validate_batch(samples, data_dir, test_patients)
    logger.info("batches 1 to %d are valid: %d images", batch, len(samples))


def train(batch: int) -> str:
    """Returns the MLflow run ID. The decision threshold aims for the gate's minimum recall."""
    _setup_logging()
    config = _training_config(batch)
    target_sensitivity = _promotion_config().min_recall
    checks = header_checks(load_config(ROOT / "config" / "ingestion.yaml", IngestionConfig))
    return train_model(config, ROOT / config.data_dir, target_sensitivity, checks)


def evaluate(run_id: str) -> dict[str, float]:
    _setup_logging()
    config = _training_config()
    return evaluate_run(run_id, ROOT / config.data_dir, config.batch_size)


def promote(run_id: str) -> bool:
    """Returns whether the model went to production. A rejection is not an error.

    Records the decision in the run and on the model version, as soon as it is made: later,
    production may have changed.
    """
    _setup_logging()
    config = _promotion_config()
    decision = promote_model(run_id, config)
    log_report(run_id, decision, config)
    return decision.promoted


def _setup_logging() -> None:
    # Steps run in fresh processes (Airflow tasks), so each one sets up its own logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def _promotion_config() -> PromotionConfig:
    return load_config(ROOT / "config" / "promotion.yaml", PromotionConfig)


def _training_config(batch: int | None = None) -> TrainingConfig:
    config = load_config(ROOT / "config" / "training.yaml", TrainingConfig)
    return config if batch is None else replace(config, batch=batch)
