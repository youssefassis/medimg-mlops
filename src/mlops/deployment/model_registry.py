"""Promote approved models in the MLflow model registry."""

import logging

logger = logging.getLogger(__name__)


def promote_to_production(mlflow_run_id: str) -> None:
    """Placeholder until phase 4: will register the model and move the `production` alias."""
    logger.info("Would promote model from MLflow run %s", mlflow_run_id)
