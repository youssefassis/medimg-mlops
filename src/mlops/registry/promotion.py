"""Registers trained models and decides which one is in production."""

import logging

import mlflow
from mlflow import MlflowClient

from mlops.config import PromotionConfig

PRODUCTION_ALIAS = "production"

# Must not drop below the production model's (minus config.max_drop).
COMPARED_METRICS = ("recall", "specificity", "roc_auc")

logger = logging.getLogger(__name__)


def check_gate(
    candidate: dict[str, float],
    production: dict[str, float] | None,
    config: PromotionConfig,
) -> list[str]:
    """Returns why the candidate can't be promoted; an empty list means it passes.

    Metrics are test metrics without their "test_" prefix. `production` is None when no model
    is in production yet.
    """
    reasons = []
    if candidate["recall"] < config.min_recall:
        reasons.append(f"recall {candidate['recall']:.3f} < {config.min_recall}")
    if candidate["specificity"] < config.min_specificity:
        reasons.append(f"specificity {candidate['specificity']:.3f} < {config.min_specificity}")
    if production is not None:
        for name in COMPARED_METRICS:
            if candidate[name] < production[name] - config.max_drop:
                reasons.append(
                    f"{name} {candidate[name]:.3f} < production {production[name]:.3f}"
                    f" - {config.max_drop}"
                )
    return reasons


def promote(run_id: str, config: PromotionConfig) -> bool:
    """Registers the run's model and gives it the production alias if it passes the gate.

    Returns whether it was promoted. Either way, the model version gets a `promotion` tag
    (passed or rejected) and a `reason` tag. Safe to retry: a run is registered only once.
    """
    client = MlflowClient()
    candidate = _test_metrics(client, run_id)
    production = _production_metrics(client, config.model_name)
    version = _register(client, run_id, config.model_name)

    reasons = check_gate(candidate, production, config)
    if reasons:
        client.set_model_version_tag(config.model_name, version, "promotion", "rejected")
        client.set_model_version_tag(config.model_name, version, "reason", "; ".join(reasons))
        logger.info("%s version %s rejected: %s", config.model_name, version, reasons)
        return False

    client.set_registered_model_alias(config.model_name, PRODUCTION_ALIAS, version)
    client.set_model_version_tag(config.model_name, version, "promotion", "passed")
    client.set_model_version_tag(config.model_name, version, "reason", _pass_reason(production))
    logger.info("%s version %s is now in production", config.model_name, version)
    return True


def _test_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    metrics = client.get_run(run_id).data.metrics
    missing = [name for name in COMPARED_METRICS if f"test_{name}" not in metrics]
    if missing:
        raise ValueError(f"Run {run_id} has no test metrics {missing}: evaluate it first")
    return {name: metrics[f"test_{name}"] for name in COMPARED_METRICS}


def _production_metrics(client: MlflowClient, model_name: str) -> dict[str, float] | None:
    models = client.search_registered_models(filter_string=f"name = '{model_name}'")
    if not models or PRODUCTION_ALIAS not in models[0].aliases:
        return None
    version = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    return _test_metrics(client, version.run_id)


def _register(client: MlflowClient, run_id: str, model_name: str) -> str:
    existing = client.search_model_versions(f"name = '{model_name}' and run_id = '{run_id}'")
    if existing:
        return existing[0].version
    return mlflow.register_model(f"runs:/{run_id}/model", model_name).version


def _pass_reason(production: dict[str, float] | None) -> str:
    if production is None:
        return "first model to pass the minimums"
    return "passes the minimums and matches production"
