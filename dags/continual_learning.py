"""Continual-learning loop: validate a new batch, retrain, evaluate, report, and deploy if approved.

The DAG only orchestrates; every step's logic lives in the `mlops` package.
"""

import logging
from dataclasses import asdict
from datetime import datetime

from airflow.sdk import Param, dag, task

from mlops.data_validation import landing_batch
from mlops.deployment import model_registry
from mlops.deployment.criteria import DeploymentCriteria, check_deployment_criteria
from mlops.evaluation import metrics
from mlops.preprocessing import dicom_standardization
from mlops.reporting import html_report
from mlops.settings import load_config
from mlops.training import trainer

logger = logging.getLogger(__name__)


@dag(
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={
        "batch_id": Param(
            "batch_001",
            type="string",
            description="Folder of the new batch inside the `landing` bucket.",
        )
    },
    tags=["mlops"],
)
def continual_learning():
    @task
    def validate_landing_batch(batch_id: str) -> int:
        return landing_batch.count_landing_files(batch_id)

    @task
    def preprocess_batch(batch_id: str, file_count: int) -> str:
        logger.info("Preprocessing %d files", file_count)
        return dicom_standardization.standardize_batch(batch_id)

    @task
    def train_model(batch_id: str, processed_prefix: str) -> str:
        logger.info("Training on %s", processed_prefix)
        return trainer.train_candidate(batch_id)

    @task
    def evaluate_model(mlflow_run_id: str) -> dict[str, float]:
        return metrics.evaluate_candidate(mlflow_run_id)

    @task(task_id="check_deployment_criteria")
    def check_criteria(candidate_metrics: dict[str, float]) -> dict:
        criteria = DeploymentCriteria(**load_config("deployment_criteria"))
        # No model registry yet, so there is no production model to compare against.
        decision = check_deployment_criteria(
            candidate_metrics, production_metrics=None, criteria=criteria
        )
        return asdict(decision)

    @task
    def publish_report(
        batch_id: str,
        mlflow_run_id: str,
        candidate_metrics: dict[str, float],
        deployment_decision: dict,
    ) -> str:
        return html_report.publish_report(
            batch_id, mlflow_run_id, candidate_metrics, deployment_decision
        )

    @task.branch
    def choose_deployment_path(deployment_decision: dict) -> str:
        return "promote_candidate" if deployment_decision["approved"] else "reject_candidate"

    @task
    def promote_candidate(mlflow_run_id: str) -> None:
        model_registry.promote_to_production(mlflow_run_id)

    @task
    def reject_candidate(deployment_decision: dict) -> None:
        unmet = "; ".join(deployment_decision["unmet_criteria"])
        logger.warning("Candidate rejected: %s", unmet)

    batch_id = "{{ params.batch_id }}"
    file_count = validate_landing_batch(batch_id)
    mlflow_run_id = train_model(batch_id, preprocess_batch(batch_id, file_count))
    candidate_metrics = evaluate_model(mlflow_run_id)
    deployment_decision = check_criteria(candidate_metrics)
    report_uri = publish_report(batch_id, mlflow_run_id, candidate_metrics, deployment_decision)
    deployment_path = choose_deployment_path(deployment_decision)
    report_uri >> deployment_path
    deployment_path >> [promote_candidate(mlflow_run_id), reject_candidate(deployment_decision)]


continual_learning()
