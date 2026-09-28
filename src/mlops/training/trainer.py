"""Train (fine-tune) a candidate model on the new batch."""

import mlflow

MLFLOW_EXPERIMENT_NAME = "cxr-pneumonia"


def train_candidate(batch_id: str) -> str:
    """Placeholder until phase 3: logs an empty MLflow run and returns its run ID."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=batch_id) as active_run:
        mlflow.log_param("batch_id", batch_id)
    return active_run.info.run_id
