"""Validates a batch, trains on it, scores the model on test/, then decides whether to promote it.

Tasks run in the mlops venv of the Airflow image (see docker/airflow.Dockerfile) and only call
mlops.pipeline. They pass the MLflow run ID to each other, never data.
"""

from datetime import timedelta

from airflow.sdk import Param, dag, task

MLOPS_PYTHON = "/opt/mlops/.venv/bin/python"


@dag(
    schedule=None,
    params={"batch": Param(1, type="integer", minimum=1, maximum=5, description="Batch number")},
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    render_template_as_native_obj=True,  # so "{{ params.batch }}" stays an int
    tags=["training"],
)
def training_pipeline():
    @task.external_python(python=MLOPS_PYTHON, expect_airflow=False)
    def validate(batch: int) -> None:
        from mlops import pipeline

        pipeline.validate(batch)

    @task.external_python(python=MLOPS_PYTHON, expect_airflow=False)
    def train(batch: int) -> str:
        from mlops import pipeline

        return pipeline.train(batch)

    @task.external_python(python=MLOPS_PYTHON, expect_airflow=False)
    def evaluate(mlflow_run_id: str) -> dict[str, float]:
        from mlops import pipeline

        return pipeline.evaluate(mlflow_run_id)

    @task.external_python(python=MLOPS_PYTHON, expect_airflow=False)
    def promote(mlflow_run_id: str) -> bool:
        from mlops import pipeline

        return pipeline.promote(mlflow_run_id)

    batch = "{{ params.batch }}"
    mlflow_run_id = train(batch)
    validate(batch) >> mlflow_run_id
    evaluate(mlflow_run_id) >> promote(mlflow_run_id)


training_pipeline()
