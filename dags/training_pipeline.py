"""Retrains whenever a batch is released.

release_batch simulates new data arriving: it releases a batch by hand, as an event on the
BATCHES asset. Each event triggers training_pipeline, which validates the batches released
so far, trains on them, scores the model on the test set, then decides whether to promote it.

Training tasks run in the mlops venv of the Airflow image (see docker/airflow.Dockerfile) and
only call mlops.pipeline. They pass the MLflow run ID to each other, never data.
"""

from datetime import timedelta

from airflow.sdk import Asset, Param, dag, task

MLOPS_PYTHON = "/opt/mlops/.venv/bin/python"
BATCHES = Asset("batches")


@dag(
    schedule=None,
    params={"batch": Param(1, type="integer", minimum=1, maximum=5, description="Batch number")},
    tags=["data"],
)
def release_batch():
    @task(outlets=[BATCHES])
    def release(*, params, outlet_events) -> None:
        outlet_events[BATCHES].extra = {"batch": params["batch"]}

    release()


@dag(
    schedule=[BATCHES],
    max_active_runs=1,  # batches released in a row train one after the other
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["training"],
)
def training_pipeline():
    @task
    def newest_batch(*, triggering_asset_events) -> int:
        """Releases made while a run was waiting can trigger one run together: the newest one
        covers them all, as training uses every batch up to it."""
        return max(event.extra["batch"] for event in triggering_asset_events[BATCHES])

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

    batch = newest_batch()
    mlflow_run_id = train(batch)
    validate(batch) >> mlflow_run_id
    evaluate(mlflow_run_id) >> promote(mlflow_run_id)


release_batch()
training_pipeline()
