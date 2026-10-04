"""Needs Airflow, which isn't in the project environment. CI runs these in a separate job."""

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp())  # keep Airflow out of ~/airflow
pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

DAGS_DIR = Path(__file__).parents[2] / "dags"


@pytest.fixture(scope="module")
def dag_bag():
    return DagBag(dag_folder=DAGS_DIR)


def test_dags_load_without_errors(dag_bag):
    assert dag_bag.import_errors == {}


def test_training_pipeline_runs_validate_then_train_then_evaluate(dag_bag):
    dag = dag_bag.dags["training_pipeline"]

    assert dag.get_task("validate").downstream_task_ids == {"train"}
    assert dag.get_task("train").downstream_task_ids == {"evaluate"}
    assert dag.get_task("evaluate").downstream_task_ids == set()


def test_training_pipeline_retries_its_tasks(dag_bag):
    dag = dag_bag.dags["training_pipeline"]

    assert all(task.retries == 2 for task in dag.tasks)
