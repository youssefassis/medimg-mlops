"""Needs Airflow, which isn't in the project environment. CI runs these in a separate job."""

import inspect
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp())  # keep Airflow out of ~/airflow
pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402
from airflow.sdk import Context  # noqa: E402

DAGS_DIR = Path(__file__).parents[2] / "dags"


@pytest.fixture(scope="module")
def dag_bag():
    return DagBag(dag_folder=DAGS_DIR)


def test_dags_load_without_errors(dag_bag):
    assert dag_bag.import_errors == {}


def test_training_pipeline_task_order(dag_bag):
    dag = dag_bag.dags["training_pipeline"]

    upstream = {task.task_id: task.upstream_task_ids for task in dag.tasks}
    assert upstream == {
        "validate": set(),
        "train": {"validate"},
        "evaluate": {"train"},
        "promote": {"train", "evaluate"},  # gets the run ID from train, runs after evaluate
    }


def test_training_pipeline_retries_its_tasks(dag_bag):
    dag = dag_bag.dags["training_pipeline"]

    assert all(task.retries == 2 for task in dag.tasks)


def test_task_arguments_do_not_clash_with_airflow_context(dag_bag):
    # Airflow refuses to run a task whose argument has the name of a context key, like run_id.
    dag = dag_bag.dags["training_pipeline"]

    for task in dag.tasks:
        arguments = inspect.signature(task.python_callable).parameters
        assert not arguments.keys() & Context.__annotations__.keys(), task.task_id
