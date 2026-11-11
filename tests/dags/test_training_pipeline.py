"""Needs Airflow, which isn't in the project environment. CI runs these in a separate job."""

import inspect
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp())  # keep Airflow out of ~/airflow
pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402
from airflow.sdk import Asset, Context  # noqa: E402

DAGS_DIR = Path(__file__).parents[2] / "dags"
BATCHES = Asset("batches")


@pytest.fixture(scope="module")
def dag_bag():
    return DagBag(dag_folder=DAGS_DIR)


def test_dags_load_without_errors(dag_bag):
    assert dag_bag.import_errors == {}


def test_training_pipeline_task_order(dag_bag):
    dag = dag_bag.dags["training_pipeline"]

    upstream = {task.task_id: task.upstream_task_ids for task in dag.tasks}
    assert upstream == {
        "newest_batch": set(),
        "validate": {"newest_batch"},
        "train": {"newest_batch", "validate"},
        "evaluate": {"train"},
        "promote": {"train", "evaluate"},  # gets the run ID from train, runs after evaluate
    }


def test_training_pipeline_retries_its_tasks(dag_bag):
    dag = dag_bag.dags["training_pipeline"]

    assert all(task.retries == 2 for task in dag.tasks)


def test_mlops_task_arguments_do_not_clash_with_airflow_context(dag_bag):
    # Airflow refuses to run a task whose argument has the name of a context key, like run_id.
    # Only newest_batch asks for a context key on purpose.
    dag = dag_bag.dags["training_pipeline"]

    for task in dag.tasks:
        if task.task_id == "newest_batch":
            continue
        arguments = inspect.signature(task.python_callable).parameters
        assert not arguments.keys() & Context.__annotations__.keys(), task.task_id


def test_releasing_a_batch_emits_its_number(dag_bag):
    (release,) = dag_bag.dags["release_batch"].tasks
    outlet_events = {BATCHES: SimpleNamespace()}

    release.python_callable(params={"batch": 3}, outlet_events=outlet_events)

    assert release.outlets == [BATCHES]
    assert outlet_events[BATCHES].extra == {"batch": 3}


def test_a_released_batch_triggers_training(dag_bag):
    dag = dag_bag.dags["training_pipeline"]

    assert dag.timetable.asset_condition.objects == (BATCHES,)
    assert dag.max_active_runs == 1


def test_training_uses_the_newest_batch_released(dag_bag):
    newest_batch = dag_bag.dags["training_pipeline"].get_task("newest_batch")
    events = [SimpleNamespace(extra={"batch": number}) for number in (2, 4, 3)]

    assert newest_batch.python_callable(triggering_asset_events={BATCHES: events}) == 4
