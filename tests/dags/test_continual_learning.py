from pathlib import Path

import pytest
from airflow.dag_processing.dagbag import DagBag

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"


@pytest.fixture(scope="module")
def dag_bag():
    return DagBag(dag_folder=DAGS_DIR)


def test_dags_import_without_errors(dag_bag):
    assert dag_bag.import_errors == {}


def test_continual_learning_has_all_stages(dag_bag):
    dag = dag_bag.dags["continual_learning"]
    assert set(dag.task_ids) == {
        "validate_landing_batch",
        "preprocess_batch",
        "train_model",
        "evaluate_model",
        "check_deployment_criteria",
        "publish_report",
        "choose_deployment_path",
        "promote_candidate",
        "reject_candidate",
    }
