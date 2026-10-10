import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from torch import nn

from mlops.config import PromotionConfig
from mlops.registry.promotion import check_gate, promote

CONFIG = PromotionConfig(
    model_name="test-model", min_recall=0.9, min_specificity=0.7, max_drop=0.01
)
GOOD = {"recall": 0.95, "specificity": 0.80, "roc_auc": 0.95}


def test_good_model_passes_when_nothing_is_in_production():
    assert check_gate(GOOD, None, CONFIG) == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"recall": 0.85}, "recall 0.850 < 0.9"),
        ({"specificity": 0.556}, "specificity 0.556 < 0.7"),
    ],
)
def test_model_below_a_minimum_fails(change, reason):
    assert check_gate(GOOD | change, None, CONFIG) == [reason]


def test_model_may_be_slightly_worse_than_production():
    candidate = {name: value - 0.009 for name, value in GOOD.items()}

    assert check_gate(candidate, GOOD, CONFIG) == []


@pytest.mark.parametrize("name", ["recall", "specificity", "roc_auc"])
def test_model_clearly_worse_than_production_fails(name):
    candidate = GOOD | {name: GOOD[name] - 0.02}

    assert check_gate(candidate, GOOD, CONFIG) == [
        f"{name} {candidate[name]:.3f} < production {GOOD[name]:.3f} - 0.01"
    ]


def test_each_view_and_sex_must_meet_the_minimums():
    candidate = GOOD | {
        "recall_view_AP": 0.95,
        "recall_view_PA": 0.85,
        "specificity_sex_F": 0.65,
        "recall_age_0-17": 0.10,  # age isn't gated: too few images per group
    }

    assert check_gate(candidate, None, CONFIG) == [
        "recall for view PA 0.850 < 0.9",
        "specificity for sex F 0.650 < 0.7",
    ]


def test_every_failed_rule_is_reported():
    candidate = {"recall": 0.5, "specificity": 0.5, "roc_auc": 0.5}

    assert len(check_gate(candidate, GOOD, CONFIG)) == 5


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts under the working directory
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")


def log_run(metrics: dict[str, float]) -> str:
    """Logs a tiny model with test metrics, like the train and evaluate steps do."""
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(
            nn.Linear(2, 2), name="model", input_example=np.zeros((2, 2), dtype=np.float32)
        )
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
    return run.info.run_id


def version_tags(run_id: str) -> dict[str, str]:
    client = MlflowClient()
    (version,) = client.search_model_versions(f"name = 'test-model' and run_id = '{run_id}'")
    return version.tags


def production_run_id() -> str:
    return MlflowClient().get_model_version_by_alias("test-model", "production").run_id


def test_first_good_model_goes_to_production(mlflow_store):
    run_id = log_run(GOOD)

    assert promote(run_id, CONFIG) is True
    assert production_run_id() == run_id
    assert version_tags(run_id)["promotion"] == "passed"


def test_worse_model_is_registered_but_not_promoted(mlflow_store):
    good_run = log_run(GOOD)
    promote(good_run, CONFIG)
    worse_run = log_run(GOOD | {"roc_auc": 0.90})

    assert promote(worse_run, CONFIG) is False
    assert production_run_id() == good_run
    assert version_tags(worse_run) == {
        "promotion": "rejected",
        "reason": "roc_auc 0.900 < production 0.950 - 0.01",
    }


def test_retrying_registers_the_run_once(mlflow_store):
    run_id = log_run(GOOD)

    promote(run_id, CONFIG)
    promote(run_id, CONFIG)

    assert len(MlflowClient().search_model_versions("name = 'test-model'")) == 1


def test_promote_fails_when_the_run_was_not_evaluated(mlflow_store):
    with mlflow.start_run() as run:
        pass

    with pytest.raises(ValueError, match="evaluate it first"):
        promote(run.info.run_id, CONFIG)
