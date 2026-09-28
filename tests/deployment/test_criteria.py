from mlops.deployment.criteria import DeploymentCriteria, check_deployment_criteria
from mlops.settings import load_config

CRITERIA = DeploymentCriteria(
    min_auroc=0.85, min_sensitivity=0.90, max_auroc_drop_vs_production=0.01
)


def test_approves_first_model_meeting_thresholds():
    decision = check_deployment_criteria(
        {"auroc": 0.90, "sensitivity": 0.92}, production_metrics=None, criteria=CRITERIA
    )
    assert decision.approved
    assert decision.unmet_criteria == []


def test_rejects_below_absolute_thresholds():
    decision = check_deployment_criteria(
        {"auroc": 0.80, "sensitivity": 0.85}, production_metrics=None, criteria=CRITERIA
    )
    assert not decision.approved
    assert len(decision.unmet_criteria) == 2


def test_tolerates_small_drop_versus_production():
    decision = check_deployment_criteria(
        {"auroc": 0.895, "sensitivity": 0.92}, production_metrics={"auroc": 0.90}, criteria=CRITERIA
    )
    assert decision.approved


def test_rejects_drop_beyond_tolerance():
    decision = check_deployment_criteria(
        {"auroc": 0.87, "sensitivity": 0.92}, production_metrics={"auroc": 0.90}, criteria=CRITERIA
    )
    assert not decision.approved
    assert "below production" in decision.unmet_criteria[0]


def test_repo_config_builds_criteria():
    assert isinstance(DeploymentCriteria(**load_config("deployment_criteria")), DeploymentCriteria)
