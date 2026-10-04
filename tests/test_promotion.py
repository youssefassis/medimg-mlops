import pytest

from mlops.config import PromotionConfig
from mlops.registry.promotion import check_gate

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


def test_every_failed_rule_is_reported():
    candidate = {"recall": 0.5, "specificity": 0.5, "roc_auc": 0.5}

    assert len(check_gate(candidate, GOOD, CONFIG)) == 5
