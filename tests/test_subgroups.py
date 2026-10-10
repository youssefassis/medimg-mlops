import math
import re

import pytest

from mlops.data.rsna import ManifestRow
from mlops.evaluation.subgroups import age_group, subgroup_results


def row(view: str, sex: str, age: int, label: str) -> ManifestRow:
    return ManifestRow("x.png", label, "p", "c", view, sex, age, 0.143, "sha")


@pytest.mark.parametrize(
    ("age", "group"),
    [(0, "0-17"), (17, "0-17"), (18, "18-39"), (40, "40-64"), (65, "65-plus"), (90, "65-plus")],
)
def test_age_group(age, group):
    assert age_group(age) == group


def test_age_groups_are_valid_in_mlflow_metric_names():
    # MLflow allows alphanumerics, "_", "-", ".", " ", ":" and "/".
    for age in range(91):
        assert re.fullmatch(r"[\w\-. :/]+", f"recall_age_{age_group(age)}")


def test_subgroup_results_score_each_group_separately():
    rows = [
        row("AP", "F", 30, "PNEUMONIA"),
        row("AP", "F", 30, "NORMAL"),
        row("PA", "M", 70, "PNEUMONIA"),
        row("PA", "M", 70, "NORMAL"),
    ]
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.1, 0.2, 0.8]  # right on AP, wrong on PA

    results = {
        (result.subgroup, result.group): result
        for result in subgroup_results(rows, labels, scores, threshold=0.5)
    }

    assert set(results) == {
        ("view", "AP"),
        ("view", "PA"),
        ("sex", "F"),
        ("sex", "M"),
        ("age", "18-39"),
        ("age", "65-plus"),
        ("pixel_spacing_mm", "0.143"),
    }
    ap, pa = results["view", "AP"], results["view", "PA"]
    assert (ap.n, ap.n_positive, ap.recall, ap.specificity) == (2, 1, 1.0, 1.0)
    assert (pa.recall, pa.specificity) == (0.0, 0.0)
    assert ap.metric_name("recall") == "recall_view_AP"


def test_group_without_positives_has_nan_recall():
    rows = [row("AP", "F", 30, "NORMAL")] * 2

    (result, *_) = subgroup_results(rows, [0, 0], [0.1, 0.2], threshold=0.5)

    assert math.isnan(result.recall)
    assert result.specificity == 1.0
