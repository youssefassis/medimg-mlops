import math

import numpy as np
import pytest

from mlops.evaluation.metrics import (
    bootstrap_intervals,
    choose_threshold,
    compute_metrics,
    sensitivity_specificity,
)


def test_compute_metrics():
    # At 0.5: 3 true positives, 1 false negative, 1 false positive, 3 true negatives.
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1]

    metrics = compute_metrics(labels, scores)

    assert metrics["recall"] == pytest.approx(3 / 4)
    assert metrics["specificity"] == pytest.approx(3 / 4)
    assert metrics["precision"] == pytest.approx(3 / 4)
    assert metrics["f1"] == pytest.approx(3 / 4)
    assert metrics["roc_auc"] == pytest.approx(15 / 16)


def test_compute_metrics_uses_the_threshold():
    metrics = compute_metrics([1, 0], [0.4, 0.2], threshold=0.3)

    assert metrics["recall"] == 1
    assert metrics["specificity"] == 1


def test_choose_threshold_is_the_highest_reaching_the_target():
    labels = [1, 1, 1, 1, 0, 0]
    scores = [0.9, 0.7, 0.5, 0.2, 0.6, 0.1]

    assert choose_threshold(labels, scores, target_sensitivity=0.75) == 0.5
    assert choose_threshold(labels, scores, target_sensitivity=0.76) == 0.2
    assert choose_threshold(labels, scores, target_sensitivity=1.0) == 0.2


def test_choose_threshold_needs_positives():
    with pytest.raises(ValueError, match="No positive"):
        choose_threshold([0, 0], [0.1, 0.2], target_sensitivity=0.9)


def test_sensitivity_specificity_is_nan_without_a_class():
    sensitivity, specificity = sensitivity_specificity([0, 0], [0.1, 0.9], threshold=0.5)

    assert math.isnan(sensitivity)
    assert specificity == 0.5


def test_bootstrap_intervals_surround_the_estimate_and_are_reproducible():
    rng = np.random.default_rng(0)
    labels = [1] * 50 + [0] * 150
    scores = list(rng.uniform(0.3, 1, 50)) + list(rng.uniform(0, 0.7, 150))
    recall = compute_metrics(labels, scores)["recall"]

    intervals = bootstrap_intervals(labels, scores, threshold=0.5, n_resamples=200)

    low, high = intervals["recall"]
    assert low < recall < high
    assert intervals == bootstrap_intervals(labels, scores, threshold=0.5, n_resamples=200)
    assert set(intervals) == {"recall", "specificity", "roc_auc"}


def test_bootstrap_intervals_are_nan_without_positives():
    intervals = bootstrap_intervals([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5, n_resamples=10)

    assert all(math.isnan(bound) for bound in intervals["recall"])
