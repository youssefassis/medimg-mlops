import pytest

from mlops.evaluation.metrics import compute_metrics


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
