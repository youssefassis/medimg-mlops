"""Classification metrics, with PNEUMONIA (label 1) as the positive class."""

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

INTERVAL_METRICS = ("recall", "specificity", "roc_auc")


def compute_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> dict[str, float]:
    """`scores` are the predicted probabilities of pneumonia."""
    predictions = [int(score >= threshold) for score in scores]
    return {
        "recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "specificity": recall_score(labels, predictions, pos_label=0, zero_division=0),
        "precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
    }


def choose_threshold(
    labels: Sequence[int], scores: Sequence[float], target_sensitivity: float
) -> float:
    """Returns the highest threshold that still finds `target_sensitivity` of the positives.

    The highest one keeps specificity as high as possible. Choose it on validation data: chosen
    on the test set, the test metrics would be optimistic.
    """
    positive_scores = sorted(
        (score for score, label in zip(scores, labels, strict=True) if label == 1), reverse=True
    )
    if not positive_scores:
        raise ValueError("No positive cases to choose a threshold from")
    needed = math.ceil(target_sensitivity * len(positive_scores))
    return positive_scores[needed - 1]


def sensitivity_specificity(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> tuple[float, float]:
    """NaN when a class is missing, e.g. in a small subgroup, instead of a misleading 0."""
    labels_array = np.asarray(labels)
    predictions = np.asarray(scores) >= threshold
    positives, negatives = labels_array == 1, labels_array == 0
    sensitivity = predictions[positives].mean() if positives.any() else math.nan
    specificity = (~predictions[negatives]).mean() if negatives.any() else math.nan
    return float(sensitivity), float(specificity)


def bootstrap_intervals(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """95% confidence intervals of recall, specificity and ROC AUC.

    Resamples the images with replacement and takes the 2.5th and 97.5th percentiles. A metric
    whose class is missing from the data gets (nan, nan).
    """
    labels_array, scores_array = np.asarray(labels), np.asarray(scores)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in INTERVAL_METRICS}
    for _ in range(n_resamples):
        index = rng.integers(0, len(labels_array), len(labels_array))
        resampled_labels, resampled_scores = labels_array[index], scores_array[index]
        recall, specificity = sensitivity_specificity(resampled_labels, resampled_scores, threshold)
        samples["recall"].append(recall)
        samples["specificity"].append(specificity)
        if len(set(resampled_labels.tolist())) == 2:
            samples["roc_auc"].append(roc_auc_score(resampled_labels, resampled_scores))

    intervals = {}
    for name, values in samples.items():
        finite = [value for value in values if not math.isnan(value)]
        low, high = np.percentile(finite, [2.5, 97.5]) if finite else (math.nan, math.nan)
        intervals[name] = (float(low), float(high))
    return intervals
