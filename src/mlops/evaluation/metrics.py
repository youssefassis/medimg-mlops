"""Classification metrics, with PNEUMONIA (label 1) as the positive class."""

from collections.abc import Sequence

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


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
