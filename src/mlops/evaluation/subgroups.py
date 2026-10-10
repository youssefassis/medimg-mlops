"""Metrics per subgroup of patients: a model can do well overall and badly for one group."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mlops.data.rsna import ManifestRow
from mlops.evaluation.metrics import bootstrap_intervals, sensitivity_specificity

# Names end up in MLflow metric names, which allow no "+" or "<".
AGE_GROUPS = ((65, "65-plus"), (40, "40-64"), (18, "18-39"), (0, "0-17"))


def age_group(age: int) -> str:
    return next(name for lower, name in AGE_GROUPS if age >= lower)


SUBGROUPS: dict[str, Callable[[ManifestRow], str]] = {
    "view": lambda row: row.view,
    "sex": lambda row: row.sex,
    "age": lambda row: age_group(row.age),
    "pixel_spacing_mm": lambda row: f"{row.pixel_spacing_mm:.3f}",
}
# Checked by the promotion gate. The other subgroups are too small for a stable rule.
GATED_SUBGROUPS = ("view", "sex")


@dataclass(frozen=True)
class SubgroupResult:
    subgroup: str  # e.g. "view"
    group: str  # e.g. "AP"
    n: int
    n_positive: int
    recall: float
    recall_low: float
    recall_high: float
    specificity: float
    specificity_low: float
    specificity_high: float

    def metric_name(self, metric: str) -> str:
        """e.g. "recall_view_AP"."""
        return f"{metric}_{self.subgroup}_{self.group}"


def subgroup_results(
    rows: Sequence[ManifestRow], labels: Sequence[int], scores: Sequence[float], threshold: float
) -> list[SubgroupResult]:
    """rows, labels and scores describe the same images, in the same order."""
    results = []
    for subgroup, group_of in SUBGROUPS.items():
        by_group: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_group.setdefault(group_of(row), []).append(index)
        for group, indexes in sorted(by_group.items()):
            group_labels = [labels[i] for i in indexes]
            group_scores = [scores[i] for i in indexes]
            recall, specificity = sensitivity_specificity(group_labels, group_scores, threshold)
            intervals = bootstrap_intervals(group_labels, group_scores, threshold)
            results.append(
                SubgroupResult(
                    subgroup,
                    group,
                    n=len(indexes),
                    n_positive=sum(group_labels),
                    recall=recall,
                    recall_low=intervals["recall"][0],
                    recall_high=intervals["recall"][1],
                    specificity=specificity,
                    specificity_low=intervals["specificity"][0],
                    specificity_high=intervals["specificity"][1],
                )
            )
    return results
