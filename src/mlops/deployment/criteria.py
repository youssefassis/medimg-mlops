"""Deployment criteria: decide whether a candidate model may replace the production model."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeploymentCriteria:
    min_auroc: float
    min_sensitivity: float
    max_auroc_drop_vs_production: float


@dataclass(frozen=True)
class DeploymentDecision:
    approved: bool
    unmet_criteria: list[str] = field(default_factory=list)


def check_deployment_criteria(
    candidate_metrics: dict[str, float],
    production_metrics: dict[str, float] | None,
    criteria: DeploymentCriteria,
) -> DeploymentDecision:
    """`production_metrics` is None when no model is deployed yet."""
    unmet_criteria = []
    if candidate_metrics["auroc"] < criteria.min_auroc:
        unmet_criteria.append(
            f"AUROC {candidate_metrics['auroc']:.3f} < minimum {criteria.min_auroc:.3f}"
        )
    if candidate_metrics["sensitivity"] < criteria.min_sensitivity:
        unmet_criteria.append(
            f"Sensitivity {candidate_metrics['sensitivity']:.3f} "
            f"< minimum {criteria.min_sensitivity:.3f}"
        )
    if production_metrics is not None:
        auroc_floor = production_metrics["auroc"] - criteria.max_auroc_drop_vs_production
        if candidate_metrics["auroc"] < auroc_floor:
            unmet_criteria.append(
                f"AUROC {candidate_metrics['auroc']:.3f} drops too far below production "
                f"{production_metrics['auroc']:.3f} "
                f"(tolerance {criteria.max_auroc_drop_vs_production:.3f})"
            )
    return DeploymentDecision(approved=not unmet_criteria, unmet_criteria=unmet_criteria)
