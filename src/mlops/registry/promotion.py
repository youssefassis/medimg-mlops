"""Decides whether a trained model may replace the production model."""

from mlops.config import PromotionConfig

# Must not drop below the production model's (minus config.max_drop).
COMPARED_METRICS = ("recall", "specificity", "roc_auc")


def check_gate(
    candidate: dict[str, float],
    production: dict[str, float] | None,
    config: PromotionConfig,
) -> list[str]:
    """Returns why the candidate can't be promoted; an empty list means it passes.

    Metrics are test metrics without their "test_" prefix. `production` is None when no model
    is in production yet.
    """
    reasons = []
    if candidate["recall"] < config.min_recall:
        reasons.append(f"recall {candidate['recall']:.3f} < {config.min_recall}")
    if candidate["specificity"] < config.min_specificity:
        reasons.append(f"specificity {candidate['specificity']:.3f} < {config.min_specificity}")
    if production is not None:
        for name in COMPARED_METRICS:
            if candidate[name] < production[name] - config.max_drop:
                reasons.append(
                    f"{name} {candidate[name]:.3f} < production {production[name]:.3f}"
                    f" - {config.max_drop}"
                )
    return reasons
