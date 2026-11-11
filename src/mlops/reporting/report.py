"""Records each run's promotion decision: report.json in the run, for machines, and a Markdown
summary as the model version's description in the registry, for people."""

from dataclasses import asdict

import mlflow
from mlflow import MlflowClient

from mlops.config import PromotionConfig
from mlops.data.lineage import run_datasets
from mlops.registry.promotion import COMPARED_METRICS, Decision

SCHEMA_VERSION = 1  # bump when a field changes meaning or disappears
REPORT_PATH = "report.json"


def build_report(run_id: str, decision: Decision, config: PromotionConfig) -> dict:
    """The data each run used, its test metrics, and what the gate decided against which
    production model."""
    run = MlflowClient().get_run(run_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": {"name": config.model_name, "version": decision.version},
        "datasets": run_datasets(run),
        "images": {split: int(run.data.params[f"n_{split}"]) for split in ("train", "val", "test")},
        "threshold": float(run.data.params["threshold"]),
        "gate": asdict(config),
        "decision": {"promoted": decision.promoted, "reasons": decision.reasons},
        "metrics": decision.metrics,
        "production": None
        if decision.production_version is None
        else {
            "version": decision.production_version,
            "compared": decision.production_metrics is not None,
            "metrics": decision.production_metrics,
        },
    }


def render_summary(report: dict) -> str:
    """Markdown: the decision, the gated metrics against production, and the data used."""
    verdict = "**Promoted** to production." if report["decision"]["promoted"] else "**Rejected**:"
    lines = [verdict, *(f"- {reason}" for reason in report["decision"]["reasons"]), ""]

    metrics, production = report["metrics"], report["production"]
    header = "none in production" if production is None else f"v{production['version']}"
    lines += [f"| Metric | This model [95% CI] | Production ({header}) |", "|---|---|---|"]
    compared = production is not None and production["compared"]
    for name in COMPARED_METRICS:
        low, high = metrics[f"{name}_ci_low"], metrics[f"{name}_ci_high"]
        current = f"{production['metrics'][name]:.3f}" if compared else "–"
        lines.append(f"| {name} | {metrics[name]:.3f} [{low:.3f}–{high:.3f}] | {current} |")

    trained_on = ", ".join(d["name"] for d in report["datasets"] if d["context"] == "training")
    lines.append("")
    if production is not None and not compared:
        lines.append("Not compared with production: it was scored on another test set.")
    lines.append(
        f"Trained on {trained_on}, threshold {report['threshold']:.3f}."
        f" Full record: `{REPORT_PATH}` in run `{report['run_id']}`."
    )
    return "\n".join(lines)


def log_report(run_id: str, decision: Decision, config: PromotionConfig) -> dict:
    """Saves the report in the run and its summary on the model version; returns the report."""
    report = build_report(run_id, decision, config)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_dict(report, REPORT_PATH)
    MlflowClient().update_model_version(
        config.model_name, decision.version, description=render_summary(report)
    )
    return report
