"""Render and publish the HTML report for a pipeline run."""

from html import escape

from mlops import object_storage


def render_report_html(
    batch_id: str,
    mlflow_run_id: str,
    candidate_metrics: dict[str, float],
    deployment_decision: dict,
) -> str:
    metric_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{value:.3f}</td></tr>"
        for name, value in candidate_metrics.items()
    )
    verdict = "APPROVED for deployment" if deployment_decision["approved"] else "REJECTED"
    unmet_items = "".join(
        f"<li>{escape(reason)}</li>" for reason in deployment_decision["unmet_criteria"]
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Report {escape(batch_id)}</title></head><body>"
        f"<h1>Batch {escape(batch_id)}</h1><p>MLflow run: {escape(mlflow_run_id)}</p>"
        f"<table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>"
        f"<h2>Deployment: {verdict}</h2><ul>{unmet_items}</ul></body></html>"
    )


def publish_report(
    batch_id: str,
    mlflow_run_id: str,
    candidate_metrics: dict[str, float],
    deployment_decision: dict,
) -> str:
    html = render_report_html(batch_id, mlflow_run_id, candidate_metrics, deployment_decision)
    key = f"{batch_id}/{mlflow_run_id}.html"
    return object_storage.upload_text(object_storage.REPORTS_BUCKET, key, html, "text/html")
