from mlops.reporting.html_report import render_report_html


def test_report_shows_unmet_criteria_escaped():
    html = render_report_html(
        "batch_001",
        "run123",
        {"auroc": 0.8},
        {"approved": False, "unmet_criteria": ["AUROC 0.800 < minimum 0.850"]},
    )
    assert "REJECTED" in html
    assert "AUROC 0.800 &lt; minimum 0.850" in html
