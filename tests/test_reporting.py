"""Tests for Markdown report rendering."""

from __future__ import annotations

from pathlib import Path

from feature_engineering_kit import (
    ChurnEvaluation,
    render_markdown_report,
    run_churn_workflow,
    save_report,
)


def test_report_contains_required_sections() -> None:
    result = run_churn_workflow(n_samples=400, seed=0, model="logreg")
    report = render_markdown_report(result)
    assert "# feature-engineering-toolkit churn workflow report" in report
    assert "## Configuration" in report
    assert "## Evaluation" in report
    assert "## Confusion matrix (test)" in report
    assert "## Top feature importances" in report
    assert "Engineered features:" in report
    assert "baseline" in report.lower()


def test_report_includes_confusion_matrix_values() -> None:
    result = run_churn_workflow(n_samples=400, seed=0, model="logreg")
    report = render_markdown_report(result)
    cm = result.confusion_matrix
    # The four integer cells appear in the report.
    for cell in (cm[0][0], cm[0][1], cm[1][0], cm[1][1]):
        assert str(cell) in report


def test_save_report_writes_file(tmp_path: Path) -> None:
    result = run_churn_workflow(n_samples=400, seed=1, model="logreg")
    report = render_markdown_report(result)
    target = save_report(report, tmp_path / "out" / "report.md")
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "## Evaluation" in text
