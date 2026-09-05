"""Run a demo of feature_engineering_kit and write a Markdown report.

Executes the full workflow (data loading -> preprocessing -> training ->
evaluation -> reporting) on the synthetic churn dataset and writes the resulting
report to ``examples/output/demo_report.md``.
"""

from __future__ import annotations

from pathlib import Path

from feature_engineering_kit import render_markdown_report, run_churn_workflow


def main() -> None:
    result = run_churn_workflow(n_samples=2000, seed=42, model="logreg")
    report = render_markdown_report(result)
    out = Path("examples/output/demo_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    print(
        f"accuracy={result.accuracy:.3f} (baseline {result.baseline_accuracy:.3f}) "
        f"roc_auc={result.roc_auc:.3f} (baseline {result.baseline_roc_auc:.3f}) "
        f"features={result.n_features_engineered} (baseline {result.n_features_baseline})"
    )


if __name__ == "__main__":
    main()
