"""Command-line interface for feature_engineering_kit."""

from __future__ import annotations

import argparse

from .data import (
    COLUMN_CATEGORICAL,
    COLUMN_DATETIME,
    COLUMN_NUMERIC,
    TARGET_COLUMN,
    column_schema,
    load_synthetic_churn_dataset,
)
from .pipeline import run_churn_workflow
from .reporting import render_markdown_report


def _build_run_parser(sub):
    run = sub.add_parser(
        "run",
        help="Run the churn workflow and emit a Markdown report",
    )
    run.add_argument("--n-samples", type=int, default=2000, help="dataset rows (default: 2000)")
    run.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    run.add_argument("--test-size", type=float, default=0.2, help="held-out fraction (default: 0.2)")
    run.add_argument(
        "--model",
        choices=["logreg", "random_forest"],
        default="logreg",
        help="classifier (default: logreg)",
    )
    run.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="drop column pairs with |corr| above this (default: 0.95)",
    )
    run.add_argument(
        "-o",
        "--output",
        default=None,
        help="write the report to this file instead of stdout",
    )
    return run


def _build_schema_parser(sub):
    schema_p = sub.add_parser(
        "schema",
        help="Print the synthetic churn dataset column schema",
    )
    schema_p.add_argument("--n-samples", type=int, default=2000)
    schema_p.add_argument("--seed", type=int, default=42)
    return schema_p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="feature-engineering-toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    _build_run_parser(sub)
    _build_schema_parser(sub)
    return parser


def _print_summary(result) -> None:
    print(
        f"accuracy={result.accuracy:.3f} (baseline {result.baseline_accuracy:.3f}) "
        f"roc_auc={result.roc_auc:.3f} (baseline {result.baseline_roc_auc:.3f}) "
        f"features={result.n_features_engineered} (baseline {result.n_features_baseline})"
    )


def cmd_run(args: argparse.Namespace) -> int:
    result = run_churn_workflow(
        n_samples=args.n_samples,
        seed=args.seed,
        test_size=args.test_size,
        model=args.model,
        drop_high_correlation=args.correlation_threshold,
    )
    report = render_markdown_report(result)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Wrote {args.output}")
    else:
        print(report)
    _print_summary(result)
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    df = load_synthetic_churn_dataset(n_samples=args.n_samples, seed=args.seed)
    schema = column_schema(df)
    print("feature_engineering_kit churn dataset schema")
    print(f"target: {TARGET_COLUMN}")
    print(f"numeric ({len(schema.numeric)}): {', '.join(schema.numeric) or '<none>'}")
    print(
        f"categorical ({len(schema.categorical)}): "
        f"{', '.join(schema.categorical) or '<none>'}"
    )
    print(f"datetime ({len(schema.datetime)}): {', '.join(schema.datetime) or '<none>'}")
    print(f"rows: {len(df)}  churn rate: {df[TARGET_COLUMN].mean():.3f}")
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "schema":
        return cmd_schema(args)
    parser.error(f"unknown command: {args.command}")
    return 2


__all__ = ["main"]
