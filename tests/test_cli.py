"""Tests for the feature-engineering-toolkit CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from feature_engineering_kit.cli import main


def test_cli_schema_prints_columns(capsys) -> None:
    rc = main(["schema"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "target: churn" in out
    assert "numeric" in out
    assert "categorical" in out
    assert "datetime" in out


def test_cli_run_writes_report_to_file(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    rc = main(
        [
            "run",
            "--n-samples", "400",
            "--seed", "0",
            "--model", "logreg",
            "-o", str(target),
        ]
    )
    assert rc == 0
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "## Evaluation" in text
    assert "Engineered features:" in text


def test_cli_run_prints_report_to_stdout(capsys) -> None:
    rc = main(["run", "--n-samples", "300", "--seed", "0", "--model", "logreg"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Configuration" in out
    assert "accuracy=" in out


def test_cli_run_woe_encode_includes_iv(tmp_path: Path) -> None:
    target = tmp_path / "woe_report.md"
    rc = main(
        [
            "run",
            "--n-samples",
            "400",
            "--seed",
            "0",
            "--model",
            "logreg",
            "--woe-encode",
            "region",
            "-o",
            str(target),
        ]
    )
    assert rc == 0
    text = target.read_text(encoding="utf-8")
    assert "## Information Value (WoE)" in text
    assert "region" in text


def test_cli_run_invalid_model_exits_nonzero() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--n-samples", "100", "--model", "bogus"])
