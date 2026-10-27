"""Tests for Boruta all-relevant feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from feature_engineering_kit import BorutaSelection, build_preprocessing_pipeline
from feature_engineering_kit.cli import main
from feature_engineering_kit.pipeline import run_churn_workflow


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 7)), columns=["s1", "s2", "s3", "n1", "n2", "n3", "n4"])
    logit = 2.0 * X["s1"] - 1.5 * X["s2"] + 1.0 * X["s3"] ** 2 - 1.0
    y = (logit + rng.logistic(size=n) > 0).astype(int)
    return X, y


def test_confirms_signal_and_rejects_noise():
    X, y = _frame()
    sel = BorutaSelection(target="y", n_estimators=60, max_iter=40, random_state=1).fit(X, y)
    for c in ("s1", "s2", "s3"):
        assert sel.status_[c] == "confirmed", sel.status_
    assert sum(sel.status_[c] == "rejected" for c in ("n1", "n2", "n3", "n4")) >= 3
    assert set(sel.selected_) <= {"s1", "s2", "s3"} and len(sel.selected_) == 3
    assert sel.selected_[0] == "s1"  # strongest effect has highest mean importance
    out = sel.transform(X.assign(label=["a"] * len(X)))
    assert list(out.columns) == [c for c in X.columns if c in sel.selected_] + ["label"]


def test_history_hits_and_early_stop():
    X, y = _frame(seed=2)
    sel = BorutaSelection(target="y", n_estimators=40, max_iter=60, random_state=0).fit(X, y)
    hist = sel.importance_history_
    assert len(hist) == sel.n_iter_ <= 60
    assert "__shadow_threshold__" in hist.columns
    for c, s in sel.status_.items():
        assert 0 <= sel.hits_[c] <= sel.n_iter_
        if s == "tentative":
            assert c not in sel.decided_at_
            continue
        k = sel.decided_at_[c]
        assert 1 <= k <= sel.n_iter_
        assert hist[c].iloc[:k].notna().all()
        if s == "rejected":
            assert hist[c].iloc[k:].isna().all()  # removed from later fits
        else:
            assert hist[c].iloc[k:].notna().all()  # confirmed stay in the model
    if all(s != "tentative" for s in sel.status_.values()):
        assert sel.n_iter_ < 60


def test_regression_target_and_custom_estimator():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=["a", "b", "z1", "z2"])
    y = 3 * X["a"] + np.sin(2 * X["b"]) + 0.1 * rng.normal(size=300)
    sel = BorutaSelection(target="y", n_estimators=50, max_iter=30).fit(X, y)
    assert {"a", "b"} <= set(sel.selected_)
    assert "z1" not in sel.selected_ and "z2" not in sel.selected_
    Xc, yc = _frame(seed=4)
    et = ExtraTreesClassifier(n_estimators=50, max_depth=6, random_state=0)
    sel2 = BorutaSelection(target="y", estimator=et, max_iter=30, max_features=2).fit(Xc, yc)
    assert len(sel2.selected_) <= 2 and set(sel2.selected_) <= {"s1", "s2", "s3"}
    with pytest.raises(AttributeError):
        BorutaSelection(target="y", estimator=LogisticRegression(), max_iter=2).fit(Xc, yc)


def test_tentative_with_tiny_budget_and_keep_tentative():
    X, y = _frame(seed=5)
    strict = BorutaSelection(target="y", n_estimators=30, max_iter=3, random_state=0).fit(X, y)
    # three iterations cannot reach p < 0.05 / 7 so nothing is decided yet
    assert set(strict.status_.values()) == {"tentative"} and strict.selected_ == []
    loose = BorutaSelection(target="y", n_estimators=30, max_iter=3, keep_tentative=True).fit(X, y)
    assert len(loose.selected_) == 7


def test_shadow_all_flag_both_modes_find_signal():
    X, y = _frame(seed=8)
    a = BorutaSelection(target="y", n_estimators=30, max_iter=25, shadow_all=True).fit(X, y)
    b = BorutaSelection(target="y", n_estimators=30, max_iter=25, shadow_all=False).fit(X, y)
    for sel in (a, b):
        assert {"s1", "s2", "s3"} <= set(sel.selected_)


def test_validation_and_no_numeric_columns():
    with pytest.raises(ValueError):
        BorutaSelection(target="y", alpha=0)
    with pytest.raises(ValueError):
        BorutaSelection(target="y", perc=0)
    with pytest.raises(ValueError):
        BorutaSelection(target="y", max_iter=0)
    with pytest.raises(ValueError):
        BorutaSelection(target="y", max_features=0)
    X, y = _frame(n=50)
    with pytest.raises(ValueError):
        BorutaSelection(target="y").fit(X)
    with pytest.raises(ValueError):
        BorutaSelection(target="y").fit(X, y[:10])
    bad = X.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        BorutaSelection(target="y").fit(bad, y)
    only_text = pd.DataFrame({"t": ["a", "b"] * 5})
    sel = BorutaSelection(target="y").fit(only_text, [0, 1] * 5)
    assert sel.selected_ == [] and list(sel.transform(only_text).columns) == ["t"]


def test_pipeline_and_cli_boruta(tmp_path):
    X, y = _frame(n=300, seed=7)
    pipe = build_preprocessing_pipeline(
        X, scale_numeric=False, drop_high_correlation=None, select_k=2, select_method="boruta",
    )
    out = pipe.fit_transform(X, y)
    assert set(out.columns) == {"s1", "s2"}  # capped at select_k by mean importance
    result = run_churn_workflow(n_samples=400, seed=0, select_k=4, select_method="boruta")
    assert result.selection_method == "boruta" and 1 <= result.n_features_engineered <= 4
    target = tmp_path / "report.md"
    assert main(["run", "--n-samples", "400", "--select-k", "3", "--select-method", "boruta",
                 "--output", str(target)]) == 0
    assert "Selected by boruta" in target.read_text(encoding="utf-8")
