"""Tests for minimum-redundancy maximum-relevance (mRMR) selection."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import (
    MRMRSelection,
    MutualInfoSelection,
    build_preprocessing_pipeline,
    discrete_mutual_info,
    quantile_discretize,
    run_churn_workflow,
)
from feature_engineering_kit.cli import main


def _redundant_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = ((a + 0.8 * b) > 0).astype(int)
    X = pd.DataFrame(
        {
            "a": a,
            "a_copy": a + rng.normal(scale=0.05, size=n),  # redundant with a
            "a_copy2": 2 * a + 1,  # exact monotone copy of a
            "b": b,
            "noise": noise,
            "label": rng.choice(["x", "y"], size=n),  # non-numeric passes through
        }
    )
    return X, y


def test_quantile_discretize_codes_and_nan():
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0, np.nan, 6.0, 7.0])
    codes = quantile_discretize(x, n_bins=2)
    assert codes[5] == -1
    finite = codes[codes >= 0]
    assert set(finite.tolist()) == {0, 1}
    assert np.bincount(finite).tolist() == [3, 4] or np.bincount(finite).tolist() == [4, 3]
    # heavy ties collapse duplicate edges
    assert set(quantile_discretize([0, 0, 0, 0, 1], 4).tolist()) <= {0, 1}
    with pytest.raises(ValueError):
        quantile_discretize(x, n_bins=1)


def test_discrete_mutual_info_known_values():
    a = np.array([0, 0, 1, 1] * 25)
    assert discrete_mutual_info(a, a) == pytest.approx(math.log(2))
    assert discrete_mutual_info(a, 1 - a) == pytest.approx(math.log(2))
    b = np.array([0, 1, 0, 1] * 25)
    assert discrete_mutual_info(a, b) == pytest.approx(0.0, abs=1e-12)
    assert discrete_mutual_info(["p", "q", "p", "q"], [1, 2, 1, 2]) == pytest.approx(math.log(2))
    with pytest.raises(ValueError):
        discrete_mutual_info([0, 1], [0])


def test_mrmr_skips_redundant_copies_where_mutual_info_does_not():
    X, y = _redundant_data()
    mrmr = MRMRSelection(target="y", k=2).fit(X, y)
    assert mrmr.selected_[0] in {"a", "a_copy", "a_copy2"}
    assert mrmr.selected_[1] == "b"
    out = mrmr.transform(X)
    assert set(out.columns) == {mrmr.selected_[0], "b", "label"}
    assert set(mrmr.drop_) == {"a", "a_copy", "a_copy2", "b", "noise"} - set(mrmr.selected_)
    # independent ranking picks two copies of `a`
    mi = MutualInfoSelection(target="y", k=2).fit(X, y)
    assert set(mi.selected_) <= {"a", "a_copy", "a_copy2"}


def test_mrmr_scores_follow_difference_criterion():
    X, y = _redundant_data(seed=1)
    sel = MRMRSelection(target="y", k=3, n_bins=8).fit(X, y)
    first, second = sel.selected_[:2]
    assert sel.scores_[first] == pytest.approx(sel.relevance_[first])
    codes = {c: quantile_discretize(X[c].to_numpy(), 8) for c in sel.relevance_}
    expected = sel.relevance_[second] - discrete_mutual_info(codes[second], codes[first])
    assert sel.scores_[second] == pytest.approx(expected)
    # relevance ordering: noise is the least relevant numeric column
    assert min(sel.relevance_, key=sel.relevance_.get) == "noise"


def test_quotient_scheme_and_regression_target():
    X, y = _redundant_data(seed=2)
    q = MRMRSelection(target="y", k=2, scheme="quotient").fit(X, y)
    assert q.selected_[1] == "b"
    rng = np.random.default_rng(3)
    y_reg = X["a"] * 2 + X["b"] ** 2 + rng.normal(scale=0.1, size=len(X))
    r = MRMRSelection(target="y", k=2, task="regression").fit(X, y_reg)
    assert set(r.selected_) & {"a", "a_copy", "a_copy2"} and "b" in r.selected_


def test_k_larger_than_features_and_validation():
    X, y = _redundant_data(n=200)
    sel = MRMRSelection(target="y", k=50).fit(X, y)
    assert len(sel.selected_) == 5 and sel.drop_ == []
    with pytest.raises(ValueError, match="requires y"):
        MRMRSelection(target="y").fit(X)
    with pytest.raises(ValueError, match="one value per row"):
        MRMRSelection(target="y").fit(X, y[:10])
    with pytest.raises(ValueError, match="scheme"):
        MRMRSelection(target="y", scheme="sum")
    with pytest.raises(ValueError, match="task"):
        MRMRSelection(target="y", task="ranking")
    with pytest.raises(ValueError, match="k must"):
        MRMRSelection(target="y", k=0)
    with pytest.raises(RuntimeError):
        MRMRSelection(target="y").transform(X)


def test_pipeline_select_step_and_workflow():
    X, y = _redundant_data(n=400)
    pipe = build_preprocessing_pipeline(
        X.drop(columns=["label"]), scale_numeric=False, drop_high_correlation=None, select_k=2
    )
    out = pipe.fit_transform(X.drop(columns=["label"]), y)
    assert out.shape[1] == 2 and "b" in out.columns
    with pytest.raises(ValueError, match="select_method"):
        build_preprocessing_pipeline(X, select_method="lasso")
    result = run_churn_workflow(n_samples=600, seed=0, select_k=5)
    assert result.n_features_engineered == 5
    assert len(result.selected_features) == 5 and result.selection_method == "mrmr"


def test_cli_select_k(tmp_path):
    target = tmp_path / "report.md"
    rc = main(["run", "--n-samples", "600", "--select-k", "4", "--output", str(target)])
    assert rc == 0
    assert "Selected by mrmr (4, in pick order)" in target.read_text(encoding="utf-8")
    rc = main(["run", "--n-samples", "600", "--select-k", "4", "--select-method", "mutual_info",
               "--output", str(target)])
    assert rc == 0
    assert "Selected by mutual_info (4" in target.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["run", "--select-k", "0"])
