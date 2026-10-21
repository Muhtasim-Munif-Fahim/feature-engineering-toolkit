"""Tests for ReliefF feature weighting and selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import (
    ReliefFSelection,
    build_preprocessing_pipeline,
    relieff_scores,
    run_churn_workflow,
)
from feature_engineering_kit.cli import main


def _reference_data():
    rng = np.random.default_rng(42)
    n = 40
    X = np.round(rng.normal(size=(n, 4)), 3)
    X[:, 1] = rng.integers(0, 3, n)  # categorical column -> mixed data
    y = np.array([0, 1, 2] * 13 + [0])
    Xc = X.copy()
    Xc[:, 1] = rng.normal(size=n)  # all continuous
    return X, y, Xc


def test_matches_scikit_rebate_reference():
    # Values from skrebate.ReliefF(n_neighbors=5).feature_importances_ (pinned).
    X, y, Xc = _reference_data()
    np.testing.assert_allclose(
        relieff_scores(X, y, n_neighbors=5),
        [-0.03929980112694729, -0.04499999999999999, -0.005824169396350019, 0.024765535427654185],
        rtol=0, atol=1e-14,
    )
    np.testing.assert_allclose(
        relieff_scores(Xc, (y == 0).astype(int), n_neighbors=5),
        [0.01947961551209811, 0.0067573029027644255, 0.006300889096864774, -0.0015340518229763856],
        rtol=0, atol=1e-14,
    )


def _xor_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n).astype(float) + 0.05 * rng.normal(size=n)
    b = rng.integers(0, 2, n).astype(float) + 0.05 * rng.normal(size=n)
    y = (np.round(a).astype(int) ^ np.round(b).astype(int))
    X = pd.DataFrame({"a": a, "b": b, "n1": rng.normal(size=n), "n2": rng.uniform(size=n),
                      "n3": rng.normal(size=n), "city": rng.choice(["x", "y"], n)})
    return X, pd.Series(y)


def test_relieff_finds_xor_interaction_that_mutual_info_misses():
    X, y = _xor_frame()
    sel = ReliefFSelection(target="y", k=2, n_neighbors=10).fit(X, y)
    assert set(sel.selected_) == {"a", "b"}
    assert sel.scores_["a"] > 0.1 and sel.scores_["b"] > 0.1
    assert max(abs(sel.scores_[c]) for c in ("n1", "n2", "n3")) < 0.05
    out = sel.transform(X)
    assert list(out.columns) == ["a", "b", "city"]  # non-numeric passes through
    # each XOR input alone carries ~no information about y
    assert abs(np.corrcoef(X["a"], y)[0, 1]) < 0.15 and abs(np.corrcoef(X["b"], y)[0, 1]) < 0.15


def test_scores_bounded_and_multiclass():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    y = np.digitize(X[:, 0], [-0.5, 0.5])  # 3 classes driven by column 0
    w = relieff_scores(X, y, n_neighbors=8)
    assert np.all(np.abs(w) <= 1)
    assert w.argmax() == 0 and w[0] > 0.1


def test_categorical_features_override_and_threshold():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 15, size=(80, 2)).astype(float)
    y = (X[:, 0] > 7).astype(int)
    auto = relieff_scores(X, y, n_neighbors=5)  # 15 levels -> continuous
    forced = relieff_scores(X, y, n_neighbors=5, categorical_features=[0, 1])
    assert not np.allclose(auto, forced)
    assert forced[0] > forced[1] and auto[0] > auto[1]
    with pytest.raises(ValueError):
        relieff_scores(X, y, categorical_features=[5])


def test_missing_values_are_skipped():
    X, y = _xor_frame(n=300, seed=3)
    Xn = X[["a", "b", "n1"]].to_numpy().copy()
    rng = np.random.default_rng(4)
    Xn[rng.random(Xn.shape) < 0.05] = np.nan
    w = relieff_scores(Xn, y, n_neighbors=10)
    assert np.all(np.isfinite(w))
    assert w[0] > w[2] and w[1] > w[2]


def test_validation():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError):
        relieff_scores(X, np.zeros(5))  # single class
    with pytest.raises(ValueError):
        relieff_scores(X, [0, 1, 0, 1])
    with pytest.raises(ValueError):
        relieff_scores(X, [0, 1, 0, 1, 0], n_neighbors=0)
    with pytest.raises(ValueError):
        ReliefFSelection(target="y", k=0)
    with pytest.raises(ValueError):
        ReliefFSelection(target="y").fit(pd.DataFrame({"a": [1.0, 2.0]}))


def test_pipeline_and_cli_relieff(tmp_path):
    X, y = _xor_frame(n=300)
    pipe = build_preprocessing_pipeline(
        X[["a", "b", "n1", "n2"]], scale_numeric=False, drop_high_correlation=None,
        select_k=2, select_method="relieff",
    )
    out = pipe.fit_transform(X[["a", "b", "n1", "n2"]], y)
    assert set(out.columns) == {"a", "b"}
    result = run_churn_workflow(n_samples=400, seed=0, select_k=4, select_method="relieff")
    assert result.n_features_engineered == 4 and result.selection_method == "relieff"
    target = tmp_path / "report.md"
    assert main(["run", "--n-samples", "400", "--select-k", "3", "--select-method", "relieff",
                 "--output", str(target)]) == 0
    assert "Selected by relieff (3" in target.read_text(encoding="utf-8")
