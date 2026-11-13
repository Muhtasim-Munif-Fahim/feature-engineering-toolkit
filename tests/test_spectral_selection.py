"""Tests for Laplacian Score and Fisher Score feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import (
    FisherScoreSelection,
    LaplacianScoreSelection,
    fisher_scores,
    laplacian_scores,
)


def _clustered(seed=0, n=60):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1, 2], n // 3)
    centers = np.array([0.0, 6.0, 12.0])
    df = pd.DataFrame(
        {
            "noise_a": rng.normal(size=n),
            "signal": centers[labels] + rng.normal(scale=0.5, size=n),
            "noise_b": rng.uniform(-3, 3, size=n),
            "signal2": -centers[labels] + rng.normal(scale=0.7, size=n),
        }
    )
    return df, labels


def _reference_laplacian(A, k, t):
    n = A.shape[0]
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    S = np.zeros((n, n))
    for i in range(n):
        order = [j for j in np.argsort(d2[i], kind="stable") if j != i][:k]
        for j in order:
            S[i, j] = S[j, i] = np.exp(-d2[i, j] / t)
    D = np.diag(S.sum(1))
    L = D - S
    one = np.ones(n)
    out = []
    for r in range(A.shape[1]):
        f = A[:, r]
        f = f - (f @ D @ one) / (one @ D @ one) * one
        out.append((f @ L @ f) / (f @ D @ f))
    return np.array(out)


def test_laplacian_matches_reference_loop():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(25, 4))
    np.testing.assert_allclose(laplacian_scores(A, n_neighbors=4, t=2.0), _reference_laplacian(A, 4, 2.0), rtol=1e-10)


def test_supervised_graph_equals_one_over_one_plus_fisher():
    df, y = _clustered()
    ls = laplacian_scores(df, graph="supervised", y=y)
    fs = fisher_scores(df, y)
    np.testing.assert_allclose(ls, 1.0 / (1.0 + fs), rtol=1e-10)


def test_fisher_score_closed_form():
    X = np.array([[1.0], [3.0], [10.0], [12.0]])
    y = np.array([0, 0, 1, 1])
    # mu=6.5, between = 2*(4.5^2)*2 = 81, within = 2*1 + 2*1 = 4
    np.testing.assert_allclose(fisher_scores(X, y), [81.0 / 4.0])
    Z = np.array([[0.0, 5.0], [0.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
    out = fisher_scores(Z, y)
    assert np.isinf(out[0]) and out[1] == 0.0


def test_scores_rank_signal_first():
    df, y = _clustered()
    ls = laplacian_scores((df - df.mean()) / df.std(ddof=0), n_neighbors=5)
    assert set(np.argsort(ls)[:2]) == {1, 3}
    assert np.all((ls >= 0) & (ls <= 2.0 + 1e-12))
    fs = fisher_scores(df, y)
    assert set(np.argsort(-fs)[:2]) == {1, 3}


def test_laplacian_selection_transformer():
    df, _ = _clustered()
    df["city"] = "x"
    df["const"] = 1.0
    sel = LaplacianScoreSelection(k=2, n_neighbors=5).fit(df)
    assert set(sel.selected_) == {"signal", "signal2"}
    assert np.isnan(sel.scores_["const"])
    out = sel.transform(df)
    assert list(out.columns) == ["signal", "signal2", "city"]
    binary = LaplacianScoreSelection(k=1, t=None).fit(df)
    assert binary.selected_[0] in {"signal", "signal2"}


def test_fisher_selection_transformer():
    df, y = _clustered()
    df["target"] = y
    sel = FisherScoreSelection(k=1, target="target").fit(df, pd.Series(y))
    assert "target" not in sel.scores_
    assert sel.selected_ == ["signal"] or sel.selected_ == ["signal2"]
    out = sel.transform(df)
    assert "noise_a" not in out.columns and "target" in out.columns


def test_validation():
    df, y = _clustered()
    with pytest.raises(ValueError):
        laplacian_scores(df, graph="other")
    with pytest.raises(ValueError):
        laplacian_scores(df, graph="supervised")
    with pytest.raises(ValueError):
        laplacian_scores(df, t=-1.0)
    with pytest.raises(ValueError):
        laplacian_scores(np.array([[np.nan, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        fisher_scores(df, np.zeros(len(df)))
    with pytest.raises(ValueError):
        fisher_scores(df, y[:5])
    with pytest.raises(ValueError):
        LaplacianScoreSelection(k=0)
    with pytest.raises(ValueError):
        FisherScoreSelection().fit(df)
    with pytest.raises(RuntimeError):
        LaplacianScoreSelection().transform(df)
