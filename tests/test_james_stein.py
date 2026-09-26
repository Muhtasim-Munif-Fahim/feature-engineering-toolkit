"""Tests for James-Stein shrinkage target encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import JamesSteinEncoder


def _df(**cols):
    return pd.DataFrame(cols)


def _reference_js(categories, y):
    """Independent James-Stein / empirical-Bayes formula."""
    categories = np.asarray(categories, dtype=object)
    y = np.asarray(y, dtype=float)
    global_mean = float(y.mean())
    cats = pd.Series(categories)
    target = pd.Series(y)
    grouped = target.groupby(cats, sort=False)
    counts = grouped.size()
    means = grouped.mean()
    ss = grouped.apply(lambda s: float(((s - s.mean()) ** 2).sum()))
    df = (counts - 1).clip(lower=0)
    denom = float(df.sum())
    sigma2 = float(ss.sum() / denom) if denom > 0 else 0.0
    n_cats = int(counts.shape[0])
    if n_cats < 2 or sigma2 == 0.0:
        tau2 = 0.0
    else:
        between = float(((means - means.mean()) ** 2).sum() / (n_cats - 1))
        expected_noise = float((sigma2 / counts).mean())
        tau2 = max(0.0, between - expected_noise)
    mapping = {}
    for cat, n_c, m_c in zip(counts.index.tolist(), counts.to_numpy(), means.to_numpy()):
        if tau2 <= 0.0:
            mapping[str(cat)] = global_mean
        else:
            shrink = sigma2 / (sigma2 + float(n_c) * tau2)
            mapping[str(cat)] = (1.0 - shrink) * float(m_c) + shrink * global_mean
    return mapping, global_mean, sigma2, tau2


def test_js_requires_y() -> None:
    enc = JamesSteinEncoder(columns=["x"], target="y")
    with pytest.raises(ValueError, match="requires y"):
        enc.fit(_df(x=["a", "b"]))


def test_js_rejects_length_mismatch() -> None:
    enc = JamesSteinEncoder(columns=["x"], target="y")
    with pytest.raises(ValueError, match="expected y to have 2 rows"):
        enc.fit(_df(x=["a", "b"]), pd.Series([1.0]))


def test_js_shrinks_rare_categories_harder() -> None:
    # Many observations in "a", few in "b" with an extreme mean.
    X = _df(x=["a"] * 40 + ["b"] * 2)
    y = pd.Series([0.0] * 20 + [1.0] * 20 + [10.0, 10.0])
    enc = JamesSteinEncoder(columns=["x"], target="y")
    out = enc.fit_transform(X, y)
    mean_a = float(out.loc[X["x"] == "a", "x"].iloc[0])
    mean_b = float(out.loc[X["x"] == "b", "x"].iloc[0])
    raw_a = 0.5
    raw_b = 10.0
    # Rare "b" should move farther toward the global mean than abundant "a".
    assert abs(mean_b - raw_b) > abs(mean_a - raw_a)
    assert mean_b < raw_b  # shrunk down toward ~0.95 global mean-ish
    assert enc.tau2_ > 0.0


def test_js_matches_reference() -> None:
    X = _df(x=["a", "a", "a", "b", "b", "c", "c", "c", "c"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    mapping, global_mean, sigma2, tau2 = _reference_js(X["x"], y)
    enc = JamesSteinEncoder(columns=["x"], target="y")
    out = enc.fit_transform(X, y)["x"].to_numpy()
    expected = np.array([mapping[str(v)] for v in X["x"].tolist()])
    assert out == pytest.approx(expected)
    assert enc.global_mean_ == pytest.approx(global_mean)
    assert enc.sigma2_ == pytest.approx(sigma2)
    assert enc.tau2_ == pytest.approx(tau2)


def test_js_unseen_maps_to_global_mean() -> None:
    X = _df(x=["a", "a", "b", "b"])
    y = pd.Series([0.0, 1.0, 0.0, 1.0])
    enc = JamesSteinEncoder(columns=["x"], target="y").fit(X, y)
    held = enc.transform(_df(x=["a", "z"]))
    assert held["x"].iloc[0] == pytest.approx(enc.maps_["x"]["a"])
    assert held["x"].iloc[1] == pytest.approx(enc.global_mean_)


def test_js_single_category_collapses_to_global_mean() -> None:
    X = _df(x=["a", "a", "a", "a"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    enc = JamesSteinEncoder(columns=["x"], target="y")
    out = enc.fit_transform(X, y)["x"].to_numpy()
    assert out == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert enc.tau2_ == 0.0
