"""Tests for supervised MDLP (Fayyad-Irani) binning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import FeatureEngineeringPipeline, MDLPBinning
from feature_engineering_kit.binning import _mdlp_cuts


def _step_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    noise = rng.uniform(0, 10, n)
    # class changes sharply at x=3 and x=7
    y = np.where(x < 3, 0, np.where(x < 7, 1, 0))
    flip = rng.random(n) < 0.05
    y = np.where(flip, 1 - y, y)
    return pd.DataFrame({"x": x, "noise": noise, "other": ["a"] * n}), pd.Series(y, name="y")


def test_finds_class_change_points():
    X, y = _step_data()
    binner = MDLPBinning(columns=["x"]).fit(X, y)
    cuts = binner.cut_points_["x"]
    assert len(cuts) >= 2
    assert min(abs(c - 3) for c in cuts) < 0.3
    assert min(abs(c - 7) for c in cuts) < 0.3
    assert binner.n_bins_["x"] == len(cuts) + 1


def test_irrelevant_feature_stays_one_bin():
    X, y = _step_data()
    binner = MDLPBinning(columns=["noise"]).fit(X, y)
    assert binner.cut_points_["noise"] == []
    out = binner.transform(X)
    assert set(out["noise"].unique()) == {0.0}


def test_perfectly_separable_single_cut():
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0] * 5)
    labels = (values > 5).astype(int)
    order = np.argsort(values, kind="mergesort")
    cuts = _mdlp_cuts(values[order], labels[order], 2, 1)
    assert cuts == [7.0]


def test_transform_codes_and_passthrough():
    X, y = _step_data()
    binner = MDLPBinning(columns=["x"])
    out = binner.fit_transform(X, y)
    assert out["other"].tolist() == X["other"].tolist()
    assert out["noise"].equals(X["noise"])
    codes = out["x"].to_numpy()
    assert codes.min() == 0 and codes.max() == binner.n_bins_["x"] - 1
    # monotone in x
    order = np.argsort(X["x"].to_numpy())
    assert np.all(np.diff(codes[order]) >= 0)


def test_onehot_and_inverse():
    X, y = _step_data()
    binner = MDLPBinning(columns=["x"], encode="onehot").fit(X, y)
    out = binner.transform(X)
    names = [f"x__bin_{i}" for i in range(binner.n_bins_["x"])]
    assert all(name in out.columns for name in names) and "x" not in out.columns
    assert (out[names].sum(axis=1) == 1).all()
    back = binner.inverse_transform(out)
    edges = binner.bin_edges_["x"]
    assert back["x"].between(edges[0], edges[-1]).all()


def test_multiclass_target_and_dataframe_y():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 9, 900)
    y = pd.DataFrame({"label": np.array(["lo", "mid", "hi"])[(x // 3).astype(int)]})
    binner = MDLPBinning(columns=["x"], target="label").fit(pd.DataFrame({"x": x}), y)
    cuts = binner.cut_points_["x"]
    assert len(cuts) == 2
    assert cuts[0] == pytest.approx(3, abs=0.1) and cuts[1] == pytest.approx(6, abs=0.1)


def test_max_bins_keeps_most_informative_cut():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, 2000)
    # strong change at 5 (0.1 -> 0.9), weak change at 8 (0.9 -> 0.6)
    p = np.where(x < 5, 0.1, np.where(x < 8, 0.9, 0.6))
    y = (rng.random(x.size) < p).astype(int)
    X = pd.DataFrame({"x": x})
    full = MDLPBinning(columns=["x"]).fit(X, y)
    capped = MDLPBinning(columns=["x"], max_bins=2).fit(X, y)
    assert len(full.cut_points_["x"]) >= 2
    assert len(capped.cut_points_["x"]) == 1
    assert capped.cut_points_["x"][0] == pytest.approx(5, abs=0.3)


def test_min_samples_leaf_and_missing_values():
    X, y = _step_data(n=300)
    X.loc[::17, "x"] = np.nan
    binner = MDLPBinning(columns=["x"], min_samples_leaf=40).fit(X, y)
    out = binner.transform(X)
    assert out["x"].isna().sum() == X["x"].isna().sum()
    codes = out["x"].dropna()
    assert codes.value_counts().min() >= 30  # leaves are large


def test_held_out_uses_training_cuts_and_pipeline():
    X, y = _step_data(seed=3)
    X_test, _ = _step_data(n=50, seed=4)
    pipe = FeatureEngineeringPipeline(steps=[("mdlp", MDLPBinning(columns=["x"]))])
    pipe.fit(X, y)
    out = pipe.transform(X_test)
    binner = pipe.steps[0][1]
    expected = np.digitize(X_test["x"], binner.bin_edges_["x"][1:-1])
    np.testing.assert_array_equal(out["x"].to_numpy(), expected.astype(float))


def test_errors():
    X, y = _step_data(n=50)
    with pytest.raises(ValueError):
        MDLPBinning(columns=["x"]).fit(X)
    with pytest.raises(ValueError):
        MDLPBinning(columns=["x"]).fit(X, y[:10])
    with pytest.raises(KeyError):
        MDLPBinning(columns=["missing"]).fit(X, y)
    with pytest.raises(ValueError):
        MDLPBinning(columns=["other"]).fit(X, y)
    with pytest.raises(ValueError):
        MDLPBinning(columns=["x"], min_samples_leaf=0)
    with pytest.raises(ValueError):
        MDLPBinning(columns=["x"], max_bins=1)
    with pytest.raises(ValueError):
        MDLPBinning(columns=["x"], encode="dense")
    with pytest.raises(RuntimeError):
        MDLPBinning(columns=["x"]).transform(X)
