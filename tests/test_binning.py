"""Tests for quantile / uniform numeric binning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import (
    FeatureEngineeringPipeline,
    QuantileBinning,
    StandardScaler,
)
from feature_engineering_kit.pipeline import build_preprocessing_pipeline


def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_quantile_bins_are_equal_frequency() -> None:
    df = _frame(a=np.arange(100, dtype=float), b=["keep"] * 100)
    original = df.copy()
    enc = QuantileBinning(columns=["a"], n_bins=5, strategy="quantile")
    out = enc.fit_transform(df)
    codes = out["a"].to_numpy()
    counts = pd.Series(codes).value_counts().sort_index()
    assert enc.n_bins_["a"] == 5
    assert len(enc.bin_edges_["a"]) == 6
    assert counts.tolist() == [20, 20, 20, 20, 20]
    assert list(codes) == sorted(codes.tolist())
    assert out["b"].tolist() == ["keep"] * 100
    pd.testing.assert_frame_equal(df, original)


def test_uniform_edges_are_equal_width_and_right_edge_is_included() -> None:
    df = _frame(a=[0.0, 2.5, 5.0, 7.5, 10.0])
    enc = QuantileBinning(columns=["a"], n_bins=4, strategy="uniform")
    out = enc.fit_transform(df)
    assert enc.bin_edges_["a"].tolist() == pytest.approx([0.0, 2.5, 5.0, 7.5, 10.0])
    assert out["a"].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 3.0])


def test_quantile_and_uniform_edges_differ_on_skewed_data() -> None:
    df = _frame(a=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    quantile = QuantileBinning(columns=["a"], n_bins=4, strategy="quantile").fit(df)
    uniform = QuantileBinning(columns=["a"], n_bins=4, strategy="uniform").fit(df)
    assert uniform.bin_edges_["a"].tolist() == pytest.approx(
        np.linspace(1.0, 100.0, 5).tolist()
    )
    assert len(quantile.bin_edges_["a"]) != len(uniform.bin_edges_["a"])


def test_sparse_quantile_bins_are_collapsed() -> None:
    df = _frame(a=[0.0] * 20 + [1.0] * 20)
    enc = QuantileBinning(columns=["a"], n_bins=8, strategy="quantile").fit(df)
    edges = enc.bin_edges_["a"]
    assert enc.n_bins_["a"] < 8
    assert enc.n_bins_["a"] == len(edges) - 1
    assert np.all(np.diff(edges) > 0)
    out = enc.transform(df)
    assert set(out["a"].tolist()) == set(range(enc.n_bins_["a"]))


def test_constant_column_is_one_bin_and_inverse_restores_value() -> None:
    df = _frame(a=[3.0, 3.0, 3.0], b=[1, 2, 3])
    df.index = [4, 5, 6]
    enc = QuantileBinning(columns=["a"], n_bins=4)
    out = enc.fit_transform(df)
    assert enc.n_bins_["a"] == 1
    assert out["a"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert out.index.tolist() == [4, 5, 6]
    inv = enc.inverse_transform(out)
    assert inv["a"].tolist() == pytest.approx([3.0, 3.0, 3.0])
    assert inv["b"].tolist() == [1, 2, 3]
    assert inv.index.tolist() == [4, 5, 6]


def test_out_of_range_values_clip_to_edge_bins() -> None:
    train = _frame(a=np.linspace(0.0, 10.0, 11))
    test = _frame(a=[-100.0, 0.0, 10.0, 100.0])
    enc = QuantileBinning(columns=["a"], n_bins=5, strategy="uniform").fit(train)
    out = enc.transform(test)
    assert out["a"].tolist() == pytest.approx([0.0, 0.0, 4.0, 4.0])


def test_ordinal_missing_stays_missing_and_inverse_uses_midpoints() -> None:
    df = _frame(a=[0.0, np.nan, 10.0])
    enc = QuantileBinning(columns=["a"], n_bins=2, strategy="uniform")
    out = enc.fit_transform(df)
    assert out["a"].iloc[0] == pytest.approx(0.0)
    assert np.isnan(out["a"].iloc[1])
    assert out["a"].iloc[2] == pytest.approx(1.0)
    inv = enc.inverse_transform(out)
    assert inv["a"].tolist() == pytest.approx([2.5, np.nan, 7.5], nan_ok=True)
    again = enc.transform(inv)
    assert again["a"].iloc[0] == pytest.approx(out["a"].iloc[0])
    assert np.isnan(again["a"].iloc[1])
    assert again["a"].iloc[2] == pytest.approx(out["a"].iloc[2])


def test_onehot_encode_and_inverse() -> None:
    df = _frame(a=[0.0, 5.0, 10.0, np.nan], b=["x", "y", "z", "w"])
    df.index = [10, 11, 12, 13]
    enc = QuantileBinning(columns=["a"], n_bins=2, strategy="uniform", encode="onehot")
    out = enc.fit_transform(df)
    assert list(out.columns) == ["a__bin_0", "a__bin_1", "b"]
    assert out["a__bin_0"].tolist() == [1, 0, 0, 0]
    assert out["a__bin_1"].tolist() == [0, 1, 1, 0]
    assert out["b"].tolist() == ["x", "y", "z", "w"]
    assert out.index.tolist() == [10, 11, 12, 13]
    inv = enc.inverse_transform(out)
    assert list(inv.columns) == ["a", "b"]
    assert inv["a"].tolist() == pytest.approx([2.5, 7.5, 7.5, np.nan], nan_ok=True)
    assert inv["b"].tolist() == ["x", "y", "z", "w"]


def test_multiple_columns_are_independent() -> None:
    df = _frame(a=np.linspace(0.0, 9.0, 10), b=np.linspace(0.0, 3.0, 10), c=[1] * 10)
    enc = QuantileBinning(columns=["a", "b"], n_bins=2, strategy="uniform")
    out = enc.fit_transform(df)
    assert enc.n_bins_["a"] == 2
    assert enc.n_bins_["b"] == 2
    assert out["c"].tolist() == [1] * 10
    assert set(out["a"].tolist()) == {0.0, 1.0}
    assert set(out["b"].tolist()) == {0.0, 1.0}


def test_train_edges_are_reused_on_transform() -> None:
    train = _frame(a=np.arange(10, dtype=float))
    test = _frame(a=[0.0, 9.0, 100.0])
    enc = QuantileBinning(columns=["a"], n_bins=5, strategy="quantile").fit(train)
    edges = enc.bin_edges_["a"].copy()
    out = enc.transform(test)
    assert np.array_equal(enc.bin_edges_["a"], edges)
    assert out["a"].iloc[0] == pytest.approx(0.0)
    assert out["a"].iloc[-1] == pytest.approx(enc.n_bins_["a"] - 1)


def test_transform_and_inverse_without_fit_raise() -> None:
    enc = QuantileBinning(columns=["a"])
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(_frame(a=[1.0]))
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.inverse_transform(_frame(a=[0.0]))


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        QuantileBinning(columns=["a"], n_bins=1)
    with pytest.raises(ValueError, match="strategy"):
        QuantileBinning(columns=["a"], strategy="kmeans")
    with pytest.raises(ValueError, match="encode"):
        QuantileBinning(columns=["a"], encode="binary")


def test_missing_column_and_non_numeric_raise() -> None:
    enc = QuantileBinning(columns=["a"])
    with pytest.raises(KeyError, match="a"):
        enc.fit(_frame(b=[1.0, 2.0]))
    with pytest.raises(ValueError, match="non-numeric"):
        enc.fit(_frame(a=["x", "y"]))
    fitted = QuantileBinning(columns=["a"], n_bins=2).fit(_frame(a=[0.0, 1.0]))
    with pytest.raises(KeyError, match="a"):
        fitted.transform(_frame(b=[0.0, 1.0]))


def test_pipeline_step_leaves_other_columns_for_scaling() -> None:
    df = _frame(a=np.arange(6, dtype=float), b=np.arange(6, dtype=float) * 10)
    pipe = FeatureEngineeringPipeline(
        [
            ("bins", QuantileBinning(columns=["a"], n_bins=3, strategy="quantile")),
            ("scale", StandardScaler(columns=["b"])),
        ]
    )
    out = pipe.fit_transform(df)
    assert set(out["a"].tolist()) <= {0.0, 1.0, 2.0}
    assert out["b"].mean() == pytest.approx(0.0, abs=1e-9)
    assert pipe.get_feature_names_out() == ["a", "b"]


def test_preprocessing_pipeline_quantile_bin_skips_scaler() -> None:
    df = _frame(
        age=[20.0, 30.0, 40.0, 50.0, 60.0],
        income=[1.0, 2.0, 3.0, 4.0, 5.0],
        plan=["a", "b", "a", "b", "a"],
    )
    pipe = build_preprocessing_pipeline(
        df,
        target_encode=[],
        one_hot=["plan"],
        poly_columns=None,
        interaction_pairs=None,
        drop_high_correlation=None,
        variance_threshold=None,
        quantile_bin=["age"],
        n_bins=3,
        bin_strategy="uniform",
    )
    steps = dict(pipe.steps)
    assert isinstance(steps["quantile_bin"], QuantileBinning)
    assert steps["quantile_bin"].n_bins == 3
    assert "age" not in steps["scale"].columns
    assert "income" in steps["scale"].columns
    out = pipe.fit_transform(df)
    assert set(out["age"].dropna().tolist()) <= {0.0, 1.0, 2.0}
    assert "age" in out.columns
