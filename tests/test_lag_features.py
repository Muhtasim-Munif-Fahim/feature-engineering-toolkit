"""Tests for LagFeatures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import LagFeatures


def test_lags_diffs_and_left_closed_rolling():
    df = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    out = LagFeatures(["x"], lags=[1, 2], diffs=[1], windows=[2], window_aggs=["mean", "max"]).fit_transform(df)
    np.testing.assert_array_equal(out["x_lag_1"].to_numpy(), [np.nan, 1, 2, 4, 8])
    np.testing.assert_array_equal(out["x_lag_2"].to_numpy(), [np.nan, np.nan, 1, 2, 4])
    np.testing.assert_array_equal(out["x_diff_1"].to_numpy(), [np.nan, 1, 2, 4, 8])
    # Window for row t covers t-2..t-1 only (never the current value).
    np.testing.assert_array_equal(out["x_roll2_mean"].to_numpy(), [np.nan, 1, 1.5, 3, 6])
    np.testing.assert_array_equal(out["x_roll2_max"].to_numpy(), [np.nan, 1, 2, 4, 8])


def test_grouped_and_ordered_preserves_row_order():
    df = pd.DataFrame(
        {
            "store": ["a", "b", "a", "b", "a"],
            "t": [3, 2, 1, 1, 2],
            "sales": [30.0, 200.0, 10.0, 100.0, 20.0],
        },
        index=list("pqrst"),
    )
    t = LagFeatures("sales", lags=[1], windows=[3], window_aggs=["sum"], group_by="store", order_by="t")
    out = t.fit_transform(df)
    assert list(out.index) == list("pqrst")
    # store a ordered: r(10) t(20) p(30); store b: s(100) q(200)
    assert out.loc["p", "sales_lag_1"] == 20.0
    assert np.isnan(out.loc["r", "sales_lag_1"])
    assert out.loc["q", "sales_lag_1"] == 100.0
    assert out.loc["p", "sales_roll3_sum"] == 30.0
    assert out.loc["t", "sales_roll3_sum"] == 10.0
    assert np.isnan(out.loc["s", "sales_roll3_sum"])


def test_fill_value_min_periods_and_drop_original():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    t = LagFeatures("x", lags=[1], windows=[3], window_aggs=["mean"], min_periods=2, fill_value=0.0, drop_original=True)
    out = t.fit_transform(df)
    assert "x" not in out.columns
    np.testing.assert_array_equal(out["x_lag_1"].to_numpy(), [0, 1, 2, 3])
    np.testing.assert_array_equal(out["x_roll3_mean"].to_numpy(), [0, 0, 1.5, 2])
    assert t.get_feature_names_out() == ["x_lag_1", "x_roll3_mean"]


def test_validation():
    with pytest.raises(ValueError):
        LagFeatures(["x"], lags=[0])
    with pytest.raises(ValueError):
        LagFeatures(["x"], lags=[], diffs=[], windows=[])
    with pytest.raises(ValueError):
        LagFeatures(["x"], windows=[2], window_aggs=["skew"])
    with pytest.raises(KeyError):
        LagFeatures(["y"]).fit(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(TypeError):
        LagFeatures(["x"]).fit(pd.DataFrame({"x": ["a", "b"]}))
    with pytest.raises(RuntimeError):
        LagFeatures(["x"]).transform(pd.DataFrame({"x": [1.0]}))
