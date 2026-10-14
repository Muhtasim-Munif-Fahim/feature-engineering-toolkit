"""Tests for CyclicalEncoder."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import CyclicalEncoder


def test_hour_wraps_around():
    df = pd.DataFrame({"hour": [0.0, 6.0, 12.0, 18.0, 23.0]})
    out = CyclicalEncoder(columns=["hour"], period=24).fit_transform(df)
    assert "hour__sin" in out.columns and "hour__cos" in out.columns
    assert "hour" not in out.columns
    # hour 0 ≈ hour 23 on the circle
    d_near = math.hypot(
        out["hour__sin"].iloc[0] - out["hour__sin"].iloc[-1],
        out["hour__cos"].iloc[0] - out["hour__cos"].iloc[-1],
    )
    d_far = math.hypot(
        out["hour__sin"].iloc[0] - out["hour__sin"].iloc[2],
        out["hour__cos"].iloc[0] - out["hour__cos"].iloc[2],
    )
    assert d_near < d_far
    assert out["hour__sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["hour__cos"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert out["hour__sin"].iloc[1] == pytest.approx(1.0, abs=1e-12)
    assert out["hour__cos"].iloc[1] == pytest.approx(0.0, abs=1e-12)


def test_per_column_periods_and_keep_original():
    df = pd.DataFrame({"hour": [0.0, 12.0], "month": [1.0, 7.0]})
    enc = CyclicalEncoder(
        columns=["hour", "month"],
        period={"hour": 24, "month": 12},
        drop_original=False,
    )
    out = enc.fit_transform(df)
    assert list(out.columns) == [
        "hour",
        "month",
        "hour__sin",
        "hour__cos",
        "month__sin",
        "month__cos",
    ]
    assert out["month__sin"].iloc[0] == pytest.approx(math.sin(2 * math.pi * 1 / 12))


def test_unit_circle_norm():
    df = pd.DataFrame({"x": np.linspace(0, 11, 12)})
    out = CyclicalEncoder(columns=["x"], period=12).fit_transform(df)
    norms = np.sqrt(out["x__sin"] ** 2 + out["x__cos"] ** 2)
    assert norms.to_numpy() == pytest.approx(np.ones(12), abs=1e-12)


def test_invalid_period_and_columns():
    with pytest.raises(ValueError, match="positive"):
        CyclicalEncoder(columns=["h"], period=0)
    with pytest.raises(ValueError, match="non-empty"):
        CyclicalEncoder(columns=[], period=24)
    with pytest.raises(ValueError, match="missing columns"):
        CyclicalEncoder(columns=["h", "m"], period={"h": 24})


def test_transform_without_fit_raises():
    enc = CyclicalEncoder(columns=["h"], period=24)
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(pd.DataFrame({"h": [1.0]}))


def test_missing_column_on_fit():
    with pytest.raises(KeyError, match="not found"):
        CyclicalEncoder(columns=["missing"], period=7).fit(pd.DataFrame({"h": [1.0]}))
