"""Tests for YeoJohnsonScaler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import YeoJohnsonScaler
from feature_engineering_kit.scaling import _yeo_johnson_transform


def test_identity_lambda_one() -> None:
    df = pd.DataFrame({"a": [-2.0, -1.0, 0.0, 1.0, 2.0]})
    out = YeoJohnsonScaler(columns=["a"], standardize=False, lmbda=1.0).fit_transform(df)
    assert out["a"].tolist() == pytest.approx(df["a"].tolist(), abs=1e-9)


def test_handles_negative_and_zero() -> None:
    df = pd.DataFrame({"a": [-5.0, -0.5, 0.0, 0.5, 5.0]})
    out = YeoJohnsonScaler(columns=["a"], standardize=True).fit_transform(df)
    assert out["a"].isna().sum() == 0
    assert out["a"].mean() == pytest.approx(0.0, abs=1e-8)
    assert out["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-8)


def test_standardize_false_keeps_scale() -> None:
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0]})
    out = YeoJohnsonScaler(columns=["a"], standardize=False, lmbda=1.0).fit_transform(df)
    assert out["a"].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0], abs=1e-9)


def test_mle_reduces_skew_on_lognormal() -> None:
    rng = np.random.default_rng(0)
    raw = rng.lognormal(mean=0.0, sigma=1.0, size=400)
    df = pd.DataFrame({"a": raw})
    scaler = YeoJohnsonScaler(columns=["a"], standardize=False).fit(df)
    # For right-skewed positive data, λ typically lands below 1.
    assert scaler.lambdas_["a"] < 1.0
    transformed = scaler.transform(df)["a"].to_numpy()
    # Skewness should drop vs the raw lognormal draws.
    def _skew(x):
        x = x - x.mean()
        return float(np.mean(x**3) / (np.std(x) ** 3 + 1e-15))

    assert abs(_skew(transformed)) < abs(_skew(raw))


def test_statistics_reuse_on_test_frame() -> None:
    train = pd.DataFrame({"a": [-2.0, -1.0, 0.0, 1.0, 4.0, 9.0]})
    test = pd.DataFrame({"a": [-3.0, 2.0, 8.0]})
    scaler = YeoJohnsonScaler(columns=["a"], standardize=True, lmbda=0.5).fit(train)
    out = scaler.transform(test)
    expected = _yeo_johnson_transform(test["a"].to_numpy(), 0.5)
    expected = (expected - scaler.mean_["a"]) / scaler.scale_["a"]
    assert out["a"].tolist() == pytest.approx(expected.tolist(), abs=1e-9)


def test_constant_column_is_safe() -> None:
    df = pd.DataFrame({"a": [3.0, 3.0, 3.0]})
    out = YeoJohnsonScaler(columns=["a"]).fit_transform(df)
    assert not out["a"].isna().any()


def test_transform_without_fit_raises() -> None:
    scaler = YeoJohnsonScaler(columns=["a"])
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.transform(pd.DataFrame({"a": [1.0, 2.0]}))


def test_export() -> None:
    from feature_engineering_kit import YeoJohnsonScaler as exported

    assert exported is YeoJohnsonScaler
