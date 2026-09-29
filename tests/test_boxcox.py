"""Tests for BoxCoxScaler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import BoxCoxScaler
from feature_engineering_kit.scaling import _box_cox_transform


def test_identity_lambda_one() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = BoxCoxScaler(columns=["a"], standardize=False, lmbda=1.0).fit_transform(df)
    # λ=1 → x - 1
    assert out["a"].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0], abs=1e-9)


def test_lambda_zero_is_log() -> None:
    df = pd.DataFrame({"a": [1.0, np.e, np.e**2]})
    out = BoxCoxScaler(columns=["a"], standardize=False, lmbda=0.0).fit_transform(df)
    assert out["a"].tolist() == pytest.approx([0.0, 1.0, 2.0], abs=1e-9)


def test_standardize_true() -> None:
    df = pd.DataFrame({"a": [0.5, 1.0, 2.0, 4.0, 8.0]})
    out = BoxCoxScaler(columns=["a"], standardize=True).fit_transform(df)
    assert out["a"].isna().sum() == 0
    assert out["a"].mean() == pytest.approx(0.0, abs=1e-8)
    assert out["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-8)


def test_mle_reduces_skew_on_lognormal() -> None:
    rng = np.random.default_rng(0)
    raw = rng.lognormal(mean=0.0, sigma=1.0, size=400)
    df = pd.DataFrame({"a": raw})
    scaler = BoxCoxScaler(columns=["a"], standardize=False).fit(df)
    # For right-skewed positive data, λ typically lands near 0 (log).
    assert scaler.lambdas_["a"] < 1.0
    transformed = scaler.transform(df)["a"].to_numpy()

    def _skew(x):
        x = x - x.mean()
        return float(np.mean(x**3) / (np.std(x) ** 3 + 1e-15))

    assert abs(_skew(transformed)) < abs(_skew(raw))


def test_statistics_reuse_on_test_frame() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 9.0]})
    test = pd.DataFrame({"a": [1.5, 2.5, 8.0]})
    scaler = BoxCoxScaler(columns=["a"], standardize=True, lmbda=0.5).fit(train)
    out = scaler.transform(test)
    expected = _box_cox_transform(test["a"].to_numpy(), 0.5)
    expected = (expected - scaler.mean_["a"]) / scaler.scale_["a"]
    assert out["a"].tolist() == pytest.approx(expected.tolist(), abs=1e-9)


def test_fit_rejects_non_positive() -> None:
    df = pd.DataFrame({"a": [1.0, 0.0, 2.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        BoxCoxScaler(columns=["a"]).fit(df)
    df_neg = pd.DataFrame({"a": [1.0, -0.5, 2.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        BoxCoxScaler(columns=["a"]).fit(df_neg)


def test_transform_non_positive_becomes_nan() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    scaler = BoxCoxScaler(columns=["a"], standardize=False, lmbda=1.0).fit(train)
    test = pd.DataFrame({"a": [1.0, 0.0, -2.0, 5.0]})
    out = scaler.transform(test)
    assert out["a"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(out["a"].iloc[1])
    assert np.isnan(out["a"].iloc[2])
    assert out["a"].iloc[3] == pytest.approx(4.0, abs=1e-9)


def test_constant_column_is_safe() -> None:
    df = pd.DataFrame({"a": [3.0, 3.0, 3.0]})
    out = BoxCoxScaler(columns=["a"]).fit_transform(df)
    assert not out["a"].isna().any()


def test_transform_without_fit_raises() -> None:
    scaler = BoxCoxScaler(columns=["a"])
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.transform(pd.DataFrame({"a": [1.0, 2.0]}))


def test_export() -> None:
    from feature_engineering_kit import BoxCoxScaler as exported

    assert exported is BoxCoxScaler
