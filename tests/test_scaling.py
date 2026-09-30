"""Tests for numeric scalers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler


def _df(a=(-2.0, 0.0, 2.0, 100.0)):
    return pd.DataFrame({"a": list(a)})


def test_standard_scaler_zero_mean_unit_std() -> None:
    df = _df(a=(-2.0, 0.0, 2.0, 4.0))
    out = StandardScaler(columns=["a"]).fit_transform(df)
    assert out["a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_standard_scaler_statistics_stored_for_test_frame() -> None:
    train = _df(a=(-2.0, 0.0, 2.0, 4.0))
    test = _df(a=(6.0, 8.0))
    scaler = StandardScaler(columns=["a"]).fit(train)
    out = scaler.transform(test)
    # Uses train mean (1.0) and train std (2.2360679...)
    expected = (np.array([6.0, 8.0]) - 1.0) / np.std([-2.0, 0.0, 2.0, 4.0], ddof=0)
    assert out["a"].tolist() == pytest.approx(expected.tolist(), abs=1e-9)


def test_minmax_scaler_into_unit_range() -> None:
    df = _df(a=(-2.0, 0.0, 2.0, 4.0))
    out = MinMaxScaler(columns=["a"]).fit_transform(df)
    assert out["a"].min() == pytest.approx(0.0, abs=1e-9)
    assert out["a"].max() == pytest.approx(1.0, abs=1e-9)


def test_robust_scaler_zero_median() -> None:
    df = pd.DataFrame({"a": [-100.0, -1.0, 0.0, 1.0, 100.0]})
    out = RobustScaler(columns=["a"]).fit_transform(df)
    assert out["a"].median() == pytest.approx(0.0, abs=1e-9)


def test_scaler_constant_column_is_safe() -> None:
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    out = StandardScaler(columns=["a"]).fit_transform(df)
    # std == 0 -> scale set to 1, no NaN
    assert not out["a"].isna().any()
    assert out["a"].tolist() == [0.0, 0.0, 0.0]


def test_transform_without_fit_raises() -> None:
    scaler = StandardScaler(columns=["a"])
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.transform(_df())



def test_maxabs_scaler_into_unit_abs_range() -> None:
    df = _df(a=(-2.0, 0.0, 2.0, 4.0))
    out = MaxAbsScaler(columns=["a"]).fit_transform(df)
    assert out["a"].min() == pytest.approx(-0.5, abs=1e-9)
    assert out["a"].max() == pytest.approx(1.0, abs=1e-9)
    assert abs(out["a"]).max() == pytest.approx(1.0, abs=1e-9)


def test_maxabs_preserves_zero_and_sign() -> None:
    df = pd.DataFrame({"a": [-4.0, 0.0, 2.0]})
    out = MaxAbsScaler(columns=["a"]).fit_transform(df)
    assert out["a"].tolist() == pytest.approx([-1.0, 0.0, 0.5], abs=1e-9)


def test_maxabs_statistics_reuse_on_test_frame() -> None:
    train = _df(a=(-2.0, 0.0, 2.0, 4.0))
    test = _df(a=(8.0, -4.0))
    scaler = MaxAbsScaler(columns=["a"]).fit(train)
    out = scaler.transform(test)
    # train max abs = 4
    assert scaler.scale_["a"] == pytest.approx(4.0)
    assert out["a"].tolist() == pytest.approx([2.0, -1.0], abs=1e-9)


def test_maxabs_constant_zero_column_is_safe() -> None:
    df = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    out = MaxAbsScaler(columns=["a"]).fit_transform(df)
    assert not out["a"].isna().any()
    assert out["a"].tolist() == [0.0, 0.0, 0.0]


def test_maxabs_transform_without_fit_raises() -> None:
    scaler = MaxAbsScaler(columns=["a"])
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.transform(_df())


def test_maxabs_export_available() -> None:
    from feature_engineering_kit import MaxAbsScaler as exported

    assert exported is MaxAbsScaler
