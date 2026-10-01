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


def test_quantile_transformer_uniform_range() -> None:
    from feature_engineering_kit import QuantileTransformer

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(5, 2, size=200)})
    out = QuantileTransformer(columns=["a"], n_quantiles=50, output_distribution="uniform").fit_transform(df)
    assert out["a"].min() >= 0.0 - 1e-9
    assert out["a"].max() <= 1.0 + 1e-9
    # Roughly uniform: median near 0.5
    assert out["a"].median() == pytest.approx(0.5, abs=0.08)


def test_quantile_transformer_normal_approx_standard() -> None:
    from feature_engineering_kit import QuantileTransformer

    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.exponential(2.0, size=400)})
    out = QuantileTransformer(
        columns=["a"], n_quantiles=100, output_distribution="normal"
    ).fit_transform(df)
    assert abs(out["a"].mean()) < 0.25
    assert 0.7 < out["a"].std(ddof=0) < 1.3


def test_quantile_transformer_reuses_fit_references() -> None:
    from feature_engineering_kit import QuantileTransformer

    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"a": [-1.0, 2.0, 5.0]})
    scaler = QuantileTransformer(columns=["a"], n_quantiles=5, output_distribution="uniform").fit(train)
    out = scaler.transform(test)
    # -1 clips to lowest quantile (0); 5 clips to 1; 2 is interior
    assert out["a"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["a"].iloc[2] == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < out["a"].iloc[1] < 1.0


def test_quantile_transformer_inverse_roundtrip() -> None:
    from feature_engineering_kit import QuantileTransformer

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0, 1, size=80), "b": rng.uniform(-3, 3, size=80)})
    scaler = QuantileTransformer(
        columns=["a", "b"], n_quantiles=40, output_distribution="uniform"
    ).fit(df)
    transformed = scaler.transform(df)
    recovered = scaler.inverse_transform(transformed)
    assert recovered["a"].to_numpy() == pytest.approx(df["a"].to_numpy(), abs=0.15)
    assert recovered["b"].to_numpy() == pytest.approx(df["b"].to_numpy(), abs=0.15)


def test_quantile_transformer_inverse_normal_roundtrip() -> None:
    from feature_engineering_kit import QuantileTransformer

    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(10, 3, size=100)})
    scaler = QuantileTransformer(
        columns=["a"], n_quantiles=50, output_distribution="normal"
    ).fit(df)
    transformed = scaler.transform(df)
    recovered = scaler.inverse_transform(transformed)
    assert recovered["a"].to_numpy() == pytest.approx(df["a"].to_numpy(), abs=0.35)


def test_quantile_transformer_constant_column_safe() -> None:
    from feature_engineering_kit import QuantileTransformer

    df = pd.DataFrame({"a": [3.0, 3.0, 3.0, 3.0]})
    out = QuantileTransformer(columns=["a"], n_quantiles=4).fit_transform(df)
    assert not out["a"].isna().any()


def test_quantile_transformer_without_fit_raises() -> None:
    from feature_engineering_kit import QuantileTransformer

    scaler = QuantileTransformer(columns=["a"])
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.transform(_df())


def test_quantile_transformer_export_available() -> None:
    from feature_engineering_kit import QuantileTransformer as exported

    assert exported.__name__ == "QuantileTransformer"
