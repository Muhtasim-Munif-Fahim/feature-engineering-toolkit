"""Tests for Winsorizer transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import Winsorizer


def test_winsorizer_clips_extremes() -> None:
    # 20 points: 0..19; 5th≈0.95, 95th≈18.05 so 0 and 19 clip inward
    df = pd.DataFrame({"a": np.arange(20, dtype=float)})
    out = Winsorizer(columns=["a"], limits=(0.05, 0.95)).fit_transform(df)
    lo = float(df["a"].quantile(0.05))
    hi = float(df["a"].quantile(0.95))
    assert out["a"].min() == pytest.approx(lo, abs=1e-9)
    assert out["a"].max() == pytest.approx(hi, abs=1e-9)
    assert out["a"].iloc[0] == pytest.approx(lo, abs=1e-9)
    assert out["a"].iloc[-1] == pytest.approx(hi, abs=1e-9)


def test_winsorizer_idempotent_on_already_clipped() -> None:
    df = pd.DataFrame({"a": np.linspace(-10, 10, 50)})
    scaler = Winsorizer(columns=["a"], limits=(0.1, 0.9)).fit(df)
    once = scaler.transform(df)
    twice = scaler.transform(once)
    assert once["a"].to_numpy() == pytest.approx(twice["a"].to_numpy(), abs=1e-12)


def test_winsorizer_reuses_fit_thresholds_on_test() -> None:
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
    test = pd.DataFrame({"a": [-100.0, 4.5, 100.0]})
    scaler = Winsorizer(columns=["a"], limits=(0.1, 0.9)).fit(train)
    out = scaler.transform(test)
    assert out["a"].iloc[0] == pytest.approx(scaler.lower_["a"], abs=1e-9)
    assert out["a"].iloc[2] == pytest.approx(scaler.upper_["a"], abs=1e-9)
    assert out["a"].iloc[1] == pytest.approx(4.5, abs=1e-9)


def test_winsorizer_invalid_limits() -> None:
    with pytest.raises(ValueError, match="limits"):
        Winsorizer(columns=["a"], limits=(0.9, 0.1))
    with pytest.raises(ValueError, match="limits"):
        Winsorizer(columns=["a"], limits=(-0.1, 0.9))
    with pytest.raises(ValueError, match="limits"):
        Winsorizer(columns=["a"], limits=(0.1, 1.1))
    with pytest.raises(ValueError, match="limits"):
        Winsorizer(columns=["a"], limits=(0.5, 0.5))


def test_winsorizer_transform_without_fit_raises() -> None:
    scaler = Winsorizer(columns=["a"])
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.transform(pd.DataFrame({"a": [1.0, 2.0]}))


def test_winsorizer_leaves_other_columns() -> None:
    df = pd.DataFrame({"a": np.arange(10, dtype=float), "b": np.arange(10, dtype=float) * 10})
    out = Winsorizer(columns=["a"], limits=(0.1, 0.9)).fit_transform(df)
    assert out["b"].tolist() == df["b"].tolist()


def test_winsorizer_export_available() -> None:
    from feature_engineering_kit import Winsorizer as exported

    assert exported is Winsorizer
