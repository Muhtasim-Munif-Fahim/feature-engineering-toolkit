"""Tests for missing-value imputation transformers."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering_kit import CategoricalImputer, NumericImputer


def _df(**cols):
    return pd.DataFrame(cols)


def test_numeric_imputer_median() -> None:
    df = _df(a=[1.0, 2.0, None, 4.0, 100.0])
    out = NumericImputer(columns=["a"], strategy="median").fit_transform(df)
    assert out["a"].isna().sum() == 0
    assert out["a"].iloc[2] == pytest.approx(3.0)


def test_numeric_imputer_mean() -> None:
    df = _df(a=[1.0, 2.0, None, 4.0])
    out = NumericImputer(columns=["a"], strategy="mean").fit_transform(df)
    assert out["a"].isna().sum() == 0
    assert out["a"].iloc[2] == pytest.approx(2.333333, abs=1e-6)


def test_numeric_imputer_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown numeric strategy"):
        NumericImputer(columns=["a"], strategy="root").fit(_df(a=[1.0]))


def test_numeric_imputer_leaves_other_columns_untouched() -> None:
    df = _df(a=[1.0, None, 3.0], b=[0.1, 0.2, 0.3])
    out = NumericImputer(columns=["a"]).fit_transform(df)
    assert out["b"].tolist() == [0.1, 0.2, 0.3]


def test_categorical_imputer_most_frequent() -> None:
    df = _df(c=["red", "red", "red", None, "blue"])
    out = CategoricalImputer(columns=["c"]).fit_transform(df)
    assert out["c"].isna().sum() == 0
    assert out["c"].iloc[3] == "red"


def test_categorical_imputer_constant() -> None:
    df = _df(c=["red", None, "blue"])
    out = CategoricalImputer(columns=["c"], strategy="constant", fill_value="unknown").fit_transform(df)
    assert out["c"].iloc[1] == "unknown"


def test_categorical_imputer_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown categorical strategy"):
        CategoricalImputer(columns=["c"], strategy="bogus").fit(_df(c=["a"]))
