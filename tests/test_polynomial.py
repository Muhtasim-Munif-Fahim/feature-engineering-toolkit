"""Tests for polynomial feature expansion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering_kit import PolynomialFeatures


def _df():
    return pd.DataFrame({"age": [1.0, 2.0, 3.0], "income": [10.0, 20.0, 30.0]})


def test_interaction_only_no_squares() -> None:
    poly = PolynomialFeatures(columns=["age", "income"], degree=2, interaction_only=True)
    out = poly.fit_transform(_df())
    assert "age_x_income" in out.columns
    assert "age_x_age" not in out.columns
    np.testing.assert_allclose(out["age_x_income"].to_numpy(), [10, 40, 90])
    # originals retained
    assert out["age"].tolist() == [1.0, 2.0, 3.0]


def test_squares_when_not_interaction_only() -> None:
    poly = PolynomialFeatures(columns=["age"], degree=2, interaction_only=False)
    out = poly.fit_transform(_df())
    assert "age_x_age" in out.columns
    np.testing.assert_allclose(out["age_x_age"].to_numpy(), [1, 4, 9])


def test_degree_one_adds_no_columns() -> None:
    poly = PolynomialFeatures(columns=["age"], degree=1)
    out = poly.fit_transform(_df())
    assert "age_x_age" not in out.columns
    assert list(out.columns) == list(_df().columns)


def test_degree_three_generates_cubes() -> None:
    poly = PolynomialFeatures(columns=["age"], degree=3, interaction_only=False)
    out = poly.fit_transform(_df())
    assert "age_x_age_x_age" in out.columns
    np.testing.assert_allclose(out["age_x_age_x_age"].to_numpy(), [1, 8, 27])


def test_missing_column_raises() -> None:
    import pytest

    poly = PolynomialFeatures(columns=["nope"], degree=2)
    with pytest.raises(KeyError):
        poly.fit(_df())


def test_include_bias() -> None:
    poly = PolynomialFeatures(columns=["age"], degree=2, include_bias=True)
    out = poly.fit_transform(_df())
    assert "poly__bias" in out.columns
    assert (out["poly__bias"] == 1.0).all()
