"""Tests for feature-selection transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from feature_engineering_kit import (
    CorrelationFilter,
    MutualInfoSelection,
    SelectFromModel,
    VarianceThreshold,
)


def _constant_df():
    return pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0], "b": [1, 2, 3, 4]})


def test_variance_threshold_drops_constant_columns() -> None:
    out = VarianceThreshold(threshold=0.0).fit_transform(_constant_df())
    assert "a" not in out.columns
    assert "b" in out.columns


def test_variance_threshold_keeps_low_variance() -> None:
    df = pd.DataFrame({"a": [1, 1, 1, 2]})
    assert VarianceThreshold(threshold=0.0).fit_transform(df)["a"].tolist() == [1, 1, 1, 2]


def test_correlation_filter_drops_redundant_column() -> None:
    df = pd.DataFrame(
        {
            "a": list(range(1, 11)),
            "b": list(range(2, 22, 2)),  # perfectly correlated with a
            "c": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    out = CorrelationFilter(threshold=0.9).fit_transform(df)
    # a kept, b dropped (later column of the correlated pair)
    assert "a" in out.columns
    assert "b" not in out.columns
    assert "c" in out.columns


def test_correlation_filter_no_redundancy_keeps_all() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 1, 2, 3]})
    out = CorrelationFilter(threshold=0.9).fit_transform(df)
    assert set(out.columns) == {"a", "b"}


def _informative_xy(random_state=0, n=200):
    rng = np.random.default_rng(random_state)
    x1 = rng.normal(0, 1, n)
    y = (x1 > 0).astype(int)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return df, pd.Series(y)


def test_mutual_info_selection_keeps_top_k() -> None:
    X, y = _informative_xy()
    sel = MutualInfoSelection(target="y", k=1, random_state=0).fit(X, y)
    out = sel.transform(X)
    assert "x1" in out.columns
    assert "x2" not in out.columns and "x3" not in out.columns


def test_mutual_info_selection_requires_y() -> None:
    with pytest.raises(ValueError, match="requires y"):
        MutualInfoSelection(target="y", k=1).fit(pd.DataFrame({"x1": [1, 2]}))


def test_select_from_model_keeps_important_feature() -> None:
    X, y = _informative_xy()
    sfm = SelectFromModel(
        target="y", estimator=DecisionTreeClassifier(random_state=0), threshold="median"
    ).fit(X, y)
    out = sfm.transform(X)
    assert "x1" in out.columns
    assert len(out.columns) <= len(X.columns)
    assert len(out.columns) >= 1


def test_select_from_model_requires_y() -> None:
    with pytest.raises(ValueError, match="requires y"):
        SelectFromModel(target="y", estimator=DecisionTreeClassifier(random_state=0)).fit(
            pd.DataFrame({"x1": [1, 2]})
        )
