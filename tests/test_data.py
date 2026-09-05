"""Tests for dataset generation and column-schema inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering_kit import (
    COLUMN_CATEGORICAL,
    COLUMN_DATETIME,
    COLUMN_NUMERIC,
    ColumnSchema,
    TARGET_COLUMN,
    column_schema,
    load_synthetic_churn_dataset,
    stratified_split,
)


def test_dataset_shape_and_columns() -> None:
    df = load_synthetic_churn_dataset(n_samples=500, seed=1)
    assert len(df) == 500
    expected = COLUMN_NUMERIC + COLUMN_CATEGORICAL + COLUMN_DATETIME + [TARGET_COLUMN]
    for col in expected:
        assert col in df.columns


def test_dataset_reproducible_with_seed() -> None:
    a = load_synthetic_churn_dataset(n_samples=300, seed=7)
    b = load_synthetic_churn_dataset(n_samples=300, seed=7)
    pd.testing.assert_series_equal(a[TARGET_COLUMN], b[TARGET_COLUMN])
    np.testing.assert_array_equal(a["age"].to_numpy(), b["age"].to_numpy())


def test_different_seed_yields_different_data() -> None:
    a = load_synthetic_churn_dataset(n_samples=300, seed=1)
    b = load_synthetic_churn_dataset(n_samples=300, seed=2)
    assert not np.array_equal(a[TARGET_COLUMN].to_numpy(), b[TARGET_COLUMN].to_numpy())


def test_target_is_binary() -> None:
    df = load_synthetic_churn_dataset(n_samples=200, seed=3)
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_missingness_present_in_numeric_and_categorical() -> None:
    df = load_synthetic_churn_dataset(n_samples=2000, seed=42)
    assert df["income"].isna().any()
    assert df["tenure"].isna().any()
    assert df["plan"].isna().any()


def test_signup_time_is_datetime() -> None:
    df = load_synthetic_churn_dataset(n_samples=100, seed=0)
    assert pd.api.types.is_datetime64_any_dtype(df["signup_time"])


def test_column_schema_grouping() -> None:
    df = load_synthetic_churn_dataset(n_samples=100, seed=0)
    schema = column_schema(df)
    assert schema.target == TARGET_COLUMN
    assert set(COLUMN_NUMERIC).issubset(set(schema.numeric))
    assert set(COLUMN_CATEGORICAL).issubset(set(schema.categorical))
    assert set(COLUMN_DATETIME).issubset(set(schema.datetime))


def test_column_schema_dataclass_contract() -> None:
    schema = ColumnSchema(numeric=["a"], categorical=["b"], datetime=["c"], target="y")
    assert schema.all_features() == ["a", "b", "c"]


def test_stratified_split_preserves_columns_and_target() -> None:
    df = load_synthetic_churn_dataset(n_samples=500, seed=5)
    train, test = stratified_split(df, test_size=0.2, seed=5)
    assert len(train) + len(test) == 500
    assert TARGET_COLUMN in train.columns and TARGET_COLUMN in test.columns
    rate_train = float(train[TARGET_COLUMN].mean())
    rate_test = float(test[TARGET_COLUMN].mean())
    assert abs(rate_train - rate_test) < 0.1


def test_invalid_n_samples_raises() -> None:
    try:
        load_synthetic_churn_dataset(n_samples=0, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n_samples <= 0")
