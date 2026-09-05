"""Tests for datetime feature extraction."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering_kit import DatetimeExtractor


def test_extract_default_features() -> None:
    df = pd.DataFrame({"ts": [pd.Timestamp("2023-06-15 03:45:00")]})
    ext = DatetimeExtractor(column="ts", drop_original=True)
    out = ext.fit_transform(df)
    assert "ts__hour" in out.columns
    assert out["ts__hour"].iloc[0] == 3
    assert out["ts__dayofweek"].iloc[0] == 3  # Thursday
    assert out["ts__month"].iloc[0] == 6
    assert out["ts__is_weekend"].iloc[0] == 0
    assert "ts" not in out.columns
    assert len(ext.get_feature_names_out()) > len(df.columns)


def test_extract_weekend_detection() -> None:
    # 2023-06-17 is a Saturday
    df = pd.DataFrame({"ts": [pd.Timestamp("2023-06-17")]})
    out = DatetimeExtractor(column="ts", drop_original=True, features=["is_weekend"]).fit_transform(df)
    assert out["ts__is_weekend"].iloc[0] == 1


def test_extract_subset_of_features() -> None:
    df = pd.DataFrame({"ts": [pd.Timestamp("2023-06-15 03:45:00")]})
    out = DatetimeExtractor(column="ts", features=["hour", "month"]).fit_transform(df)
    assert "ts__hour" in out.columns
    assert "ts__month" in out.columns
    assert "ts__dayofweek" not in out.columns


def test_extract_unknown_feature_raises() -> None:
    with pytest.raises(ValueError, match="unknown datetime features"):
        DatetimeExtractor(column="ts", features=["decade"]).fit(
            pd.DataFrame({"ts": [pd.Timestamp("2023-01-01")]})
        )


def test_extract_missing_column_raises() -> None:
    with pytest.raises(KeyError, match="ts"):
        DatetimeExtractor(column="ts").fit(pd.DataFrame({"x": [1]}))


def test_transform_without_fit_raises() -> None:
    ext = DatetimeExtractor(column="ts")
    with pytest.raises(RuntimeError, match="not fitted"):
        ext.transform(pd.DataFrame({"ts": [pd.Timestamp("2023-01-01")]}))
