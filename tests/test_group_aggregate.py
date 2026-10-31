"""Tests for GroupAggregateFeatures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import FeatureEngineeringPipeline, GroupAggregateFeatures


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "region": ["a", "a", "a", "b", "b", None],
            "tier": ["x", "y", "x", "x", "x", "y"],
            "spend": [10.0, 20.0, 30.0, 5.0, 7.0, 100.0],
        }
    )


def test_basic_aggregates_match_pandas(df):
    t = GroupAggregateFeatures("region", ["spend"], aggs=["mean", "std", "min", "max", "count"])
    out = t.fit_transform(df)
    assert out["spend_mean_by_region"].tolist() == [20.0, 20.0, 20.0, 6.0, 6.0, 100.0]
    assert out["spend_min_by_region"].tolist() == [10.0, 10.0, 10.0, 5.0, 5.0, 100.0]
    assert out["spend_count_by_region"].tolist() == [3, 3, 3, 2, 2, 1]
    np.testing.assert_allclose(out["spend_std_by_region"][:3], 10.0)
    assert out["spend_std_by_region"].iloc[5] == 0.0  # singleton (missing-key) group
    assert t.n_groups_ == 3
    assert list(out.columns[:3]) == ["region", "tier", "spend"]


def test_unseen_group_falls_back_to_global(df):
    t = GroupAggregateFeatures("region", ["spend"], aggs=["mean", "nunique"]).fit(df)
    out = t.transform(pd.DataFrame({"region": ["zzz", "a"], "tier": ["x", "x"], "spend": [1.0, 2.0]}))
    assert out["spend_mean_by_region"].tolist() == [pytest.approx(df.spend.mean()), 20.0]
    assert out["spend_nunique_by_region"].tolist() == [6.0, 3.0]


def test_relative_features(df):
    t = GroupAggregateFeatures("region", ["spend"], aggs=[], relative=["diff", "ratio", "zscore"])
    out = t.fit_transform(df)
    assert out["spend_diff_mean_by_region"].tolist() == [-10.0, 0.0, 10.0, -1.0, 1.0, 0.0]
    np.testing.assert_allclose(out["spend_ratio_mean_by_region"][:3], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(out["spend_zscore_by_region"][:3], [-1.0, 0.0, 1.0])
    assert out["spend_zscore_by_region"].iloc[5] == 0.0
    assert "spend_mean_by_region" not in out.columns


def test_multi_key_and_min_group_size(df):
    t = GroupAggregateFeatures(["region", "tier"], ["spend"], min_group_size=2, drop_original=True)
    out = t.fit_transform(df)
    assert "spend" not in out.columns
    col = "spend_mean_by_region_tier"
    # (a, x) has 2 rows -> own mean; (a, y) singleton -> global mean
    assert out[col].iloc[0] == 20.0 and out[col].iloc[2] == 20.0
    assert out[col].iloc[1] == pytest.approx(df.spend.mean())
    assert out[col].iloc[3] == 6.0


def test_works_in_pipeline_and_validates(df):
    pipe = FeatureEngineeringPipeline([("agg", GroupAggregateFeatures("region", ["spend"]))])
    out = pipe.fit_transform(df)
    assert "spend_mean_by_region" in out.columns
    with pytest.raises(ValueError):
        GroupAggregateFeatures("region", ["spend"], aggs=["mode"])
    with pytest.raises(ValueError):
        GroupAggregateFeatures("region", ["spend"], relative=["log"])
    with pytest.raises(ValueError):
        GroupAggregateFeatures("spend", ["spend"])
    with pytest.raises(ValueError):
        GroupAggregateFeatures("region", ["spend"], min_group_size=0)
    with pytest.raises(KeyError):
        GroupAggregateFeatures("nope", ["spend"]).fit(df)
    with pytest.raises(RuntimeError):
        GroupAggregateFeatures("region", ["spend"]).transform(df)
