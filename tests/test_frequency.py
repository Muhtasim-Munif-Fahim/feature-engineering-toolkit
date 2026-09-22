"""Tests for rare-category grouping and frequency encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import (
    FeatureEngineeringPipeline,
    FrequencyEncoder,
    RareCategoryGrouper,
    TargetEncoder,
)


def _df(**cols):
    return pd.DataFrame(cols)


def test_grouper_default_min_count_collapses_singletons() -> None:
    enc = RareCategoryGrouper(columns=["x"])
    assert enc.min_count == 2
    out = enc.fit_transform(_df(x=["a", "b", "b"]))
    assert out["x"].tolist() == ["other", "b", "b"]
    assert enc.counts_["x"] == {"a": 1, "b": 2}
    assert enc.kept_["x"] == {"b"}


def test_grouper_keeps_levels_at_min_count() -> None:
    enc = RareCategoryGrouper(columns=["x"], min_count=3)
    out = enc.fit_transform(_df(x=["a", "a", "a", "b", "b", "c"]))
    assert out["x"].tolist() == ["a", "a", "a", "other", "other", "other"]
    assert enc.kept_["x"] == {"a"}


def test_grouper_unseen_maps_to_other_and_missing_is_preserved() -> None:
    enc = RareCategoryGrouper(columns=["x"], min_count=1)
    enc.fit(_df(x=["a", "b"]))
    out = enc.transform(_df(x=["a", "c", None]))
    assert out["x"].iloc[0] == "a"
    assert out["x"].iloc[1] == "other"
    assert pd.isna(out["x"].iloc[2])


def test_grouper_fit_counts_are_frozen_at_transform() -> None:
    enc = RareCategoryGrouper(columns=["x"], min_count=2)
    enc.fit(_df(x=["a", "a", "b"]))
    out = enc.transform(_df(x=["a", "b", "b", "b"]))
    # "a" was frequent in training; "b" was rare. Test frequencies do not matter.
    assert out["x"].tolist() == ["a", "other", "other", "other"]


def test_grouper_custom_label_and_existing_bucket() -> None:
    custom = RareCategoryGrouper(columns=["x"], min_count=2, other_label="RARE")
    assert custom.fit_transform(_df(x=["a", "b", "b"]))["x"].tolist() == ["RARE", "b", "b"]

    merged = RareCategoryGrouper(columns=["x"], min_count=2, other_label="other")
    out = merged.fit_transform(_df(x=["other", "other", "a", "b", "b"]))
    assert out["x"].tolist() == ["other", "other", "other", "b", "b"]
    assert merged.kept_["x"] == {"other", "b"}


def test_grouper_multiple_columns_and_passthrough() -> None:
    df = _df(x=["a", "a", "b"], z=["q", "r", "r"], age=[1, 2, 3])
    original = df.copy()
    out = RareCategoryGrouper(columns=["x", "z"], min_count=2).fit_transform(df)
    assert out["x"].tolist() == ["a", "a", "other"]
    assert out["z"].tolist() == ["other", "r", "r"]
    assert out["age"].tolist() == [1, 2, 3]
    pd.testing.assert_frame_equal(df, original)


def test_grouper_integer_levels_and_index() -> None:
    df = _df(x=[1, 1, 1, 2, 2, 3])
    df.index = [5, 4, 3, 2, 1, 0]
    out = RareCategoryGrouper(columns=["x"], min_count=2).fit_transform(df)
    assert out.index.tolist() == [5, 4, 3, 2, 1, 0]
    assert out["x"].tolist() == [1, 1, 1, 2, 2, "other"]


def test_grouper_min_count_above_n_maps_everything_observed() -> None:
    out = RareCategoryGrouper(columns=["x"], min_count=10).fit_transform(
        _df(x=["a", "a", "b"])
    )
    assert out["x"].tolist() == ["other", "other", "other"]


def test_grouper_rejects_bad_min_count_and_label() -> None:
    with pytest.raises(ValueError, match="min_count must be an integer >= 1"):
        RareCategoryGrouper(columns=["x"], min_count=0)
    with pytest.raises(ValueError, match="min_count must be an integer >= 1"):
        RareCategoryGrouper(columns=["x"], min_count=-1)
    with pytest.raises(ValueError, match="min_count must be an integer >= 1"):
        RareCategoryGrouper(columns=["x"], min_count=1.5)
    with pytest.raises(ValueError, match="min_count must be an integer >= 1"):
        RareCategoryGrouper(columns=["x"], min_count=True)
    with pytest.raises(ValueError, match="other_label must be a non-missing"):
        RareCategoryGrouper(columns=["x"], other_label=None)
    # Integral floats are accepted.
    assert RareCategoryGrouper(columns=["x"], min_count=2.0).min_count == 2


def test_grouper_missing_column_and_unfitted() -> None:
    enc = RareCategoryGrouper(columns=["x"])
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(_df(x=["a"]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.fit(_df(z=["a"]))
    enc.fit(_df(x=["a", "a"]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.transform(_df(z=["a"]))


def test_grouped_rare_levels_share_one_target_mean() -> None:
    X = _df(city=["a", "a", "a", "a", "b", "c"])
    y = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    grouped = RareCategoryGrouper(columns=["city"], min_count=2).fit_transform(X)
    enc = TargetEncoder(columns=["city"], target="y", smoothing=0.0, cv=None)
    out = enc.fit_transform(grouped, y)["city"].tolist()
    # "b" and "c" share the pooled mean (1 + 0) / 2.
    assert out == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])


def test_frequency_relative_and_raw_counts() -> None:
    X = _df(x=["a", "a", "b"], age=[1, 2, 3])
    original = X.copy()
    relative = FrequencyEncoder(columns=["x"])
    assert relative.cv is None
    assert relative.normalize is True
    assert relative.min_count is None
    out = relative.fit_transform(X)
    assert out["x"].tolist() == pytest.approx([2 / 3, 2 / 3, 1 / 3])
    assert out["age"].tolist() == [1, 2, 3]
    assert relative.n_samples_ == 3
    assert relative.counts_["x"] == {"a": 2, "b": 1}
    assert sum(relative.maps_["x"].values()) == pytest.approx(1.0)
    pd.testing.assert_frame_equal(X, original)

    raw = FrequencyEncoder(columns=["x"], normalize=False)
    assert raw.fit_transform(X)["x"].tolist() == pytest.approx([2.0, 2.0, 1.0])


def test_frequency_unseen_is_zero_and_missing_uses_missing_rate() -> None:
    enc = FrequencyEncoder(columns=["x"])
    enc.fit(_df(x=["a", "a", None, "b"]))
    out = enc.transform(_df(x=["a", "c", None]))
    assert out["x"].tolist() == pytest.approx([0.5, 0.0, 0.25])

    no_missing = FrequencyEncoder(columns=["x"])
    no_missing.fit(_df(x=["a", "a", "b"]))
    assert no_missing.transform(_df(x=[None]))["x"].tolist() == pytest.approx([0.0])


def test_frequency_multiple_columns() -> None:
    out = FrequencyEncoder(columns=["x", "z"]).fit_transform(
        _df(x=["a", "a", "b"], z=["q", "r", "q"])
    )
    assert out["x"].tolist() == pytest.approx([2 / 3, 2 / 3, 1 / 3])
    assert out["z"].tolist() == pytest.approx([2 / 3, 1 / 3, 2 / 3])


def test_frequency_min_count_pools_rare_levels() -> None:
    X = _df(x=["a", "a", "a", "b", "b", "c", "d"])
    enc = FrequencyEncoder(columns=["x"], min_count=2, normalize=False, cv=None)
    out = enc.fit_transform(X)
    # c and d share the pooled count 2; unseen joins that bucket.
    assert out["x"].tolist() == pytest.approx([3, 3, 3, 2, 2, 2, 2])
    assert enc.transform(_df(x=["z"]))["x"].tolist() == pytest.approx([2])
    assert enc.maps_["x"]["other"] == pytest.approx(2)
    assert enc.other_frequency_["x"] == pytest.approx(2)
    assert sum(FrequencyEncoder(columns=["x"], min_count=2).fit(X).maps_["x"].values()) == (
        pytest.approx(1.0)
    )


def test_frequency_min_count_keeps_missing_separate() -> None:
    X = _df(x=["a", "a", None, "b"])
    enc = FrequencyEncoder(columns=["x"], min_count=2, cv=None)
    out = enc.fit_transform(X)["x"].tolist()
    # n=4; "a" kept (2/4); "b" pooled (1/4); missing stays its own level (1/4).
    assert out == pytest.approx([0.5, 0.5, 0.25, 0.25])
    assert enc.maps_["x"][None] == pytest.approx(0.25)
    assert enc.transform(_df(x=[None, "z"]))["x"].tolist() == pytest.approx([0.25, 0.25])


def test_frequency_min_count_matches_group_then_encode() -> None:
    X = _df(city=["a", "a", "a", "b", "b", "c", "d"])
    grouped = RareCategoryGrouper(columns=["city"], min_count=2).fit_transform(X)
    chained = FrequencyEncoder(columns=["city"], cv=None).fit_transform(grouped)
    direct = FrequencyEncoder(columns=["city"], min_count=2, cv=None).fit_transform(X)
    np.testing.assert_allclose(chained["city"], direct["city"])
    unseen = _df(city=["z"])
    grouped_unseen = RareCategoryGrouper(columns=["city"], min_count=2).fit(X).transform(
        unseen
    )
    chained_unseen = FrequencyEncoder(columns=["city"], cv=None).fit(grouped).transform(
        grouped_unseen
    )
    direct_unseen = FrequencyEncoder(columns=["city"], min_count=2, cv=None).fit(X).transform(
        unseen
    )
    np.testing.assert_allclose(chained_unseen["city"], direct_unseen["city"])


def test_frequency_custom_other_label() -> None:
    enc = FrequencyEncoder(
        columns=["x"], min_count=2, other_label="RARE", normalize=False, cv=None
    )
    out = enc.fit_transform(_df(x=["a", "b", "b"]))
    assert out["x"].tolist() == pytest.approx([1, 2, 2])
    assert enc.maps_["x"]["RARE"] == pytest.approx(1)


def test_frequency_global_mapping_ignores_cv() -> None:
    X = _df(x=["a", "a", "b"])
    enc = FrequencyEncoder(columns=["x"], cv=5, shuffle=False)
    enc.fit(X)
    assert enc.transform(X)["x"].tolist() == pytest.approx([2 / 3, 2 / 3, 1 / 3])


def test_oof_frequency_matches_complementary_folds() -> None:
    enc = FrequencyEncoder(columns=["x"], cv=2, shuffle=False)
    X = _df(x=["a", "a", "b", "a", "b", "b"])
    out = enc.fit_transform(X)["x"].tolist()
    # KFold(n_splits=2, shuffle=False) on 6 rows:
    #   fold 0 test [0,1,2] = a,a,b encoded from [a,b,b] -> a=1/3, b=2/3
    #   fold 1 test [3,4,5] = a,b,b encoded from [a,a,b] -> a=2/3, b=1/3
    assert out == pytest.approx([1 / 3, 1 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3])
    assert enc.transform(X)["x"].tolist() == pytest.approx([0.5] * 6)

    raw = FrequencyEncoder(columns=["x"], normalize=False, cv=2, shuffle=False)
    assert raw.fit_transform(X)["x"].tolist() == pytest.approx([1, 1, 2, 2, 1, 1])


def test_oof_frequency_differs_from_global_and_ignores_y() -> None:
    X = _df(x=["a", "a", "b", "a", "b", "b"])
    enc = FrequencyEncoder(columns=["x"], cv=2, shuffle=False)
    oof = enc.fit_transform(X, pd.Series([0, 1, 0, 1, 0, 1]))["x"].to_numpy()
    again = FrequencyEncoder(columns=["x"], cv=2, shuffle=False).fit_transform(X)["x"]
    np.testing.assert_allclose(oof, again.to_numpy())
    assert not np.allclose(oof, enc.transform(X)["x"].to_numpy())


def test_oof_unique_level_frequency_excludes_itself() -> None:
    X = _df(x=[f"id_{i}" for i in range(6)])
    enc = FrequencyEncoder(columns=["x"], cv=2, shuffle=False)
    oof = enc.fit_transform(X)["x"].to_numpy()
    assert oof.tolist() == pytest.approx([0.0] * 6)
    assert enc.transform(X)["x"].tolist() == pytest.approx([1 / 6] * 6)


def test_leave_one_out_frequency() -> None:
    X = _df(x=["a", "a", "a", "b"])
    enc = FrequencyEncoder(columns=["x"], normalize=False, cv=4, shuffle=False)
    # Each "a" row sees the other two "a"s; "b" is absent from the other rows.
    assert enc.fit_transform(X)["x"].tolist() == pytest.approx([2, 2, 2, 0])
    assert enc.transform(X)["x"].tolist() == pytest.approx([3, 3, 3, 1])


def test_oof_min_count_uses_fold_counts() -> None:
    X = _df(x=["a", "a", "a", "b", "b", "c"])
    enc = FrequencyEncoder(
        columns=["x"], min_count=2, normalize=False, cv=2, shuffle=False
    )
    out = enc.fit_transform(X)["x"].tolist()
    # Fold 0 train [b,b,c]: "c" is rare, so unseen "a" joins that bucket (count 1).
    # Fold 1 train [a,a,a]: nothing is rare, so unseen "b" and "c" encode to 0.
    assert out == pytest.approx([1, 1, 1, 0, 0, 0])
    assert enc.transform(X)["x"].tolist() == pytest.approx([3, 3, 3, 2, 2, 1])

    plain = FrequencyEncoder(columns=["x"], normalize=False, cv=2, shuffle=False)
    assert plain.fit_transform(X)["x"].tolist() == pytest.approx([0, 0, 0, 0, 0, 0])


def test_oof_frequency_preserves_index_and_is_reproducible() -> None:
    X = _df(x=["a", "a", "b", "a", "b", "b"])
    X.index = [10, 20, 30, 40, 50, 60]
    enc = FrequencyEncoder(columns=["x"], normalize=False, cv=2, shuffle=False)
    out = enc.fit_transform(X)
    assert out.index.tolist() == [10, 20, 30, 40, 50, 60]
    assert out["x"].tolist() == pytest.approx([1, 1, 2, 2, 1, 1])

    kwargs = dict(columns=["x"], cv=3, shuffle=True, random_state=0)
    left = FrequencyEncoder(**kwargs).fit_transform(_df(x=list("abacabacbc")))
    right = FrequencyEncoder(**kwargs).fit_transform(_df(x=list("abacabacbc")))
    np.testing.assert_allclose(left["x"], right["x"])


def test_oof_falls_back_when_sample_is_too_small() -> None:
    X = _df(x=["a"])
    enc = FrequencyEncoder(columns=["x"], cv=5, shuffle=False)
    assert enc.fit_transform(X)["x"].tolist() == pytest.approx([1.0])


def test_frequency_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="cv must be None"):
        FrequencyEncoder(columns=["x"], cv=1)
    with pytest.raises(ValueError, match="min_count must be an integer >= 1"):
        FrequencyEncoder(columns=["x"], min_count=0)
    with pytest.raises(ValueError, match="other_label must be a non-missing"):
        FrequencyEncoder(columns=["x"], other_label=float("nan"))
    assert FrequencyEncoder(columns=["x"], min_count=None).min_count is None


def test_frequency_missing_column_and_unfitted() -> None:
    enc = FrequencyEncoder(columns=["x"])
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(_df(x=["a"]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.fit(_df(z=["a"]))
    enc.fit(_df(x=["a", "b"]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.transform(_df(z=["a"]))


def test_pipeline_chains_grouper_and_frequency() -> None:
    X = _df(city=["a", "a", "a", "b", "b", "c"], age=[1, 2, 3, 4, 5, 6])
    pipe = FeatureEngineeringPipeline(
        steps=[
            ("rare", RareCategoryGrouper(columns=["city"], min_count=2)),
            ("freq", FrequencyEncoder(columns=["city"], cv=None)),
        ]
    )
    out = pipe.fit_transform(X)
    assert out["city"].tolist() == pytest.approx([0.5, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 6])
    assert out["age"].tolist() == [1, 2, 3, 4, 5, 6]
    transformed = pipe.transform(_df(city=["a", "z"], age=[7, 8]))
    assert transformed["city"].tolist() == pytest.approx([0.5, 1 / 6])
    assert transformed["age"].tolist() == [7, 8]


def test_pipeline_fit_transform_uses_oof_frequencies() -> None:
    X = _df(x=["a", "a", "b", "a", "b", "b"])
    pipe = FeatureEngineeringPipeline(
        steps=[("freq", FrequencyEncoder(columns=["x"], cv=2, shuffle=False))]
    )
    oof = pipe.fit_transform(X)["x"].tolist()
    full = pipe.transform(X)["x"].tolist()
    assert oof == pytest.approx([1 / 3, 1 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3])
    assert full == pytest.approx([0.5] * 6)
