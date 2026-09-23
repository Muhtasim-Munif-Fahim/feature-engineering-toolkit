"""Tests for leave-one-out target encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import (
    FeatureEngineeringPipeline,
    LeaveOneOutEncoder,
    TargetEncoder,
)


def _df(**cols):
    return pd.DataFrame(cols)


def _reference_loo(categories, y, smoothing: float) -> np.ndarray:
    """Independent leave-one-out formula used to check the encoder."""
    y = np.asarray(y, dtype=float)
    categories = np.asarray(categories, dtype=object)
    global_mean = float(y.mean())
    out = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        same = categories == categories[i]
        count = int(same.sum())
        total = float(y[same].sum())
        denom = (count - 1) + smoothing
        if denom == 0.0:
            out[i] = global_mean
        else:
            out[i] = (total - y[i] + smoothing * global_mean) / denom
    return out


def test_loo_requires_y() -> None:
    enc = LeaveOneOutEncoder(columns=["x"], target="y")
    with pytest.raises(ValueError, match="requires y"):
        enc.fit(_df(x=["a", "b"]))


def test_loo_rejects_negative_smoothing_and_length_mismatch() -> None:
    with pytest.raises(ValueError, match="smoothing must be >= 0"):
        LeaveOneOutEncoder(columns=["x"], target="y", smoothing=-0.1)
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    with pytest.raises(ValueError, match="expected y to have 2 rows"):
        enc.fit(_df(x=["a", "b"]), pd.Series([1.0]))


def test_loo_smoothing_zero_is_mean_of_other_rows() -> None:
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    X = _df(x=["a", "a", "a", "b", "b"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0])
    out = enc.fit_transform(X, y)["x"].to_numpy()
    # Other "a" rows: (0+1)/2, (1+1)/2, (1+0)/2. Other "b" row: 1 and 0.
    assert out.tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0, 0.0])
    # Full-sample means still include the row itself.
    assert enc.transform(X)["x"].tolist() == pytest.approx(
        [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5]
    )
    assert enc.global_mean_ == pytest.approx(0.6)


def test_loo_smoothing_shrinks_toward_global_mean() -> None:
    X = _df(x=["a", "a", "a", "b", "b"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0])
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=2.0)
    out = enc.fit_transform(X, y)["x"].tolist()
    # m = 0.6
    # a rows: (sum_a - y + 2*0.6) / (3 - 1 + 2), sum_a = 2
    # b rows: (sum_b - y + 1.2) / (2 - 1 + 2), sum_b = 1
    assert out == pytest.approx([0.55, 0.8, 0.55, 2.2 / 3.0, 0.4])
    heavy = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=1e6)
    shrunk = heavy.fit_transform(X, y)["x"].to_numpy()
    assert shrunk == pytest.approx([0.6] * 5, abs=1e-4)


def test_loo_matches_reference_on_mixed_categories() -> None:
    rng = np.random.default_rng(0)
    categories = rng.choice(["a", "b", "c", "d"], size=40)
    y = rng.normal(size=40)
    smoothing = 3.5
    enc = LeaveOneOutEncoder(columns=["city"], target="y", smoothing=smoothing)
    out = enc.fit_transform(_df(city=categories), y)["city"].to_numpy()
    expected = _reference_loo(categories, y, smoothing)
    np.testing.assert_allclose(out, expected)


def test_loo_binary_target_excludes_the_row_even_when_cv_would_stratify() -> None:
    """``TargetEncoder(cv=n)`` cannot leave one binary row out; this encoder can."""
    X = _df(x=["a", "a", "a", "a"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    out = enc.fit_transform(X, y)["x"].to_numpy()
    # Each row is the mean of the other three labels.
    assert out.tolist() == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0])

    folded = TargetEncoder(columns=["x"], target="y", smoothing=0.0, cv=4, shuffle=False)
    folded_out = folded.fit_transform(X, y)["x"].to_numpy()
    assert not np.allclose(folded_out, out)


def test_loo_singleton_and_unique_ids_use_global_mean() -> None:
    X = _df(x=["a", "b", "b"])
    y = pd.Series([1.0, 0.0, 0.5])
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    out = enc.fit_transform(X, y)["x"].tolist()
    # "a" has no partner. "b" rows see only the other "b".
    assert out == pytest.approx([0.5, 0.5, 0.0])

    rng = np.random.default_rng(1)
    n = 80
    y_id = rng.integers(0, 2, size=n).astype(float)
    ids = _df(x=[f"id_{i}" for i in range(n)])
    loo = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    encoded = loo.fit_transform(ids, y_id)["x"].to_numpy()
    assert encoded == pytest.approx([float(y_id.mean())] * n)
    naive = TargetEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    np.testing.assert_allclose(naive.fit_transform(ids, y_id)["x"].to_numpy(), y_id)
    assert not np.allclose(encoded, y_id)


def test_loo_two_row_category_swaps_labels_when_smoothing_is_zero() -> None:
    X = _df(x=["a", "a"])
    y = pd.Series([1.0, 0.0])
    out = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0).fit_transform(X, y)
    assert out["x"].tolist() == pytest.approx([0.0, 1.0])


def test_transform_matches_target_encoder_global_mapping() -> None:
    X = _df(x=["a", "a", "b", "b", "c"], other=[1, 2, 3, 4, 5])
    y = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0])
    smoothing = 2.5
    loo = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=smoothing)
    te = TargetEncoder(columns=["x"], target="y", smoothing=smoothing, cv=None)
    loo.fit(X, y)
    te.fit(X, y)
    assert loo.global_mean_ == pytest.approx(te.global_mean_)
    assert loo.maps_["x"].keys() == te.maps_["x"].keys()
    for key in loo.maps_["x"]:
        assert loo.maps_["x"][key] == pytest.approx(te.maps_["x"][key])
    pd.testing.assert_series_equal(loo.transform(X)["x"], te.transform(X)["x"])
    unseen = _df(x=["z", "a"], other=[9, 8])
    pd.testing.assert_series_equal(
        loo.transform(unseen)["x"], te.transform(unseen)["x"]
    )
    # Training encodings exclude the row, so they are not the global mapping.
    assert not np.allclose(
        loo.fit_transform(X, y)["x"].to_numpy(), loo.transform(X)["x"].to_numpy()
    )


def test_loo_unseen_category_uses_global_mean() -> None:
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    enc.fit(_df(x=["a", "a", "b"]), pd.Series([1.0, 0.0, 1.0]))
    out = enc.transform(_df(x=["c"]))
    assert out["x"].tolist() == pytest.approx([2.0 / 3.0])


def test_loo_accepts_dataframe_and_array_targets() -> None:
    X = _df(x=["a", "a", "b"])
    y = np.array([1.0, 0.0, 1.0])
    from_array = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    from_frame = LeaveOneOutEncoder(columns=["x"], target="label", smoothing=0.0)
    array_out = from_array.fit_transform(X, y)["x"].to_numpy()
    frame_out = from_frame.fit_transform(X, pd.DataFrame({"label": y}))["x"].to_numpy()
    np.testing.assert_allclose(array_out, frame_out)
    # Other "a" is 0 or 1. The lone "b" falls back to the global mean 2/3.
    assert array_out.tolist() == pytest.approx([0.0, 1.0, 2.0 / 3.0])


def test_loo_multiple_columns_passthrough_index_and_no_mutation() -> None:
    df = _df(x=["a", "a", "b"], z=["q", "r", "r"], age=[1, 2, 3])
    df.index = [10, 20, 30]
    original = df.copy()
    y = pd.Series([1.0, 0.0, 1.0])
    enc = LeaveOneOutEncoder(columns=["x", "z"], target="y", smoothing=0.0)
    out = enc.fit_transform(df, y)
    pd.testing.assert_frame_equal(df, original)
    assert out.index.tolist() == [10, 20, 30]
    assert out["age"].tolist() == [1, 2, 3]
    # x: other "a" is 0 / 1; "b" is the only "b" -> global mean 2/3.
    assert out["x"].tolist() == pytest.approx([0.0, 1.0, 2.0 / 3.0])
    # z: "q" is unique -> 2/3; each "r" sees the other "r".
    assert out["z"].tolist() == pytest.approx([2.0 / 3.0, 1.0, 0.0])


def test_loo_integer_levels_and_missing_values() -> None:
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    X = _df(x=[1, 1, 2, None])
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    out = enc.fit_transform(X, y)["x"].tolist()
    # Missing values have no stored level and use the global mean (0.5).
    assert out == pytest.approx([0.0, 1.0, 0.5, 0.5])
    transformed = enc.transform(_df(x=[1, 2, None, 9]))["x"].tolist()
    assert transformed == pytest.approx([0.5, 1.0, 0.5, 0.5])


def test_loo_single_row_falls_back_to_its_mean() -> None:
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    out = enc.fit_transform(_df(x=["a"]), pd.Series([3.0]))
    assert out["x"].tolist() == pytest.approx([3.0])


def test_loo_is_deterministic() -> None:
    X = _df(x=list("abacabacbc"))
    y = pd.Series(np.linspace(0.0, 1.0, num=10))
    left = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=1.0).fit_transform(X, y)
    right = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=1.0).fit_transform(X, y)
    np.testing.assert_allclose(left["x"], right["x"])


def test_loo_missing_column_and_unfitted() -> None:
    enc = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(_df(x=["a"]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.fit(_df(z=["a", "b"]), pd.Series([0.0, 1.0]))
    enc.fit(_df(x=["a", "b"]), pd.Series([0.0, 1.0]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.transform(_df(z=["a"]))


def test_pipeline_fit_transform_uses_loo_and_transform_uses_global_means() -> None:
    X = _df(x=["a", "a", "b"])
    y = pd.Series([1.0, 0.0, 1.0])
    pipe = FeatureEngineeringPipeline(
        steps=[("loo", LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0))]
    )
    trained = pipe.fit_transform(X, y)["x"].tolist()
    assert trained == pytest.approx([0.0, 1.0, 2.0 / 3.0])
    held_out = pipe.transform(X)["x"].tolist()
    assert held_out == pytest.approx([0.5, 0.5, 1.0])
    unseen = pipe.transform(_df(x=["c"]))["x"].tolist()
    assert unseen == pytest.approx([2.0 / 3.0])
