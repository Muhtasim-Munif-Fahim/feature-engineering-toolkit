"""Tests for CatBoost-style ordered target encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import (
    CatBoostEncoder,
    FeatureEngineeringPipeline,
    LeaveOneOutEncoder,
    TargetEncoder,
)


def _df(**cols):
    return pd.DataFrame(cols)


def _reference_ordered(categories, y, smoothing: float, order: np.ndarray) -> np.ndarray:
    """Expanding-category means along a fixed order (independent of the encoder)."""
    y = np.asarray(y, dtype=float)
    categories = np.asarray(categories, dtype=object)
    global_mean = float(y.mean())
    out = np.full(len(y), global_mean, dtype=float)
    running_sum: dict[object, float] = {}
    running_count: dict[object, float] = {}
    for pos in order:
        cat = categories[pos]
        if cat is None or (isinstance(cat, float) and np.isnan(cat)):
            out[pos] = global_mean
            continue
        prev_sum = running_sum.get(cat, 0.0)
        prev_count = running_count.get(cat, 0.0)
        denom = prev_count + smoothing
        if denom == 0.0:
            out[pos] = global_mean
        else:
            out[pos] = (prev_sum + smoothing * global_mean) / denom
        running_sum[cat] = prev_sum + float(y[pos])
        running_count[cat] = prev_count + 1.0
    return out


def test_catboost_requires_y() -> None:
    enc = CatBoostEncoder(columns=["x"], target="y", random_state=0)
    with pytest.raises(ValueError, match="requires y"):
        enc.fit(_df(x=["a", "b"]))


def test_catboost_rejects_bad_hyperparameters_and_length_mismatch() -> None:
    with pytest.raises(ValueError, match="smoothing must be >= 0"):
        CatBoostEncoder(columns=["x"], target="y", smoothing=-0.1)
    with pytest.raises(ValueError, match="n_permutations must be an integer >= 1"):
        CatBoostEncoder(columns=["x"], target="y", n_permutations=0)
    with pytest.raises(ValueError, match="random_state must be an integer or None"):
        CatBoostEncoder(columns=["x"], target="y", random_state=True)
    enc = CatBoostEncoder(columns=["x"], target="y", smoothing=0.0, random_state=0)
    with pytest.raises(ValueError, match="expected y to have 2 rows"):
        enc.fit(_df(x=["a", "b"]), pd.Series([1.0]))


def test_catboost_identity_order_is_expanding_mean() -> None:
    """With a fixed identity permutation the encoding is the expanding mean."""
    X = _df(x=["a", "a", "a", "b", "b"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0])
    enc = CatBoostEncoder(
        columns=["x"], target="y", smoothing=0.0, n_permutations=1, random_state=0
    )
    # Monkeypatch the RNG so the only permutation is identity.
    class _IdentityRng:
        def permutation(self, n):
            return np.arange(n)

    enc.fit(X, y)
    values = enc._ordered_values(X["x"], y, _IdentityRng())  # type: ignore[arg-type]
    expected = _reference_ordered(X["x"].to_numpy(), y.to_numpy(), 0.0, np.arange(5))
    np.testing.assert_allclose(values, expected)
    # First "a" and first "b" have no history -> global mean 0.6.
    assert values.tolist() == pytest.approx([0.6, 1.0, 0.5, 0.6, 0.0])


def test_catboost_matches_reference_on_seeded_permutation() -> None:
    rng = np.random.default_rng(7)
    categories = rng.choice(["a", "b", "c"], size=30)
    y = rng.normal(size=30)
    smoothing = 2.5
    # Reproduce the first (and only) permutation the encoder will draw.
    order_rng = np.random.default_rng(11)
    order = order_rng.permutation(30)
    expected = _reference_ordered(categories, y, smoothing, order)
    enc = CatBoostEncoder(
        columns=["city"],
        target="y",
        smoothing=smoothing,
        n_permutations=1,
        random_state=11,
    )
    out = enc.fit_transform(_df(city=categories), y)["city"].to_numpy()
    np.testing.assert_allclose(out, expected)


def test_catboost_averages_multiple_permutations() -> None:
    X = _df(x=["a", "a", "b", "b"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    single = CatBoostEncoder(
        columns=["x"], target="y", smoothing=0.0, n_permutations=1, random_state=3
    )
    multi = CatBoostEncoder(
        columns=["x"], target="y", smoothing=0.0, n_permutations=8, random_state=3
    )
    one = single.fit_transform(X, y)["x"].to_numpy()
    many = multi.fit_transform(X, y)["x"].to_numpy()
    # Multi-permutation average should differ from a single draw in general,
    # and stay inside the convex hull of possible expanding means.
    assert not np.allclose(one, many)
    assert np.all(many >= 0.0 - 1e-12)
    assert np.all(many <= 1.0 + 1e-12)


def test_catboost_transform_matches_target_encoder_global_mapping() -> None:
    X = _df(x=["a", "a", "b", "b", "c"], other=[1, 2, 3, 4, 5])
    y = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0])
    smoothing = 2.5
    cb = CatBoostEncoder(
        columns=["x"], target="y", smoothing=smoothing, random_state=0
    )
    te = TargetEncoder(columns=["x"], target="y", smoothing=smoothing, cv=None)
    cb.fit(X, y)
    te.fit(X, y)
    assert cb.global_mean_ == pytest.approx(te.global_mean_)
    assert cb.maps_["x"].keys() == te.maps_["x"].keys()
    for key in cb.maps_["x"]:
        assert cb.maps_["x"][key] == pytest.approx(te.maps_["x"][key])
    pd.testing.assert_series_equal(cb.transform(X)["x"], te.transform(X)["x"])
    unseen = _df(x=["z", "a"], other=[9, 8])
    pd.testing.assert_series_equal(
        cb.transform(unseen)["x"], te.transform(unseen)["x"]
    )


def test_catboost_training_encodings_differ_from_global_means() -> None:
    X = _df(x=["a", "a", "a", "b", "b"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0])
    enc = CatBoostEncoder(
        columns=["x"], target="y", smoothing=0.0, n_permutations=4, random_state=1
    )
    trained = enc.fit_transform(X, y)["x"].to_numpy()
    held = enc.transform(X)["x"].to_numpy()
    assert not np.allclose(trained, held)


def test_catboost_does_not_copy_unique_ids_like_naive_target_encoder() -> None:
    rng = np.random.default_rng(1)
    n = 60
    y = rng.integers(0, 2, size=n).astype(float)
    ids = _df(x=[f"id_{i}" for i in range(n)])
    cb = CatBoostEncoder(
        columns=["x"], target="y", smoothing=0.0, n_permutations=3, random_state=0
    )
    encoded = cb.fit_transform(ids, y)["x"].to_numpy()
    # Every id is unique, so every expanding history is empty -> global mean.
    assert encoded == pytest.approx([float(y.mean())] * n)
    naive = TargetEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    np.testing.assert_allclose(naive.fit_transform(ids, y)["x"].to_numpy(), y)
    assert not np.allclose(encoded, y)


def test_catboost_is_reproducible_with_seed() -> None:
    X = _df(x=list("abacabacbc"))
    y = pd.Series(np.linspace(0.0, 1.0, num=10))
    left = CatBoostEncoder(
        columns=["x"], target="y", smoothing=1.0, n_permutations=3, random_state=42
    ).fit_transform(X, y)
    right = CatBoostEncoder(
        columns=["x"], target="y", smoothing=1.0, n_permutations=3, random_state=42
    ).fit_transform(X, y)
    np.testing.assert_allclose(left["x"], right["x"])
    other = CatBoostEncoder(
        columns=["x"], target="y", smoothing=1.0, n_permutations=3, random_state=43
    ).fit_transform(X, y)
    assert not np.allclose(left["x"], other["x"])


def test_catboost_multiple_columns_passthrough_and_no_mutation() -> None:
    df = _df(x=["a", "a", "b"], z=["q", "r", "r"], age=[1, 2, 3])
    df.index = [10, 20, 30]
    original = df.copy()
    y = pd.Series([1.0, 0.0, 1.0])
    enc = CatBoostEncoder(
        columns=["x", "z"], target="y", smoothing=0.0, n_permutations=2, random_state=0
    )
    out = enc.fit_transform(df, y)
    pd.testing.assert_frame_equal(df, original)
    assert out.index.tolist() == [10, 20, 30]
    assert out["age"].tolist() == [1, 2, 3]


def test_catboost_missing_column_and_unfitted() -> None:
    enc = CatBoostEncoder(columns=["x"], target="y", smoothing=0.0, random_state=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        enc.transform(_df(x=["a"]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.fit(_df(z=["a", "b"]), pd.Series([0.0, 1.0]))
    enc.fit(_df(x=["a", "b"]), pd.Series([0.0, 1.0]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.transform(_df(z=["a"]))


def test_pipeline_fit_transform_uses_ordered_encoding() -> None:
    X = _df(x=["a", "a", "b"])
    y = pd.Series([1.0, 0.0, 1.0])
    pipe = FeatureEngineeringPipeline(
        steps=[
            (
                "cb",
                CatBoostEncoder(
                    columns=["x"],
                    target="y",
                    smoothing=0.0,
                    n_permutations=1,
                    random_state=0,
                ),
            )
        ]
    )
    trained = pipe.fit_transform(X, y)["x"].to_numpy()
    held_out = pipe.transform(X)["x"].to_numpy()
    assert trained.shape == (3,)
    assert held_out.tolist() == pytest.approx([0.5, 0.5, 1.0])
    unseen = pipe.transform(_df(x=["c"]))["x"].tolist()
    assert unseen == pytest.approx([2.0 / 3.0])


def test_catboost_accepts_dataframe_and_array_targets() -> None:
    X = _df(x=["a", "a", "b"])
    y = np.array([1.0, 0.0, 1.0])
    from_array = CatBoostEncoder(
        columns=["x"], target="y", smoothing=0.0, n_permutations=1, random_state=5
    )
    from_frame = CatBoostEncoder(
        columns=["x"], target="label", smoothing=0.0, n_permutations=1, random_state=5
    )
    array_out = from_array.fit_transform(X, y)["x"].to_numpy()
    frame_out = from_frame.fit_transform(X, pd.DataFrame({"label": y}))["x"].to_numpy()
    np.testing.assert_allclose(array_out, frame_out)


def test_catboost_export_and_differs_from_loo_in_general() -> None:
    X = _df(x=["a", "a", "a", "b", "b", "b"])
    y = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    cb = CatBoostEncoder(
        columns=["x"], target="y", smoothing=0.0, n_permutations=1, random_state=2
    )
    loo = LeaveOneOutEncoder(columns=["x"], target="y", smoothing=0.0)
    cb_out = cb.fit_transform(X, y)["x"].to_numpy()
    loo_out = loo.fit_transform(X, y)["x"].to_numpy()
    # Ordered encoding uses a prefix, LOO uses all other rows — generally differ.
    assert not np.allclose(cb_out, loo_out)
