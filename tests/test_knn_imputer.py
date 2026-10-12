"""Tests for KNNImputer and nan_euclidean_distances."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer as SkKNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances as sk_nan_euclidean

from feature_engineering_kit import KNNImputer, nan_euclidean_distances


def _frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": z + rng.normal(scale=0.3, size=n),
            "b": 2 * z + rng.normal(scale=0.3, size=n),
            "c": -z + rng.normal(scale=0.3, size=n),
            "d": rng.normal(size=n),
            "label": rng.choice(["x", "y"], size=n),
        }
    )
    # At most one missing value per row, so every row shares coordinates.
    cols = ["a", "b", "c", "d"]
    for i in rng.choice(n, size=30, replace=False):
        df.loc[i, cols[rng.integers(0, 4)]] = np.nan
    return df


def test_nan_euclidean_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    A = rng.normal(size=(7, 4))
    B = rng.normal(size=(5, 4))
    A[0, 1] = A[3, 0] = A[3, 2] = np.nan
    B[2, 3] = np.nan
    np.testing.assert_allclose(
        nan_euclidean_distances(A, B), sk_nan_euclidean(A, B), rtol=1e-10, atol=1e-12
    )


def test_nan_euclidean_no_shared_coordinate_is_nan() -> None:
    A = np.array([[1.0, np.nan]])
    B = np.array([[np.nan, 2.0]])
    assert np.isnan(nan_euclidean_distances(A, B)[0, 0])


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_matches_sklearn(weights: str, k: int) -> None:
    df = _frame()
    cols = ["a", "b", "c", "d"]
    ours = KNNImputer(cols, n_neighbors=k, weights=weights).fit_transform(df)
    ref = SkKNNImputer(n_neighbors=k, weights=weights).fit_transform(df[cols])
    np.testing.assert_allclose(ours[cols].to_numpy(), ref, rtol=1e-10, atol=1e-12)
    assert list(ours["label"]) == list(df["label"])
    assert not ours[cols].isna().any().any()


def test_train_test_split_uses_training_donors() -> None:
    df = _frame()
    cols = ["a", "b", "c", "d"]
    train, test = df.iloc[:80], df.iloc[80:]
    imp = KNNImputer(cols, n_neighbors=4).fit(train)
    ref = SkKNNImputer(n_neighbors=4).fit(train[cols])
    np.testing.assert_allclose(
        imp.transform(test)[cols].to_numpy(), ref.transform(test[cols]), rtol=1e-10
    )


def test_hand_example_and_correlated_recovery() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0], "y": [1.0, 2.0, np.nan, 10.0]})
    out = KNNImputer(["x", "y"], n_neighbors=2).fit_transform(df)
    # Nearest rows to x=3 with y observed: x=2 (d=sqrt(2)) and x=1 (d=2*sqrt(2)).
    assert out.loc[2, "y"] == pytest.approx(1.5)
    out_w = KNNImputer(["x", "y"], n_neighbors=2, weights="distance").fit_transform(df)
    assert out_w.loc[2, "y"] == pytest.approx((2 / 1 + 1 / 2) / (1 + 1 / 2))


def test_zero_distance_donor_takes_all_weight() -> None:
    df = pd.DataFrame({"x": [1.0, 1.0, 5.0], "y": [7.0, np.nan, 100.0]})
    out = KNNImputer(["x", "y"], n_neighbors=2, weights="distance").fit_transform(df)
    assert out.loc[1, "y"] == pytest.approx(7.0)


def test_falls_back_to_mean_without_shared_coordinates() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 6.0, 8.0]})
    test = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
    out = KNNImputer(["x", "y"]).fit(train).transform(test)
    assert out.loc[0, "x"] == pytest.approx(2.0)
    assert out.loc[0, "y"] == pytest.approx(6.0)


def test_add_indicator() -> None:
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, 3.0]})
    imp = KNNImputer(["x", "y"], n_neighbors=1, add_indicator=True)
    out = imp.fit_transform(df)
    assert imp.indicator_columns_ == ["x"]
    assert list(out["x__was_missing"]) == [0, 1, 0]
    assert "y__was_missing" not in out.columns
    assert imp.get_feature_names_out()[-1] == "x__was_missing"


def test_validation_and_fitted_state() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="n_neighbors"):
        KNNImputer(["x"], n_neighbors=0).fit(df)
    with pytest.raises(ValueError, match="weights"):
        KNNImputer(["x"], weights="gaussian").fit(df)
    with pytest.raises(KeyError, match="nope"):
        KNNImputer(["nope"]).fit(df)
    with pytest.raises(RuntimeError, match="not fitted"):
        KNNImputer(["x"]).transform(df)
    with pytest.raises(ValueError, match="infinite"):
        KNNImputer(["x"]).fit(pd.DataFrame({"x": [1.0, np.inf]}))


def test_preprocessing_pipeline_knn_option() -> None:
    from feature_engineering_kit import build_preprocessing_pipeline

    df = _frame().drop(columns=["label"])
    pipe = build_preprocessing_pipeline(
        df, numeric_impute="knn", knn_neighbors=3, drop_high_correlation=None
    )
    step = dict(pipe.steps)["impute_numeric"]
    assert isinstance(step, KNNImputer) and step.n_neighbors == 3
    out = pipe.fit_transform(df)
    assert not out.isna().any().any()

    mean_pipe = build_preprocessing_pipeline(df, numeric_impute="mean")
    assert dict(mean_pipe.steps)["impute_numeric"].strategy == "mean"
    with pytest.raises(ValueError, match="numeric_impute"):
        build_preprocessing_pipeline(df, numeric_impute="mode")
