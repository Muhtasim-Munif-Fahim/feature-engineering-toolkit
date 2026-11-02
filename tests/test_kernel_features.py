"""Tests for RandomFourierFeatures / NystroemFeatures kernel maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import laplacian_kernel, polynomial_kernel, rbf_kernel

from feature_engineering_kit import NystroemFeatures, RandomFourierFeatures, kernel_matrix


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "cat": rng.choice(["x", "y"], n)}
    )


def test_kernel_matrix_matches_sklearn():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    np.testing.assert_allclose(kernel_matrix(A, B, "rbf", gamma=0.3), rbf_kernel(A, B, gamma=0.3))
    np.testing.assert_allclose(kernel_matrix(A, B, "laplacian", gamma=0.3), laplacian_kernel(A, B, gamma=0.3))
    np.testing.assert_allclose(
        kernel_matrix(A, B, "polynomial", gamma=0.5, degree=2, coef0=1.0),
        polynomial_kernel(A, B, degree=2, gamma=0.5, coef0=1.0),
    )
    with pytest.raises(ValueError):
        kernel_matrix(A, B, "bogus")


@pytest.mark.parametrize("kernel", ["rbf", "laplacian"])
def test_rff_approximates_kernel(kernel):
    df = _frame(200)
    rff = RandomFourierFeatures(["a", "b"], n_components=4000, kernel=kernel, gamma=0.5, random_state=0)
    Z = rff.fit(df).transform_array(df)
    exact = kernel_matrix(df[["a", "b"]].to_numpy(), df[["a", "b"]].to_numpy(), kernel, gamma=0.5)
    assert np.abs(Z @ Z.T - exact).max() < 0.1
    assert np.abs(Z @ Z.T - exact).mean() < 0.02


def test_rff_columns_passthrough_and_determinism():
    df = _frame()
    out = RandomFourierFeatures(["a", "b"], n_components=10, random_state=3).fit_transform(df)
    assert list(out.columns) == ["cat"] + [f"rff_{i}" for i in range(10)]
    again = RandomFourierFeatures(["a", "b"], n_components=10, random_state=3).fit_transform(df)
    pd.testing.assert_frame_equal(out, again)
    kept = RandomFourierFeatures(["a"], n_components=3, keep_original=True, prefix="k").fit_transform(df)
    assert list(kept.columns) == ["a", "b", "cat", "k_0", "k_1", "k_2"]


def test_nystroem_exact_with_all_landmarks():
    df = _frame(60)
    X = df[["a", "b"]].to_numpy()
    for kernel in ("rbf", "laplacian", "polynomial", "linear"):
        nys = NystroemFeatures(["a", "b"], n_components=60, kernel=kernel, gamma=0.7, random_state=0)
        Z = nys.fit(df).transform_array(df)
        exact = kernel_matrix(X, X, kernel, gamma=0.7, degree=3, coef0=1.0)
        np.testing.assert_allclose(Z @ Z.T, exact, atol=1e-6 * max(1.0, np.abs(exact).max()))
    lin = NystroemFeatures(["a", "b"], n_components=60, kernel="linear").fit(df)
    assert lin.n_features_out_ == 2  # rank-deficient landmark Gram matrix is truncated


def test_nystroem_beats_rff_at_same_budget():
    df = _frame(400)
    X = df[["a", "b"]].to_numpy()
    exact = kernel_matrix(X, X, "rbf", gamma=0.5)
    nys = NystroemFeatures(["a", "b"], n_components=50, gamma=0.5, random_state=0).fit(df).transform_array(df)
    rff = RandomFourierFeatures(["a", "b"], n_components=50, gamma=0.5, random_state=0).fit(df).transform_array(df)
    assert np.abs(nys @ nys.T - exact).mean() < np.abs(rff @ rff.T - exact).mean()


def test_linear_model_on_features_fits_nonlinear_target():
    df = _frame(800, seed=2)
    y = np.sin(2 * df["a"]) * np.cos(df["b"])
    train, test = df.iloc[:600], df.iloc[600:]
    for tr in (
        RandomFourierFeatures(["a", "b"], n_components=300, gamma=1.0, random_state=0),
        NystroemFeatures(["a", "b"], n_components=150, gamma=1.0, random_state=0),
    ):
        Ztr = tr.fit(train).transform_array(train)
        Zte = tr.transform_array(test)
        r2_kernel = Ridge(alpha=1e-3).fit(Ztr, y[:600]).score(Zte, y[600:])
        r2_linear = Ridge(alpha=1e-3).fit(train[["a", "b"]], y[:600]).score(test[["a", "b"]], y[600:])
        assert r2_kernel > 0.9 and r2_linear < 0.2


def test_default_gamma_and_validation():
    df = _frame(50)
    rff = RandomFourierFeatures(["a", "b"], n_components=5).fit(df)
    assert rff.gamma_ == 0.5
    with pytest.raises(ValueError):
        RandomFourierFeatures(["a"], n_components=0)
    with pytest.raises(ValueError):
        RandomFourierFeatures(["a"], kernel="poly")
    with pytest.raises(ValueError):
        RandomFourierFeatures(["a"], gamma=-1.0).fit(df)
    with pytest.raises(ValueError):
        RandomFourierFeatures(["zzz"]).fit(df)
    with pytest.raises(ValueError):
        RandomFourierFeatures([])
    bad = df.copy()
    bad.loc[0, "a"] = np.nan
    with pytest.raises(ValueError):
        NystroemFeatures(["a", "b"]).fit(bad)
    with pytest.raises(RuntimeError):
        NystroemFeatures(["a"]).transform(df)
    with pytest.raises(ValueError):
        RandomFourierFeatures(["a"], n_components=2, prefix="ca").fit_transform(df.rename(columns={"b": "ca_0"}))
