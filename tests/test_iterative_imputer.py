"""Tests for IterativeImputer and fit_bayesian_ridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer as SkIterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression

from feature_engineering_kit import IterativeImputer, build_preprocessing_pipeline, fit_bayesian_ridge

COLS = ["a", "b", "c", "d"]


def _frame(n: int = 150, seed: int = 0, frac: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": z + rng.normal(scale=0.3, size=n),
            "b": 2 * z + 1 + rng.normal(scale=0.3, size=n),
            "c": -z + rng.normal(scale=0.3, size=n),
            "d": rng.normal(size=n),
            "label": rng.choice(["x", "y"], size=n),
        }
    )
    M = df[COLS].to_numpy(copy=True)
    drop = rng.random(M.shape) < frac * np.array([0.5, 1.0, 1.5, 0.8])
    M[drop] = np.nan
    df[COLS] = M
    return df


@pytest.mark.parametrize("n,p", [(80, 3), (6, 10)])
def test_bayesian_ridge_matches_sklearn(n, p) -> None:
    rng = np.random.default_rng(n)
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 0.5 + rng.normal(scale=0.2, size=n)
    coef, b, alpha, lam = fit_bayesian_ridge(X, y)
    sk = BayesianRidge().fit(X, y)
    np.testing.assert_allclose(coef, sk.coef_, rtol=1e-8, atol=1e-10)
    assert b == pytest.approx(sk.intercept_, abs=1e-9)
    assert alpha == pytest.approx(sk.alpha_, rel=1e-8)
    assert lam == pytest.approx(sk.lambda_, rel=1e-8)


@pytest.mark.parametrize("order", ["ascending", "descending", "roman", "arabic"])
def test_matches_sklearn_bayesian_ridge(order) -> None:
    df = _frame()
    ours = IterativeImputer(COLS, imputation_order=order, max_iter=10)
    out = ours.fit_transform(df)
    sk = SkIterativeImputer(max_iter=10, imputation_order=order, random_state=0)
    ref = sk.fit_transform(df[COLS].to_numpy())
    np.testing.assert_allclose(out[COLS].to_numpy(), ref, rtol=1e-7, atol=1e-8)
    assert ours.n_iter_ == sk.n_iter_


def test_matches_sklearn_linear_and_transform_replay() -> None:
    train, test = _frame(seed=1), _frame(n=40, seed=2)
    ours = IterativeImputer(COLS, estimator="linear", max_iter=5, tol=0.0)
    ours.fit(train)
    sk = SkIterativeImputer(estimator=LinearRegression(), max_iter=5, tol=0.0, random_state=0)
    sk.fit(train[COLS].to_numpy())
    np.testing.assert_allclose(
        ours.transform(test)[COLS].to_numpy(), sk.transform(test[COLS].to_numpy()), rtol=1e-8, atol=1e-9
    )
    assert ours.n_iter_ == 5


def test_beats_mean_imputation_on_correlated_data() -> None:
    rng = np.random.default_rng(3)
    full = _frame(n=300, seed=3, frac=0.0)
    df = full.copy()
    mask = rng.random((300, 3)) < 0.2
    for j, c in enumerate(["a", "b", "c"]):
        df.loc[mask[:, j], c] = np.nan
    out = IterativeImputer(COLS).fit_transform(df)
    for j, c in enumerate(["a", "b", "c"]):
        m = mask[:, j]
        err_it = np.mean((out.loc[m, c] - full.loc[m, c]) ** 2)
        err_mean = np.mean((df[c].mean() - full.loc[m, c]) ** 2)
        assert err_it < 0.5 * err_mean


def test_observed_untouched_clipping_and_indicator() -> None:
    df = _frame(seed=4)
    out = IterativeImputer(COLS, min_value=-0.5, max_value=0.5, add_indicator=True).fit_transform(df)
    obs = df[COLS].notna().to_numpy()
    np.testing.assert_array_equal(out[COLS].to_numpy()[obs], df[COLS].to_numpy()[obs])
    imputed = out[COLS].to_numpy()[~obs]
    assert imputed.min() >= -0.5 and imputed.max() <= 0.5
    assert out["label"].tolist() == df["label"].tolist()
    for c in COLS:
        assert out[f"{c}__was_missing"].tolist() == df[c].isna().astype(int).tolist()
    assert not out[COLS].isna().any().any()


def test_converges_and_records_history() -> None:
    imp = IterativeImputer(COLS, max_iter=50, tol=1e-3)
    imp.fit(_frame(seed=5))
    assert 1 <= imp.n_iter_ < 50
    assert len(imp.history_) == imp.n_iter_


def test_all_missing_column_and_validation() -> None:
    df = _frame(seed=6)
    df["e"] = np.nan
    out = IterativeImputer(COLS + ["e"]).fit_transform(df)
    assert (out["e"] == 0.0).all()
    assert not out[COLS].isna().any().any()
    with pytest.raises(ValueError):
        IterativeImputer(COLS, estimator="rf").fit(df)
    with pytest.raises(ValueError):
        IterativeImputer(COLS, imputation_order="random").fit(df)
    with pytest.raises(ValueError):
        IterativeImputer(COLS, max_iter=-1).fit(df)
    with pytest.raises(KeyError):
        IterativeImputer(["zz"]).fit(df)
    with pytest.raises(RuntimeError):
        IterativeImputer(COLS).transform(df)


def test_pipeline_iterative_option() -> None:
    df = _frame(seed=7)
    pipe = build_preprocessing_pipeline(df.drop(columns=["label"]), numeric_impute="iterative")
    names = [n for n, _ in pipe.steps]
    assert "impute_numeric" in names
    step = dict(pipe.steps)["impute_numeric"]
    assert isinstance(step, IterativeImputer)
