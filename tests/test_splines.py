"""Tests for SplineTransformer (B-spline basis features)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import SplineTransformer, bspline_basis


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.uniform(-3.0, 3.0, n), "z": rng.normal(size=n)})


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
def test_partition_of_unity_and_shape(degree, knots) -> None:
    df = _frame()
    st = SplineTransformer(columns=["x"], n_knots=6, degree=degree, knots=knots)
    out = st.fit_transform(df)
    names = [f"x__spline_{i}" for i in range(6 + degree - 1)]
    assert [c for c in out.columns if c.startswith("x__")] == names
    assert "x" not in out.columns
    block = out[names].to_numpy()
    assert np.all(block >= -1e-12)
    np.testing.assert_allclose(block.sum(axis=1), 1.0, atol=1e-10)
    # local support: at most degree + 1 non-zero basis values per row
    assert int((block > 1e-12).sum(axis=1).max()) <= degree + 1
    assert out["z"].tolist() == df["z"].tolist()


def test_linear_hat_functions_exact() -> None:
    df = pd.DataFrame({"x": [0.0, 0.5, 1.0, 1.25, 2.0]})
    out = SplineTransformer(columns=["x"], n_knots=3, degree=1).fit_transform(df)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.75, 0.25],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-12)


def test_matches_scipy_bspline() -> None:
    interpolate = pytest.importorskip("scipy.interpolate")
    df = _frame(n=50, seed=1)
    st = SplineTransformer(columns=["x"], n_knots=5, degree=3).fit(df)
    t = st.knots_["x"]
    x = np.linspace(st.base_knots_["x"][0], st.base_knots_["x"][-1], 37)
    ours = bspline_basis(x, t, 3)
    n_basis = len(t) - 4
    for i in range(n_basis):
        coef = np.zeros(n_basis)
        coef[i] = 1.0
        ref = interpolate.BSpline(t, coef, 3, extrapolate=False)(x)
        np.testing.assert_allclose(ours[:, i], np.nan_to_num(ref), atol=1e-10)


def test_spline_features_fit_nonlinear_signal() -> None:
    df = _frame(n=400, seed=2)
    y = np.sin(df["x"].to_numpy())
    feats = SplineTransformer(columns=["x"], n_knots=8).fit_transform(df[["x"]]).to_numpy()
    coef, *_ = np.linalg.lstsq(feats, y, rcond=None)
    spline_mse = float(np.mean((feats @ coef - y) ** 2))
    lin = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    lcoef, *_ = np.linalg.lstsq(lin, y, rcond=None)
    linear_mse = float(np.mean((lin @ lcoef - y) ** 2))
    assert spline_mse < 1e-4
    assert spline_mse < linear_mse / 100


def test_fit_knots_reused_on_test_and_constant_extrapolation() -> None:
    train = pd.DataFrame({"x": np.linspace(0.0, 10.0, 21)})
    st = SplineTransformer(columns=["x"], n_knots=4, degree=2).fit(train)
    test = pd.DataFrame({"x": [-5.0, 0.0, 10.0, 50.0]})
    out = st.transform(test).to_numpy()
    np.testing.assert_allclose(out[0], out[1], atol=1e-12)
    np.testing.assert_allclose(out[2], out[3], atol=1e-12)
    np.testing.assert_allclose(st.base_knots_["x"], [0.0, 10 / 3, 20 / 3, 10.0])


def test_continue_and_error_extrapolation() -> None:
    train = pd.DataFrame({"x": np.linspace(0.0, 1.0, 11)})
    cont = SplineTransformer(columns=["x"], n_knots=3, degree=1, extrapolation="continue").fit(train)
    out = cont.transform(pd.DataFrame({"x": [-0.25, 5.0]})).to_numpy()
    # Below the range the first hat decays linearly toward the padding knot
    # (-0.5); far above the padded support every basis value is zero.
    assert out[0, 0] == pytest.approx(0.5)
    assert np.all(out[1] == 0.0)
    err = SplineTransformer(columns=["x"], n_knots=3, extrapolation="error").fit(train)
    err.transform(pd.DataFrame({"x": [0.0, 1.0]}))
    with pytest.raises(ValueError, match="outside the fitted range"):
        err.transform(pd.DataFrame({"x": [1.5]}))


def test_include_bias_false_and_keep_original() -> None:
    df = _frame(n=30)
    out = SplineTransformer(columns=["x"], n_knots=4, degree=3, include_bias=False).fit_transform(df)
    assert [c for c in out.columns if c.startswith("x__")] == [f"x__spline_{i}" for i in range(5)]
    kept = SplineTransformer(columns=["x"], n_knots=4, keep_original=True).fit_transform(df)
    assert list(kept.columns[:2]) == ["x", "x__spline_0"]
    assert kept["x"].tolist() == df["x"].tolist()


def test_quantile_knots_follow_data_and_collapse_ties() -> None:
    df = pd.DataFrame({"x": np.r_[np.zeros(50), np.linspace(0.0, 1.0, 50)]})
    st = SplineTransformer(columns=["x"], n_knots=5, knots="quantile").fit(df)
    knots = st.base_knots_["x"]
    assert np.all(np.diff(knots) > 0)
    assert st.n_knots_["x"] < 5
    assert knots[0] == 0.0 and knots[-1] == 1.0


def test_explicit_knots_and_missing_values() -> None:
    df = pd.DataFrame({"x": [0.0, np.nan, 2.0, 4.0]})
    st = SplineTransformer(columns=["x"], knots=[0.0, 2.0, 4.0], degree=1)
    out = st.fit_transform(df)
    assert out.iloc[1].isna().all()
    np.testing.assert_allclose(out.iloc[2].to_numpy(), [0.0, 1.0, 0.0])


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="n_knots"):
        SplineTransformer(columns=["x"], n_knots=1)
    with pytest.raises(ValueError, match="degree"):
        SplineTransformer(columns=["x"], degree=-1)
    with pytest.raises(ValueError, match="knots"):
        SplineTransformer(columns=["x"], knots="kmeans")
    with pytest.raises(ValueError, match="strictly increasing"):
        SplineTransformer(columns=["x"], knots=[1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="extrapolation"):
        SplineTransformer(columns=["x"], extrapolation="periodic")
    with pytest.raises(ValueError, match="constant"):
        SplineTransformer(columns=["x"]).fit(pd.DataFrame({"x": [3.0, 3.0, 3.0]}))
    with pytest.raises(ValueError, match="non-numeric"):
        SplineTransformer(columns=["x"]).fit(pd.DataFrame({"x": ["a", "b"]}))
    with pytest.raises(KeyError):
        SplineTransformer(columns=["missing"]).fit(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(RuntimeError, match="not fitted"):
        SplineTransformer(columns=["x"]).transform(pd.DataFrame({"x": [1.0]}))


def test_works_in_pipeline() -> None:
    from feature_engineering_kit import FeatureEngineeringPipeline, StandardScaler

    df = _frame(n=40)
    pipe = FeatureEngineeringPipeline(
        steps=[("spline", SplineTransformer(columns=["x"], n_knots=4)), ("scale", StandardScaler(columns=["z"]))]
    )
    out = pipe.fit_transform(df)
    assert "x__spline_5" in out.columns
    np.testing.assert_allclose(pipe.transform(df).to_numpy(), out.to_numpy())
