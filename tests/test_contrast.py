"""Tests for classical contrast coding (ContrastEncoder / contrast_matrix)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import ContrastEncoder, contrast_matrix

LEVELS = ["a", "b", "c", "d"]
MEANS = np.array([2.0, 5.0, 3.0, 10.0])


def _data(n_per=(5, 9, 6, 12)):
    rows = []
    for level, n in zip(LEVELS, n_per):
        rows += [level] * n
    df = pd.DataFrame({"g": rows, "x": np.arange(len(rows), dtype=float)})
    y = MEANS[[LEVELS.index(v) for v in rows]]
    return df, y


def _coefs(scheme):
    df, y = _data()
    enc = ContrastEncoder(["g"], scheme=scheme)
    Z = enc.fit_transform(df).drop(columns=["x"]).to_numpy()
    design = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta, enc


def test_treatment_coefficients_are_differences_from_reference():
    beta, enc = _coefs("treatment")
    assert beta[0] == pytest.approx(MEANS[0])
    assert beta[1:] == pytest.approx(MEANS[1:] - MEANS[0])
    assert enc.feature_names_["g"] == ["g__b", "g__c", "g__d"]


def test_sum_coding_deviations_from_unweighted_grand_mean():
    beta, enc = _coefs("sum")  # unbalanced groups: still the unweighted mean
    assert beta[0] == pytest.approx(MEANS.mean())
    assert beta[1:] == pytest.approx(MEANS[:-1] - MEANS.mean())
    assert enc.feature_names_["g"] == ["g__sum_a", "g__sum_b", "g__sum_c"]


def test_helmert_contrasts_level_against_previous_mean():
    beta, _ = _coefs("helmert")
    assert beta[0] == pytest.approx(MEANS.mean())
    for j in range(1, 4):
        assert beta[j] == pytest.approx((MEANS[j] - MEANS[:j].mean()) / (j + 1))


def test_backward_difference_gives_successive_differences():
    beta, enc = _coefs("backward_difference")
    assert beta[0] == pytest.approx(MEANS.mean())
    assert beta[1:] == pytest.approx(np.diff(MEANS))
    assert enc.feature_names_["g"][0] == "g__diff_b_vs_a"


def test_polynomial_matches_r_contr_poly_and_is_orthonormal():
    mat = contrast_matrix("polynomial", 4)
    expected = np.array(
        [
            [-0.6708204, 0.5, -0.2236068],
            [-0.2236068, -0.5, 0.6708204],
            [0.2236068, -0.5, -0.6708204],
            [0.6708204, 0.5, 0.2236068],
        ]
    )
    assert mat == pytest.approx(expected, abs=1e-6)
    assert mat.T @ mat == pytest.approx(np.eye(3))
    assert mat.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)
    # a purely linear trend loads only on the linear column
    df = pd.DataFrame({"dose": ["lo", "mid", "hi", "max"] * 3})
    enc = ContrastEncoder(["dose"], scheme="polynomial", categories={"dose": ["lo", "mid", "hi", "max"]})
    Z = enc.fit_transform(df).to_numpy()
    y = 1.0 + 2.0 * df["dose"].map({"lo": 0, "mid": 1, "hi": 2, "max": 3}).to_numpy()
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z]), y, rcond=None)
    assert beta[1] > 0 and beta[2:] == pytest.approx([0.0, 0.0], abs=1e-10)
    assert list(enc.coding_["dose"].columns) == ["dose__poly_linear", "dose__poly_quadratic", "dose__poly_cubic"]
    # unequally spaced scores: linear column is proportional to centred scores
    s = np.array([1.0, 2.0, 4.0, 8.0])
    lin = contrast_matrix("polynomial", 4, scores=s)[:, 0]
    assert lin == pytest.approx((s - s.mean()) / np.linalg.norm(s - s.mean()))


def test_contrast_matrix_edge_cases():
    assert contrast_matrix("helmert", 1).shape == (1, 0)
    for scheme in ("treatment", "sum", "helmert", "backward_difference", "polynomial"):
        mat = contrast_matrix(scheme, 6)
        assert mat.shape == (6, 5)
        assert np.linalg.matrix_rank(np.column_stack([np.ones(6), mat])) == 6
    with pytest.raises(ValueError):
        contrast_matrix("deviation", 3)
    with pytest.raises(ValueError):
        contrast_matrix("polynomial", 3, scores=[1, 1, 2])
    with pytest.raises(ValueError):
        contrast_matrix("polynomial", 3, scores=[1, 2])


def test_unknown_missing_and_per_column_schemes():
    train = pd.DataFrame({"g": ["a", "b", "c", None], "h": ["x", "y", "x", "y"], "v": [1, 2, 3, 4]})
    enc = ContrastEncoder(["g", "h"], scheme={"g": "sum", "h": "treatment"})
    out = enc.fit_transform(train)
    assert list(out.columns) == ["v", "g__sum_a", "g__sum_b", "h__y"]
    assert out.iloc[3][["g__sum_a", "g__sum_b"]].tolist() == [0.0, 0.0]
    test = pd.DataFrame({"g": ["zzz", "c"], "h": ["y", "x"], "v": [0, 0]})
    res = enc.transform(test)
    assert res[["g__sum_a", "g__sum_b"]].to_numpy().tolist() == [[0.0, 0.0], [-1.0, -1.0]]
    assert res["h__y"].tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match="unknown"):
        ContrastEncoder(["g"], handle_unknown="error").fit(train.dropna()).transform(test)
    with pytest.raises(ValueError, match="missing"):
        ContrastEncoder(["g"], handle_missing="error").fit(train)
    cat = ContrastEncoder(["g"], scheme="treatment", handle_missing="category").fit(train)
    assert cat.categories_["g"] == ["a", "b", "c", "__missing__"]
    assert cat.transform(train)["g____missing__"].tolist() == [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        ContrastEncoder(["g"], scheme="bogus").fit(train)
    with pytest.raises(ValueError, match="duplicates"):
        ContrastEncoder(["g"], categories={"g": ["a", "a"]}).fit(train)
    with pytest.raises(RuntimeError):
        ContrastEncoder(["g"]).transform(train)


def test_explicit_order_and_single_level_column():
    df = pd.DataFrame({"size": ["S", "M", "L", "M"], "k": ["only"] * 4})
    enc = ContrastEncoder(["size", "k"], scheme="backward_difference", categories={"size": ["S", "M", "L"]})
    out = enc.fit_transform(df)
    assert list(out.columns) == ["size__diff_M_vs_S", "size__diff_L_vs_M"]
    assert enc.get_feature_names_out() == ["size__diff_M_vs_S", "size__diff_L_vs_M"]
