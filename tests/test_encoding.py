"""Tests for categorical encoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import OneHotEncoder, OrdinalEncoder, TargetEncoder


def _df(**cols):
    return pd.DataFrame(cols)


def test_onehot_creates_indicator_columns() -> None:
    enc = OneHotEncoder(columns=["color"])
    df = _df(color=["red", "blue", "red", "green", "blue"])
    out = enc.fit_transform(df)
    expected_cols = ["color__blue", "color__green", "color__red"]
    assert [c for c in out.columns if c.startswith("color__")] == expected_cols
    # blue -> color__blue == 1 for the blue rows
    assert out["color__blue"].tolist() == [0, 1, 0, 0, 1]
    assert out["color__red"].tolist() == [1, 0, 1, 0, 0]
    assert "color" not in out.columns


def test_onehot_drop_first_reduces_columns() -> None:
    enc = OneHotEncoder(columns=["color"], drop="first")
    df = _df(color=["red", "blue", "red", "green", "blue"])
    out = enc.fit_transform(df)
    cols = [c for c in out.columns if c.startswith("color__")]
    assert cols == ["color__green", "color__red"]


def test_onehot_unseen_category_is_all_zeros() -> None:
    enc = OneHotEncoder(columns=["color"])
    train = _df(color=["red", "blue", "red"])
    enc.fit(train)
    test = _df(color=["red", "purple"])
    out = enc.transform(test)
    assert out["color__red"].tolist() == [1, 0]
    assert out["color__blue"].tolist() == [0, 0]
    assert "color__purple" not in out.columns


def test_ordinal_encoder_codes_categories() -> None:
    enc = OrdinalEncoder(columns=["size"])
    df = _df(size=["small", "large", "medium", "small"])
    out = enc.fit_transform(df)
    # categories sorted -> large=1, medium=2, small=3
    assert out["size"].tolist() == [3, 1, 2, 3]
    assert enc.categories_["size"] == ["large", "medium", "small"]


def test_ordinal_encoder_unseen_maps_to_zero() -> None:
    enc = OrdinalEncoder(columns=["size"])
    enc.fit(_df(size=["small", "large"]))
    out = enc.transform(_df(size=["medium"]))
    assert out["size"].tolist() == [0]


def test_target_encoder_requires_y() -> None:
    enc = TargetEncoder(columns=["x"], target="y")
    with pytest.raises(ValueError, match="requires y"):
        enc.fit(_df(x=["a", "b"]))


def test_target_encoder_smoothing_zero_recovers_group_mean() -> None:
    enc = TargetEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    df = _df(x=["a", "a", "b", "b"], y=[1, 0, 1, 1])
    out = enc.fit_transform(df[["x"]], df["y"])
    assert out["x"].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert enc.global_mean_ == pytest.approx(0.75)


def test_target_encoder_global_maps_ignore_cv() -> None:
    """``fit`` always stores the full-data mapping used by ``transform``."""
    enc = TargetEncoder(columns=["x"], target="y", smoothing=0.0, cv=5, random_state=0)
    df = _df(x=["a", "a", "b", "b"], y=[1, 0, 1, 1])
    enc.fit(df[["x"]], df["y"])
    out = enc.transform(df[["x"]])
    assert out["x"].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_target_encoder_rejects_invalid_cv() -> None:
    with pytest.raises(ValueError, match="cv must be None"):
        TargetEncoder(columns=["x"], target="y", cv=1)


def test_target_encoder_unseen_falls_back_to_global_mean() -> None:
    enc = TargetEncoder(columns=["x"], target="y", smoothing=0.0)
    df = _df(x=["a", "a", "b", "b"], y=[1, 0, 1, 1])
    enc.fit(df[["x"]], df["y"])
    out = enc.transform(_df(x=["c"]))
    assert out["x"].tolist() == pytest.approx([0.75])


def test_oof_fit_transform_matches_complementary_fold_means() -> None:
    """With ``shuffle=False`` and a continuous target, encodings equal the other fold's group mean."""
    enc = TargetEncoder(
        columns=["x"], target="y", smoothing=0.0, cv=2, shuffle=False
    )
    # Distinct y values so the splitter uses KFold (not StratifiedKFold).
    X = _df(x=["a", "a", "b", "a", "b", "b"])
    y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    out = enc.fit_transform(X, y)["x"].tolist()
    # KFold(n_splits=2, shuffle=False) on 6 rows:
    #   fold 0 test [0,1,2] encoded from rows [3,4,5]
    #     a (row 3, y=0.4) -> 0.4; b (rows 4-5, y=0.5,0.6) -> 0.55
    #   fold 1 test [3,4,5] encoded from rows [0,1,2]
    #     a (rows 0-1, y=0.1,0.2) -> 0.15; b (row 2, y=0.3) -> 0.3
    assert out == pytest.approx([0.4, 0.4, 0.55, 0.15, 0.3, 0.3])
    # transform() still uses the full-data mapping.
    global_out = enc.transform(X)["x"].tolist()
    a_mean = (0.1 + 0.2 + 0.4) / 3
    b_mean = (0.3 + 0.5 + 0.6) / 3
    assert global_out == pytest.approx([a_mean, a_mean, b_mean, a_mean, b_mean, b_mean])


def test_oof_fit_transform_differs_from_global_transform_on_train() -> None:
    enc = TargetEncoder(
        columns=["x"], target="y", smoothing=0.0, cv=2, shuffle=False
    )
    X = _df(x=["a", "a", "b", "a", "b", "b"])
    y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    oof = enc.fit_transform(X, y)["x"].to_numpy()
    global_enc = enc.transform(X)["x"].to_numpy()
    assert not np.allclose(oof, global_enc)
    naive = TargetEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    naive.fit(X, y)
    np.testing.assert_allclose(global_enc, naive.transform(X)["x"].to_numpy())


def test_oof_leaks_less_than_naive_fit_on_all() -> None:
    """A unique-id column is a textbook leakage trap.

    Naive fit-on-all mean encoding copies the row's own label (correlation 1).
    Out-of-fold encoding never sees that row's target, so the encoded column
    is far less informative about ``y``.
    """
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n).astype(float)
    X = pd.DataFrame({"x": [f"id_{i}" for i in range(n)]})

    naive = TargetEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    oof = TargetEncoder(
        columns=["x"], target="y", smoothing=0.0, cv=5, shuffle=True, random_state=0
    )
    naive_enc = naive.fit_transform(X, y)["x"].to_numpy()
    oof_enc = oof.fit_transform(X, y)["x"].to_numpy()

    np.testing.assert_allclose(naive_enc, y)
    naive_corr = float(np.corrcoef(naive_enc, y)[0, 1])
    oof_corr = float(np.corrcoef(oof_enc, y)[0, 1])
    assert naive_corr == pytest.approx(1.0)
    assert abs(oof_corr) < 0.2
    assert abs(oof_corr) < naive_corr
    # Each unique category is unseen in other folds, so OOF values are the
    # complementary-fold global mean — never the row's own label.
    assert not np.allclose(oof_enc, y)


def test_oof_held_out_row_does_not_use_its_own_target() -> None:
    """Leave-one-out (cv == n, no shuffle): encoding equals the other rows' mean."""
    X = _df(x=["a", "a", "a", "a"])
    # Four distinct values so KFold (not stratified) can emit 4 folds.
    y = pd.Series([1.0, 0.0, 0.5, 0.25])
    enc = TargetEncoder(
        columns=["x"], target="y", smoothing=0.0, cv=4, shuffle=False
    )
    out = enc.fit_transform(X, y)["x"].to_numpy()
    expected = np.array(
        [
            (0.0 + 0.5 + 0.25) / 3,
            (1.0 + 0.5 + 0.25) / 3,
            (1.0 + 0.0 + 0.25) / 3,
            (1.0 + 0.0 + 0.5) / 3,
        ]
    )
    np.testing.assert_allclose(out, expected)
    assert not np.allclose(out, y.to_numpy())


def test_target_encoder_missing_column_raises() -> None:
    enc = TargetEncoder(columns=["x"], target="y", cv=None)
    enc.fit(_df(x=["a", "b"]), pd.Series([0, 1]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.transform(_df(z=["a"]))
