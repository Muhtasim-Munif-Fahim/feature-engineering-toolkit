"""Tests for categorical encoders."""

from __future__ import annotations

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
    enc = TargetEncoder(columns=["x"], target="y", smoothing=0.0)
    df = _df(x=["a", "a", "b", "b"], y=[1, 0, 1, 1])
    out = enc.fit_transform(df, df["y"])
    assert out["x"].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert enc.global_mean_ == pytest.approx(0.75)


def test_target_encoder_unseen_falls_back_to_global_mean() -> None:
    enc = TargetEncoder(columns=["x"], target="y", smoothing=0.0)
    df = _df(x=["a", "a", "b", "b"], y=[1, 0, 1, 1])
    enc.fit(df[["x"]], df["y"])
    out = enc.transform(_df(x=["c"]))
    assert out["x"].tolist() == pytest.approx([0.75])
