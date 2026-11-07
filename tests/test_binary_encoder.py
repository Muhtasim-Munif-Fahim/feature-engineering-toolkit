"""Tests for BinaryEncoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import BinaryEncoder, FeatureEngineeringPipeline


def _df():
    return pd.DataFrame(
        {
            "city": ["dhaka", "paris", "dhaka", None, "tokyo"],
            "plan": ["a", "b", "a", "c", "b"],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_shape_names_and_passthrough():
    enc = BinaryEncoder(columns=["city", "plan"])
    out = enc.fit_transform(_df())
    assert "city" not in out.columns and "plan" not in out.columns
    assert out["amount"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    # city: 3 levels + indicator -> 4 codes -> 2 bits; plan: 3 levels + ind -> 2 bits
    assert enc.n_bits_["city"] == 2
    assert enc.n_bits_["plan"] == 2
    city_cols = [c for c in out.columns if c.startswith("city__bin")]
    assert len(city_cols) == 2
    # bits are 0/1
    assert set(out[city_cols].to_numpy().ravel()).issubset({0.0, 1.0})


def test_same_category_same_bits_and_indicator_for_missing():
    enc = BinaryEncoder(columns=["city"]).fit(_df())
    out = enc.transform(pd.DataFrame({"city": ["dhaka", "dhaka", None, "never-seen"]}))
    bits = out.to_numpy()
    assert np.array_equal(bits[0], bits[1])
    # missing and unseen share indicator code 0 -> all-zero bits
    assert np.array_equal(bits[2], bits[3])
    assert np.all(bits[2] == 0.0)


def test_error_on_unknown():
    enc = BinaryEncoder(columns=["plan"], handle_unknown="error").fit(_df())
    with pytest.raises(ValueError, match="unknown"):
        enc.transform(pd.DataFrame({"plan": ["z"]}))


def test_zeros_unknown_keeps_fitted_at_zero_index():
    enc = BinaryEncoder(columns=["plan"], handle_unknown="zeros").fit(_df())
    # 3 levels -> codes 0,1,2 -> 2 bits; unknown -> zeros
    out = enc.transform(pd.DataFrame({"plan": ["b", "missing-level"]}))
    assert enc.n_bits_["plan"] == 2
    assert np.all(out.iloc[1].to_numpy() == 0.0)
    # "b" is not code 0 under sorted order (a=0,b=1,c=2)
    assert not np.all(out.iloc[0].to_numpy() == 0.0)


def test_rejects_bad_handle_unknown_and_empty_columns():
    with pytest.raises(ValueError):
        BinaryEncoder(columns=["x"], handle_unknown="drop")
    with pytest.raises(ValueError):
        BinaryEncoder(columns=[])


def test_pipeline_integration():
    pipe = FeatureEngineeringPipeline([("bin", BinaryEncoder(columns=["city", "plan"]))])
    out = pipe.fit_transform(_df())
    assert any(c.startswith("city__bin") for c in out.columns)
