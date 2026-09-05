"""Tests for pairwise interaction features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering_kit import InteractionFeatures


def _df():
    return pd.DataFrame({"age": [10.0, 20.0], "income": [1.0, 0.0]})


def test_product_and_sum() -> None:
    ix = InteractionFeatures(operations=[("age", "income", "product"), ("age", "income", "sum")])
    out = ix.fit_transform(_df())
    np.testing.assert_allclose(out["age_product_income"].to_numpy(), [10, 0])
    np.testing.assert_allclose(out["age_sum_income"].to_numpy(), [11, 20])


def test_ratio_handles_zero_denominator() -> None:
    ix = InteractionFeatures(operations=[("age", "income", "ratio")])
    out = ix.fit_transform(_df())
    # income == 0 -> ratio is 0, no inf/nan
    assert out["age_ratio_income"].tolist() == [10.0, 0.0]
    assert not np.isinf(out["age_ratio_income"].to_numpy()).any()
    assert not np.isnan(out["age_ratio_income"].to_numpy()).any()


def test_diff() -> None:
    ix = InteractionFeatures(operations=[("age", "income", "diff")])
    out = ix.fit_transform(_df())
    np.testing.assert_allclose(out["age_diff_income"].to_numpy(), [9, 20])


def test_unknown_how_raises() -> None:
    import pytest

    ix = InteractionFeatures(operations=[("age", "income", "power")])
    with pytest.raises(ValueError, match="unknown interaction kind"):
        ix.fit(_df())
