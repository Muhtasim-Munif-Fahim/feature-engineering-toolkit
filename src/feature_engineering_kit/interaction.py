"""Pairwise feature-interaction transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer

_VALID_HOW = {"product", "ratio", "sum", "diff"}


class InteractionFeatures(Transformer):
    """Add pairwise interaction columns to a DataFrame.

    Parameters
    ----------
    operations:
        Iterable of ``(col_a, col_b, how)`` triples where ``how`` is one of
        ``"product"``, ``"ratio"``, ``"sum"``, or ``"diff"``. The resulting
        column is named ``"{col_a}_{how}_{col_b}"``.
    """

    def __init__(self, operations):
        self.operations = [tuple(op) for op in operations]

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        for a, b, how in self.operations:
            if how not in _VALID_HOW:
                raise ValueError(f"unknown interaction kind: {how!r}; expected one of {sorted(_VALID_HOW)}")
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for a, b, how in self.operations:
            if a not in out.columns or b not in out.columns:
                raise KeyError(f"interaction columns not found: {a!r}, {b!r}")
            av = pd.to_numeric(out[a], errors="coerce").to_numpy(dtype=float)
            bv = pd.to_numeric(out[b], errors="coerce").to_numpy(dtype=float)
            if how == "product":
                values = av * bv
            elif how == "sum":
                values = av + bv
            elif how == "diff":
                values = av - bv
            else:  # ratio
                values = np.divide(
                    av, bv, out=np.zeros_like(av), where=bv != 0
                )
            out[f"{a}_{how}_{b}"] = values
        return out


__all__ = ["InteractionFeatures"]
