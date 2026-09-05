"""Polynomial feature expansion (products of feature powers)."""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

import numpy as np
import pandas as pd

from .base import Transformer


class PolynomialFeatures(Transformer):
    """Generate polynomial combinations of the given columns.

    Unlike an scikit-learn-style replacement, this transformer *appends* new
    columns to the frame (the originals are retained) and only adds terms of
    degree >= 2, so it composes naturally with the other transformers.

    Parameters
    ----------
    columns:
        Numeric columns to expand.
    degree:
        Maximum degree of the generated combinations (>= 1).
    interaction_only:
        If ``True``, only combinations of *distinct* columns are generated
        (no squared/cubed terms).
    include_bias:
        If ``True``, prepend a constant ``1.0`` bias column.
    """

    def __init__(self, columns, degree=2, interaction_only=False, include_bias=False):
        self.columns = list(columns)
        self.degree = int(degree)
        self.interaction_only = bool(interaction_only)
        self.include_bias = bool(include_bias)

    def _build_combinations(self) -> list[tuple[str, ...]]:
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        combos: list[tuple[str, ...]] = []
        source = combinations if self.interaction_only else combinations_with_replacement
        for deg in range(2, self.degree + 1):
            for combo in source(self.columns, deg):
                combos.append(combo)
        return combos

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"columns not found during fit: {missing}")
        self.combinations_ = self._build_combinations()
        self.output_columns_ = list(X.columns)
        if self.include_bias:
            self.output_columns_.append("poly__bias")
        for combo in self.combinations_:
            self.output_columns_.append("_x_".join(combo))
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.include_bias:
            out["poly__bias"] = 1.0
        for combo in self.combinations_:
            name = "_x_".join(combo)
            values = np.ones(len(out), dtype=float)
            for col in combo:
                values = values * pd.to_numeric(out[col], errors="coerce").to_numpy()
            out[name] = values
        return out


__all__ = ["PolynomialFeatures"]
