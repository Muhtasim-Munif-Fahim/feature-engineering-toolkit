"""Numeric feature scalers.

Each scaler is column-wise and stores per-column statistics at ``fit`` time so
that the same statistics can be applied to a held-out/test frame at ``transform``
time, preventing leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer


def _to_float(series: pd.Series) -> pd.Series:
    return series.astype(float)


class StandardScaler(Transformer):
    """Standardize columns to zero mean and unit variance (``ddof=0``)."""

    def __init__(self, columns):
        self.columns = list(columns)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.mean_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        for col in self.columns:
            s = _to_float(X[col])
            std = float(s.std(ddof=0))
            self.mean_[col] = float(s.mean())
            self.scale_[col] = std if std > 0 else 1.0
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = (_to_float(out[col]) - self.mean_[col]) / self.scale_[col]
        return out


class MinMaxScaler(Transformer):
    """Scale columns into ``feature_range`` (default ``[0, 1]``)."""

    def __init__(self, columns, feature_range=(0.0, 1.0)):
        self.columns = list(columns)
        self.feature_range = tuple(feature_range)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        lo, hi = self.feature_range
        self.lo_: dict[str, float] = {}
        self.hi_: dict[str, float] = {}
        self.mn_: dict[str, float] = {}
        self.rng_: dict[str, float] = {}
        for col in self.columns:
            s = _to_float(X[col])
            mn, mx = float(s.min()), float(s.max())
            rng = mx - mn
            if rng == 0 or np.isnan(rng):
                rng = 1.0
            self.lo_[col] = lo
            self.hi_[col] = hi
            self.mn_[col] = mn
            self.rng_[col] = rng
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col in out.columns:
                span = self.hi_[col] - self.lo_[col]
                out[col] = (
                    (_to_float(out[col]) - self.mn_[col]) / self.rng_[col] * span
                    + self.lo_[col]
                )
        return out


class RobustScaler(Transformer):
    """Scale columns by median and inter-quartile range (robust to outliers)."""

    def __init__(self, columns, quantile_range=(25.0, 75.0)):
        self.columns = list(columns)
        self.quantile_range = tuple(quantile_range)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.center_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        lo_q = self.quantile_range[0] / 100.0
        hi_q = self.quantile_range[1] / 100.0
        for col in self.columns:
            s = _to_float(X[col])
            center = float(s.quantile(0.5))
            scale = float(s.quantile(hi_q)) - float(s.quantile(lo_q))
            self.center_[col] = center
            self.scale_[col] = scale if scale > 0 else 1.0
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = (_to_float(out[col]) - self.center_[col]) / self.scale_[col]
        return out


__all__ = ["StandardScaler", "MinMaxScaler", "RobustScaler"]
