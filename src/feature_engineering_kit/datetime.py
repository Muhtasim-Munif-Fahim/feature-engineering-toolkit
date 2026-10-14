"""Datetime feature extraction transformer.

Converts a timestamp-like column into its calendar / cyclical components so
that models can exploit temporal structure directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer

_EXTRACTORS: dict[str, callable] = {
    "hour": lambda s: s.dt.hour,
    "dayofweek": lambda s: s.dt.dayofweek,
    "day": lambda s: s.dt.day,
    "month": lambda s: s.dt.month,
    "quarter": lambda s: s.dt.quarter,
    "year": lambda s: s.dt.year,
    "is_weekend": lambda s: (s.dt.dayofweek >= 5),
    "is_month_start": lambda s: s.dt.is_month_start,
    "is_month_end": lambda s: s.dt.is_month_end,
    "weekofyear": lambda s: s.dt.isocalendar().week,
}


class DatetimeExtractor(Transformer):
    """Extract calendar features from a datetime column.

    Parameters
    ----------
    column:
        Name of the datetime column to transform.
    drop_original:
        If ``True`` (default), drop the source column after extraction.
    features:
        Iterable of feature names to extract. Must be a subset of the keys of
        ``DatetimeExtractor.available_features()``.
    """

    def __init__(self, column, drop_original=True, features=None):
        self.column = column
        self.drop_original = drop_original
        self.features = list(features) if features else list(_EXTRACTORS)

    @staticmethod
    def available_features() -> list[str]:
        return list(_EXTRACTORS)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        unknown = [f for f in self.features if f not in _EXTRACTORS]
        if unknown:
            raise ValueError(
                f"unknown datetime features: {unknown}; "
                f"expected a subset of {self.available_features()}"
            )
        if self.column not in X.columns:
            raise KeyError(f"column '{self.column}' not found during fit")
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.column not in out.columns:
            raise KeyError(f"column '{self.column}' not found during transform")
        ts = pd.to_datetime(out[self.column], errors="coerce")
        for feature in self.features:
            values = _EXTRACTORS[feature](ts)
            if feature == "weekofyear":
                values = values.astype("float64")
            else:
                values = values.astype("float64")
            out[f"{self.column}__{feature}"] = values
        if self.drop_original:
            out = out.drop(columns=[self.column])
        return out




class CyclicalEncoder(Transformer):
    """Map a periodic numeric column to sine / cosine features.

    Models cannot treat hour 23 and hour 0 as adjacent when the raw integer
    is used. Encoding ``x`` on a known period ``P`` as

        sin(2 π x / P),  cos(2 π x / P)

    places neighbouring values next to each other on the unit circle. Typical
    periods: hour→24, dayofweek→7, month→12, day-of-year→365.

    Parameters
    ----------
    columns:
        Column names to encode. Each becomes ``{col}__sin`` and ``{col}__cos``.
    period:
        Shared period applied to every column, or a mapping of column → period.
        Must be a positive finite number.
    drop_original:
        If ``True`` (default), drop the source columns after encoding.
    """

    def __init__(self, columns, period, drop_original: bool = True):
        if not columns:
            raise ValueError("columns must be a non-empty sequence")
        self.columns = list(columns)
        if isinstance(period, dict):
            missing = [c for c in self.columns if c not in period]
            if missing:
                raise ValueError(f"period mapping missing columns: {missing}")
            self.period = {c: float(period[c]) for c in self.columns}
        else:
            p = float(period)
            self.period = {c: p for c in self.columns}
        for col, p in self.period.items():
            if not (p > 0) or p != p or p == float("inf"):
                raise ValueError(f"period for '{col}' must be a positive finite number")
        self.drop_original = bool(drop_original)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"columns not found during fit: {missing}")
        self.period_ = dict(self.period)
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        missing = [c for c in self.columns if c not in out.columns]
        if missing:
            raise KeyError(f"columns not found during transform: {missing}")
        for column in self.columns:
            period = self.period_[column]
            values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
            angle = 2.0 * np.pi * values / period
            out[f"{column}__sin"] = np.sin(angle)
            out[f"{column}__cos"] = np.cos(angle)
        if self.drop_original:
            out = out.drop(columns=list(self.columns))
        return out


__all__ = ["DatetimeExtractor", "CyclicalEncoder"]
