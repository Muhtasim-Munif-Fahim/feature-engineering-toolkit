"""Datetime feature extraction transformer.

Converts a timestamp-like column into its calendar / cyclical components so
that models can exploit temporal structure directly.
"""

from __future__ import annotations

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


__all__ = ["DatetimeExtractor"]
