"""Categorical feature encoders.

Each encoder is a :class:`~feature_engineering_kit.base.Transformer` that
replaces one or more object/string columns with numeric representations so the
engineered frame is consumable by scikit-learn estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer


class OneHotEncoder(Transformer):
    """One-hot encode categorical columns.

    Parameters
    ----------
    columns:
        Categorical columns to encode.
    drop:
        ``None`` (keep every category) or ``"first"`` (drop the first sorted
        category of each column to reduce multicollinearity).
    dtype:
        Output dtype for the indicator columns (default ``"int"``).
    """

    def __init__(self, columns, drop=None, dtype="int"):
        self.columns = list(columns)
        self.drop = drop
        self.dtype = dtype

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.categories_: dict[str, list[str]] = {}
        for col in self.columns:
            cats = sorted({str(v) for v in X[col].dropna().tolist()})
            if self.drop == "first" and cats:
                cats = cats[1:]
            self.categories_[col] = cats
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            vals = out[col].astype(object)
            for cat in self.categories_[col]:
                out[f"{col}__{cat}"] = (vals == cat).fillna(False).astype(self.dtype)
            out = out.drop(columns=[col])
        return out


class OrdinalEncoder(Transformer):
    """Ordinal-encode categorical columns to integer codes.

    Categories are sorted and mapped to ``1..k``; unseen or missing values map
    to ``0``.
    """

    def __init__(self, columns, dtype="int"):
        self.columns = list(columns)
        self.dtype = dtype

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.categories_: dict[str, list[str]] = {}
        self.mapping_: dict[str, dict[str, int]] = {}
        for col in self.columns:
            cats = sorted({str(v) for v in X[col].dropna().tolist()})
            self.categories_[col] = cats
            self.mapping_[col] = {cat: i + 1 for i, cat in enumerate(cats)}
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            mapping = self.mapping_[col]
            out[col] = out[col].astype(object).map(
                lambda v: mapping.get(str(v), 0)
            ).astype(self.dtype)
        return out


class TargetEncoder(Transformer):
    """Mean (target) encoding with Bayesian smoothing.

    Each category is replaced by a convex combination of the per-category target
    mean and the global target mean, controlled by ``smoothing``. A larger
    ``smoothing`` shrinks estimates harder toward the global mean and guards
    against high-cardinality / rare-category leakage.

    Parameters
    ----------
    columns:
        Categorical columns to encode.
    target:
        Name of the target column in ``y`` (when ``y`` is a DataFrame) or the
        target values (when ``y`` is array-like / Series).
    smoothing:
        Smoothing weight; ``0.0`` recovers the raw per-category mean.
    """

    def __init__(self, columns, target, smoothing=10.0):
        self.columns = list(columns)
        self.target = target
        self.smoothing = float(smoothing)

    def _as_target_series(self, y) -> pd.Series:
        if y is None:
            raise ValueError("TargetEncoder.fit requires y")
        if isinstance(y, pd.DataFrame):
            s = y[self.target]
        elif isinstance(y, pd.Series):
            s = y
        else:
            s = pd.Series(np.asarray(y).ravel())
        return s.reset_index(drop=True)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        target = self._as_target_series(y)
        self.global_mean_ = float(target.mean())
        self.maps_: dict[str, dict[str, float]] = {}
        for col in self.columns:
            s = X[col].astype(object).reset_index(drop=True)
            grouped = target.groupby(s, sort=False)
            means = grouped.mean()
            counts = grouped.size()
            smooth = (counts * means + self.smoothing * self.global_mean_) / (
                counts + self.smoothing
            )
            self.maps_[col] = {
                str(k): float(v) for k, v in smooth.items() if k is not None
            }
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            mapping = self.maps_[col]
            gm = self.global_mean_
            out[col] = (
                out[col].astype(object).map(lambda v: mapping.get(str(v), gm))
                .astype(float)
            )
        return out


__all__ = ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"]
