"""Missing-value imputation transformers."""

from __future__ import annotations

import pandas as pd

from .base import Transformer


class NumericImputer(Transformer):
    """Impute missing numeric values with a central-statistic fill value.

    Parameters
    ----------
    columns:
        Numeric columns to impute.
    strategy:
        ``"median"`` (default) or ``"mean"``.
    """

    _STRATEGIES = {"median", "mean"}

    def __init__(self, columns, strategy="median"):
        self.columns = list(columns)
        self.strategy = strategy

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        if self.strategy not in self._STRATEGIES:
            raise ValueError(
                f"unknown numeric strategy: {self.strategy!r}; "
                f"expected one of {sorted(self._STRATEGIES)}"
            )
        self.statistics_: dict[str, float] = {}
        for col in self.columns:
            s = X[col]
            if self.strategy == "mean":
                value = float(s.mean())
            else:
                value = float(s.median())
            self.statistics_[col] = value
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = out[col].fillna(self.statistics_[col])
        return out


class CategoricalImputer(Transformer):
    """Impute missing categorical values with the mode or a constant.

    Parameters
    ----------
    columns:
        Categorical columns to impute.
    strategy:
        ``"most_frequent"`` (default, fills with the training-mode) or
        ``"constant"`` (fills with ``fill_value``).
    fill_value:
        Value to use with ``strategy="constant"`` (default ``"missing"``).
    """

    def __init__(self, columns, strategy="most_frequent", fill_value="missing"):
        self.columns = list(columns)
        self.strategy = strategy
        self.fill_value = fill_value

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.statistics_: dict[str, object] = {}
        for col in self.columns:
            s = X[col]
            if self.strategy == "most_frequent":
                mode = s.mode(dropna=True)
                value = mode.iloc[0] if not mode.empty else self.fill_value
            elif self.strategy == "constant":
                value = self.fill_value
            else:
                raise ValueError(
                    f"unknown categorical strategy: {self.strategy!r}; "
                    "expected 'most_frequent' or 'constant'"
                )
            self.statistics_[col] = value
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = out[col].fillna(self.statistics_[col])
        return out


__all__ = ["NumericImputer", "CategoricalImputer"]
