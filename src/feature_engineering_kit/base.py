"""Base class for DataFrame transformers in feature_engineering_kit."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Transformer(ABC):
    """Minimal fit/transform interface for ``pandas`` DataFrame transformers.

    Subclasses implement :meth:`_fit` and :meth:`_transform`. The public
    :meth:`fit` / :meth:`transform` / :meth:`fit_transform` methods handle
    fitted-state checks and column-name tracking. ``y`` is accepted and may be
    required by target-aware transformers (e.g. target encoding) but is optional
    with the default of ``None``.
    """

    columns_in_: list[str]
    columns_out_: list[str]

    def fit(self, X: pd.DataFrame, y=None) -> "Transformer":
        self.columns_in_ = list(X.columns)
        self._fit(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.columns_in_ = list(X.columns)
        self._fit(X, y)
        out = self._transform(X)
        self.columns_out_ = list(out.columns)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        out = self._transform(X)
        self.columns_out_ = list(out.columns)
        return out

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y=None) -> None:
        """Estimate transformer parameters from ``X`` (and optionally ``y``)."""

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformation and return a new DataFrame."""

    def get_feature_names_out(self) -> list[str] | None:
        return getattr(self, "columns_out_", None) or list(self.columns_in_)

    def _check_fitted(self) -> None:
        if "columns_in_" not in self.__dict__:
            cls = type(self).__name__
            raise RuntimeError(
                f"{cls} is not fitted yet; call fit() before transform()."
            )


__all__ = ["Transformer"]
