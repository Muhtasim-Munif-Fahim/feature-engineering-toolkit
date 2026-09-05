"""Feature-selection transformers.

These selectors are column-wise and designed to run after encoding so every
column is numeric. They expose the same ``fit``/``transform`` interface, so they
compose inside :class:`~feature_engineering_kit.pipeline.FeatureEngineeringPipeline`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone as _clone_estimator
from sklearn.feature_selection import mutual_info_classif

from .base import Transformer


def _numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


class VarianceThreshold(Transformer):
    """Drop numeric columns whose variance is below ``threshold``.

    Constant (or near-constant) columns carry little signal and can destabilise
    estimators; ``threshold=0`` drops only exactly-constant columns.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        numeric = _numeric_columns(X)
        if numeric:
            variances = X[numeric].var(ddof=0)
        else:
            variances = pd.Series(dtype=float)
        self.drop_ = [c for c in numeric if float(variances[c]) <= self.threshold]
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


class CorrelationFilter(Transformer):
    """Drop one column from each pair of highly-correlated features.

    Parameters
    ----------
    threshold:
        Absolute correlation above which a pair is considered redundant.
    method:
        Correlation method (``"pearson"``, ``"spearman"``, or ``"kendall"``).
    """

    def __init__(self, threshold: float = 0.95, method: str = "pearson"):
        self.threshold = abs(float(threshold))
        self.method = method

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.drop_: list[str] = []
        numeric = _numeric_columns(X)
        if len(numeric) < 2:
            return None
        corr = X[numeric].corr(method=self.method)
        for i, candidate in enumerate(numeric):
            if candidate in self.drop_:
                continue
            for other in numeric[i + 1 :]:
                if other in self.drop_:
                    continue
                value = corr.loc[candidate, other]
                if pd.isna(value):
                    continue
                if abs(value) > self.threshold:
                    # Drop the later column of the pair; keep iteration stable.
                    self.drop_.append(other)
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


class MutualInfoSelection(Transformer):
    """Select the top-``k`` features by mutual information with the target.

    Parameters
    ----------
    target:
        Name used to label the target (the actual values come from ``y``).
    k:
        Number of top features to retain. If ``k`` >= number of numeric
        features, all numeric features are kept.
    random_state:
        Forwarded to :func:`sklearn.feature_selection.mutual_info_classif`.
    """

    def __init__(self, target: str, k: int = 10, random_state: int = 0):
        self.target = target
        self.k = int(k)
        self.random_state = int(random_state)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        if y is None:
            raise ValueError("MutualInfoSelection.fit requires y")
        numeric = _numeric_columns(X)
        if not numeric:
            self.drop_ = []
            return None
        y_arr = np.asarray(pd.Series(y).to_numpy()).ravel()
        scores = mutual_info_classif(
            X[numeric].to_numpy(), y_arr, random_state=self.random_state
        )
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        ordered = sorted(zip(numeric, scores), key=lambda t: t[1], reverse=True)
        self.selected_ = [c for c, _ in ordered[: self.k]]
        self.drop_ = [c for c in numeric if c not in self.selected_]
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


class SelectFromModel(Transformer):
    """Select features whose importance meets a threshold.

    The estimator is cloned and fit on the numeric columns of ``X`` and ``y`` at
    ``fit`` time; importances are read from ``feature_importances_`` (trees) or
    ``np.abs(coef_)`` (linear models).

    Parameters
    ----------
    target:
        Name used to label the target (actual values come from ``y``).
    estimator:
        A scikit-learn-compatible estimator supporting ``fit`` and either
        ``feature_importances_`` or ``coef_``.
    threshold:
        ``"median"`` (default), ``"mean"``, or a numeric cutoff.
    """

    def __init__(self, target: str, estimator, threshold: str | float = "median"):
        self.target = target
        self.estimator = estimator
        self.threshold = threshold

    def _importances(self, fitted, columns: list[str]) -> np.ndarray:
        if hasattr(fitted, "feature_importances_"):
            imp = np.asarray(fitted.feature_importances_, dtype=float)
        elif hasattr(fitted, "coef_"):
            coef = np.asarray(fitted.coef_, dtype=float)
            imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef).ravel()
        else:
            raise AttributeError("estimator exposes neither feature_importances_ nor coef_")
        if imp.shape[0] != len(columns):
            imp = np.resize(imp, len(columns))
        return imp

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        if y is None:
            raise ValueError("SelectFromModel.fit requires y")
        numeric = _numeric_columns(X)
        if not numeric:
            self.drop_ = []
            return None
        y_arr = np.asarray(pd.Series(y).to_numpy()).ravel()
        self.estimator_ = _clone_estimator(self.estimator)
        self.estimator_.fit(X[numeric].to_numpy(), y_arr)
        importances = self._importances(self.estimator_, numeric)
        kind: Literal["median", "mean"] = "median"
        if isinstance(self.threshold, str):
            kind = self.threshold
        if kind == "median":
            cutoff = float(np.median(importances))
        elif kind == "mean":
            cutoff = float(np.mean(importances))
        else:
            cutoff = float(self.threshold)
        self.drop_ = [
            c for c, imp in zip(numeric, importances) if float(imp) < cutoff
        ]
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


__all__ = [
    "VarianceThreshold",
    "CorrelationFilter",
    "MutualInfoSelection",
    "SelectFromModel",
]
