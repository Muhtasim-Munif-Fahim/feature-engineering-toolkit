"""Missing-value imputation transformers."""

from __future__ import annotations

import numpy as np
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


def nan_euclidean_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances that skip missing coordinates.

    For rows ``a`` and ``b`` with ``m`` of ``p`` coordinates observed in both,
    ``d(a, b) = sqrt(p / m * sum((a_j - b_j) ** 2 over shared j))``. The
    ``p / m`` factor rescales partial distances to the full dimension
    (Dixon, 1979; same convention as scikit-learn). With no shared coordinate
    the distance is ``nan``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("A and B must be 2-D with the same number of columns")
    p = A.shape[1]
    a_obs = ~np.isnan(A)
    b_obs = ~np.isnan(B)
    A0 = np.where(a_obs, A, 0.0)
    B0 = np.where(b_obs, B, 0.0)
    # sum over shared j of a^2 + b^2 - 2ab, built from masked products.
    sq = (
        (A0 ** 2) @ b_obs.T.astype(float)
        + a_obs.astype(float) @ (B0 ** 2).T
        - 2.0 * A0 @ B0.T
    )
    shared = a_obs.astype(float) @ b_obs.T.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist2 = np.where(shared > 0, np.maximum(sq, 0.0) * p / shared, np.nan)
    return np.sqrt(dist2)


class KNNImputer(Transformer):
    """Impute missing numeric values from the k nearest training rows.

    Distances use :func:`nan_euclidean_distances` over ``columns``, so rows
    with missing values can still be compared on the coordinates they share.
    For every missing cell, the donors are the ``n_neighbors`` closest
    training rows that have that column observed. Their values are averaged,
    uniformly or with ``1 / distance`` weights. If no donor shares any
    coordinate with the row, the cell falls back to the training column mean.

    Parameters
    ----------
    columns:
        Numeric columns used both for the distance and for imputation.
    n_neighbors:
        Number of donors per missing cell (default 5).
    weights:
        ``"uniform"`` (default) or ``"distance"``. With ``"distance"``,
        donors at distance 0 take all of the weight.
    add_indicator:
        If true, append a ``{column}__was_missing`` 0/1 column for every
        column that had missing values during ``fit``.
    """

    _WEIGHTS = ("uniform", "distance")

    def __init__(self, columns, n_neighbors=5, weights="uniform", add_indicator=False):
        self.columns = list(columns)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.add_indicator = add_indicator

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")
        return X[self.columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        if not self.columns:
            raise ValueError("columns must be non-empty")
        if (
            isinstance(self.n_neighbors, bool)
            or not isinstance(self.n_neighbors, (int, np.integer))
            or self.n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer")
        if self.weights not in self._WEIGHTS:
            raise ValueError(
                f"unknown weights: {self.weights!r}; expected one of {list(self._WEIGHTS)}"
            )
        M = self._matrix(X)
        if np.isinf(M).any():
            raise ValueError("KNNImputer does not accept infinite values")
        self.fit_X_ = M
        with np.errstate(invalid="ignore"):
            means = np.nanmean(np.where(np.isnan(M).all(axis=0), 0.0, M), axis=0)
        self.statistics_ = {c: float(v) for c, v in zip(self.columns, means)}
        self.indicator_columns_ = [
            c for c, has in zip(self.columns, np.isnan(M).any(axis=0)) if has
        ]
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        M = self._matrix(X)
        mask = np.isnan(M)
        out_M = M.copy()
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size:
            dist = nan_euclidean_distances(M[rows], self.fit_X_)
            fit_obs = ~np.isnan(self.fit_X_)
            for j, col in enumerate(self.columns):
                need = np.flatnonzero(mask[rows, j])
                if need.size == 0:
                    continue
                donors = np.flatnonzero(fit_obs[:, j])
                if donors.size == 0:
                    out_M[rows[need], j] = self.statistics_[col]
                    continue
                values = self.fit_X_[donors, j]
                for r in need:
                    d = dist[r, donors]
                    finite = np.flatnonzero(np.isfinite(d))
                    if finite.size == 0:
                        out_M[rows[r], j] = self.statistics_[col]
                        continue
                    k = min(int(self.n_neighbors), finite.size)
                    order = finite[np.argsort(d[finite], kind="stable")[:k]]
                    dk = d[order]
                    if self.weights == "distance":
                        zero = dk == 0.0
                        w = zero.astype(float) if zero.any() else 1.0 / dk
                    else:
                        w = np.ones_like(dk)
                    out_M[rows[r], j] = float(np.sum(w * values[order]) / np.sum(w))
        out = X.copy()
        for j, col in enumerate(self.columns):
            out[col] = out_M[:, j]
        if self.add_indicator:
            for col in self.indicator_columns_:
                out[f"{col}__was_missing"] = mask[:, self.columns.index(col)].astype(np.int64)
        return out


__all__ = ["NumericImputer", "CategoricalImputer", "KNNImputer", "nan_euclidean_distances"]
