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


def _center(X: np.ndarray, y: np.ndarray):
    x_off = X.mean(axis=0)
    y_off = float(y.mean())
    return X - x_off, y - y_off, x_off, y_off


def fit_bayesian_ridge(
    X,
    y,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
):
    """Bayesian ridge regression by evidence maximization (MacKay 1992).

    Gamma priors on the noise precision ``alpha`` and the weight precision
    ``lambda`` give the fixed-point updates of Tipping (2001). The algorithm
    and defaults follow scikit-learn's ``BayesianRidge``, with an intercept.
    Returns ``(coef, intercept, alpha, lambda)``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    Xc, yc, x_off, y_off = _center(X, y)
    eps = np.finfo(np.float64).eps
    alpha = 1.0 / (np.var(yc) + eps)
    lam = 1.0
    XT_y = Xc.T @ yc
    U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2

    def update(alpha, lam):
        if n > p:
            coef = Vh.T @ ((Vh / (ev + lam / alpha)[:, None]) @ XT_y)
        else:
            coef = Xc.T @ ((U / (ev + lam / alpha)[None, :]) @ (U.T @ yc))
        return coef, float(np.sum((yc - Xc @ coef) ** 2))

    coef_old = None
    for it in range(max_iter):
        coef, rss = update(alpha, lam)
        gamma = float(np.sum((alpha * ev) / (lam + alpha * ev)))
        lam = (gamma + 2 * lambda_1) / (float(np.sum(coef ** 2)) + 2 * lambda_2)
        alpha = (n - gamma + 2 * alpha_1) / (rss + 2 * alpha_2)
        if it != 0 and np.sum(np.abs(coef_old - coef)) < tol:
            break
        coef_old = coef.copy()
    coef, _ = update(alpha, lam)
    return coef, y_off - float(x_off @ coef), float(alpha), float(lam)


def _fit_linear(X, y):
    Xc, yc, x_off, y_off = _center(np.asarray(X, float), np.asarray(y, float))
    coef = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    return coef, y_off - float(x_off @ coef)


class IterativeImputer(Transformer):
    """Multivariate (MICE-style) imputation by chained regressions.

    Missing cells start at the training column mean. Each round then visits
    every column in turn and regresses it on all other columns, using the rows
    where it is observed, and overwrites its missing cells with the
    predictions. Updates are Gauss-Seidel: later columns see the new values.
    Rounds stop after ``max_iter`` or when
    ``||X_t - X_{t-1}||_inf < tol * max|X_observed|``. The norm is the
    matrix infinity norm, i.e. the largest absolute row sum.
    This follows van Buuren & Groothuis-Oudshoorn (2011) and matches
    scikit-learn's ``IterativeImputer`` with ``sample_posterior=False``.

    Parameters
    ----------
    columns:
        Numeric columns used as predictors and imputed.
    estimator:
        ``"bayesian_ridge"`` (default; evidence-maximized ridge) or ``"linear"``
        (ordinary least squares).
    max_iter, tol:
        Round limit and relative convergence tolerance.
    imputation_order:
        ``"ascending"`` (fewest missing first, default), ``"descending"``,
        ``"roman"`` (column order) or ``"arabic"`` (reverse column order).
    min_value, max_value:
        Optional clipping bounds for imputed values.
    add_indicator:
        Append ``{column}__was_missing`` 0/1 columns for columns that had
        missing values during ``fit``.

    ``transform`` replays the fitted regression sequence, so new data is
    imputed with the models learned during ``fit``. Columns that were
    entirely missing at fit time are filled with 0.
    """

    _ESTIMATORS = ("bayesian_ridge", "linear")
    _ORDERS = ("ascending", "descending", "roman", "arabic")

    def __init__(
        self,
        columns,
        estimator="bayesian_ridge",
        max_iter=10,
        tol=1e-3,
        imputation_order="ascending",
        min_value=None,
        max_value=None,
        add_indicator=False,
    ):
        self.columns = list(columns)
        self.estimator = estimator
        self.max_iter = max_iter
        self.tol = tol
        self.imputation_order = imputation_order
        self.min_value = min_value
        self.max_value = max_value
        self.add_indicator = add_indicator

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")
        return X[self.columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    def _order(self, frac: np.ndarray) -> np.ndarray:
        if self.imputation_order == "ascending":
            return np.argsort(frac, kind="mergesort")
        if self.imputation_order == "descending":
            return np.argsort(frac, kind="mergesort")[::-1]
        if self.imputation_order == "roman":
            return np.arange(frac.size)
        return np.arange(frac.size)[::-1]

    def _clip(self, v: np.ndarray) -> np.ndarray:
        lo = -np.inf if self.min_value is None else self.min_value
        hi = np.inf if self.max_value is None else self.max_value
        return np.clip(v, lo, hi)

    def _fit_one(self, Xtr, ytr):
        if self.estimator == "linear":
            return _fit_linear(Xtr, ytr)
        coef, b, _, _ = fit_bayesian_ridge(Xtr, ytr)
        return coef, b

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        if not self.columns:
            raise ValueError("columns must be non-empty")
        if self.estimator not in self._ESTIMATORS:
            raise ValueError(
                f"unknown estimator: {self.estimator!r}; expected one of {list(self._ESTIMATORS)}"
            )
        if self.imputation_order not in self._ORDERS:
            raise ValueError(
                f"unknown imputation_order: {self.imputation_order!r}; "
                f"expected one of {list(self._ORDERS)}"
            )
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer")
        if not self.tol >= 0:
            raise ValueError("tol must be >= 0")
        M = self._matrix(X)
        if np.isinf(M).any():
            raise ValueError("IterativeImputer does not accept infinite values")
        mask = np.isnan(M)
        all_missing = mask.all(axis=0)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(np.where(all_missing, 0.0, M), axis=0)
        means = np.where(all_missing, 0.0, means)
        self.statistics_ = {c: float(v) for c, v in zip(self.columns, means)}
        self.indicator_columns_ = [c for c, has in zip(self.columns, mask.any(axis=0)) if has]
        self._means = means
        active = np.flatnonzero(~all_missing)
        Xt = np.where(mask, means[None, :], M)
        self.imputation_sequence_: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        self.n_iter_ = 0
        self.history_: list[float] = []
        if active.size < 2 or not mask[:, active].any() or self.max_iter == 0:
            self._fit_out = Xt
            return None
        order = active[self._order(mask[:, active].mean(axis=0))]
        obs = M[~mask]
        tol_abs = self.tol * float(np.max(np.abs(obs))) if obs.size else 0.0
        for it in range(int(self.max_iter)):
            prev = Xt.copy()
            for j in order:
                others = active[active != j]
                rows = ~mask[:, j]
                coef, b = self._fit_one(Xt[rows][:, others], Xt[rows, j])
                self.imputation_sequence_.append((int(j), others, coef, b))
                miss = mask[:, j]
                if miss.any():
                    Xt[miss, j] = self._clip(Xt[miss][:, others] @ coef + b)
            self.n_iter_ = it + 1
            change = float(np.linalg.norm(Xt - prev, ord=np.inf, axis=None))
            self.history_.append(change)
            if change < tol_abs:
                break
        self._fit_out = Xt
        return None

    def _impute(self, M: np.ndarray) -> np.ndarray:
        mask = np.isnan(M)
        Xt = np.where(mask, self._means[None, :], M)
        for j, others, coef, b in self.imputation_sequence_:
            miss = mask[:, j]
            if miss.any():
                Xt[miss, j] = self._clip(Xt[miss][:, others] @ coef + b)
        return Xt

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.columns_in_ = list(X.columns)
        self._fit(X, y)
        out = self._frame(X, self._fit_out, np.isnan(self._matrix(X)))
        self.columns_out_ = list(out.columns)
        return out

    def _frame(self, X: pd.DataFrame, Xt: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
        out = X.copy()
        for j, col in enumerate(self.columns):
            out[col] = Xt[:, j]
        if self.add_indicator:
            for col in self.indicator_columns_:
                out[f"{col}__was_missing"] = mask[:, self.columns.index(col)].astype(np.int64)
        return out

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        M = self._matrix(X)
        if np.isinf(M).any():
            raise ValueError("IterativeImputer does not accept infinite values")
        return self._frame(X, self._impute(M), np.isnan(M))


__all__ = [
    "NumericImputer",
    "CategoricalImputer",
    "KNNImputer",
    "IterativeImputer",
    "fit_bayesian_ridge",
    "nan_euclidean_distances",
]
