"""Numeric feature scalers.

Each scaler is column-wise and stores per-column statistics at ``fit`` time so
that the same statistics can be applied to a held-out/test frame at ``transform``
time, preventing leakage. ``YeoJohnsonScaler`` additionally estimates a power
parameter per column before (optionally) standardising.
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




def _yeo_johnson_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Apply the Yeo-Johnson power transform with a fixed ``lmbda``."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    if abs(lmbda) < 1e-12:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lmbda) - 1.0) / lmbda
    if abs(lmbda - 2.0) < 1e-12:
        out[neg] = -np.log1p(-x[neg])
    else:
        out[neg] = -(np.power(-x[neg] + 1.0, 2.0 - lmbda) - 1.0) / (2.0 - lmbda)
    return out


def _yeo_johnson_loglik(x: np.ndarray, lmbda: float) -> float:
    """Gaussian log-likelihood of Yeo-Johnson-transformed ``x`` (for MLE)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return -np.inf
    y = _yeo_johnson_transform(x, lmbda)
    var = float(np.var(y))
    if var <= 0 or not np.isfinite(var):
        return -np.inf
    # Jacobian: for x>=0, (x+1)^{λ-1}; for x<0, (-x+1)^{1-λ}
    pos = x >= 0
    neg = ~pos
    log_jac = np.zeros(n, dtype=float)
    if pos.any():
        log_jac[pos] = (lmbda - 1.0) * np.log1p(x[pos])
    if neg.any():
        log_jac[neg] = (1.0 - lmbda) * np.log1p(-x[neg])
    return float(-0.5 * n * np.log(2.0 * np.pi * var) - 0.5 * n + np.sum(log_jac))


def _brent_maximize(func, lo: float, hi: float, tol: float = 1e-5, max_iter: int = 80) -> float:
    """Maximize a unimodal scalar function on ``[lo, hi]`` via golden-section search."""
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    resphi = 2.0 - phi
    a, b = lo, hi
    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc, fd = func(c), func(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = a + resphi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = b - resphi * (b - a)
            fd = func(d)
    return 0.5 * (a + b)


class YeoJohnsonScaler(Transformer):
    """Yeo-Johnson power transform that stabilises variance and reduces skew.

    Unlike Box-Cox, Yeo-Johnson is defined for non-positive values. For each
    column the power parameter ``λ`` is estimated by maximising the Gaussian
    log-likelihood of the transformed values (same objective as scikit-learn's
    ``PowerTransformer(method="yeo-johnson")``). After the power transform the
    column is optionally standardised to zero mean / unit variance.

    Parameters
    ----------
    columns :
        Column names to transform.
    standardize :
        If ``True`` (default), subtract the post-transform mean and divide by
        the post-transform standard deviation (``ddof=0``).
    lmbda :
        Optional fixed ``λ`` applied to every column. When ``None`` (default)
        each column gets its own MLE estimate in ``[-2, 2]``.
    """

    def __init__(self, columns, standardize: bool = True, lmbda: float | None = None):
        self.columns = list(columns)
        self.standardize = bool(standardize)
        self.lmbda = lmbda

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.lambdas_: dict[str, float] = {}
        self.mean_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        for col in self.columns:
            s = _to_float(X[col])
            values = s.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if self.lmbda is not None:
                lam = float(self.lmbda)
            elif values.size < 2 or float(np.nanstd(values)) == 0.0:
                lam = 1.0  # identity when there is nothing to estimate
            else:
                lam = _brent_maximize(lambda L: _yeo_johnson_loglik(values, L), -2.0, 2.0)
            self.lambdas_[col] = lam
            transformed = _yeo_johnson_transform(s.to_numpy(dtype=float), lam)
            finite = transformed[np.isfinite(transformed)]
            mu = float(np.mean(finite)) if finite.size else 0.0
            sigma = float(np.std(finite)) if finite.size else 1.0
            if sigma <= 0 or not np.isfinite(sigma):
                sigma = 1.0
            self.mean_[col] = mu
            self.scale_[col] = sigma
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                continue
            values = _to_float(out[col]).to_numpy(dtype=float)
            transformed = _yeo_johnson_transform(values, self.lambdas_[col])
            if self.standardize:
                transformed = (transformed - self.mean_[col]) / self.scale_[col]
            out[col] = transformed
        return out


__all__ = ["StandardScaler", "MinMaxScaler", "RobustScaler", "YeoJohnsonScaler"]
