"""Numeric feature scalers.

Each scaler is column-wise and stores per-column statistics at ``fit`` time so
that the same statistics can be applied to a held-out/test frame at ``transform``
time, preventing leakage. ``YeoJohnsonScaler`` and ``BoxCoxScaler`` additionally
estimate a power parameter per column before (optionally) standardising.
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



class MaxAbsScaler(Transformer):
    """Scale each column by its maximum absolute value into ``[-1, 1]``.

    For every selected column the fit-time statistic is
    ``max_abs = max(|x|)`` over finite values. Transform divides by that
    scale (or ``1.0`` when the column is all zeros / empty), so the
    training values land in ``[-1, 1]`` and the origin is preserved —
    useful for sparse / already-centered data where shifting would densify.

    Parameters
    ----------
    columns :
        Column names to scale.
    """

    def __init__(self, columns):
        self.columns = list(columns)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.scale_: dict[str, float] = {}
        for col in self.columns:
            s = _to_float(X[col])
            values = s.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                max_abs = 1.0
            else:
                max_abs = float(np.max(np.abs(finite)))
            self.scale_[col] = max_abs if max_abs > 0 else 1.0
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = _to_float(out[col]) / self.scale_[col]
        return out



class Normalizer(Transformer):
    """Scale each *row* to unit norm over the selected columns.

    Unlike column-wise scalers, :class:`Normalizer` divides every sample by
    its vector norm computed on ``columns``. Supported norms are ``"l1"``,
    ``"l2"`` (default), and ``"max"`` (infinity / max absolute value).
    ``fit`` stores the column list only (no statistics); zero-norm rows are
    left unchanged at transform time.

    Parameters
    ----------
    columns :
        Column names that form the feature vector for each row.
    norm :
        One of ``"l1"``, ``"l2"``, or ``"max"``.
    """

    def __init__(self, columns, norm: str = "l2"):
        if norm not in {"l1", "l2", "max"}:
            raise ValueError("norm must be one of {'l1', 'l2', 'max'}")
        self.columns = list(columns)
        self.norm = norm

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        # Row-wise normalizer has no column statistics to learn; store the
        # column order so transform can rebuild the same feature matrix.
        self.columns_ = list(self.columns)
        self.norm_ = self.norm
        return None

    def _row_norms(self, values: np.ndarray) -> np.ndarray:
        if self.norm_ == "l1":
            norms = np.nansum(np.abs(values), axis=1)
        elif self.norm_ == "l2":
            norms = np.sqrt(np.nansum(np.square(values), axis=1))
        else:  # max
            norms = np.nanmax(np.abs(values), axis=1)
        norms = np.asarray(norms, dtype=float)
        norms[~np.isfinite(norms)] = 0.0
        return norms

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        cols = [c for c in self.columns_ if c in out.columns]
        if not cols:
            return out
        mat = out[cols].apply(_to_float).to_numpy(dtype=float)
        norms = self._row_norms(mat)
        # Avoid divide-by-zero: leave zero-norm rows unchanged
        safe = norms.copy()
        safe[safe == 0.0] = 1.0
        scaled = mat / safe[:, None]
        for j, col in enumerate(cols):
            out[col] = scaled[:, j]
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





class Winsorizer(Transformer):
    """Clip numeric columns at lower/upper empirical percentiles (winsorize).

    At ``fit`` time, each selected column's lower and upper percentile
    thresholds are estimated from the training frame (default 5th and 95th).
    ``transform`` clips values outside those bounds to the fitted thresholds,
    which shrinks extreme outliers without discarding rows. Mirrors the
    sklearn-style ``Transformer`` API used by :class:`RobustScaler` and
    :class:`Normalizer`.

    Parameters
    ----------
    columns :
        Column names to winsorize.
    limits :
        Pair ``(lower, upper)`` of percentiles in ``[0, 1]``. ``lower`` must
        be strictly less than ``upper``. Defaults to ``(0.05, 0.95)``.
    """

    def __init__(self, columns, limits=(0.05, 0.95)):
        if (
            not isinstance(limits, (tuple, list))
            or len(limits) != 2
        ):
            raise ValueError("limits must be a (lower, upper) pair")
        lo, hi = float(limits[0]), float(limits[1])
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("limits must satisfy 0 <= lower < upper <= 1")
        self.columns = list(columns)
        self.limits = (lo, hi)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.lower_: dict[str, float] = {}
        self.upper_: dict[str, float] = {}
        lo_q, hi_q = self.limits
        for col in self.columns:
            s = _to_float(X[col])
            self.lower_[col] = float(s.quantile(lo_q))
            self.upper_[col] = float(s.quantile(hi_q))
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = _to_float(out[col]).clip(
                    lower=self.lower_[col], upper=self.upper_[col]
                )
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




def _box_cox_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Apply the Box-Cox power transform with a fixed ``lmbda`` (requires x > 0)."""
    x = np.asarray(x, dtype=float)
    if abs(lmbda) < 1e-12:
        return np.log(x)
    return (np.power(x, lmbda) - 1.0) / lmbda


def _box_cox_loglik(x: np.ndarray, lmbda: float) -> float:
    """Gaussian log-likelihood of Box-Cox-transformed ``x`` (for MLE)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return -np.inf
    if np.any(x <= 0):
        return -np.inf
    y = _box_cox_transform(x, lmbda)
    var = float(np.var(y))
    if var <= 0 or not np.isfinite(var):
        return -np.inf
    # Jacobian: x^{λ-1}
    log_jac = (lmbda - 1.0) * np.sum(np.log(x))
    return float(-0.5 * n * np.log(2.0 * np.pi * var) - 0.5 * n + log_jac)


class BoxCoxScaler(Transformer):
    """Box-Cox power transform for strictly positive columns.

    Unlike Yeo-Johnson, Box-Cox is only defined for ``x > 0``. For each column
    the power parameter ``λ`` is estimated by maximising the Gaussian
    log-likelihood of the transformed values (same objective as scikit-learn's
    ``PowerTransformer(method="box-cox")``). After the power transform the
    column is optionally standardised to zero mean / unit variance.

    Non-positive handling
    ---------------------
    At ``fit`` time, every selected column must contain only finite, strictly
    positive values among the non-missing entries; otherwise a ``ValueError``
    is raised naming the offending column. At ``transform`` time, non-positive
    or non-finite values become ``NaN`` in the output (documented rather than
    silently clipped), so callers can detect them.

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
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(
                    f"BoxCoxScaler column {col!r} has no finite values to fit"
                )
            if np.any(finite <= 0):
                raise ValueError(
                    f"BoxCoxScaler requires strictly positive values; "
                    f"column {col!r} has non-positive entries"
                )
            if self.lmbda is not None:
                lam = float(self.lmbda)
            elif finite.size < 2 or float(np.nanstd(finite)) == 0.0:
                lam = 1.0  # identity when there is nothing to estimate
            else:
                lam = _brent_maximize(lambda L: _box_cox_loglik(finite, L), -2.0, 2.0)
            self.lambdas_[col] = lam
            transformed = _box_cox_transform(finite, lam)
            mu = float(np.mean(transformed)) if transformed.size else 0.0
            sigma = float(np.std(transformed)) if transformed.size else 1.0
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
            transformed = np.full(values.shape, np.nan, dtype=float)
            valid = np.isfinite(values) & (values > 0)
            if np.any(valid):
                transformed[valid] = _box_cox_transform(values[valid], self.lambdas_[col])
            if self.standardize:
                transformed = (transformed - self.mean_[col]) / self.scale_[col]
            out[col] = transformed
        return out


__all__ = ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "Normalizer", "RobustScaler", "Winsorizer", "YeoJohnsonScaler", "BoxCoxScaler", "QuantileTransformer"]



def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse CDF of the standard normal (``scipy.special.ndtri``)."""
    from scipy.special import ndtri

    p = np.asarray(p, dtype=float)
    out = np.empty_like(p, dtype=float)
    finite = np.isfinite(p)
    out[:] = np.nan
    if np.any(finite):
        # Clip away from {0,1} so ndtri stays finite
        clipped = np.clip(p[finite], 1e-12, 1.0 - 1e-12)
        out[finite] = ndtri(clipped)
    return out


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """CDF of the standard normal (``scipy.special.ndtr``)."""
    from scipy.special import ndtr

    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    finite = np.isfinite(x)
    out[:] = np.nan
    if np.any(finite):
        out[finite] = ndtr(x[finite])
    return out


class QuantileTransformer(Transformer):
    """Map each column to a uniform or normal distribution via its empirical CDF.

    At ``fit`` time a grid of ``n_quantiles`` reference values is stored per
    column (empirical quantiles of the finite training values). ``transform``
    interpolates each value onto that CDF, yielding roughly uniform ``[0, 1]``
    ranks; when ``output_distribution="normal"`` those ranks are mapped
    through the inverse standard-normal CDF. ``inverse_transform`` reverses
    the mapping (normal → uniform → original scale).

    Column-wise and leakage-safe: test frames reuse the fit-time references,
    matching the :class:`StandardScaler` / :class:`RobustScaler` /
    :class:`MaxAbsScaler` :class:`~feature_engineering_kit.base.Transformer`
    API (``fit`` / ``transform`` / ``fit_transform``) plus ``inverse_transform``.

    Parameters
    ----------
    columns :
        Column names to transform.
    n_quantiles :
        Number of quantiles to estimate. Clamped to ``[2, n_samples]`` at fit.
    output_distribution :
        ``"uniform"`` (default) or ``"normal"``.
    subsample :
        If the column has more than this many finite values, a random subset
        of that size is used to estimate quantiles (``None`` disables).
    random_state :
        Seed for subsample draws.
    """

    def __init__(
        self,
        columns,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
        subsample: int | None = 100_000,
        random_state: int | None = None,
    ):
        if output_distribution not in ("uniform", "normal"):
            raise ValueError(
                "output_distribution must be 'uniform' or 'normal', "
                f"got {output_distribution!r}"
            )
        if int(n_quantiles) < 2:
            raise ValueError("n_quantiles must be >= 2")
        self.columns = list(columns)
        self.n_quantiles = int(n_quantiles)
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = random_state

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.references_: dict[str, np.ndarray] = {}
        self.quantiles_: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(self.random_state)
        for col in self.columns:
            s = _to_float(X[col])
            values = s.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                # Degenerate: identity map around 0
                self.references_[col] = np.array([0.0, 0.0], dtype=float)
                self.quantiles_[col] = np.array([0.0, 1.0], dtype=float)
                continue
            if self.subsample is not None and finite.size > int(self.subsample):
                finite = rng.choice(finite, size=int(self.subsample), replace=False)
            finite = np.sort(finite)
            n_q = int(min(self.n_quantiles, finite.size))
            n_q = max(n_q, 2)
            # Evenly spaced quantile probabilities in [0, 1]
            probs = np.linspace(0.0, 1.0, n_q)
            refs = np.quantile(finite, probs)
            # Ensure strictly non-decreasing for interpolation; collapse ties softly
            for i in range(1, refs.size):
                if refs[i] < refs[i - 1]:
                    refs[i] = refs[i - 1]
            self.references_[col] = refs.astype(float)
            self.quantiles_[col] = probs.astype(float)
        return None

    def _to_uniform(self, values: np.ndarray, col: str) -> np.ndarray:
        refs = self.references_[col]
        probs = self.quantiles_[col]
        out = np.full(values.shape, np.nan, dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return out
        # np.interp handles values outside [min, max] by clipping to endpoints
        out[finite] = np.interp(values[finite], refs, probs)
        return out

    def _from_uniform(self, uniforms: np.ndarray, col: str) -> np.ndarray:
        refs = self.references_[col]
        probs = self.quantiles_[col]
        out = np.full(uniforms.shape, np.nan, dtype=float)
        finite = np.isfinite(uniforms)
        if not np.any(finite):
            return out
        clipped = np.clip(uniforms[finite], 0.0, 1.0)
        out[finite] = np.interp(clipped, probs, refs)
        return out

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                continue
            values = _to_float(out[col]).to_numpy(dtype=float)
            uniforms = self._to_uniform(values, col)
            if self.output_distribution == "normal":
                # Avoid exact 0/1 which map to ±inf
                eps = 1e-7
                clipped = np.clip(uniforms, eps, 1.0 - eps)
                transformed = _norm_ppf(clipped)
                # Preserve NaNs from non-finite inputs
                transformed[~np.isfinite(uniforms)] = np.nan
                out[col] = transformed
            else:
                out[col] = uniforms
        return out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map transformed columns back to the original scale."""
        self._check_fitted()
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                continue
            values = _to_float(out[col]).to_numpy(dtype=float)
            if self.output_distribution == "normal":
                uniforms = _norm_cdf(values)
            else:
                uniforms = values
            out[col] = self._from_uniform(uniforms, col)
        return out


