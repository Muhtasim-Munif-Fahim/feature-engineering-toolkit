"""B-spline basis expansion for numeric columns.

:class:`SplineTransformer` replaces each selected numeric column with a set of
B-spline basis functions (de Boor, *A Practical Guide to Splines*, 1978). A
linear model fit on these features can represent smooth, non-monotone
relationships with a single input while staying well conditioned, which is
why spline features are a common alternative to high-degree polynomials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer

_KNOT_STRATEGIES = ("uniform", "quantile")
_EXTRAPOLATIONS = ("constant", "continue", "error")
_KNOT_TOL = 1e-10


def _validate_choice(value, allowed: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"{name} must be one of {choices}")
    return value


def _validate_int(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _spline_column(column: str, index: int) -> str:
    return f"{column}__spline_{index}"


def _numeric_values(series: pd.Series, column: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    raw_missing = series.isna().to_numpy()
    failed = ~raw_missing & numeric.isna().to_numpy()
    if bool(np.any(failed)):
        raise ValueError(f"column '{column}' contains non-numeric values")
    return numeric.to_numpy(dtype=float)


def _extend_knots(base: np.ndarray, degree: int) -> np.ndarray:
    """Pad ``base`` knots with ``degree`` equally spaced knots on each side.

    The padding uses the spacing of the first/last interior interval, so the
    basis evaluated on ``[base[0], base[-1]]`` forms a partition of unity.
    """
    if degree == 0:
        return base.copy()
    left_step = base[1] - base[0]
    right_step = base[-1] - base[-2]
    left = base[0] - left_step * np.arange(degree, 0, -1)
    right = base[-1] + right_step * np.arange(1, degree + 1)
    return np.concatenate([left, base, right])


def bspline_basis(x, knots, degree: int) -> np.ndarray:
    """Evaluate all B-spline basis functions of ``degree`` at ``x``.

    Uses the Cox-de Boor recursion on a full (already padded) knot vector
    ``knots`` of length ``m``; returns an array of shape ``(len(x), m - degree - 1)``.
    Each basis function is supported on ``[knots[i], knots[i + degree + 1])``.
    Non-finite inputs produce rows of ``NaN``.
    """
    x = np.asarray(x, dtype=float).ravel()
    t = np.asarray(knots, dtype=float)
    degree = _validate_int(degree, "degree", 0)
    n_basis = len(t) - degree - 1
    if n_basis < 1:
        raise ValueError("knot vector too short for the requested degree")
    finite = np.isfinite(x)
    xf = np.where(finite, x, t[0])
    # degree-0 indicators on half-open intervals [t_i, t_{i+1}).
    basis = ((xf[:, None] >= t[None, :-1]) & (xf[:, None] < t[None, 1:])).astype(float)
    for k in range(1, degree + 1):
        n = basis.shape[1] - 1
        left_den = t[k : k + n] - t[:n]
        right_den = t[k + 1 : k + 1 + n] - t[1 : 1 + n]
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(left_den > 0, (xf[:, None] - t[None, :n]) / left_den, 0.0)
            right = np.where(
                right_den > 0, (t[None, k + 1 : k + 1 + n] - xf[:, None]) / right_den, 0.0
            )
        basis = left * basis[:, :n] + right * basis[:, 1 : n + 1]
    basis[~finite] = np.nan
    return basis


class SplineTransformer(Transformer):
    """Expand numeric columns into B-spline basis features.

    Each selected column ``c`` is replaced (by default) with
    ``n_knots + degree - 1`` columns ``c__spline_0 .. c__spline_{k-1}``. Knot
    positions are learned on ``fit`` from the training frame and reused by
    ``transform``. Inside the training range ``[min, max]`` the basis is a
    partition of unity (each row sums to one) and at most ``degree + 1``
    features are non-zero for any value, so the expansion is sparse and
    numerically stable.

    Parameters
    ----------
    columns:
        Numeric columns to expand. Other columns pass through unchanged.
    n_knots:
        Number of boundary + interior knots (``>= 2``). With ``"quantile"``
        knots, tied quantiles are collapsed, so fewer knots may be used; the
        fitted count is on ``n_knots_``.
    degree:
        Polynomial degree of the pieces (``>= 0``). ``3`` (cubic) is the
        default; ``1`` gives piecewise-linear "hat" features and ``0`` gives
        indicator (binning) features.
    knots:
        ``"uniform"`` (equally spaced between training min and max) or
        ``"quantile"`` (at empirical quantiles, more knots where data is dense).
        An explicit increasing array of knot positions is also accepted and
        applied to every selected column.
    extrapolation:
        Behaviour for values outside the training range. ``"constant"``
        (default) clips them to the boundary, ``"continue"`` evaluates the
        padded basis (values far outside become all zeros), and ``"error"``
        raises ``ValueError``.
    include_bias:
        If ``False``, drop the last basis column of each feature. Because the
        full basis sums to one, dropping one column removes the collinearity
        with a model intercept.
    keep_original:
        If ``True``, keep the source column and append the spline columns after
        it; by default the source column is replaced.

    Missing values produce ``NaN`` in every spline column of that row.
    """

    def __init__(
        self,
        columns,
        n_knots=5,
        degree=3,
        knots="uniform",
        extrapolation="constant",
        include_bias=True,
        keep_original=False,
    ):
        self.columns = list(columns)
        self.n_knots = _validate_int(n_knots, "n_knots", 2)
        self.degree = _validate_int(degree, "degree", 0)
        if isinstance(knots, str):
            self.knots = _validate_choice(knots, _KNOT_STRATEGIES, "knots")
        else:
            arr = np.asarray(knots, dtype=float).ravel()
            if arr.size < 2 or not np.isfinite(arr).all() or np.any(np.diff(arr) <= 0):
                raise ValueError("explicit knots must be >= 2 finite, strictly increasing values")
            self.knots = arr
        self.extrapolation = _validate_choice(extrapolation, _EXTRAPOLATIONS, "extrapolation")
        self.include_bias = bool(include_bias)
        self.keep_original = bool(keep_original)

    def _base_knots(self, values: np.ndarray, column: str) -> np.ndarray:
        if not isinstance(self.knots, str):
            return self.knots.copy()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"column '{column}' has no finite values")
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi - lo <= _KNOT_TOL * max(abs(lo), abs(hi), 1.0):
            raise ValueError(f"column '{column}' is constant; cannot place spline knots")
        if self.knots == "uniform":
            return np.linspace(lo, hi, self.n_knots)
        raw = np.quantile(finite, np.linspace(0.0, 1.0, self.n_knots))
        tol = _KNOT_TOL * (hi - lo)
        kept = [float(raw[0])]
        for v in raw[1:]:
            if float(v) - kept[-1] > tol:
                kept.append(float(v))
        kept[-1] = hi
        return np.asarray(kept, dtype=float)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.base_knots_: dict[str, np.ndarray] = {}
        self.knots_: dict[str, np.ndarray] = {}
        self.n_knots_: dict[str, int] = {}
        self.n_features_out_: dict[str, int] = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"column '{col}' not found during fit")
            base = self._base_knots(_numeric_values(X[col], col), col)
            self.base_knots_[col] = base
            self.knots_[col] = _extend_knots(base, self.degree)
            self.n_knots_[col] = int(base.size)
            n_basis = int(base.size + self.degree - 1)
            self.n_features_out_[col] = n_basis if self.include_bias else n_basis - 1
        return None

    def _expand(self, values: np.ndarray, col: str) -> np.ndarray:
        base = self.base_knots_[col]
        lo, hi = float(base[0]), float(base[-1])
        finite = np.isfinite(values)
        outside = finite & ((values < lo) | (values > hi))
        if self.extrapolation == "error" and bool(np.any(outside)):
            raise ValueError(
                f"column '{col}' has values outside the fitted range [{lo}, {hi}]"
            )
        x = values.copy()
        if self.extrapolation == "constant":
            x[finite] = np.clip(x[finite], lo, hi)
        basis = bspline_basis(x, self.knots_[col], self.degree)
        if self.degree == 0:
            # Half-open intervals leave the right boundary uncovered; map it to the last bin.
            at_hi = finite & (x == hi)
            basis[at_hi, -1] = 1.0
        return basis[:, : self.n_features_out_[col]]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            block = self._expand(_numeric_values(out[col], col), col)
            loc = int(out.columns.get_loc(col))
            if self.keep_original:
                loc += 1
            else:
                out = out.drop(columns=[col])
            for i in range(block.shape[1]):
                out.insert(loc + i, _spline_column(col, i), block[:, i])
        return out


__all__ = ["SplineTransformer", "bspline_basis"]
