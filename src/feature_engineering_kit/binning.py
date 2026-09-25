"""Numeric discretizers.

:class:`QuantileBinning` is an equal-frequency (or equal-width) binning
transformer. Bin edges are learned on ``fit`` and reused by ``transform``, so a
held-out frame is cut with the training thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer

_STRATEGIES = ("quantile", "uniform")
_ENCODINGS = ("ordinal", "onehot")
# Drop bin edges closer than this. Matches the usual KBinsDiscretizer rule for
# duplicate quantiles (empty / sparse bins caused by ties).
_EDGE_TOL = 1e-8


def _validate_n_bins(n_bins) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise ValueError("n_bins must be an integer >= 2")
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    return n_bins


def _validate_choice(value, allowed: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"{name} must be one of {choices}")
    return value


def _finite_values(series: pd.Series, column: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    raw_missing = series.isna()
    failed = ~raw_missing.to_numpy() & numeric.isna().to_numpy()
    if bool(np.any(failed)):
        raise ValueError(f"column '{column}' contains non-numeric values")
    values = numeric.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"column '{column}' has no finite values to bin")
    return values


def _collapse_edges(edges: np.ndarray) -> np.ndarray:
    """Drop duplicate or near-duplicate edges so empty bins are removed.

    A constant column collapses to a single bin whose two edges are equal.
    The rightmost edge is kept so the fitted range still covers the training
    maximum when only the last cut was redundant.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.size == 0 or not np.isfinite(edges).all():
        raise ValueError("bin edges must be finite")
    span = float(edges[-1] - edges[0])
    tol = _EDGE_TOL * max(span, 1.0)
    if span <= tol:
        value = float(edges[0])
        return np.array([value, value], dtype=float)

    kept = [float(edges[0])]
    for edge in edges[1:-1]:
        edge = float(edge)
        if edge - kept[-1] > tol:
            kept.append(edge)
    last = float(edges[-1])
    if last - kept[-1] > tol:
        kept.append(last)
    else:
        kept[-1] = last
    if len(kept) < 2 or kept[-1] - kept[0] <= tol:
        value = float(edges[0])
        return np.array([value, value], dtype=float)
    return np.asarray(kept, dtype=float)


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map finite values to bin codes. Missing values stay NaN.

    Interior cuts use the half-open intervals ``[e_i, e_{i+1})``. Values below
    the first edge clip to bin ``0``; values at or above the last interior edge
    (including the training maximum) clip to the last bin.
    """
    codes = np.full(np.shape(values), np.nan, dtype=float)
    mask = np.isfinite(values)
    n_bins = int(len(edges) - 1)
    if n_bins <= 1 or not mask.any():
        codes[mask] = 0.0
        return codes
    codes[mask] = np.digitize(values[mask], edges[1:-1], right=False)
    return codes


def _bin_column(column: str, index: int) -> str:
    return f"{column}__bin_{index}"


class QuantileBinning(Transformer):
    """Discretize numeric columns into ordinal codes or one-hot indicators.

    ``strategy="quantile"`` (the default) places cuts at empirical quantiles so
    each bin holds about the same number of training rows (equal frequency).
    ``strategy="uniform"`` places cuts at equal widths between the training
    minimum and maximum.

    Ties and repeated quantiles produce empty intervals. Those sparse bins are
    dropped: ``bin_edges_`` keeps only strictly increasing cuts, and
    ``n_bins_`` records how many bins remain. A constant column becomes one
    bin. Requested ``n_bins`` is therefore an upper bound, not a guarantee.

    ``encode="ordinal"`` replaces each column with codes ``0 .. n_bins_-1``.
    ``encode="onehot"`` replaces it with ``{column}__bin_{i}`` indicators and
    drops the original column. Missing values stay missing in the ordinal
    encoding. In the one-hot encoding a missing row is all zeros, and
    :meth:`inverse_transform` reads that row back as missing.

    Values outside the training range clip to the nearest edge bin.
    :meth:`inverse_transform` maps each code to the midpoint of its bin. It is
    not a perfect reconstruction of the original numbers.

    Parameters
    ----------
    columns:
        Numeric columns to discretize. Other columns pass through unchanged.
    n_bins:
        Number of bins to request. Must be an integer ``>= 2``. Fewer bins are
        used when cuts collapse.
    strategy:
        ``"quantile"`` (equal frequency) or ``"uniform"`` (equal width).
    encode:
        ``"ordinal"`` (default) or ``"onehot"``.
    """

    def __init__(self, columns, n_bins=5, strategy="quantile", encode="ordinal"):
        self.columns = list(columns)
        self.n_bins = _validate_n_bins(n_bins)
        self.strategy = _validate_choice(strategy, _STRATEGIES, "strategy")
        self.encode = _validate_choice(encode, _ENCODINGS, "encode")

    def _edges_for(self, values: np.ndarray) -> np.ndarray:
        if self.strategy == "uniform":
            lo = float(np.min(values))
            hi = float(np.max(values))
            raw = np.linspace(lo, hi, self.n_bins + 1)
        else:
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
            raw = np.quantile(values, quantiles)
        return _collapse_edges(raw)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.n_bins_: dict[str, int] = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"column '{col}' not found during fit")
            edges = self._edges_for(_finite_values(X[col], col))
            self.bin_edges_[col] = edges
            self.n_bins_[col] = int(len(edges) - 1)
        return None

    def _centers(self, column: str) -> np.ndarray:
        edges = self.bin_edges_[column]
        if len(edges) == 2 and edges[0] == edges[1]:
            return np.array([float(edges[0])], dtype=float)
        return (edges[:-1] + edges[1:]) / 2.0

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            raw_missing = out[col].isna().to_numpy()
            failed = ~raw_missing & ~np.isfinite(values)
            if bool(np.any(failed)):
                raise ValueError(f"column '{col}' contains non-numeric values")
            codes = _digitize(values, self.bin_edges_[col])
            if self.encode == "ordinal":
                out[col] = codes
                continue
            loc = int(out.columns.get_loc(col))
            out = out.drop(columns=[col])
            for i in range(self.n_bins_[col]):
                indicator = (codes == i).astype(np.int64)
                out.insert(loc + i, _bin_column(col, i), indicator)
        return out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map bin codes (or one-hot indicators) back to bin midpoints.

        Ordinal codes are rounded to the nearest integer and clipped into
        ``0 .. n_bins_-1``. An all-zero one-hot row is treated as missing.
        Other columns and the index are preserved.
        """
        self._check_fitted()
        out = X.copy()
        for col in self.columns:
            centers = self._centers(col)
            if self.encode == "ordinal":
                if col not in out.columns:
                    raise KeyError(f"column '{col}' not found during inverse_transform")
                codes = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
                restored = np.full(len(out), np.nan, dtype=float)
                known = np.isfinite(codes)
                idx = np.rint(codes[known]).astype(int)
                idx = np.clip(idx, 0, self.n_bins_[col] - 1)
                restored[known] = centers[idx]
                out[col] = restored
                continue

            names = [_bin_column(col, i) for i in range(self.n_bins_[col])]
            missing = [name for name in names if name not in out.columns]
            if missing:
                raise KeyError(
                    f"column '{missing[0]}' not found during inverse_transform"
                )
            block = out[names].to_numpy(dtype=float)
            chosen = np.argmax(np.nan_to_num(block, nan=-1.0), axis=1)
            row_sum = np.nansum(block, axis=1)
            restored = centers[chosen].astype(float)
            restored[row_sum <= 0] = np.nan
            loc = int(out.columns.get_loc(names[0]))
            out = out.drop(columns=names)
            out.insert(loc, col, restored)
        return out


__all__ = ["QuantileBinning"]
