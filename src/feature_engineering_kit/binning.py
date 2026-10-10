"""Numeric discretizers.

:class:`QuantileBinning` is an equal-frequency (or equal-width) binning
transformer. :class:`MDLPBinning` is the supervised Fayyad-Irani entropy /
MDL discretizer, which places cuts where the class distribution changes. Bin
edges are learned on ``fit`` and reused by ``transform``, so a held-out frame
is cut with the training thresholds.
"""

from __future__ import annotations

import heapq

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


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _mdlp_best_split(values, labels, lo, hi, n_classes, min_samples_leaf):
    """Best entropy split of ``[lo, hi)`` if it passes the MDL test, else ``None``.

    Returns ``(entropy_reduction, cut, split_index)`` where the reduction is
    ``n * gain`` (bits saved over the whole segment).
    """
    n = hi - lo
    if n < 2 or n < 2 * min_samples_leaf:
        return None
    seg_vals = values[lo:hi]
    seg_lab = labels[lo:hi]
    onehot = np.zeros((n, n_classes), dtype=float)
    onehot[np.arange(n), seg_lab] = 1.0
    left = np.cumsum(onehot, axis=0)[:-1]  # class counts in the first i+1 rows
    total = left[-1] + onehot[-1]
    right = total - left
    sizes = np.arange(1, n, dtype=float)
    # candidate boundaries: only between distinct consecutive values
    valid = seg_vals[1:] > seg_vals[:-1]
    valid &= (sizes >= min_samples_leaf) & (n - sizes >= min_samples_leaf)
    if not valid.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        pl = left / sizes[:, None]
        pr = right / (n - sizes)[:, None]
        hl = -np.sum(np.where(pl > 0, pl * np.log2(np.where(pl > 0, pl, 1.0)), 0.0), axis=1)
        hr = -np.sum(np.where(pr > 0, pr * np.log2(np.where(pr > 0, pr, 1.0)), 0.0), axis=1)
    weighted = (sizes * hl + (n - sizes) * hr) / n
    weighted[~valid] = np.inf
    best = int(np.argmin(weighted))
    ent_s = _entropy(total)
    gain = ent_s - float(weighted[best])
    k = int(np.count_nonzero(total))
    k1 = int(np.count_nonzero(left[best]))
    k2 = int(np.count_nonzero(right[best]))
    delta = np.log2(3.0**k - 2.0) - (
        k * ent_s - k1 * _entropy(left[best]) - k2 * _entropy(right[best])
    )
    threshold = (np.log2(n - 1.0) + delta) / n
    if not gain > threshold:
        return None
    cut = 0.5 * (float(seg_vals[best]) + float(seg_vals[best + 1]))
    return n * gain, cut, lo + best + 1


def _mdlp_cuts(
    values: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    min_samples_leaf: int,
    max_cuts: int | None = None,
) -> list[float]:
    """Fayyad-Irani (1993) recursive entropy splits with the MDL stop rule.

    ``values`` must be sorted ascending and ``labels`` integer-coded in the
    same order. Segments are expanded best-first (largest entropy reduction),
    so ``max_cuts`` keeps the most informative cuts. Without a cap the result
    equals the classic depth-first recursion. Returns sorted cut points.
    """
    cuts: list[float] = []
    heap: list[tuple[float, int, float, int, int, int]] = []
    counter = 0

    def push(lo: int, hi: int) -> None:
        nonlocal counter
        found = _mdlp_best_split(values, labels, lo, hi, n_classes, min_samples_leaf)
        if found is not None:
            reduction, cut, split = found
            heapq.heappush(heap, (-reduction, counter, cut, split, lo, hi))
            counter += 1

    push(0, values.size)
    while heap and (max_cuts is None or len(cuts) < max_cuts):
        _, _, cut, split, lo, hi = heapq.heappop(heap)
        cuts.append(cut)
        push(lo, split)
        push(split, hi)
    return sorted(cuts)


class MDLPBinning(QuantileBinning):
    """Supervised entropy-based discretization (Fayyad & Irani, 1993).

    Each numeric column is split recursively at the boundary that minimizes
    the class-weighted entropy of the target. A split is kept only if its
    information gain exceeds the Minimum Description Length threshold

    ``Gain > (log2(N - 1) + log2(3^k - 2) - [k Ent(S) - k1 Ent(S1) - k2 Ent(S2)]) / N``

    where ``k``, ``k1`` and ``k2`` count the classes present in the parent and
    the two children. Unlike :class:`QuantileBinning` the number of bins is
    data-driven. A feature unrelated to the target typically stays a single
    bin, while a feature with sharp class changes gets a cut at each change.

    ``fit`` requires ``y`` (any discrete target: binary or multiclass labels).
    Rows whose feature value is missing are ignored when learning cuts.
    ``transform``, ``encode`` and ``inverse_transform`` behave exactly as in
    :class:`QuantileBinning`; ``cut_points_`` stores the interior cuts.

    Parameters
    ----------
    columns:
        Numeric columns to discretize. Other columns pass through unchanged.
    target:
        Name of the target column when ``y`` is a DataFrame.
    encode:
        ``"ordinal"`` (default) or ``"onehot"``.
    max_bins:
        Optional cap on the number of bins per column. Segments are split
        best-first by entropy reduction, so the most informative cuts are kept.
    min_samples_leaf:
        Minimum rows on each side of a cut (default ``1``, the classic MDLP).
    """

    def __init__(self, columns, target=None, encode="ordinal", max_bins=None, min_samples_leaf=1):
        self.columns = list(columns)
        self.target = target
        self.encode = _validate_choice(encode, _ENCODINGS, "encode")
        if max_bins is not None:
            max_bins = _validate_n_bins(max_bins)
        self.max_bins = max_bins
        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, (int, np.integer))
            or min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be an integer >= 1")
        self.min_samples_leaf = int(min_samples_leaf)
        self.strategy = "mdlp"
        self.n_bins = max_bins

    def _target(self, y, n_rows: int) -> np.ndarray:
        if y is None:
            raise ValueError("MDLPBinning.fit requires y")
        if isinstance(y, pd.DataFrame):
            if self.target is None or self.target not in y.columns:
                raise KeyError("target column not found in y")
            series = y[self.target]
        elif isinstance(y, pd.Series):
            series = y
        else:
            series = pd.Series(np.asarray(y).ravel())
        if len(series) != n_rows:
            raise ValueError("X and y must have the same number of rows")
        if series.isna().any():
            raise ValueError("y must not contain missing values")
        _, codes = np.unique(series.to_numpy(), return_inverse=True)
        return codes.astype(int)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        labels_all = self._target(y, len(X))
        n_classes = int(labels_all.max()) + 1
        self.bin_edges_ = {}
        self.n_bins_ = {}
        self.cut_points_: dict[str, list[float]] = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"column '{col}' not found during fit")
            _finite_values(X[col], col)  # validates numeric content
            numeric = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(numeric)
            values = numeric[mask]
            labels = labels_all[mask]
            order = np.argsort(values, kind="mergesort")
            values = values[order]
            labels = labels[order]
            max_cuts = None if self.max_bins is None else self.max_bins - 1
            cuts = _mdlp_cuts(values, labels, n_classes, self.min_samples_leaf, max_cuts)
            edges = _collapse_edges(
                np.asarray([float(values[0]), *cuts, float(values[-1])], dtype=float)
            )
            self.cut_points_[col] = cuts
            self.bin_edges_[col] = edges
            self.n_bins_[col] = int(len(edges) - 1)
        return None


__all__ = ["MDLPBinning", "QuantileBinning"]
