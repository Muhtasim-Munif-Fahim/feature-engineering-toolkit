"""Graph-based filter feature selection: Laplacian Score and Fisher Score.

**Laplacian Score** (He, Cai & Niyogi, NIPS 2005) is an *unsupervised*
filter. Build a k-nearest-neighbour graph with heat-kernel weights
``S_ij = exp(-||x_i - x_j||^2 / t)`` (symmetrised), degree matrix
``D = diag(S 1)`` and Laplacian ``L = D - S``. For each feature ``f``,
remove its degree-weighted mean ``f~ = f - (f'D1 / 1'D1) 1`` and score

    L_r = (f~' L f~) / (f~' D f~).

Small scores mean the feature varies little across graph edges relative to
its overall (degree-weighted) variance, i.e. it respects the local manifold
structure; features are therefore kept in *ascending* order.

**Fisher Score** is the classic supervised filter

    F_r = sum_c n_c (mu_{c,r} - mu_r)^2 / sum_c n_c sigma_{c,r}^2,

kept in *descending* order. With the supervised graph ``S_ij = 1/n_c`` for
pairs in the same class ``c`` (``graph="supervised"``), He et al. show the
Laplacian Score equals ``1 / (1 + F_r)``; ``laplacian_scores`` supports this
graph so the two filters can be compared directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer

__all__ = [
    "laplacian_scores",
    "fisher_scores",
    "LaplacianScoreSelection",
    "FisherScoreSelection",
]


def _as_matrix(X) -> np.ndarray:
    A = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    if A.shape[0] < 2:
        raise ValueError("X must have at least two rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("X must not contain NaN or infinite values; impute first")
    return A


def _check_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _knn_heat_graph(A: np.ndarray, n_neighbors: int, t) -> np.ndarray:
    n = A.shape[0]
    k = min(n_neighbors, n - 1)
    sq = np.sum(A * A, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * A @ A.T, 0.0)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1, kind="stable")[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    mask[np.repeat(np.arange(n), k), nn.ravel()] = True
    mask |= mask.T
    if t == "auto":
        # Mean squared distance over graph edges (a common heat-kernel width).
        edge = d2[mask]
        t_val = float(edge.mean()) if edge.size and edge.mean() > 0 else 1.0
    elif t is None:
        t_val = None
    else:
        t_val = float(t)
        if not np.isfinite(t_val) or t_val <= 0:
            raise ValueError('t must be a positive float, "auto", or None (binary weights)')
    S = np.zeros((n, n))
    if t_val is None:
        S[mask] = 1.0
    else:
        S[mask] = np.exp(-d2[mask] / t_val)
    return S


def _supervised_graph(y: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    same = inv[:, None] == inv[None, :]
    return np.where(same, 1.0 / counts[inv][:, None], 0.0)


def laplacian_scores(X, n_neighbors: int = 5, t="auto", graph: str = "knn", y=None) -> np.ndarray:
    """Laplacian Score of every column (smaller = better).

    Parameters
    ----------
    X:
        Numeric matrix or DataFrame without missing values.
    n_neighbors:
        ``k`` for the symmetrised kNN graph (``graph="knn"``).
    t:
        Heat-kernel width: positive float, ``"auto"`` (mean squared edge
        distance) or ``None`` for binary 0/1 edge weights.
    graph:
        ``"knn"`` (unsupervised) or ``"supervised"`` (requires ``y``; edges
        ``1/n_c`` within each class, giving ``1 / (1 + Fisher score)``).
    y:
        Class labels for ``graph="supervised"``.

    Constant features get ``nan`` (their score is undefined).
    """
    A = _as_matrix(X)
    if graph == "knn":
        S = _knn_heat_graph(A, _check_positive_int(n_neighbors, "n_neighbors"), t)
    elif graph == "supervised":
        if y is None:
            raise ValueError('graph="supervised" requires y')
        y_arr = np.asarray(pd.Series(y).to_numpy()).ravel()
        if y_arr.shape[0] != A.shape[0]:
            raise ValueError("y must have one value per row of X")
        S = _supervised_graph(y_arr)
    else:
        raise ValueError('graph must be "knn" or "supervised"')
    d = S.sum(axis=1)
    L = np.diag(d) - S
    mean = (d @ A) / d.sum()
    F = A - mean
    num = np.einsum("ij,ik,kj->j", F, L, F)
    den = np.einsum("ij,i,ij->j", F, d, F)
    scale = np.maximum(np.abs(A).max(axis=0), 1.0)
    out = np.full(A.shape[1], np.nan)
    ok = den > 1e-12 * (scale ** 2) * d.sum()
    out[ok] = num[ok] / den[ok]
    return out


def fisher_scores(X, y) -> np.ndarray:
    """Fisher Score of every column (larger = better).

    Between-class over within-class scatter, with population (``ddof=0``)
    class variances. A feature constant overall scores 0; a feature constant
    within every class but varying between classes scores ``inf``.
    """
    A = _as_matrix(X)
    y_arr = np.asarray(pd.Series(y).to_numpy()).ravel()
    if y_arr.shape[0] != A.shape[0]:
        raise ValueError("y must have one value per row of X")
    classes, inv = np.unique(y_arr, return_inverse=True)
    if classes.size < 2:
        raise ValueError("fisher_scores needs at least two classes")
    mu = A.mean(axis=0)
    between = np.zeros(A.shape[1])
    within = np.zeros(A.shape[1])
    for c in range(classes.size):
        Ac = A[inv == c]
        n_c = Ac.shape[0]
        between += n_c * (Ac.mean(axis=0) - mu) ** 2
        within += n_c * Ac.var(axis=0)
    tiny = 1e-12 * np.maximum(np.abs(A).max(axis=0), 1.0) ** 2 * A.shape[0]
    out = np.zeros(A.shape[1])
    pos_w = within > tiny
    out[pos_w] = between[pos_w] / within[pos_w]
    out[~pos_w & (between > tiny)] = np.inf
    return out


def _numeric_columns(X: pd.DataFrame, exclude=None) -> list[str]:
    return [c for c in X.columns if c != exclude and pd.api.types.is_numeric_dtype(X[c])]


class _ScoreSelection(Transformer):
    def __init__(self, k: int):
        self.k = _check_positive_int(k, "k")

    def _select(self, numeric: list[str], scores: np.ndarray, ascending: bool) -> None:
        self.scores_ = {c: float(s) for c, s in zip(numeric, scores)}
        key = np.where(np.isnan(scores), np.inf, scores if ascending else -scores)
        order = sorted(range(len(numeric)), key=lambda i: (key[i], i))
        self.selected_ = [numeric[i] for i in order[: self.k]]
        self.drop_ = [c for c in numeric if c not in self.selected_]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


class LaplacianScoreSelection(_ScoreSelection):
    """Keep the ``k`` numeric columns with the *lowest* Laplacian Score.

    Unsupervised (``y`` is ignored), so it works for clustering pipelines.
    Columns are optionally standardised first (``standardize=True``), which
    keeps one wide-range column from dominating the kNN graph. Constant
    columns score ``nan`` and are ranked last. Non-numeric columns pass
    through.

    Attributes: ``scores_``, ``selected_`` (best first), ``drop_``.
    """

    def __init__(self, k: int = 10, n_neighbors: int = 5, t="auto", standardize: bool = True):
        super().__init__(k)
        self.n_neighbors = _check_positive_int(n_neighbors, "n_neighbors")
        self.t = t
        self.standardize = bool(standardize)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        numeric = _numeric_columns(X)
        if not numeric:
            self.scores_, self.selected_, self.drop_ = {}, [], []
            return None
        A = _as_matrix(X[numeric])
        if self.standardize:
            sd = A.std(axis=0)
            A = (A - A.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        self._select(numeric, laplacian_scores(A, n_neighbors=self.n_neighbors, t=self.t), ascending=True)
        return None


class FisherScoreSelection(_ScoreSelection):
    """Keep the ``k`` numeric columns with the *highest* Fisher Score.

    Requires class labels ``y`` at ``fit``; a column named ``target`` in
    ``X`` is never scored. Non-numeric columns pass through.

    Attributes: ``scores_``, ``selected_`` (best first), ``drop_``.
    """

    def __init__(self, k: int = 10, target: str | None = None):
        super().__init__(k)
        self.target = target

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        if y is None:
            raise ValueError("FisherScoreSelection.fit requires y")
        numeric = _numeric_columns(X, exclude=self.target)
        if not numeric:
            self.scores_, self.selected_, self.drop_ = {}, [], []
            return None
        self._select(numeric, fisher_scores(X[numeric], y), ascending=False)
        return None
