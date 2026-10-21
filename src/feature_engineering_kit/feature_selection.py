"""Feature-selection transformers.

These selectors are column-wise and designed to run after encoding so every
column is numeric. They expose the same ``fit``/``transform`` interface, so they
compose inside :class:`~feature_engineering_kit.pipeline.FeatureEngineeringPipeline`.
:class:`MRMRSelection` adds minimum-redundancy maximum-relevance selection on
discretized mutual information (Peng, Long & Ding, 2005), and
:class:`ReliefFSelection` ranks features by ReliefF nearest-hit/miss weights
(Kononenko, 1994), which also catch features that only matter jointly.
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


def quantile_discretize(values, n_bins: int = 10) -> np.ndarray:
    """Equal-frequency integer codes for a numeric vector.

    Bin edges are the empirical quantiles; duplicate edges (heavy ties) are
    merged, so a column with ``m < n_bins`` distinct values gets at most
    ``m`` codes. ``NaN`` values get their own code (``-1``), so missingness
    can carry information.
    """
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    x = np.asarray(values, dtype=float).ravel()
    codes = np.full(x.shape[0], -1, dtype=int)
    finite = ~np.isnan(x)
    if not finite.any():
        return codes
    edges = np.unique(np.quantile(x[finite], np.linspace(0, 1, int(n_bins) + 1)[1:-1]))
    codes[finite] = np.searchsorted(edges, x[finite], side="right")
    return codes


def discrete_mutual_info(a, b) -> float:
    """Plug-in mutual information (nats) between two discrete vectors."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same length")
    n = a.shape[0]
    if n == 0:
        return 0.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(joint, (ai, bi), 1.0)
    joint /= n
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(max(0.0, np.sum(joint[nz] * np.log(joint[nz] / (pa @ pb)[nz]))))


class MRMRSelection(Transformer):
    """Minimum-redundancy maximum-relevance selection (Peng, Long & Ding, 2005).

    Greedily picks ``k`` features. The first is the one with the highest
    mutual information with the target; each later step picks the feature
    maximizing

    * ``"difference"`` (MID): ``I(f; y) - mean_{s in S} I(f; s)``, or
    * ``"quotient"`` (MIQ): ``I(f; y) / (eps + mean_{s in S} I(f; s))``

    over the already selected set ``S``. Unlike :class:`MutualInfoSelection`,
    which ranks features independently, a feature that duplicates one already
    chosen is penalized, so the selected set is informative *and*
    non-redundant.

    Mutual information is the plug-in estimate on discretized values: numeric
    features are cut into ``n_bins`` equal-frequency bins (NaN is its own
    bin); the target is used as-is for ``task="classification"`` and
    quantile-binned for ``task="regression"``. This is the discretized
    setting of the original paper and makes the selection deterministic.

    Attributes
    ----------
    selected_:
        Selected columns in pick order.
    relevance_:
        ``I(f; y)`` for every numeric column (nats).
    scores_:
        mRMR criterion value of each selected column at the step it was
        picked.
    drop_:
        Numeric columns not selected (non-numeric columns pass through).
    """

    def __init__(
        self,
        target: str,
        k: int = 10,
        scheme: str = "difference",
        task: str = "classification",
        n_bins: int = 10,
    ):
        if scheme not in ("difference", "quotient"):
            raise ValueError("scheme must be 'difference' or 'quotient'")
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError("k must be a positive integer")
        self.target = target
        self.k = int(k)
        self.scheme = scheme
        self.task = task
        self.n_bins = int(n_bins)
        quantile_discretize([0.0, 1.0], self.n_bins)  # validate n_bins early

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        if y is None:
            raise ValueError("MRMRSelection.fit requires y")
        numeric = _numeric_columns(X)
        y_arr = np.asarray(pd.Series(y).to_numpy()).ravel()
        if y_arr.shape[0] != len(X):
            raise ValueError("y must have one value per row of X")
        self.selected_: list[str] = []
        self.scores_: dict[str, float] = {}
        self.relevance_: dict[str, float] = {}
        if not numeric:
            self.drop_ = []
            return None
        if self.task == "regression":
            y_codes = quantile_discretize(y_arr.astype(float), self.n_bins)
        else:
            y_codes = pd.factorize(pd.Series(y_arr), use_na_sentinel=False)[0]
        codes = {c: quantile_discretize(X[c].to_numpy(dtype=float), self.n_bins) for c in numeric}
        relevance = {c: discrete_mutual_info(codes[c], y_codes) for c in numeric}
        self.relevance_ = relevance
        redundancy_sum = {c: 0.0 for c in numeric}
        remaining = list(numeric)
        eps = 1e-12
        while remaining and len(self.selected_) < self.k:
            m = len(self.selected_)
            best, best_score = None, -np.inf
            for c in remaining:
                if m == 0:
                    score = relevance[c]
                else:
                    red = redundancy_sum[c] / m
                    if self.scheme == "difference":
                        score = relevance[c] - red
                    else:
                        score = relevance[c] / (red + eps) if relevance[c] > 0 else 0.0
                if score > best_score + 1e-15:
                    best, best_score = c, score
            self.selected_.append(best)
            self.scores_[best] = float(best_score)
            remaining.remove(best)
            for c in remaining:
                redundancy_sum[c] += discrete_mutual_info(codes[c], codes[best])
        self.drop_ = [c for c in numeric if c not in self.selected_]
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


def _relieff_feature_types(
    X: np.ndarray, categorical_features, categorical_threshold: int
) -> np.ndarray:
    """Boolean mask, True for categorical (discrete) columns."""
    p = X.shape[1]
    if categorical_features is not None:
        mask = np.zeros(p, dtype=bool)
        for j in categorical_features:
            if not 0 <= int(j) < p:
                raise ValueError(f"categorical feature index {j} out of range")
            mask[int(j)] = True
        return mask
    out = np.zeros(p, dtype=bool)
    for j in range(p):
        col = X[:, j]
        out[j] = np.unique(col[~np.isnan(col)]).size <= categorical_threshold
    return out


def relieff_scores(
    X,
    y,
    n_neighbors: int = 10,
    categorical_features=None,
    categorical_threshold: int = 10,
) -> np.ndarray:
    """ReliefF feature weights (Kononenko, 1994), one per column of ``X``.

    Each instance looks at its ``n_neighbors`` nearest *hits* (same class) and,
    for every other class, its ``n_neighbors`` nearest *misses*. A feature
    gains weight when it differs on misses and loses weight when it differs on
    hits, so it captures interactions that univariate filters miss (XOR-style
    targets). For each instance, hits and misses are averaged separately, and
    the total is divided by ``n``, so the weights lie in ``[-1, 1]``. For
    multiclass targets, each miss class is weighted by its share of the
    misses (the ReliefF rule).

    Conventions follow scikit-rebate's ``ReliefF``, so for complete data the
    scores agree to floating-point precision:

    * columns with at most ``categorical_threshold`` distinct values are
      categorical (unless ``categorical_features`` lists the categorical
      column indices explicitly). Their diff is 0/1 mismatch;
    * a continuous diff is ``|a - b| / range``. With mixed data the diff
      saturates at 1 once ``|a - b|`` exceeds the column's standard deviation
      (ReBATE's ramp);
    * the neighbour distance is the Manhattan distance on range-scaled
      continuous columns. With mixed data, the Hamming fraction over the
      categorical columns, multiplied by the number of features, is added.

    Missing values (NaN) are skipped: an instance with a missing feature
    value contributes nothing to that feature, and neighbours that are
    missing it are left out of its average. The neighbour distance is taken
    over the coordinates present in both rows and rescaled by the fraction
    present. This NaN distance is this kit's own choice and is not
    ReBATE's.

    An instance is never its own neighbour. When a class has ``n_neighbors``
    or fewer members, every other member is used, so this matches ReBATE
    whenever ``n_neighbors`` is smaller than the smallest class. (ReBATE
    falls back to counting the instance itself as a hit.)
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    ya = np.asarray(y).ravel()
    n, p = Xa.shape
    if ya.shape[0] != n:
        raise ValueError("y must have one value per row of X")
    if isinstance(n_neighbors, bool) or int(n_neighbors) != n_neighbors or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    k = int(n_neighbors)
    if pd.isna(pd.Series(ya)).any():
        raise ValueError("y must not contain missing values")
    classes, y_codes = np.unique(ya, return_inverse=True)
    if classes.size < 2:
        raise ValueError("ReliefF needs at least two classes")
    if p == 0:
        return np.zeros(0)
    cat = _relieff_feature_types(Xa, categorical_features, int(categorical_threshold))
    cont = ~cat
    nan = np.isnan(Xa)
    with np.errstate(all="ignore"):
        mn = np.where(cont, np.nanmin(np.where(nan, np.inf, Xa), axis=0), 0.0)
        mx = np.where(cont, np.nanmax(np.where(nan, -np.inf, Xa), axis=0), 0.0)
        sd = np.array([np.nanstd(Xa[:, j]) if cont[j] and (~nan[:, j]).any() else 0.0 for j in range(p)])
    rng_ = mx - mn
    rng_safe = np.where(rng_ > 0, rng_, 1.0)
    mixed = bool(cat.any() and cont.any())

    # neighbour distances
    Xc = (Xa[:, cont] - mn[cont]) / rng_safe[cont]
    Xd = Xa[:, cat]
    if not nan.any():
        D = np.zeros((n, n))
        if cont.any():
            D += np.abs(Xc[:, None, :] - Xc[None, :, :]).sum(axis=2)
        if cat.any():
            ham = (Xd[:, None, :] != Xd[None, :, :]).sum(axis=2) / Xd.shape[1]
            D += ham * p if mixed else ham
    else:
        both_c = (~np.isnan(Xc))[:, None, :] & (~np.isnan(Xc))[None, :, :]
        both_d = (~np.isnan(Xd))[:, None, :] & (~np.isnan(Xd))[None, :, :]
        dc = np.where(both_c, np.abs(np.nan_to_num(Xc)[:, None, :] - np.nan_to_num(Xc)[None, :, :]), 0.0).sum(axis=2)
        dd = np.where(both_d, np.nan_to_num(Xd)[:, None, :] != np.nan_to_num(Xd)[None, :, :], False).sum(axis=2)
        present = both_c.sum(axis=2) + both_d.sum(axis=2)
        n_d = max(int(cat.sum()), 1)
        raw = dc + (dd / n_d * p if mixed else dd / n_d)
        D = np.where(present > 0, raw * (p / np.maximum(present, 1)), np.inf)
    np.fill_diagonal(D, np.inf)

    def feature_diff(j: int, a: float, b: np.ndarray) -> np.ndarray:
        if cat[j]:
            return (b != a).astype(float)
        raw = np.abs(a - b)
        if rng_[j] <= 0:
            return np.zeros_like(raw)
        if mixed:
            return np.where(raw > sd[j], 1.0, raw / rng_[j])
        return raw / rng_[j]

    n_classes = classes.size
    weights = np.zeros(p)
    for i in range(n):
        order = np.argsort(D[i])
        order = order[np.isfinite(D[i, order])]
        yo = y_codes[order]
        hits = order[yo == y_codes[i]][:k]
        miss_by_class = {c: order[yo == c][:k] for c in range(n_classes) if c != y_codes[i]}
        for j in range(p):
            a = Xa[i, j]
            if np.isnan(a):
                continue
            h = hits[~nan[hits, j]]
            if n_classes == 2:
                (m,) = miss_by_class.values()
                m = m[~nan[m, j]]
                ch, cm = h.size, m.size
                if ch == 0 and cm == 0:
                    continue
                dh = -feature_diff(j, a, Xa[h, j]).sum() if ch else 0.0
                dm = feature_diff(j, a, Xa[m, j]).sum() if cm else 0.0
                val = (dh / ch if ch else 0.0) + (dm / cm if cm else 0.0)
            else:
                ms = {c: idx[~nan[idx, j]] for c, idx in miss_by_class.items()}
                cm = sum(v.size for v in ms.values())
                ch = h.size
                if ch == 0 and cm == 0:
                    continue
                val = 0.0
                if cm:
                    acc = 0.0
                    for v in ms.values():
                        if v.size:
                            acc += feature_diff(j, a, Xa[v, j]).sum() * (v.size / cm) * len(ms)
                    val += acc / cm
                if ch:
                    val += -feature_diff(j, a, Xa[h, j]).sum() / ch
            weights[j] += val / n
    return weights


class ReliefFSelection(Transformer):
    """Keep the ``k`` numeric columns with the highest ReliefF weight.

    ReliefF (Kononenko, 1994) scores each feature by how well it separates
    nearby instances of different classes, compared with nearby instances
    of the same class. Because neighbours are found in the full feature
    space, it rewards features that matter only *jointly* (for example
    ``y = a XOR b``), which ranking by mutual information or correlation
    cannot detect. See :func:`relieff_scores` for the exact scoring
    conventions. Columns are only scale-normalised (by range), so encode
    categoricals numerically first. Non-numeric columns pass through.

    Attributes
    ----------
    scores_:
        ReliefF weight for every numeric column.
    selected_:
        Kept columns, best first.
    drop_:
        Numeric columns not kept.
    """

    def __init__(
        self,
        target: str,
        k: int = 10,
        n_neighbors: int = 10,
        categorical_threshold: int = 10,
    ):
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError("k must be a positive integer")
        if isinstance(n_neighbors, bool) or int(n_neighbors) != n_neighbors or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        self.target = target
        self.k = int(k)
        self.n_neighbors = int(n_neighbors)
        self.categorical_threshold = int(categorical_threshold)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        if y is None:
            raise ValueError("ReliefFSelection.fit requires y")
        numeric = _numeric_columns(X)
        y_arr = np.asarray(pd.Series(y).to_numpy()).ravel()
        if y_arr.shape[0] != len(X):
            raise ValueError("y must have one value per row of X")
        self.scores_: dict[str, float] = {}
        self.selected_: list[str] = []
        if not numeric:
            self.drop_ = []
            return None
        w = relieff_scores(
            X[numeric].to_numpy(dtype=float),
            y_arr,
            n_neighbors=self.n_neighbors,
            categorical_threshold=self.categorical_threshold,
        )
        self.scores_ = {c: float(v) for c, v in zip(numeric, w)}
        order = sorted(range(len(numeric)), key=lambda i: (-w[i], i))
        self.selected_ = [numeric[i] for i in order[: self.k]]
        self.drop_ = [c for c in numeric if c not in self.selected_]
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.drop_ if c in X.columns])


__all__ = [
    "VarianceThreshold",
    "CorrelationFilter",
    "MutualInfoSelection",
    "SelectFromModel",
    "MRMRSelection",
    "ReliefFSelection",
    "relieff_scores",
    "discrete_mutual_info",
    "quantile_discretize",
]
