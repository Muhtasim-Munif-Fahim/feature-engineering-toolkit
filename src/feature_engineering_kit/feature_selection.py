"""Feature-selection transformers.

These selectors are column-wise and designed to run after encoding so every
column is numeric. They expose the same ``fit``/``transform`` interface, so they
compose inside :class:`~feature_engineering_kit.pipeline.FeatureEngineeringPipeline`.
:class:`MRMRSelection` adds minimum-redundancy maximum-relevance selection on
discretized mutual information (Peng, Long & Ding, 2005), and
:class:`ReliefFSelection` ranks features by ReliefF nearest-hit/miss weights
(Kononenko, 1994), which also catch features that only matter jointly.
:class:`BorutaSelection` is the all-relevant Boruta wrapper (Kursa &
Rudnicki, 2010): features must beat randomly permuted "shadow" copies of
themselves in tree importance more often than chance.
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


def _is_classification_target(y: np.ndarray) -> bool:
    if y.dtype.kind in "OUSb":
        return True
    if y.dtype.kind in "iu":
        return len(np.unique(y)) <= max(20, int(0.05 * len(y)))
    return False


class BorutaSelection(Transformer):
    """All-relevant feature selection with Boruta (Kursa & Rudnicki, 2010).

    Each iteration appends a *shadow* copy of every still-active numeric
    column with its rows independently permuted, fits a tree ensemble on the
    real + shadow matrix, and records a **hit** for every real feature whose
    importance exceeds the ``perc``-th percentile of the shadow importances
    (``perc=100`` is the classic "max shadow" rule). After ``t`` iterations
    a feature's hit count is ``Binomial(t, 0.5)`` under the null of being no
    better than noise, so two one-sided binomial tests decide it:

    * **confirmed** when ``P(H >= hits) < alpha``;
    * **rejected** when ``P(H <= hits) < alpha`` (it is then removed from
      later fits);
    * **tentative** if still undecided after ``max_iter`` iterations.

    With ``bonferroni=True`` (default, as in the reference R package)
    ``alpha`` is divided by the number of undecided features at each test.

    Parameters
    ----------
    target:
        Name used to label the target (values come from ``y``).
    estimator:
        Optional scikit-learn estimator exposing ``feature_importances_``. By
        default a ``RandomForestClassifier`` or ``RandomForestRegressor``
        (``max_depth=5``, balanced classes) is chosen from ``y``.
    n_estimators, max_iter, alpha, perc, bonferroni:
        See above.
    keep_tentative:
        Also keep tentative features (default only confirmed ones are kept).
    max_features:
        Optional cap: keep at most this many kept features, ranked by mean
        importance across iterations.
    random_state:
        Seed for shadow permutations and the default forest.
    shadow_all:
        Keep a shadow for *every* numeric column in each iteration, including
        rejected ones (default). The reference implementations only shadow
        still-active columns, which makes the max-shadow threshold easier to
        beat as features are rejected and inflates false confirmations on
        small samples; ``shadow_all=False`` restores that behaviour.

    Attributes
    ----------
    status_:
        ``"confirmed"``, ``"tentative"`` or ``"rejected"`` per numeric column.
    hits_:
        Hit counts per numeric column.
    decided_at_:
        Iteration at which each confirmed/rejected column was decided.
    importance_history_:
        DataFrame (iterations x numeric columns) of importances; ``NaN`` once a
        column is rejected. Column ``"__shadow_threshold__"`` holds the shadow
        cut-off used each iteration.
    n_iter_:
        Iterations actually run (stops early once every feature is decided).
    selected_, drop_:
        Kept and dropped numeric columns.
    """

    def __init__(
        self,
        target: str,
        estimator=None,
        n_estimators: int = 100,
        max_iter: int = 100,
        alpha: float = 0.05,
        perc: float = 100.0,
        bonferroni: bool = True,
        keep_tentative: bool = False,
        max_features: int | None = None,
        random_state: int | None = 0,
        shadow_all: bool = True,
    ):
        if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if isinstance(n_estimators, bool) or int(n_estimators) != n_estimators or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < float(perc) <= 100.0:
            raise ValueError("perc must be in (0, 100]")
        if max_features is not None and (
            isinstance(max_features, bool) or int(max_features) != max_features or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer or None")
        self.target = target
        self.estimator = estimator
        self.n_estimators = int(n_estimators)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.perc = float(perc)
        self.bonferroni = bool(bonferroni)
        self.keep_tentative = bool(keep_tentative)
        self.max_features = None if max_features is None else int(max_features)
        self.random_state = random_state
        self.shadow_all = bool(shadow_all)

    def _make_estimator(self, y: np.ndarray):
        if self.estimator is not None:
            return _clone_estimator(self.estimator)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if _is_classification_target(y):
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                class_weight="balanced",
                n_jobs=1,
                random_state=self.random_state,
            )
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=5,
            n_jobs=1,
            random_state=self.random_state,
        )

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        from scipy.stats import binom

        if y is None:
            raise ValueError("BorutaSelection.fit requires y")
        numeric = _numeric_columns(X)
        y_arr = np.asarray(pd.Series(y).to_numpy()).ravel()
        if y_arr.shape[0] != len(X):
            raise ValueError("y must have one value per row of X")
        self.status_: dict[str, str] = {}
        self.hits_: dict[str, int] = {}
        self.decided_at_: dict[str, int] = {}
        self.selected_: list[str] = []
        self.drop_: list[str] = []
        self.n_iter_ = 0
        if not numeric:
            self.importance_history_ = pd.DataFrame()
            return None
        values = X[numeric].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("BorutaSelection needs finite numeric columns; impute first")
        p = len(numeric)
        rng = np.random.default_rng(self.random_state)
        hits = np.zeros(p, dtype=int)
        status = np.zeros(p, dtype=int)  # 0 undecided, 1 confirmed, -1 rejected
        decided = np.zeros(p, dtype=int)
        history: list[np.ndarray] = []
        for it in range(1, self.max_iter + 1):
            active = np.flatnonzero(status >= 0)
            real = values[:, active]
            shadow_src = values if self.shadow_all else real
            shadow = np.column_stack(
                [rng.permutation(shadow_src[:, j]) for j in range(shadow_src.shape[1])]
            )
            model = self._make_estimator(y_arr)
            if hasattr(model, "random_state") and self.estimator is None:
                model.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
            model.fit(np.hstack([real, shadow]), y_arr)
            imp = np.asarray(getattr(model, "feature_importances_"), dtype=float)
            if imp.shape[0] != len(active) + shadow.shape[1]:
                raise AttributeError("estimator must expose one feature_importances_ value per column")
            imp_real, imp_shadow = imp[: len(active)], imp[len(active):]
            threshold = float(np.percentile(imp_shadow, self.perc))
            row = np.full(p + 1, np.nan)
            row[active] = imp_real
            row[p] = threshold
            history.append(row)
            hits[active[imp_real > threshold]] += 1
            undecided = np.flatnonzero(status == 0)
            level = self.alpha / len(undecided) if (self.bonferroni and len(undecided)) else self.alpha
            for j in undecided:
                p_accept = float(binom.sf(hits[j] - 1, it, 0.5))
                p_reject = float(binom.cdf(hits[j], it, 0.5))
                if p_accept < level:
                    status[j] = 1
                    decided[j] = it
                elif p_reject < level:
                    status[j] = -1
                    decided[j] = it
            self.n_iter_ = it
            if not np.any(status == 0):
                break
        names = {1: "confirmed", 0: "tentative", -1: "rejected"}
        self.status_ = {c: names[int(s)] for c, s in zip(numeric, status)}
        self.hits_ = {c: int(h) for c, h in zip(numeric, hits)}
        self.decided_at_ = {c: int(d) for c, d, s in zip(numeric, decided, status) if s != 0}
        self.importance_history_ = pd.DataFrame(history, columns=numeric + ["__shadow_threshold__"])
        mean_imp = self.importance_history_[numeric].mean(axis=0, skipna=True).fillna(0.0)
        keep_codes = (1, 0) if self.keep_tentative else (1,)
        kept = [c for c, s in zip(numeric, status) if int(s) in keep_codes]
        kept.sort(key=lambda c: (-float(mean_imp[c]), numeric.index(c)))
        if self.max_features is not None:
            kept = kept[: self.max_features]
        self.selected_ = kept
        self.drop_ = [c for c in numeric if c not in kept]
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
    "BorutaSelection",
    "relieff_scores",
    "discrete_mutual_info",
    "quantile_discretize",
]
