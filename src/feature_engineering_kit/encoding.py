"""Categorical feature encoders.

Each encoder is a :class:`~feature_engineering_kit.base.Transformer`.
:class:`RareCategoryGrouper` collapses infrequent levels and stays categorical.
The other encoders replace object/string columns with numeric representations
so the engineered frame is consumable by scikit-learn estimators. Target-aware
encoders include K-fold :class:`TargetEncoder`, leave-one-out
:class:`LeaveOneOutEncoder`, James-Stein shrinkage
:class:`JamesSteinEncoder`, and :class:`WoEEncoder`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer


class OneHotEncoder(Transformer):
    """One-hot encode categorical columns.

    Parameters
    ----------
    columns:
        Categorical columns to encode.
    drop:
        ``None`` (keep every category) or ``"first"`` (drop the first sorted
        category of each column to reduce multicollinearity).
    dtype:
        Output dtype for the indicator columns (default ``"int"``).
    """

    def __init__(self, columns, drop=None, dtype="int"):
        self.columns = list(columns)
        self.drop = drop
        self.dtype = dtype

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.categories_: dict[str, list[str]] = {}
        for col in self.columns:
            cats = sorted({str(v) for v in X[col].dropna().tolist()})
            if self.drop == "first" and cats:
                cats = cats[1:]
            self.categories_[col] = cats
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            vals = out[col].astype(object)
            for cat in self.categories_[col]:
                out[f"{col}__{cat}"] = (vals == cat).fillna(False).astype(self.dtype)
            out = out.drop(columns=[col])
        return out


class OrdinalEncoder(Transformer):
    """Ordinal-encode categorical columns to integer codes.

    Categories are sorted and mapped to ``1..k``; unseen or missing values map
    to ``0``.
    """

    def __init__(self, columns, dtype="int"):
        self.columns = list(columns)
        self.dtype = dtype

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.categories_: dict[str, list[str]] = {}
        self.mapping_: dict[str, dict[str, int]] = {}
        for col in self.columns:
            cats = sorted({str(v) for v in X[col].dropna().tolist()})
            self.categories_[col] = cats
            self.mapping_[col] = {cat: i + 1 for i, cat in enumerate(cats)}
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            mapping = self.mapping_[col]
            out[col] = out[col].astype(object).map(
                lambda v: mapping.get(str(v), 0)
            ).astype(self.dtype)
        return out


class TargetEncoder(Transformer):
    """Mean (target) encoding with Bayesian smoothing and out-of-fold fitting.

    Each category is replaced by a convex combination of the per-category target
    mean and the global target mean, controlled by ``smoothing``. A larger
    ``smoothing`` shrinks estimates harder toward the global mean and guards
    against high-cardinality / rare-category leakage.

    Naive fit-on-all encoding leaks the row's own label into the feature,
    especially when a category is rare. By default ``fit_transform`` uses
    ``cv``-fold *out-of-fold* (OOF) encodings: each training row is encoded
    from the folds it does not belong to. ``transform`` (and therefore any
    held-out / test frame) always uses the global mapping learned from the
    full training set.

    Parameters
    ----------
    columns:
        Categorical columns to encode.
    target:
        Name of the target column in ``y`` (when ``y`` is a DataFrame) or the
        target values (when ``y`` is array-like / Series).
    smoothing:
        Smoothing weight; ``0.0`` recovers the raw per-category mean.
    cv:
        Number of folds for out-of-fold encoding during ``fit_transform``.
        ``None`` disables OOF encoding (fit-on-all; leaky on the training
        rows). Must be ``None`` or an integer ``>= 2``.
    shuffle:
        Whether to shuffle rows before splitting folds. Ignored when
        ``cv is None``.
    random_state:
        Seed forwarded to the fold splitter when ``shuffle`` is True.
    """

    def __init__(
        self,
        columns,
        target,
        smoothing=10.0,
        cv=5,
        shuffle=True,
        random_state=None,
    ):
        self.columns = list(columns)
        self.target = target
        self.smoothing = float(smoothing)
        if cv is not None:
            cv = int(cv)
            if cv < 2:
                raise ValueError("cv must be None or an integer >= 2")
        self.cv = cv
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def _as_target_series(self, y) -> pd.Series:
        if y is None:
            raise ValueError("TargetEncoder.fit requires y")
        if isinstance(y, pd.DataFrame):
            s = y[self.target]
        elif isinstance(y, pd.Series):
            s = y
        else:
            s = pd.Series(np.asarray(y).ravel())
        return s.reset_index(drop=True)

    def _smoothed_mapping(
        self, categories: pd.Series, target: pd.Series, global_mean: float
    ) -> dict[str, float]:
        grouped = target.groupby(categories, sort=False)
        means = grouped.mean()
        counts = grouped.size()
        smooth = (counts * means + self.smoothing * global_mean) / (
            counts + self.smoothing
        )
        return {str(k): float(v) for k, v in smooth.items() if k is not None}

    def _apply_mapping(
        self, series: pd.Series, mapping: dict[str, float], default: float
    ) -> pd.Series:
        return (
            series.astype(object)
            .map(lambda v: mapping.get(str(v), default))
            .astype(float)
        )

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        target = self._as_target_series(y)
        self.global_mean_ = float(target.mean())
        self.maps_: dict[str, dict[str, float]] = {}
        for col in self.columns:
            s = X[col].astype(object).reset_index(drop=True)
            self.maps_[col] = self._smoothed_mapping(s, target, self.global_mean_)
        return None

    def _iter_oof_splits(self, target: pd.Series):
        """Yield ``(train_idx, test_idx)`` for OOF encoding, or nothing.

        Falls back to no splits (caller uses the global mapping) when the
        sample is too small for ``cv`` folds.
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        n = len(target)
        n_splits = min(int(self.cv), n)
        y_arr = np.asarray(target)
        unique, counts = np.unique(y_arr, return_counts=True)
        # Stratify only for low-cardinality (classification-like) targets.
        use_stratified = 2 <= unique.size <= 10 and int(counts.min()) >= 2
        if use_stratified:
            n_splits = min(n_splits, int(counts.min()))
        if n_splits < 2:
            return
        kwargs: dict = {"n_splits": n_splits, "shuffle": self.shuffle}
        if self.shuffle:
            kwargs["random_state"] = self.random_state
        if use_stratified:
            splitter = StratifiedKFold(**kwargs)
            yield from splitter.split(np.zeros(n), y_arr)
        else:
            splitter = KFold(**kwargs)
            yield from splitter.split(np.zeros(n))

    def _encode_oof(self, X: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        X_pos = X.reset_index(drop=True)
        n = len(X_pos)
        oof: dict[str, np.ndarray] = {
            col: np.full(n, self.global_mean_, dtype=float) for col in self.columns
        }
        n_folds = 0
        for train_idx, test_idx in self._iter_oof_splits(target):
            n_folds += 1
            y_tr = target.iloc[train_idx].reset_index(drop=True)
            fold_mean = float(y_tr.mean())
            for col in self.columns:
                s_tr = X_pos[col].astype(object).iloc[train_idx].reset_index(drop=True)
                mapping = self._smoothed_mapping(s_tr, y_tr, fold_mean)
                s_te = X_pos[col].astype(object).iloc[test_idx]
                oof[col][test_idx] = self._apply_mapping(
                    s_te, mapping, fold_mean
                ).to_numpy(dtype=float)
        out = X.copy()
        if n_folds == 0:
            return self._transform(out)
        for col in self.columns:
            out[col] = pd.Series(oof[col], index=X.index, dtype=float)
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.columns_in_ = list(X.columns)
        target = self._as_target_series(y)
        # Always store the full-data mapping so transform() is leakage-free
        # for held-out rows.
        self._fit(X, target)
        if self.cv is None:
            out = self._transform(X)
        else:
            out = self._encode_oof(X, target)
        self.columns_out_ = list(out.columns)
        return out

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            out[col] = self._apply_mapping(
                out[col], self.maps_[col], self.global_mean_
            )
        return out


class LeaveOneOutEncoder(Transformer):
    """Leave-one-out mean target encoding with Bayesian smoothing.

    For each training row, ``fit_transform`` encodes a category as the smoothed
    mean of the target on the *other* rows with that category. The row's own
    label is left out, so a level cannot copy ``y`` the way a fit-on-all mean
    does.

    With category count ``n_c``, category target sum ``S_c``, global target
    mean ``m``, and smoothing weight ``a``:

    ``LOO_i = (S_c - y_i + a * m) / (n_c - 1 + a)``

    ``a = 0`` is the mean of the other rows in the category. A level that
    appears once has no other row; that denominator is zero when ``a = 0``,
    and the value falls back to ``m``. The same fallback is used for missing
    statistics. Every unique level therefore becomes the constant ``m``, not
    that row's label.

    ``transform`` does not leave a row out. Held-out data uses the smoothed
    means learned on the full training set — the same mapping as
    :class:`TargetEncoder` with ``cv=None`` and the same ``smoothing``.
    Unseen categories map to ``m``.

    This is not ``TargetEncoder(cv=n_rows, shuffle=False)``. That setting is
    only leave-one-out for a continuous target. Binary targets use stratified
    folds, which cannot hold out one row at a time. ``LeaveOneOutEncoder``
    always excludes the current row, including for binary ``y``.

    A two-row category with ``a = 0`` swaps the two labels (each row is
    encoded as the other row's target). Prefer :class:`TargetEncoder` with
    ``cv >= 2`` when groups are that small and the encoded column will be
    used to train a model.

    Parameters
    ----------
    columns:
        Categorical columns to encode. Other columns pass through unchanged.
    target:
        Name of the target column in ``y`` (when ``y`` is a DataFrame) or the
        target values (when ``y`` is array-like / Series).
    smoothing:
        Shrinkage toward the global mean. ``0.0`` recovers the raw
        leave-one-out category mean. Must be ``>= 0``. Default ``10.0``,
        matching :class:`TargetEncoder`.
    """

    def __init__(self, columns, target, smoothing=10.0):
        self.columns = list(columns)
        self.target = target
        smoothing = float(smoothing)
        if smoothing < 0.0:
            raise ValueError("smoothing must be >= 0")
        self.smoothing = smoothing

    def _as_target_series(self, y) -> pd.Series:
        if y is None:
            raise ValueError("LeaveOneOutEncoder.fit requires y")
        return _as_target_series(y, self.target)

    def _check_length(self, X: pd.DataFrame, target: pd.Series) -> None:
        if len(target) != len(X):
            raise ValueError(
                "LeaveOneOutEncoder expected y to have "
                f"{len(X)} rows, got {len(target)}"
            )

    def _smoothed_mapping(
        self, categories: pd.Series, target: pd.Series, global_mean: float
    ) -> dict[str, float]:
        grouped = target.groupby(categories, sort=False)
        means = grouped.mean()
        counts = grouped.size()
        smooth = (counts * means + self.smoothing * global_mean) / (
            counts + self.smoothing
        )
        return {str(k): float(v) for k, v in smooth.items() if k is not None}

    def _apply_mapping(
        self, series: pd.Series, mapping: dict[str, float], default: float
    ) -> pd.Series:
        return (
            series.astype(object)
            .map(lambda v: mapping.get(str(v), default))
            .astype(float)
        )

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        target = self._as_target_series(y)
        self._check_length(X, target)
        self.global_mean_ = float(target.mean())
        self.maps_: dict[str, dict[str, float]] = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"column '{col}' not found during fit")
            categories = X[col].astype(object).reset_index(drop=True)
            self.maps_[col] = self._smoothed_mapping(
                categories, target, self.global_mean_
            )
        return None

    def _loo_values(self, categories: pd.Series, target: pd.Series) -> np.ndarray:
        """Smoothed leave-one-out means, aligned to ``categories`` by position."""
        labels = categories.astype(object).reset_index(drop=True)
        y = target.reset_index(drop=True).astype(float)
        y_values = y.to_numpy(dtype=float)
        n = len(y_values)
        global_mean = float(self.global_mean_)
        encoded = np.full(n, global_mean, dtype=float)
        if n == 0:
            return encoded

        stats = pd.DataFrame({"cat": labels, "y": y_values}).groupby(
            "cat", sort=False, dropna=True
        )["y"].agg(sum="sum", count="size")
        cat_sum = labels.map(stats["sum"]).to_numpy(dtype=float)
        cat_count = labels.map(stats["count"]).to_numpy(dtype=float)
        known = np.isfinite(cat_sum) & np.isfinite(cat_count)
        loo_sum = cat_sum - y_values
        loo_count = cat_count - 1.0
        denom = loo_count + self.smoothing
        numer = loo_sum + self.smoothing * global_mean
        np.divide(
            numer,
            denom,
            out=encoded,
            where=known & (denom != 0.0),
        )
        return encoded

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        target = self._as_target_series(y)
        out = X.copy()
        for col in self.columns:
            values = self._loo_values(out[col], target)
            out[col] = pd.Series(values, index=out.index, dtype=float)
        self.columns_out_ = list(out.columns)
        return out

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            out[col] = self._apply_mapping(
                out[col], self.maps_[col], self.global_mean_
            )
        return out


def iv_strength(iv: float) -> str:
    """Siddiqi scorecard label for a column's Information Value.

    Thresholds follow Siddiqi, *Credit Risk Scorecards*:

    * ``< 0.02`` — unpredictive
    * ``0.02``–``0.1`` — weak
    * ``0.1``–``0.3`` — medium
    * ``0.3``–``0.5`` — strong
    * ``>= 0.5`` — suspicious (possible leakage or overfit)
    """
    value = float(iv)
    if value < 0.02:
        return "unpredictive"
    if value < 0.1:
        return "weak"
    if value < 0.3:
        return "medium"
    if value < 0.5:
        return "strong"
    return "suspicious"


def _as_target_series(y, target) -> pd.Series:
    if y is None:
        raise ValueError("fit requires y")
    if isinstance(y, pd.DataFrame):
        s = y[target]
    elif isinstance(y, pd.Series):
        s = y
    else:
        s = pd.Series(np.asarray(y).ravel())
    return s.reset_index(drop=True)


class WoEEncoder(Transformer):
    """Weight of Evidence encoding with Information Value reporting.

    For a binary target, each category ``i`` is replaced by

    ``WoE_i = ln( P(X=i | y=0) / P(X=i | y=1) )``

    after Laplace-style additive ``smoothing``. Categories associated with the
    non-event (``y != event_value``) receive positive WoE; event-heavy
    categories receive negative WoE. Unseen or missing-at-transform-time
    categories map to ``0.0`` (no evidence).

    Information Value is the column-level companion statistic

    ``IV = Σ (P(X=i | y=0) - P(X=i | y=1)) * WoE_i``

    stored on :attr:`iv_` after ``fit``. Use :meth:`iv_report` for the ranked
    column summary and :attr:`iv_table_` for the per-category breakdown.

    Naive fit-on-all WoE leaks a row's own label when a category is rare
    (a unique id encodes to a large |WoE| that reconstructs ``y``). By default
    ``fit_transform`` uses ``cv``-fold *out-of-fold* encodings: each training
    row is encoded from the folds it does not belong to. ``transform`` (and
    therefore any held-out / test frame) always uses the global mapping and
    IV tables learned from the full training set.

    Parameters
    ----------
    columns:
        Categorical columns to encode.
    target:
        Name of the target column in ``y`` (when ``y`` is a DataFrame) or an
        identifier stored for reporting (when ``y`` is array-like / Series).
    smoothing:
        Additive Laplace adjustment applied to event and non-event counts.
        ``0.0`` recovers raw WoE when every cell is positive; empty cells are
        floored so ``log`` stays finite.
    event_value:
        Label treated as the event (default ``1``, e.g. churn / default).
        All other values are the non-event.
    cv:
        Number of folds for out-of-fold encoding during ``fit_transform``.
        ``None`` disables OOF encoding (fit-on-all; leaky on the training
        rows). Must be ``None`` or an integer ``>= 2``.
    shuffle:
        Whether to shuffle rows before splitting folds. Ignored when
        ``cv is None``.
    random_state:
        Seed forwarded to the fold splitter when ``shuffle`` is True.
    """

    def __init__(
        self,
        columns,
        target,
        smoothing=0.5,
        event_value=1,
        cv=5,
        shuffle=True,
        random_state=None,
    ):
        self.columns = list(columns)
        self.target = target
        self.smoothing = float(smoothing)
        if self.smoothing < 0:
            raise ValueError("smoothing must be >= 0")
        self.event_value = event_value
        if cv is not None:
            cv = int(cv)
            if cv < 2:
                raise ValueError("cv must be None or an integer >= 2")
        self.cv = cv
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def _event_mask(self, target: pd.Series) -> pd.Series:
        return (target == self.event_value).astype(int)

    def _validate_target(self, target: pd.Series) -> None:
        if not bool((target == self.event_value).any()):
            raise ValueError(
                f"event_value={self.event_value!r} not found in y; "
                "WoE encoding requires both event and non-event classes"
            )
        if bool((target == self.event_value).all()):
            raise ValueError(
                "y contains only the event class; "
                "WoE encoding requires both event and non-event classes"
            )

    def _woe_mapping(
        self, categories: pd.Series, target: pd.Series
    ) -> tuple[dict[str, float], pd.DataFrame]:
        event = self._event_mask(target)
        n_event = float(event.sum())
        n_non = float(len(event) - n_event)
        if n_event <= 0.0 or n_non <= 0.0:
            return {}, pd.DataFrame(
                columns=[
                    "column",
                    "category",
                    "count",
                    "n_event",
                    "n_non_event",
                    "dist_event",
                    "dist_non_event",
                    "woe",
                    "iv_contribution",
                ]
            )

        cats = categories.astype(object).map(lambda v: "nan" if pd.isna(v) else str(v))
        grouped = event.groupby(cats, sort=False)
        n_i = grouped.size().astype(float)
        n_event_i = grouped.sum().astype(float)
        n_non_i = n_i - n_event_i
        n_cats = float(len(n_i))
        floor = 0.0 if self.smoothing > 0.0 else 1e-12
        dist_non = (n_non_i + self.smoothing).clip(lower=floor) / (
            n_non + self.smoothing * n_cats
        )
        dist_evt = (n_event_i + self.smoothing).clip(lower=floor) / (
            n_event + self.smoothing * n_cats
        )
        woe = np.log(dist_non / dist_evt)
        iv_contrib = (dist_non - dist_evt) * woe
        table = pd.DataFrame(
            {
                "category": [str(k) for k in n_i.index],
                "count": n_i.to_numpy(dtype=float),
                "n_event": n_event_i.to_numpy(dtype=float),
                "n_non_event": n_non_i.to_numpy(dtype=float),
                "dist_event": dist_evt.to_numpy(dtype=float),
                "dist_non_event": dist_non.to_numpy(dtype=float),
                "woe": woe.to_numpy(dtype=float),
                "iv_contribution": iv_contrib.to_numpy(dtype=float),
            }
        )
        mapping = {
            str(cat): float(woe_value)
            for cat, woe_value in zip(table["category"], table["woe"])
        }
        return mapping, table

    def _apply_mapping(self, series: pd.Series, mapping: dict[str, float]) -> pd.Series:
        def _lookup(v):
            if pd.isna(v):
                key = "nan"
            else:
                key = str(v)
            return mapping.get(key, 0.0)

        return series.astype(object).map(_lookup).astype(float)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        target = _as_target_series(y, self.target)
        self._validate_target(target)
        self.maps_: dict[str, dict[str, float]] = {}
        self.iv_: dict[str, float] = {}
        tables: list[pd.DataFrame] = []
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"column '{col}' not found during fit")
            s = X[col].reset_index(drop=True)
            mapping, table = self._woe_mapping(s, target)
            table = table.copy()
            table.insert(0, "column", col)
            self.maps_[col] = mapping
            self.iv_[col] = float(table["iv_contribution"].sum()) if len(table) else 0.0
            tables.append(table)
        self.iv_table_ = (
            pd.concat(tables, ignore_index=True)
            if tables
            else pd.DataFrame(
                columns=[
                    "column",
                    "category",
                    "count",
                    "n_event",
                    "n_non_event",
                    "dist_event",
                    "dist_non_event",
                    "woe",
                    "iv_contribution",
                ]
            )
        )
        return None

    def _iter_oof_splits(self, target: pd.Series):
        """Yield ``(train_idx, test_idx)`` for OOF encoding, or nothing.

        Falls back to no splits (caller uses the global mapping) when the
        sample is too small for ``cv`` folds.
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        n = len(target)
        n_splits = min(int(self.cv), n)
        y_arr = self._event_mask(target).to_numpy()
        unique, counts = np.unique(y_arr, return_counts=True)
        use_stratified = unique.size == 2 and int(counts.min()) >= 2
        if use_stratified:
            n_splits = min(n_splits, int(counts.min()))
        if n_splits < 2:
            return
        kwargs: dict = {"n_splits": n_splits, "shuffle": self.shuffle}
        if self.shuffle:
            kwargs["random_state"] = self.random_state
        if use_stratified:
            splitter = StratifiedKFold(**kwargs)
            yield from splitter.split(np.zeros(n), y_arr)
        else:
            splitter = KFold(**kwargs)
            yield from splitter.split(np.zeros(n))

    def _encode_oof(self, X: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        X_pos = X.reset_index(drop=True)
        n = len(X_pos)
        oof: dict[str, np.ndarray] = {
            col: np.zeros(n, dtype=float) for col in self.columns
        }
        n_folds = 0
        for train_idx, test_idx in self._iter_oof_splits(target):
            n_folds += 1
            y_tr = target.iloc[train_idx].reset_index(drop=True)
            for col in self.columns:
                s_tr = X_pos[col].iloc[train_idx].reset_index(drop=True)
                mapping, _ = self._woe_mapping(s_tr, y_tr)
                s_te = X_pos[col].iloc[test_idx]
                oof[col][test_idx] = self._apply_mapping(s_te, mapping).to_numpy(
                    dtype=float
                )
        out = X.copy()
        if n_folds == 0:
            return self._transform(out)
        for col in self.columns:
            out[col] = pd.Series(oof[col], index=X.index, dtype=float)
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.columns_in_ = list(X.columns)
        target = _as_target_series(y, self.target)
        # Always store the full-data mapping and IV so transform() / iv_report()
        # describe the training set, not a single fold.
        self._fit(X, target)
        if self.cv is None:
            out = self._transform(X)
        else:
            out = self._encode_oof(X, target)
        self.columns_out_ = list(out.columns)
        return out

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            out[col] = self._apply_mapping(out[col], self.maps_[col])
        return out

    def iv_report(self) -> pd.DataFrame:
        """Ranked per-column Information Value table.

        Returns
        -------
        pandas.DataFrame
            Columns ``column``, ``iv``, ``strength``, sorted strongest-first.
        """
        self._check_fitted()
        rows = [
            {
                "column": col,
                "iv": float(self.iv_[col]),
                "strength": iv_strength(self.iv_[col]),
            }
            for col in self.columns
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)


def information_value(
    X: pd.DataFrame,
    y,
    columns,
    *,
    target="y",
    event_value=1,
    smoothing=0.5,
) -> pd.DataFrame:
    """Compute Information Value for ``columns`` without encoding a matrix.

    Fits a :class:`WoEEncoder` on the full sample (``cv=None``) and returns
    :meth:`WoEEncoder.iv_report`. Use the encoder itself when you also need
    leakage-safe WoE features.
    """
    enc = WoEEncoder(
        columns=columns,
        target=target,
        smoothing=smoothing,
        event_value=event_value,
        cv=None,
    )
    enc.fit(X, y)
    return enc.iv_report()


def _category_key(value):
    """Hashable category key. Missing values collapse to ``None``."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _raw_counts(series: pd.Series, *, drop_missing: bool) -> dict:
    counts: dict = {}
    for raw, count in series.value_counts(dropna=drop_missing).items():
        key = _category_key(raw)
        if key is None and drop_missing:
            continue
        counts[key] = counts.get(key, 0) + int(count)
    return counts


def _validate_min_count(min_count, *, allow_none: bool):
    if min_count is None:
        if allow_none:
            return None
        raise ValueError("min_count must be an integer >= 1")
    if isinstance(min_count, bool):
        raise ValueError("min_count must be an integer >= 1")
    if isinstance(min_count, (float, np.floating)):
        if not float(min_count).is_integer():
            raise ValueError("min_count must be an integer >= 1")
        min_count = int(min_count)
    elif isinstance(min_count, (int, np.integer)):
        min_count = int(min_count)
    else:
        raise ValueError("min_count must be an integer >= 1")
    if min_count < 1:
        raise ValueError("min_count must be an integer >= 1")
    return min_count


def _validate_other_label(other_label):
    label = _category_key(other_label)
    if label is None:
        raise ValueError("other_label must be a non-missing value")
    return label


class RareCategoryGrouper(Transformer):
    """Group infrequent categorical levels into a single bucket.

    Levels whose training count is strictly below ``min_count`` are replaced
    by ``other_label``. Levels at or above ``min_count`` are kept as they
    appeared in the data. Categories unseen at transform time are treated as
    rare and mapped to the same bucket. Missing values are preserved so a
    later imputer can still see them.

    The mapping is learned on ``fit`` and reused by ``transform``. A level that
    is rare in a test fold stays rare if it was rare in training, and a level
    that was common in training is kept even if it shows up only once at
    transform time.

    Parameters
    ----------
    columns:
        Categorical columns to group. Other columns pass through unchanged.
    min_count:
        Minimum training count required to keep a level. Default ``2`` groups
        singletons. Must be an integer ``>= 1``.
    other_label:
        Replacement label for rare and unseen levels. Default ``"other"``.
        Must be non-missing. If that label already exists and is itself
        frequent, rare levels are merged into it.
    """

    def __init__(self, columns, min_count=2, other_label="other"):
        self.columns = list(columns)
        self.min_count = _validate_min_count(min_count, allow_none=False)
        self.other_label = _validate_other_label(other_label)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.counts_: dict[str, dict] = {}
        self.kept_: dict[str, set] = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"column '{col}' not found during fit")
            counts = _raw_counts(X[col], drop_missing=True)
            self.counts_[col] = counts
            self.kept_[col] = {
                key for key, count in counts.items() if count >= self.min_count
            }
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            kept = self.kept_[col]
            other = self.other_label
            replaced = []
            for value in out[col].tolist():
                key = _category_key(value)
                if key is None or key in kept:
                    replaced.append(value)
                else:
                    replaced.append(other)
            out[col] = pd.Series(replaced, index=out.index, dtype=object)
        return out


class FrequencyEncoder(Transformer):
    """Replace categories with their training-set frequency.

    By default each level is mapped to its relative frequency ``count / n_rows``.
    Pass ``normalize=False`` to emit raw counts instead. Unseen categories map
    to ``0``. Missing values are their own level and encode to the missing rate
    of the sample the mapping was fit on (``0`` when that sample had no
    missing rows).

    ``min_count`` optionally pools levels below that count into ``other_label``
    before the frequencies are computed. Every pooled level, and any unseen
    non-missing level, then receives the pooled bucket's frequency. When the
    column has no missing values and ``cv is None``, that matches
    :class:`RareCategoryGrouper` followed by a frequency encoder. Missing
    values are not pooled into the rare bucket.

    A row counted in its own frequency slightly inflates rare levels: a unique
    id encodes to ``1/n`` only because of itself. When ``cv`` is an integer
    ``>= 2``, ``fit_transform`` writes *out-of-fold* frequencies. Each training
    row is encoded from the folds it does not belong to, and both the rare
    bucket and the frequencies are recomputed inside each fold. ``transform``
    (and ``fit`` followed by ``transform``) always uses the global mapping
    learned from the full training set. Folds are a plain ``KFold`` because
    frequency encoding does not use ``y``. ``cv=None`` (the default) encodes
    with full-sample frequencies. Setting ``cv`` to the number of rows and
    ``shuffle=False`` is leave-one-out.

    Parameters
    ----------
    columns:
        Categorical columns to encode. Other columns pass through unchanged.
    normalize:
        If True (default), encode ``count / n_rows``. If False, encode raw
        counts.
    min_count:
        If set, levels with training count ``< min_count`` are pooled into
        ``other_label`` before frequencies are computed. ``None`` (default)
        keeps every level. Must be ``None`` or an integer ``>= 1``.
    other_label:
        Bucket label used when ``min_count`` is set. Default ``"other"``.
    cv:
        Number of folds for out-of-fold encoding during ``fit_transform``.
        ``None`` (default) disables it. Must be ``None`` or an integer ``>= 2``.
    shuffle:
        Whether to shuffle rows before splitting folds. Ignored when
        ``cv is None``.
    random_state:
        Seed forwarded to the fold splitter when ``shuffle`` is True.
    """

    def __init__(
        self,
        columns,
        normalize=True,
        min_count=None,
        other_label="other",
        cv=None,
        shuffle=True,
        random_state=None,
    ):
        self.columns = list(columns)
        self.normalize = bool(normalize)
        self.min_count = _validate_min_count(min_count, allow_none=True)
        self.other_label = _validate_other_label(other_label)
        if cv is not None:
            cv = int(cv)
            if cv < 2:
                raise ValueError("cv must be None or an integer >= 2")
        self.cv = cv
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def _mapping_from_counts(self, counts: dict, n: int) -> tuple[dict, float]:
        def encode(count: int) -> float:
            if not self.normalize:
                return float(count)
            if n == 0:
                return 0.0
            return float(count) / float(n)

        if self.min_count is None:
            return {key: encode(count) for key, count in counts.items()}, 0.0

        missing_count = int(counts.get(None, 0))
        kept: dict = {}
        rare_total = 0
        for key, count in counts.items():
            if key is None:
                continue
            if count >= self.min_count:
                kept[key] = count
            else:
                rare_total += int(count)
        if rare_total or self.other_label in kept:
            pooled = int(kept.get(self.other_label, 0)) + int(rare_total)
            kept[self.other_label] = pooled
        mapping = {key: encode(count) for key, count in kept.items()}
        if missing_count:
            mapping[None] = encode(missing_count)
        return mapping, float(mapping.get(self.other_label, 0.0))

    def _apply_mapping(
        self, series: pd.Series, mapping: dict, other_freq: float
    ) -> pd.Series:
        # Walk values explicitly so missing entries are encoded too. ``Series.map``
        # can leave NA untouched, which would drop the fitted missing rate.
        encoded = []
        for value in series.tolist():
            key = _category_key(value)
            if key in mapping:
                encoded.append(mapping[key])
            elif key is None:
                encoded.append(0.0)
            else:
                encoded.append(other_freq)
        return pd.Series(encoded, index=series.index, dtype=float)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.n_samples_ = int(len(X))
        self.counts_: dict[str, dict] = {}
        self.maps_: dict[str, dict] = {}
        self.other_frequency_: dict[str, float] = {}
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"column '{col}' not found during fit")
            counts = _raw_counts(X[col], drop_missing=False)
            mapping, other_freq = self._mapping_from_counts(counts, self.n_samples_)
            self.counts_[col] = counts
            self.maps_[col] = mapping
            self.other_frequency_[col] = other_freq
        return None

    def _iter_oof_splits(self, n: int):
        """Yield ``(train_idx, test_idx)`` for OOF encoding, or nothing.

        Falls back to no splits (caller uses the global mapping) when the
        sample is too small for ``cv`` folds.
        """
        from sklearn.model_selection import KFold

        n_splits = min(int(self.cv), n)
        if n_splits < 2:
            return
        kwargs: dict = {"n_splits": n_splits, "shuffle": self.shuffle}
        if self.shuffle:
            kwargs["random_state"] = self.random_state
        splitter = KFold(**kwargs)
        yield from splitter.split(np.zeros(n))

    def _encode_oof(self, X: pd.DataFrame) -> pd.DataFrame:
        X_pos = X.reset_index(drop=True)
        n = len(X_pos)
        oof: dict[str, np.ndarray] = {
            col: np.zeros(n, dtype=float) for col in self.columns
        }
        n_folds = 0
        for train_idx, test_idx in self._iter_oof_splits(n):
            n_folds += 1
            for col in self.columns:
                train = X_pos[col].iloc[train_idx]
                counts = _raw_counts(train, drop_missing=False)
                mapping, other_freq = self._mapping_from_counts(counts, int(len(train)))
                test = X_pos[col].iloc[test_idx]
                oof[col][test_idx] = self._apply_mapping(
                    test, mapping, other_freq
                ).to_numpy(dtype=float)
        out = X.copy()
        if n_folds == 0:
            return self._transform(out)
        for col in self.columns:
            out[col] = pd.Series(oof[col], index=X.index, dtype=float)
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        if self.cv is None:
            out = self._transform(X)
        else:
            out = self._encode_oof(X)
        self.columns_out_ = list(out.columns)
        return out

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            out[col] = self._apply_mapping(
                out[col], self.maps_[col], self.other_frequency_[col]
            )
        return out


class JamesSteinEncoder(Transformer):
    """James-Stein / empirical-Bayes shrinkage target encoding.

    Each category mean is shrunk toward the global target mean by an amount
    that grows as the category gets rarer. With category count ``n_c``,
    category mean ``m_c``, global mean ``m``, residual variance ``sigma2``,
    and between-category variance ``tau2``:

    ``B_c = sigma2 / (sigma2 + n_c * tau2)``
    ``JS_c = (1 - B_c) * m_c + B_c * m``

    ``sigma2`` is the pooled within-category variance of ``y``. ``tau2`` is
    the positive part of the method-of-moments estimate
    ``Var(m_c) - mean(sigma2 / n_c)``. When ``tau2`` collapses to zero (or
    there are fewer than two categories), every level maps to ``m``.

    Unseen categories at transform time map to ``m``. Missing values are
    treated as their own stringified category when present at fit time.

    Parameters
    ----------
    columns:
        Categorical columns to encode. Other columns pass through unchanged.
    target:
        Name of the target column in ``y`` (when ``y`` is a DataFrame) or the
        target values (when ``y`` is array-like / Series).
    """

    def __init__(self, columns, target):
        self.columns = list(columns)
        self.target = target

    def _as_target_series(self, y) -> pd.Series:
        if y is None:
            raise ValueError("JamesSteinEncoder.fit requires y")
        return _as_target_series(y, self.target)

    def _check_length(self, X: pd.DataFrame, target: pd.Series) -> None:
        if len(target) != len(X):
            raise ValueError(
                "JamesSteinEncoder expected y to have "
                f"{len(X)} rows, got {len(target)}"
            )

    def _js_mapping(
        self, categories: pd.Series, target: pd.Series, global_mean: float
    ) -> dict[str, float]:
        labels = categories.astype(object).reset_index(drop=True)
        y = target.reset_index(drop=True).astype(float)
        frame = pd.DataFrame({"cat": labels, "y": y.to_numpy(dtype=float)})
        grouped = frame.groupby("cat", sort=False, dropna=True)["y"]
        counts = grouped.size()
        means = grouped.mean()
        # Pooled within-category residual variance (ddof=1 when possible).
        ss = grouped.apply(lambda s: float(((s - s.mean()) ** 2).sum()))
        df = (counts - 1).clip(lower=0)
        denom = float(df.sum())
        if denom > 0.0:
            sigma2 = float(ss.sum() / denom)
        else:
            sigma2 = 0.0
        if not np.isfinite(sigma2) or sigma2 < 0.0:
            sigma2 = 0.0

        n_cats = int(counts.shape[0])
        if n_cats < 2 or sigma2 == 0.0:
            tau2 = 0.0
        else:
            mean_of_means = float(means.mean())
            between = float(((means - mean_of_means) ** 2).sum() / (n_cats - 1))
            expected_noise = float((sigma2 / counts).mean())
            tau2 = max(0.0, between - expected_noise)

        mapping: dict[str, float] = {}
        for cat, n_c, m_c in zip(counts.index.tolist(), counts.to_numpy(), means.to_numpy()):
            n_c = float(n_c)
            m_c = float(m_c)
            if tau2 <= 0.0:
                mapping[str(cat)] = float(global_mean)
            else:
                shrink = sigma2 / (sigma2 + n_c * tau2)
                mapping[str(cat)] = float((1.0 - shrink) * m_c + shrink * global_mean)
        self.sigma2_ = float(sigma2)
        self.tau2_ = float(tau2)
        return mapping

    def _apply_mapping(
        self, series: pd.Series, mapping: dict[str, float], default: float
    ) -> pd.Series:
        return (
            series.astype(object)
            .map(lambda v: mapping.get(str(v), default))
            .astype(float)
        )

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        target = self._as_target_series(y)
        self._check_length(X, target)
        if not np.all(np.isfinite(target.to_numpy(dtype=float))):
            raise ValueError("JamesSteinEncoder requires finite target values")
        self.global_mean_ = float(target.mean())
        self.maps_: dict[str, dict[str, float]] = {}
        # Per-column variance estimates; last fit wins for the attrs used in tests.
        self.sigma2_ = 0.0
        self.tau2_ = 0.0
        for col in self.columns:
            if col not in X.columns:
                raise KeyError(f"column '{col}' not found during fit")
            categories = X[col].astype(object).reset_index(drop=True)
            self.maps_[col] = self._js_mapping(categories, target, self.global_mean_)
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not found during transform")
            out[col] = self._apply_mapping(
                out[col], self.maps_[col], self.global_mean_
            )
        return out



__all__ = [
    "OneHotEncoder",
    "OrdinalEncoder",
    "RareCategoryGrouper",
    "FrequencyEncoder",
    "TargetEncoder",
    "LeaveOneOutEncoder",
    "JamesSteinEncoder",
    "WoEEncoder",
    "information_value",
    "iv_strength",
]
