"""Categorical feature encoders.

Each encoder is a :class:`~feature_engineering_kit.base.Transformer` that
replaces one or more object/string columns with numeric representations so the
engineered frame is consumable by scikit-learn estimators.
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


__all__ = [
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "WoEEncoder",
    "information_value",
    "iv_strength",
]
