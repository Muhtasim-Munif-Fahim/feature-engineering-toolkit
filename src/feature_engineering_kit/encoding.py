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


__all__ = ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"]
