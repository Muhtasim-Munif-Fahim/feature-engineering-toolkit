"""Group-by aggregation features.

``GroupAggregateFeatures`` learns per-group statistics of numeric columns on
the training data (``mean``, ``std``, ``min``, ``max``, ``median``, ``sum``,
``count``, ``nunique``) and maps them back onto each row by its group key —
the classic "average spend of this customer's region" feature. It can also
emit *relative* features: ``diff`` (value minus group mean) and ``ratio``
(value divided by group mean), and z-scores within the group (``zscore``).

Rows whose group was not seen during ``fit`` (or whose group was too small,
see ``min_group_size``) fall back to the global statistic computed on the
whole training column, so ``transform`` never produces NaN for unseen keys.
Missing group keys form their own group.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .base import Transformer

__all__ = ["GroupAggregateFeatures"]

_AGGS = ("mean", "std", "min", "max", "median", "sum", "count", "nunique")
_RELATIVE = ("diff", "ratio", "zscore")
_MISSING_KEY = "__missing__"


def _global_stat(series: pd.Series, agg: str) -> float:
    s = series.dropna()
    if agg == "count":
        return float(s.size)
    if agg == "nunique":
        return float(s.nunique())
    if s.empty:
        return float("nan")
    if agg == "std":
        return float(s.std(ddof=1)) if s.size > 1 else 0.0
    return float(getattr(s, agg)())


class GroupAggregateFeatures(Transformer):
    """Append per-group aggregate statistics as new columns.

    Parameters
    ----------
    group_by:
        A column name or list of column names forming the group key.
    columns:
        Numeric columns to aggregate.
    aggs:
        Aggregations to compute; subset of
        ``("mean", "std", "min", "max", "median", "sum", "count", "nunique")``.
    relative:
        Optional relative features; subset of ``("diff", "ratio", "zscore")``.
        ``diff = x - group_mean``, ``ratio = x / group_mean`` and
        ``zscore = (x - group_mean) / group_std`` (0 when ``group_std`` is 0).
    min_group_size:
        Groups with fewer non-missing training values than this use the
        global statistic instead (shrinks noisy small-group estimates).
    drop_original:
        Drop the aggregated source columns from the output.
    """

    def __init__(
        self,
        group_by,
        columns: Sequence[str],
        aggs: Sequence[str] = ("mean",),
        relative: Sequence[str] = (),
        min_group_size: int = 1,
        drop_original: bool = False,
    ):
        self.group_by = [group_by] if isinstance(group_by, str) else list(group_by)
        self.columns = list(columns)
        self.aggs = list(aggs)
        self.relative = list(relative)
        self.min_group_size = min_group_size
        self.drop_original = drop_original
        if not self.group_by:
            raise ValueError("group_by must name at least one column")
        if not self.columns:
            raise ValueError("columns must not be empty")
        if set(self.group_by) & set(self.columns):
            raise ValueError("group_by and columns must not overlap")
        bad = [a for a in self.aggs if a not in _AGGS]
        if bad:
            raise ValueError(f"unknown aggs {bad}; choose from {_AGGS}")
        bad = [r for r in self.relative if r not in _RELATIVE]
        if bad:
            raise ValueError(f"unknown relative features {bad}; choose from {_RELATIVE}")
        if int(min_group_size) != min_group_size or min_group_size < 1:
            raise ValueError("min_group_size must be an integer >= 1")

    # -- helpers ---------------------------------------------------------
    @property
    def _key_name(self) -> str:
        return "_".join(self.group_by)

    def _keys(self, X: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.group_by if c not in X.columns]
        if missing:
            raise KeyError(f"group_by columns not in X: {missing}")
        parts = X[self.group_by].astype(object).where(X[self.group_by].notna(), _MISSING_KEY)
        if len(self.group_by) == 1:
            return parts.iloc[:, 0]
        return pd.Series(list(map(tuple, parts.to_numpy())), index=X.index)

    def _needed_stats(self) -> list[str]:
        stats = list(self.aggs)
        if self.relative and "mean" not in stats:
            stats.append("mean")
        if "zscore" in self.relative and "std" not in stats:
            stats.append("std")
        return stats

    # -- Transformer API -------------------------------------------------
    def _fit(self, X: pd.DataFrame, y=None) -> None:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"columns not in X: {missing}")
        keys = self._keys(X)
        stats = self._needed_stats()
        self.group_stats_: dict[str, dict[str, dict]] = {}
        self.global_stats_: dict[str, dict[str, float]] = {}
        for col in self.columns:
            values = pd.to_numeric(X[col], errors="raise").astype(float)
            grouped = values.groupby(keys.to_numpy(), sort=False)
            sizes = grouped.count()
            small = sizes[sizes < self.min_group_size].index
            self.global_stats_[col] = {a: _global_stat(values, a) for a in stats}
            per_stat = {}
            for agg in stats:
                if agg == "std":
                    s = grouped.std(ddof=1).fillna(0.0)
                elif agg == "count":
                    s = sizes.astype(float)
                else:
                    s = getattr(grouped, agg)().astype(float)
                s = s.drop(index=small, errors="ignore")
                per_stat[agg] = s.to_dict()
            self.group_stats_[col] = per_stat
        self.n_groups_ = int(keys.nunique())

    def _lookup(self, keys: pd.Series, col: str, agg: str) -> np.ndarray:
        mapping = self.group_stats_[col][agg]
        fallback = self.global_stats_[col][agg]
        return np.array([mapping.get(k, fallback) for k in keys], dtype=float)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keys = self._keys(X)
        out = X.copy()
        key = self._key_name
        for col in self.columns:
            for agg in self.aggs:
                out[f"{col}_{agg}_by_{key}"] = self._lookup(keys, col, agg)
            if self.relative:
                x = pd.to_numeric(X[col], errors="raise").astype(float).to_numpy()
                mean = self._lookup(keys, col, "mean")
                if "diff" in self.relative:
                    out[f"{col}_diff_mean_by_{key}"] = x - mean
                if "ratio" in self.relative:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(mean != 0, x / np.where(mean == 0, 1.0, mean), np.nan)
                    out[f"{col}_ratio_mean_by_{key}"] = ratio
                if "zscore" in self.relative:
                    std = self._lookup(keys, col, "std")
                    z = np.where(std > 0, (x - mean) / np.where(std > 0, std, 1.0), 0.0)
                    out[f"{col}_zscore_by_{key}"] = z
        if self.drop_original:
            out = out.drop(columns=self.columns)
        return out
