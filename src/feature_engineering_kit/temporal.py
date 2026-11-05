"""Leakage-safe lag, difference and rolling-window features.

``LagFeatures`` builds the standard time-series / panel features from ordered
rows: ``x_lag_k`` (value ``k`` steps back), ``x_diff_k`` (``x - x_lag_k``) and
rolling statistics over the previous ``w`` rows (``x_roll{w}_mean`` ...).

Rolling windows are *closed on the left*: the window for row ``t`` covers rows
``t - w .. t - 1`` and never includes the current value, so the features can
be used to predict ``x_t`` without target leakage. Optional ``group_by``
computes everything within each entity (e.g. per store or per customer) and
``order_by`` sorts by time inside each group; the output always preserves the
original row order and index.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .base import Transformer

__all__ = ["LagFeatures"]

_WINDOW_AGGS = ("mean", "std", "min", "max", "sum", "median")


def _positive_ints(values, name: str) -> list[int]:
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must contain positive integers, got {v!r}")
        out.append(int(v))
    return out


class LagFeatures(Transformer):
    """Append lag, diff and left-closed rolling-window features.

    Parameters
    ----------
    columns:
        Numeric columns to derive features from.
    lags:
        Lag offsets; creates ``{col}_lag_{k}``.
    diffs:
        Difference offsets; creates ``{col}_diff_{k} = x - x.shift(k)``.
    windows:
        Rolling window sizes over previous rows; creates
        ``{col}_roll{w}_{agg}`` for every ``agg`` in ``window_aggs``.
    window_aggs:
        Subset of ``("mean", "std", "min", "max", "sum", "median")``.
    group_by:
        Optional column name(s); features are computed within each group.
    order_by:
        Optional column to sort by (stable) before shifting.
    min_periods:
        Minimum number of past observations for a rolling value (else NaN).
    fill_value:
        Value used to fill the leading NaNs (``None`` keeps NaN).
    drop_original:
        Drop the source columns from the output.
    """

    def __init__(
        self,
        columns: Sequence[str],
        lags: Sequence[int] = (1,),
        diffs: Sequence[int] = (),
        windows: Sequence[int] = (),
        window_aggs: Sequence[str] = ("mean",),
        group_by=None,
        order_by: str | None = None,
        min_periods: int = 1,
        fill_value: float | None = None,
        drop_original: bool = False,
    ):
        self.columns = [columns] if isinstance(columns, str) else list(columns)
        if not self.columns:
            raise ValueError("columns must not be empty")
        self.lags = _positive_ints(lags, "lags")
        self.diffs = _positive_ints(diffs, "diffs")
        self.windows = _positive_ints(windows, "windows")
        bad = [a for a in window_aggs if a not in _WINDOW_AGGS]
        if bad:
            raise ValueError(f"unknown window_aggs {bad}; choose from {_WINDOW_AGGS}")
        self.window_aggs = list(window_aggs)
        if not (self.lags or self.diffs or self.windows):
            raise ValueError("at least one of lags, diffs or windows is required")
        if group_by is None:
            self.group_by: list[str] = []
        else:
            self.group_by = [group_by] if isinstance(group_by, str) else list(group_by)
        self.order_by = order_by
        if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
            raise ValueError("min_periods must be a positive integer")
        self.min_periods = min_periods
        self.fill_value = fill_value
        self.drop_original = drop_original

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        needed = self.columns + self.group_by + ([self.order_by] if self.order_by else [])
        missing = [c for c in needed if c not in X.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")
        for col in self.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise TypeError(f"column {col!r} must be numeric")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._fit(X)
        out = X.copy()
        pos = pd.RangeIndex(len(X))
        work = X.reset_index(drop=True)
        if self.order_by:
            order = work.sort_values(self.order_by, kind="mergesort").index
        else:
            order = pos
        ordered = work.loc[order]

        def shifted(series: pd.Series, k: int) -> pd.Series:
            if self.group_by:
                return series.groupby([ordered[g] for g in self.group_by], dropna=False, sort=False).shift(k)
            return series.shift(k)

        new: dict[str, pd.Series] = {}
        for col in self.columns:
            s = ordered[col].astype(float)
            for k in self.lags:
                new[f"{col}_lag_{k}"] = shifted(s, k)
            for k in self.diffs:
                new[f"{col}_diff_{k}"] = s - shifted(s, k)
            if self.windows:
                prev = shifted(s, 1)
                for w in self.windows:
                    if self.group_by:
                        keys = [ordered[g] for g in self.group_by]
                        roll = prev.groupby(keys, dropna=False, sort=False).rolling(w, min_periods=self.min_periods)
                    else:
                        roll = prev.rolling(w, min_periods=self.min_periods)
                    for agg in self.window_aggs:
                        vals = getattr(roll, agg)()
                        if self.group_by:
                            vals = vals.reset_index(level=list(range(len(self.group_by))), drop=True)
                        new[f"{col}_roll{w}_{agg}"] = vals
        for name, series in new.items():
            series = series.reindex(pos)
            if self.fill_value is not None:
                series = series.fillna(self.fill_value)
            out[name] = series.to_numpy()
        if self.drop_original:
            out = out.drop(columns=self.columns)
        return out
