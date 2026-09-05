"""Synthetic dataset generation and column-schema utilities.

The dataset models customer churn for a subscription business. The target is a
synthetic Bernoulli outcome whose log-odds are a function of numeric,
categorical, and datetime features, with a small amount of injected missingness
so that the imputation and encoding transformers are exercised on realistic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas.api.types as ptypes
from sklearn.model_selection import train_test_split as _sk_train_test_split

TARGET_COLUMN = "churn"
COLUMN_NUMERIC = ["age", "income", "tenure"]
COLUMN_CATEGORICAL = ["plan", "region"]
COLUMN_DATETIME = ["signup_time"]


@dataclass(frozen=True)
class ColumnSchema:
    """Static description of a dataset's feature column groups."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    target: str | None = None

    def all_features(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.datetime]


def column_schema(df: pd.DataFrame, target: str = TARGET_COLUMN) -> ColumnSchema:
    """Infer a :class:`ColumnSchema` from a DataFrame's dtypes.

    Columns are classified by dtype using :mod:`pandas.api.types`, so the
    classification is robust to pandas string-backed dtypes and missing values:

    * datetime64 columns -> ``datetime``
    * boolean columns are ignored
    * numeric columns -> ``numeric``
    * everything else (object/string) -> ``categorical``

    The ``target`` column, if present, is excluded from the feature groups.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    datetime: list[str] = []
    for col in df.columns:
        if col == target:
            continue
        dtype = df[col].dtype
        if ptypes.is_datetime64_any_dtype(dtype):
            datetime.append(col)
        elif ptypes.is_bool_dtype(dtype):
            continue
        elif ptypes.is_numeric_dtype(dtype):
            numeric.append(col)
        else:
            categorical.append(col)
    return ColumnSchema(
        numeric=numeric, categorical=categorical, datetime=datetime, target=target
    )


def load_synthetic_churn_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic churn dataset.

    Parameters
    ----------
    n_samples:
        Number of rows to generate.
    seed:
        Seed for ``numpy.random.default_rng``. Different seeds produce different
        data; the same seed is fully reproducible.

    Returns
    -------
    pandas.DataFrame with columns ``age``, ``income``, ``tenure``, ``plan``,
    ``region``, ``signup_time`` (datetime64), and ``churn`` (0/1 int).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    rng = np.random.default_rng(seed)
    n = int(n_samples)

    age = np.clip(rng.normal(42, 12, n), 18, 90).astype(np.float64)
    income = np.clip(rng.normal(65000, 18000, n), 10000, 250000).astype(np.float64)
    tenure = np.clip(rng.exponential(14, n), 0, 200).astype(np.float64)

    plans = np.array(["basic", "standard", "premium", "enterprise"])
    regions = np.array(["north", "south", "east", "west"])
    plan = plans[rng.integers(0, len(plans), n)]
    region = regions[rng.integers(0, len(regions), n)]

    base = pd.Timestamp("2023-01-01")
    day_offsets = rng.integers(0, 7 * 365, n)
    hour_offsets = rng.integers(0, 24, n)
    signup_time = base + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(
        hour_offsets, unit="h"
    )
    signup_time = pd.Series(signup_time, dtype="datetime64[ns]")

    # Churn log-odds as a function of the features (known, reproducible signal).
    # Coefficients are chosen so the base churn rate is ~0.33 and the dominant
    # signal lives in the categorical (`plan`) and temporal (`late_night`)
    # features, which only the engineered model can capture.
    late_night = signup_time.dt.hour.between(2, 6).to_numpy()
    logit = (
        -1.1
        + 1.3 * (plan == "basic")
        - 0.7 * (plan == "premium")
        + 0.00002 * (income - 65000)
        - 0.05 * (tenure - 14)
        + 1.0 * late_night
        - 0.02 * (age - 42)
        + 0.6 * (region == "west")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(np.int64)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "tenure": tenure,
            "plan": plan,
            "region": region,
            "signup_time": signup_time,
            TARGET_COLUMN: churn,
        }
    )
    # Inject missingness so the imputation transformers have something to do.
    df.loc[rng.random(n) < 0.02, "age"] = np.nan
    df.loc[rng.random(n) < 0.05, "income"] = np.nan
    df.loc[rng.random(n) < 0.04, "tenure"] = np.nan
    df.loc[rng.random(n) < 0.03, "plan"] = pd.NA
    df.loc[rng.random(n) < 0.02, "region"] = pd.NA
    return df


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving the target rate.

    Parameters
    ----------
    df:
        DataFrame containing the ``TARGET_COLUMN``.
    test_size:
        Fraction held out for testing.
    seed:
        Reproducibility seed forwarded to scikit-learn.

    Returns
    -------
    (train_df, test_df) with the same columns as ``df``.
    """
    train, test = _sk_train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET_COLUMN]
    )
    return train, test


__all__ = [
    "TARGET_COLUMN",
    "COLUMN_NUMERIC",
    "COLUMN_CATEGORICAL",
    "COLUMN_DATETIME",
    "ColumnSchema",
    "column_schema",
    "load_synthetic_churn_dataset",
    "stratified_split",
]
