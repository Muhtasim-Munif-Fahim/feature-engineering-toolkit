"""feature_engineering_kit: transformers, ML workflow, and reporting."""

from .base import Transformer
from .data import (
    COLUMN_CATEGORICAL,
    COLUMN_DATETIME,
    COLUMN_NUMERIC,
    TARGET_COLUMN,
    ColumnSchema,
    column_schema,
    load_synthetic_churn_dataset,
    stratified_split,
)
from .encoding import OneHotEncoder, OrdinalEncoder, TargetEncoder
from .scaling import MinMaxScaler, RobustScaler, StandardScaler

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Transformer",
    "ColumnSchema",
    "TARGET_COLUMN",
    "COLUMN_NUMERIC",
    "COLUMN_CATEGORICAL",
    "COLUMN_DATETIME",
    "column_schema",
    "load_synthetic_churn_dataset",
    "stratified_split",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
]
