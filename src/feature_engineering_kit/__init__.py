"""feature_engineering_kit: transformers, ML workflow, and reporting."""

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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "COLUMN_NUMERIC",
    "COLUMN_CATEGORICAL",
    "COLUMN_DATETIME",
    "TARGET_COLUMN",
    "ColumnSchema",
    "column_schema",
    "load_synthetic_churn_dataset",
    "stratified_split",
]
