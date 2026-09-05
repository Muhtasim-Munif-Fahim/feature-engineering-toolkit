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
from .datetime import DatetimeExtractor
from .encoding import OneHotEncoder, OrdinalEncoder, TargetEncoder
from .feature_selection import (
    CorrelationFilter,
    MutualInfoSelection,
    SelectFromModel,
    VarianceThreshold,
)
from .imputation import CategoricalImputer, NumericImputer
from .interaction import InteractionFeatures
from .pipeline import (
    ChurnEvaluation,
    FeatureEngineeringPipeline,
    build_baseline_pipeline,
    build_preprocessing_pipeline,
    run_churn_workflow,
)
from .polynomial import PolynomialFeatures
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
    "NumericImputer",
    "CategoricalImputer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "DatetimeExtractor",
    "PolynomialFeatures",
    "InteractionFeatures",
    "VarianceThreshold",
    "CorrelationFilter",
    "MutualInfoSelection",
    "SelectFromModel",
    "FeatureEngineeringPipeline",
    "build_preprocessing_pipeline",
    "build_baseline_pipeline",
    "run_churn_workflow",
    "ChurnEvaluation",
]
