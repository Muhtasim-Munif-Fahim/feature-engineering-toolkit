"""feature_engineering_kit: transformers, ML workflow, and reporting."""

from .base import Transformer
from .binning import MDLPBinning, QuantileBinning
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
from .datetime import CyclicalEncoder, DatetimeExtractor
from .encoding import (
    CatBoostEncoder,
    FrequencyEncoder,
    HashingEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareCategoryGrouper,
    TargetEncoder,
    WoEEncoder,
    information_value,
    iv_strength,
)
from .feature_selection import (
    BorutaSelection,
    CorrelationFilter,
    MRMRSelection,
    MutualInfoSelection,
    ReliefFSelection,
    SelectFromModel,
    VarianceThreshold,
    discrete_mutual_info,
    quantile_discretize,
    relieff_scores,
)
from .imputation import (
    CategoricalImputer,
    IterativeImputer,
    KNNImputer,
    NumericImputer,
    fit_bayesian_ridge,
    nan_euclidean_distances,
)
from .interaction import InteractionFeatures
from .pipeline import (
    ChurnEvaluation,
    FeatureEngineeringPipeline,
    build_baseline_pipeline,
    build_preprocessing_pipeline,
    run_churn_workflow,
)
from .polynomial import PolynomialFeatures
from .reporting import render_markdown_report, save_report
from .scaling import BoxCoxScaler, MaxAbsScaler, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler, StandardScaler, Winsorizer, YeoJohnsonScaler
from .splines import SplineTransformer, bspline_basis

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Transformer",
    "IterativeImputer",
    "fit_bayesian_ridge",
    "QuantileBinning",
    "MDLPBinning",
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
    "KNNImputer",
    "nan_euclidean_distances",
    "CatBoostEncoder",
    "FrequencyEncoder",
    "HashingEncoder",
    "JamesSteinEncoder",
    "LeaveOneOutEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "RareCategoryGrouper",
    "TargetEncoder",
    "WoEEncoder",
    "information_value",
    "iv_strength",
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "Normalizer",
    "Winsorizer",
    "RobustScaler",
    "YeoJohnsonScaler",
    "BoxCoxScaler",
    "QuantileTransformer",
    "DatetimeExtractor",
    "CyclicalEncoder",
    "PolynomialFeatures",
    "SplineTransformer",
    "bspline_basis",
    "InteractionFeatures",
    "VarianceThreshold",
    "CorrelationFilter",
    "MutualInfoSelection",
    "MRMRSelection",
    "ReliefFSelection",
    "BorutaSelection",
    "relieff_scores",
    "discrete_mutual_info",
    "quantile_discretize",
    "SelectFromModel",
    "FeatureEngineeringPipeline",
    "build_preprocessing_pipeline",
    "build_baseline_pipeline",
    "run_churn_workflow",
    "ChurnEvaluation",
    "render_markdown_report",
    "save_report",
]
