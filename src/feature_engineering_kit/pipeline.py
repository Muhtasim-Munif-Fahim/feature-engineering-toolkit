"""Feature-engineering pipeline and end-to-end ML workflow.

The pipeline chains :class:`~feature_engineering_kit.base.Transformer` steps so
that a single ``fit_transform(X, y)`` builds the engineered training matrix and
``transform(X)`` reproduces the same mapping on a test matrix (no target needed
at transform time). The bundled :func:`run_churn_workflow` exercises the full
data-loading -> preprocessing -> training -> evaluation arc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Transformer
from .binning import QuantileBinning
from .data import (
    COLUMN_CATEGORICAL,
    COLUMN_NUMERIC,
    TARGET_COLUMN,
    column_schema,
    load_synthetic_churn_dataset,
    stratified_split,
)
from .datetime import DatetimeExtractor
from .encoding import OneHotEncoder, TargetEncoder, WoEEncoder
from .feature_selection import CorrelationFilter, VarianceThreshold
from .imputation import CategoricalImputer, NumericImputer
from .interaction import InteractionFeatures
from .polynomial import PolynomialFeatures
from .scaling import StandardScaler


class FeatureEngineeringPipeline:
    """Ordered chain of transformers with ``fit`` / ``transform`` semantics.

    Parameters
    ----------
    steps:
        Sequence of ``(name, Transformer)`` pairs applied left-to-right.
    """

    def __init__(self, steps):
        self.steps: list[tuple[str, Transformer]] = list(steps)
        self.feature_names_out_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineeringPipeline":
        # Use fit_transform per step so out-of-fold target encoding (and any
        # other fit-time mapping) is what downstream steps see during fit.
        Xt = X.copy()
        for _, step in self.steps:
            Xt = step.fit_transform(Xt, y)
        self.feature_names_out_ = list(Xt.columns)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Xt = X.copy()
        for _, step in self.steps:
            Xt = step.fit_transform(Xt, y)
        self.feature_names_out_ = list(Xt.columns)
        return Xt

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = X.copy()
        for _, step in self.steps:
            Xt = step.transform(Xt)
        self.feature_names_out_ = list(Xt.columns)
        return Xt

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_out_)


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    *,
    target_encode=None,
    woe_encode=None,
    one_hot=None,
    poly_columns=None,
    poly_degree: int = 2,
    poly_interaction_only: bool = True,
    interaction_pairs=None,
    scale_numeric: bool = True,
    datetime_features=None,
    variance_threshold: float = 0.0,
    drop_high_correlation: float | None = 0.95,
    target_encode_cv=5,
    woe_encode_cv=5,
    random_state=None,
    quantile_bin=None,
    n_bins: int = 5,
    bin_strategy: str = "quantile",
    bin_encode: str = "ordinal",
) -> FeatureEngineeringPipeline:
    """Build a standard preprocessing pipeline from a feature DataFrame.

    The pipeline imputes, optionally discretizes numeric columns, encodes,
    extracts datetime features, scales, expands polynomial/interaction
    features, then filters redundant columns.

    Categorical columns in ``target_encode`` use K-fold out-of-fold mean
    encoding (``target_encode_cv`` folds, seeded by ``random_state``) so the
    training matrix does not see a row's own label. Columns in ``woe_encode``
    use out-of-fold Weight of Evidence (``woe_encode_cv`` folds) and expose
    Information Value on the fitted encoder. A column cannot appear in both
    ``target_encode`` and ``woe_encode``.

    Columns in ``quantile_bin`` are discretized after numeric imputation.
    ``bin_strategy`` is ``"quantile"`` or ``"uniform"`` and ``bin_encode`` is
    ``"ordinal"`` or ``"onehot"``. Binned columns are left out of
    ``StandardScaler`` so ordinal codes stay ``0 .. n_bins-1``. One-hot
    encoding drops the source column.
    """
    schema = column_schema(X, target=None)
    numeric = list(schema.numeric)
    categorical = list(schema.categorical)
    datetime = list(schema.datetime)
    target_encode = list(target_encode or [])
    woe_encode = list(woe_encode or [])
    one_hot = list(one_hot or [])
    quantile_bin = list(quantile_bin or [])
    overlap = sorted(set(target_encode) & set(woe_encode))
    if overlap:
        raise ValueError(
            "columns cannot be both target-encoded and WoE-encoded: "
            + ", ".join(overlap)
        )

    steps: list[tuple[str, Transformer]] = []
    if numeric:
        steps.append(("impute_numeric", NumericImputer(numeric, strategy="median")))
    if quantile_bin:
        steps.append(
            (
                "quantile_bin",
                QuantileBinning(
                    columns=quantile_bin,
                    n_bins=n_bins,
                    strategy=bin_strategy,
                    encode=bin_encode,
                ),
            )
        )
    if categorical:
        steps.append(
            ("impute_categorical", CategoricalImputer(categorical, strategy="most_frequent"))
        )
    for col in target_encode:
        steps.append(
            (
                f"target_encode_{col}",
                TargetEncoder(
                    columns=[col],
                    target=TARGET_COLUMN,
                    cv=target_encode_cv,
                    random_state=random_state,
                ),
            )
        )
    for col in woe_encode:
        steps.append(
            (
                f"woe_encode_{col}",
                WoEEncoder(
                    columns=[col],
                    target=TARGET_COLUMN,
                    cv=woe_encode_cv,
                    random_state=random_state,
                ),
            )
        )
    for col in one_hot:
        steps.append((f"one_hot_{col}", OneHotEncoder(columns=[col], drop="first")))
    for col in datetime:
        steps.append(
            (
                f"datetime_{col}",
                DatetimeExtractor(column=col, drop_original=True, features=datetime_features),
            )
        )
    scale_columns = [
        col for col in numeric + target_encode if col not in set(quantile_bin)
    ]
    if scale_numeric and scale_columns:
        steps.append(("scale", StandardScaler(columns=scale_columns)))
    if poly_columns is not None:
        steps.append(
            (
                "poly",
                PolynomialFeatures(
                    columns=list(poly_columns),
                    degree=poly_degree,
                    interaction_only=poly_interaction_only,
                ),
            )
        )
    if interaction_pairs:
        steps.append(
            ("interactions", InteractionFeatures(operations=list(interaction_pairs)))
        )
    if variance_threshold is not None:
        steps.append(("variance_threshold", VarianceThreshold(variance_threshold)))
    if drop_high_correlation is not None:
        steps.append(("correlation_filter", CorrelationFilter(drop_high_correlation)))
    return FeatureEngineeringPipeline(steps)


def build_baseline_pipeline(X: pd.DataFrame) -> FeatureEngineeringPipeline:
    """Numeric-only imputation + scaling (the comparison baseline).

    Non-numeric columns are dropped so the resulting matrix is strictly numeric
    and consumable by scikit-learn estimators.
    """

    class _NumericSelector(Transformer):
        def _fit(self, X: pd.DataFrame, y=None) -> None:
            self.numeric_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

        def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
            return X[[c for c in self.numeric_ if c in X.columns]]

    schema = column_schema(X, target=None)
    steps: list[tuple[str, Transformer]] = [("_select_numeric", _NumericSelector())]
    if schema.numeric:
        steps.append(("impute", NumericImputer(schema.numeric, strategy="median")))
        steps.append(("scale", StandardScaler(columns=schema.numeric)))
    return FeatureEngineeringPipeline(steps)


@dataclass
class ChurnEvaluation:
    """Result of :func:`run_churn_workflow`."""

    seed: int
    n_samples: int
    test_size: float
    model_name: str
    n_features_engineered: int
    n_features_baseline: int
    accuracy: float
    baseline_accuracy: float
    roc_auc: float
    baseline_roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    target_rate: float
    feature_importances: list[tuple[str, float]] = field(default_factory=list)
    dropped_by_variance: list[str] = field(default_factory=list)
    dropped_by_correlation: list[str] = field(default_factory=list)
    information_values: list[tuple[str, float]] = field(default_factory=list)
    iv_details: list[dict] = field(default_factory=list)


def _build_model(name: str, seed: int):
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown model: {name!r}; expected 'logreg' or 'random_forest'")


def _classification_metrics(y_true, y_pred, proba) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _feature_importances(model, columns: list[str]) -> list[tuple[str, float]]:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float).ravel()
        imp = np.abs(coef)
    else:
        return [(c, 0.0) for c in columns]
    if imp.shape[0] != len(columns):
        return [(c, 0.0) for c in columns]
    return [(c, float(v)) for c, v in zip(columns, imp)]


def _step_drop(pipeline: FeatureEngineeringPipeline, name: str) -> list[str]:
    lookup = dict(pipeline.steps)
    step = lookup.get(name)
    if step is None:
        return []
    return list(getattr(step, "drop_", []))


def _collect_information_values(
    pipeline: FeatureEngineeringPipeline,
) -> tuple[list[tuple[str, float]], list[dict]]:
    values: list[tuple[str, float]] = []
    details: list[dict] = []
    for _, step in pipeline.steps:
        iv_map = getattr(step, "iv_", None)
        if iv_map:
            values.extend((str(col), float(iv)) for col, iv in iv_map.items())
        table = getattr(step, "iv_table_", None)
        if table is not None and len(table):
            details.extend(table.to_dict(orient="records"))
    values.sort(key=lambda item: item[1], reverse=True)
    return values, details


def run_churn_workflow(
    n_samples: int = 2000,
    seed: int = 42,
    test_size: float = 0.2,
    model: str = "logreg",
    *,
    target_encode=None,
    woe_encode=None,
    one_hot=None,
    poly_columns=None,
    poly_degree: int = 2,
    interaction_pairs=None,
    drop_high_correlation: float | None = 0.95,
    variance_threshold: float = 0.0,
    quantile_bin=None,
    n_bins: int = 5,
    bin_strategy: str = "quantile",
    bin_encode: str = "ordinal",
) -> ChurnEvaluation:
    """Run a reproducible churn-classification workflow end-to-end.

    Steps: load synthetic data -> stratified split -> feature engineering
    (``fit_transform`` on train, ``transform`` on test to prevent leakage;
    target encoding uses K-fold out-of-fold means on train and the global
    mapping on test; optional ``woe_encode`` columns use out-of-fold Weight
    of Evidence with Information Value recorded on the result; optional
    ``quantile_bin`` columns are discretized after imputation) -> train
    classifier -> evaluate -> return metrics plus a baseline (numeric-only)
    for comparison.
    """
    df = load_synthetic_churn_dataset(n_samples=n_samples, seed=seed)
    train, test = stratified_split(df, test_size=test_size, seed=seed)
    y_train = train[TARGET_COLUMN]
    X_train = train.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]
    X_test = test.drop(columns=[TARGET_COLUMN])

    woe_encode = list(woe_encode or [])
    if target_encode is None:
        target_encode = [c for c in [COLUMN_CATEGORICAL[0]] if c not in woe_encode]
    else:
        target_encode = list(target_encode)
    one_hot = list(
        one_hot
        or [
            c
            for c in COLUMN_CATEGORICAL
            if c not in target_encode and c not in woe_encode
        ]
    )
    poly_columns = list(poly_columns or ["income", "tenure"])
    interaction_pairs = list(
        interaction_pairs
        or [("age", "tenure", "product"), ("income", "tenure", "product")]
    )
    quantile_bin = list(quantile_bin or [])
    # One-hot binning removes the source column. Skip expansions that still
    # name it so the optional hook stays runnable.
    if bin_encode == "onehot" and quantile_bin:
        binned = set(quantile_bin)
        poly_columns = [col for col in poly_columns if col not in binned]
        interaction_pairs = [
            pair
            for pair in interaction_pairs
            if pair[0] not in binned and pair[1] not in binned
        ]
    datetime_features = ["hour", "dayofweek", "month", "is_weekend"]

    pipeline = build_preprocessing_pipeline(
        X_train,
        target_encode=target_encode,
        woe_encode=woe_encode,
        one_hot=one_hot,
        poly_columns=poly_columns,
        poly_degree=poly_degree,
        interaction_pairs=interaction_pairs,
        datetime_features=datetime_features,
        variance_threshold=variance_threshold,
        drop_high_correlation=drop_high_correlation,
        random_state=seed,
        quantile_bin=quantile_bin,
        n_bins=n_bins,
        bin_strategy=bin_strategy,
        bin_encode=bin_encode,
    )
    X_train_eng = pipeline.fit_transform(X_train, y_train)
    X_test_eng = pipeline.transform(X_test)

    y_tr = y_train.to_numpy()
    y_te = y_test.to_numpy()

    estimator = _build_model(model, seed)
    estimator.fit(X_train_eng, y_tr)
    preds = estimator.predict(X_test_eng)
    proba = (
        estimator.predict_proba(X_test_eng)[:, 1]
        if hasattr(estimator, "predict_proba")
        else preds.astype(float)
    )
    metrics = _classification_metrics(y_te, preds, proba)

    baseline = build_baseline_pipeline(X_train)
    X_tr_b = baseline.fit_transform(X_train)
    X_te_b = baseline.transform(X_test)
    base_estimator = _build_model(model, seed)
    base_estimator.fit(X_tr_b, y_tr)
    b_preds = base_estimator.predict(X_te_b)
    b_proba = (
        base_estimator.predict_proba(X_te_b)[:, 1]
        if hasattr(base_estimator, "predict_proba")
        else b_preds.astype(float)
    )
    baseline_metrics = _classification_metrics(y_te, b_preds, b_proba)

    importances = _feature_importances(estimator, list(X_train_eng.columns))
    top_features = sorted(importances, key=lambda t: abs(t[1]), reverse=True)[:10]
    information_values, iv_details = _collect_information_values(pipeline)

    return ChurnEvaluation(
        seed=seed,
        n_samples=n_samples,
        test_size=test_size,
        model_name=model,
        n_features_engineered=int(X_train_eng.shape[1]),
        n_features_baseline=int(X_tr_b.shape[1]),
        accuracy=metrics["accuracy"],
        baseline_accuracy=baseline_metrics["accuracy"],
        roc_auc=metrics["roc_auc"],
        baseline_roc_auc=baseline_metrics["roc_auc"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        confusion_matrix=metrics["confusion_matrix"],
        target_rate=float(np.mean(y_tr)),
        feature_importances=[(c, v) for c, v in top_features],
        dropped_by_variance=_step_drop(pipeline, "variance_threshold"),
        dropped_by_correlation=_step_drop(pipeline, "correlation_filter"),
        information_values=information_values,
        iv_details=iv_details,
    )


__all__ = [
    "FeatureEngineeringPipeline",
    "build_preprocessing_pipeline",
    "build_baseline_pipeline",
    "run_churn_workflow",
    "ChurnEvaluation",
]
