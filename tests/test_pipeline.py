"""Tests for the feature-engineering pipeline and churn workflow."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering_kit import (
    FeatureEngineeringPipeline,
    NumericImputer,
    StandardScaler,
    TargetEncoder,
    WoEEncoder,
    run_churn_workflow,
)
from feature_engineering_kit.pipeline import build_preprocessing_pipeline


def _small_df():
    return pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [10, 20, 30, 40]})


def _two_step_pipeline():
    return FeatureEngineeringPipeline(
        [
            ("impute", NumericImputer(columns=["a"], strategy="mean")),
            ("scale", StandardScaler(columns=["a", "b"])),
        ]
    )


def test_pipeline_fit_transform_equals_fit_then_transform() -> None:
    df = _small_df()
    a = _two_step_pipeline().fit_transform(df)
    pipe = _two_step_pipeline()
    pipe.fit(df)
    b = pipe.transform(df)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)
    assert pipe.get_feature_names_out() == list(a.columns)


def test_pipeline_transform_without_fit_raises() -> None:
    pipe = FeatureEngineeringPipeline([("scale", StandardScaler(columns=["a", "b"]))])
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.transform(_small_df())


def test_run_churn_workflow_metrics_in_range() -> None:
    r = run_churn_workflow(n_samples=1500, seed=42, model="logreg")
    assert r.n_features_engineered == 13
    assert r.n_features_baseline == 3
    for value in (r.accuracy, r.roc_auc, r.precision, r.recall, r.f1):
        assert 0.0 <= value <= 1.0
    for value in (r.baseline_accuracy, r.baseline_roc_auc):
        assert 0.0 <= value <= 1.0
    cm = r.confusion_matrix
    assert len(cm) == 2 and all(len(row) == 2 for row in cm)
    assert sum(sum(row) for row in cm) == 300
    assert 0.0 < r.target_rate < 1.0
    assert len(r.feature_importances) <= 10


def test_run_churn_workflow_engineered_beats_baseline() -> None:
    r = run_churn_workflow(n_samples=1500, seed=42, model="logreg")
    # Engineered features (plan, region, time, interactions) clearly lift AUC.
    assert r.roc_auc >= r.baseline_roc_auc
    assert r.accuracy >= r.baseline_accuracy


def test_run_churn_workflow_feature_importances_ordered() -> None:
    r = run_churn_workflow(n_samples=1500, seed=42, model="logreg")
    magnitudes = [abs(v) for _, v in r.feature_importances]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_run_churn_workflow_random_forest_runs() -> None:
    r = run_churn_workflow(n_samples=600, seed=0, model="random_forest")
    assert 0.0 <= r.accuracy <= 1.0
    assert 0.0 <= r.roc_auc <= 1.0
    assert len(r.feature_importances) <= 10


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        run_churn_workflow(n_samples=200, seed=0, model="nope")


def test_preprocessing_pipeline_target_encoder_is_out_of_fold() -> None:
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "plan": ["a", "b", "a", "b"],
        }
    )
    pipe = build_preprocessing_pipeline(
        df, target_encode=["plan"], one_hot=[], random_state=0
    )
    enc = dict(pipe.steps)["target_encode_plan"]
    assert isinstance(enc, TargetEncoder)
    assert enc.cv == 5
    assert enc.random_state == 0


def test_preprocessing_pipeline_woe_encoder_is_out_of_fold() -> None:
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "plan": ["a", "b", "a", "b"],
        }
    )
    pipe = build_preprocessing_pipeline(
        df, target_encode=[], woe_encode=["plan"], one_hot=[], random_state=0
    )
    enc = dict(pipe.steps)["woe_encode_plan"]
    assert isinstance(enc, WoEEncoder)
    assert enc.cv == 5
    assert enc.random_state == 0


def test_preprocessing_pipeline_rejects_overlapping_encoders() -> None:
    df = pd.DataFrame({"plan": ["a", "b"], "age": [1.0, 2.0]})
    with pytest.raises(ValueError, match="cannot be both"):
        build_preprocessing_pipeline(
            df, target_encode=["plan"], woe_encode=["plan"], one_hot=[]
        )


def test_run_churn_workflow_woe_hook_records_iv() -> None:
    r = run_churn_workflow(
        n_samples=600, seed=0, model="logreg", woe_encode=["region"]
    )
    assert r.information_values
    cols = [name for name, _ in r.information_values]
    assert "region" in cols
    assert all(iv >= 0.0 for _, iv in r.information_values)
    assert r.iv_details
    assert {row["column"] for row in r.iv_details} == {"region"}
    for value in (r.accuracy, r.roc_auc):
        assert 0.0 <= value <= 1.0


def test_run_churn_workflow_default_has_no_iv() -> None:
    r = run_churn_workflow(n_samples=400, seed=0, model="logreg")
    assert r.information_values == []
    assert r.iv_details == []
