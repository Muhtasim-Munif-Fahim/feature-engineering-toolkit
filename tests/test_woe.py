"""Tests for Weight of Evidence encoding and Information Value reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from feature_engineering_kit import (
    WoEEncoder,
    information_value,
    iv_strength,
)


def _df(**cols):
    return pd.DataFrame(cols)


def _raw_woe(n_non_i, n_event_i, n_non, n_event, smoothing, n_cats):
    floor = 0.0 if smoothing > 0.0 else 1e-12
    dist_non = max(n_non_i + smoothing, floor) / (n_non + smoothing * n_cats)
    dist_evt = max(n_event_i + smoothing, floor) / (n_event + smoothing * n_cats)
    return float(np.log(dist_non / dist_evt))


def test_woe_encoder_requires_y() -> None:
    enc = WoEEncoder(columns=["x"], target="y")
    with pytest.raises(ValueError, match="requires y"):
        enc.fit(_df(x=["a", "b"]))


def test_woe_encoder_requires_both_classes() -> None:
    enc = WoEEncoder(columns=["x"], target="y", cv=None)
    with pytest.raises(ValueError, match="event_value"):
        enc.fit(_df(x=["a", "b"]), pd.Series([0, 0]))
    with pytest.raises(ValueError, match="only the event class"):
        enc.fit(_df(x=["a", "b"]), pd.Series([1, 1]))


def test_woe_encoder_rejects_invalid_cv_and_smoothing() -> None:
    with pytest.raises(ValueError, match="cv must be None"):
        WoEEncoder(columns=["x"], target="y", cv=1)
    with pytest.raises(ValueError, match="smoothing must be"):
        WoEEncoder(columns=["x"], target="y", smoothing=-0.1)


def test_woe_smoothing_zero_recovers_raw_log_odds() -> None:
    enc = WoEEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    df = _df(x=["a", "a", "a", "b", "b", "b"], y=[1, 1, 0, 0, 0, 1])
    out = enc.fit_transform(df[["x"]], df["y"])
    # 3 events, 3 non-events; a: 1 non / 2 event; b: 2 non / 1 event
    woe_a = _raw_woe(1, 2, 3, 3, 0.0, 2)
    woe_b = _raw_woe(2, 1, 3, 3, 0.0, 2)
    assert out["x"].tolist() == pytest.approx([woe_a, woe_a, woe_a, woe_b, woe_b, woe_b])
    assert enc.maps_["x"]["a"] == pytest.approx(woe_a)
    assert enc.maps_["x"]["b"] == pytest.approx(woe_b)
    expected_iv = (1 / 3 - 2 / 3) * woe_a + (2 / 3 - 1 / 3) * woe_b
    assert enc.iv_["x"] == pytest.approx(expected_iv)


def test_woe_default_smoothing_matches_laplace_formula() -> None:
    enc = WoEEncoder(columns=["x"], target="y", smoothing=0.5, cv=None)
    df = _df(x=["a", "a", "a", "b", "b", "b"], y=[1, 1, 0, 0, 0, 1])
    enc.fit(df[["x"]], df["y"])
    woe_a = _raw_woe(1, 2, 3, 3, 0.5, 2)
    woe_b = _raw_woe(2, 1, 3, 3, 0.5, 2)
    assert enc.maps_["x"]["a"] == pytest.approx(woe_a)
    assert enc.maps_["x"]["b"] == pytest.approx(woe_b)
    assert woe_a < 0 < woe_b


def test_woe_unseen_and_missing_map_to_zero() -> None:
    enc = WoEEncoder(columns=["x"], target="y", smoothing=0.5, cv=None)
    enc.fit(_df(x=["a", "a", "b", "b"]), pd.Series([1, 0, 0, 1]))
    out = enc.transform(_df(x=["c", None]))
    assert out["x"].tolist() == pytest.approx([0.0, 0.0])


def test_woe_global_maps_ignore_cv() -> None:
    """``fit`` always stores the full-data mapping used by ``transform``."""
    enc = WoEEncoder(columns=["x"], target="y", smoothing=0.0, cv=5, random_state=0)
    df = _df(x=["a", "a", "a", "b", "b", "b"], y=[1, 1, 0, 0, 0, 1])
    enc.fit(df[["x"]], df["y"])
    out = enc.transform(df[["x"]])
    naive = WoEEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    naive.fit(df[["x"]], df["y"])
    np.testing.assert_allclose(out["x"].to_numpy(), naive.transform(df[["x"]])["x"].to_numpy())


def test_woe_missing_column_raises() -> None:
    enc = WoEEncoder(columns=["x"], target="y", cv=None)
    enc.fit(_df(x=["a", "b"]), pd.Series([0, 1]))
    with pytest.raises(KeyError, match="column 'x'"):
        enc.transform(_df(z=["a"]))


def test_iv_report_ranks_predictive_column_first() -> None:
    rng = np.random.default_rng(0)
    n = 400
    signal = np.where(rng.random(n) < 0.5, "high_risk", "low_risk")
    y = np.where(signal == "high_risk", rng.random(n) < 0.7, rng.random(n) < 0.2).astype(int)
    noise = rng.choice(["p", "q", "r", "s"], size=n)
    X = pd.DataFrame({"signal": signal, "noise": noise})
    enc = WoEEncoder(columns=["signal", "noise"], target="y", cv=None)
    enc.fit(X, y)
    report = enc.iv_report()
    assert list(report.columns) == ["column", "iv", "strength"]
    assert report.iloc[0]["column"] == "signal"
    assert report.loc[report["column"] == "signal", "iv"].iloc[0] > report.loc[
        report["column"] == "noise", "iv"
    ].iloc[0]
    standalone = information_value(X, y, columns=["signal", "noise"])
    pd.testing.assert_frame_equal(report.reset_index(drop=True), standalone.reset_index(drop=True))
    assert enc.iv_table_["column"].tolist().count("signal") >= 2


def test_iv_strength_thresholds() -> None:
    assert iv_strength(0.01) == "unpredictive"
    assert iv_strength(0.05) == "weak"
    assert iv_strength(0.2) == "medium"
    assert iv_strength(0.4) == "strong"
    assert iv_strength(0.6) == "suspicious"


def test_oof_woe_matches_complementary_fold() -> None:
    X = _df(x=["a", "a", "b", "a", "b", "b"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    enc = WoEEncoder(
        columns=["x"], target="y", smoothing=0.0, cv=2, shuffle=False
    )
    out = enc.fit_transform(X, y)["x"].to_numpy()

    expected = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=2, shuffle=False)
    for train_idx, test_idx in splitter.split(X, y):
        fold = WoEEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
        fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        expected[test_idx] = fold.transform(X.iloc[test_idx])["x"].to_numpy()
    np.testing.assert_allclose(out, expected)
    global_enc = enc.transform(X)["x"].to_numpy()
    assert not np.allclose(out, global_enc)


def test_oof_woe_leaks_less_than_naive_fit_on_all() -> None:
    """A unique-id column is a textbook leakage trap.

    Naive fit-on-all WoE with no smoothing reconstructs the row's label
    (one class per category). Out-of-fold encoding never sees that row's
    target, so unseen ids map to 0.
    """
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    X = pd.DataFrame({"x": [f"id_{i}" for i in range(n)]})

    naive = WoEEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    oof = WoEEncoder(
        columns=["x"], target="y", smoothing=0.0, cv=5, shuffle=True, random_state=0
    )
    naive_enc = naive.fit_transform(X, y)["x"].to_numpy()
    oof_enc = oof.fit_transform(X, y)["x"].to_numpy()

    naive_sign = np.sign(naive_enc)
    # Event (y=1) -> negative WoE; non-event -> positive WoE.
    assert np.all(naive_sign[y == 1] < 0)
    assert np.all(naive_sign[y == 0] > 0)
    assert np.allclose(oof_enc, 0.0)
    naive_corr = float(np.corrcoef(naive_enc, y.astype(float))[0, 1])
    assert abs(naive_corr) > 0.9


def test_oof_held_out_row_does_not_use_its_own_target() -> None:
    """Stratified leave-two-out: each row is encoded without its own label."""
    X = _df(x=["a", "a", "a", "a", "b", "b", "b", "b"])
    # a is event-heavy, b is non-event-heavy, so raw WoE is nonzero.
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 1])
    enc = WoEEncoder(
        columns=["x"], target="y", smoothing=0.0, cv=4, shuffle=False
    )
    out = enc.fit_transform(X, y)["x"].to_numpy()
    expected = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=4, shuffle=False)
    n_splits = 0
    for train_idx, test_idx in splitter.split(X, y):
        n_splits += 1
        fold = WoEEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
        fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        expected[test_idx] = fold.transform(X.iloc[test_idx])["x"].to_numpy()
    assert n_splits == 4
    np.testing.assert_allclose(out, expected)
    naive = WoEEncoder(columns=["x"], target="y", smoothing=0.0, cv=None)
    naive_enc = naive.fit_transform(X, y)["x"].to_numpy()
    assert not np.allclose(out, naive_enc)
