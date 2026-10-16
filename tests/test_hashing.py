"""Tests for the signed feature-hashing encoder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_engineering_kit import FeatureEngineeringPipeline, HashingEncoder


def _df():
    return pd.DataFrame(
        {
            "city": ["dhaka", "paris", "dhaka", None, "tokyo"],
            "device": ["ios", "android", "web", "ios", "ios"],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_shape_names_and_passthrough():
    enc = HashingEncoder(columns=["city", "device"], n_components=4)
    out = enc.fit_transform(_df())
    assert "city" not in out.columns and "device" not in out.columns
    assert out["amount"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = [f"city__hash{j}" for j in range(4)] + [f"device__hash{j}" for j in range(4)]
    assert [c for c in out.columns if c != "amount"] == expected
    assert enc.n_features_out_ == 8
    # exactly one signed unit per row per column block
    city = out[[f"city__hash{j}" for j in range(4)]].to_numpy()
    assert np.allclose(np.abs(city).sum(axis=1), 1.0)


def test_same_value_same_encoding_and_unseen_values_work():
    enc = HashingEncoder(columns=["city"], n_components=16).fit(_df())
    out = enc.transform(pd.DataFrame({"city": ["dhaka", "dhaka", "never-seen"]}))
    block = out.to_numpy()
    assert np.array_equal(block[0], block[1])
    assert np.abs(block[2]).sum() == 1.0


def test_missing_handling():
    df = _df()
    as_value = HashingEncoder(columns=["city"], n_components=8).fit_transform(df)
    zero = HashingEncoder(columns=["city"], n_components=8, handle_missing="zero").fit_transform(df)
    cols = [f"city__hash{j}" for j in range(8)]
    assert np.abs(as_value.loc[3, cols].to_numpy()).sum() == 1.0
    assert np.abs(zero.loc[3, cols].to_numpy()).sum() == 0.0


def test_unsigned_mode_is_nonnegative_counts():
    out = HashingEncoder(columns=["city", "device"], n_components=1, alternate_sign=False, combine=True).fit_transform(_df())
    # one bucket, two columns -> each row counts 2 tokens
    assert out["hash__0"].tolist() == [2.0] * 5


def test_combined_block_distinguishes_columns():
    df = pd.DataFrame({"a": ["x"] * 200, "b": ["x"] * 200})
    enc = HashingEncoder(columns=["a", "b"], n_components=1024, combine=True, alternate_sign=False)
    out = enc.fit_transform(df)
    # "a=x" and "b=x" are different tokens -> two distinct buckets (fixed seed, no collision)
    assert (out.iloc[0] > 0).sum() == 2


def test_seed_changes_buckets():
    values = pd.DataFrame({"c": [f"level{i}" for i in range(50)]})
    a = HashingEncoder(columns=["c"], n_components=64, seed=0).fit_transform(values).to_numpy()
    b = HashingEncoder(columns=["c"], n_components=64, seed=1).fit_transform(values).to_numpy()
    assert not np.array_equal(a, b)


def test_signed_hashing_preserves_inner_products_in_expectation():
    levels = pd.DataFrame({"c": [f"id-{i}" for i in range(4000)]})
    out = HashingEncoder(columns=["c"], n_components=32).fit_transform(levels).to_numpy()
    gram = out @ out.T
    assert np.allclose(np.diag(gram), 1.0)
    off = gram[np.triu_indices_from(gram, k=1)]
    # distinct levels: one-hot inner product 0, hashed inner product mean ~ 0
    assert abs(off.mean()) < 0.003
    unsigned = HashingEncoder(columns=["c"], n_components=32, alternate_sign=False).fit_transform(levels).to_numpy()
    off_u = (unsigned @ unsigned.T)[np.triu_indices(4000, k=1)]
    assert off_u.mean() == pytest.approx(1 / 32, rel=0.1)


def test_stable_across_processes_with_different_hash_seeds():
    code = (
        "import json, pandas as pd;"
        "from feature_engineering_kit import HashingEncoder;"
        "df = pd.DataFrame({'c': ['alpha', 'beta', 'gamma', 'delta']});"
        "out = HashingEncoder(columns=['c'], n_components=16, seed=7).fit_transform(df);"
        "print(json.dumps(out.to_numpy().tolist()))"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    results = []
    for hash_seed in ("1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=src)
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        results.append(json.loads(proc.stdout))
    assert results[0] == results[1]


def test_validation_and_errors():
    with pytest.raises(ValueError):
        HashingEncoder(columns=["c"], n_components=0)
    with pytest.raises(ValueError):
        HashingEncoder(columns=["c"], handle_missing="drop")
    with pytest.raises(ValueError):
        HashingEncoder(columns=["c"], seed=-1)
    with pytest.raises(KeyError):
        HashingEncoder(columns=["nope"]).fit(_df())
    with pytest.raises(RuntimeError):
        HashingEncoder(columns=["city"]).transform(_df())
    clash = pd.DataFrame({"c": ["a"], "c__hash0": [1.0]})
    with pytest.raises(ValueError, match="collide"):
        HashingEncoder(columns=["c"], n_components=2).fit_transform(clash)


def test_in_pipeline():
    pipe = FeatureEngineeringPipeline(steps=[("hash", HashingEncoder(columns=["city"], n_components=4))])
    out = pipe.fit_transform(_df())
    assert out.shape == (5, 2 + 4)
