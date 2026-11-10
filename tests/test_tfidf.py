"""Tests for TfidfFeatures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from feature_engineering_kit import TfidfFeatures

DOCS = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are friends",
    "The mat is red, the log is brown",
    "dog dog dog barks",
]


def _frame(docs=DOCS):
    return pd.DataFrame({"text": docs, "x": np.arange(len(docs), dtype=float)}, index=[f"r{i}" for i in range(len(docs))])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"ngram_range": (1, 2)},
        {"sublinear_tf": True},
        {"smooth_idf": False},
        {"norm": "l1"},
        {"norm": None, "use_idf": False},
        {"min_df": 2},
        {"max_df": 0.5},
    ],
)
def test_matches_sklearn(kwargs):
    df = _frame()
    tf = TfidfFeatures("text", **kwargs)
    out = tf.fit_transform(df)
    sk = TfidfVectorizer(**kwargs)
    ref = sk.fit_transform(DOCS).toarray()
    assert tf.vocabulary_["text"] == list(sk.get_feature_names_out())
    feats = out[[c for c in out.columns if c.startswith("text__tfidf__")]].to_numpy()
    np.testing.assert_allclose(feats, ref, atol=1e-12)


def test_columns_index_and_drop_original():
    df = _frame()
    out = TfidfFeatures("text").fit_transform(df)
    assert "text" not in out.columns and "x" in out.columns
    assert list(out.index) == list(df.index)
    assert "text__tfidf__cat" in out.columns
    kept = TfidfFeatures("text", drop_original=False).fit_transform(df)
    assert "text" in kept.columns
    bigram = TfidfFeatures("text", ngram_range=(2, 2)).fit_transform(df)
    assert "text__tfidf__sat_on" in bigram.columns


def test_max_features_keeps_most_frequent():
    tf = TfidfFeatures("text", max_features=2).fit(_frame())
    assert tf.vocabulary_["text"] == ["dog", "the"]


def test_transform_unseen_and_missing():
    tf = TfidfFeatures("text").fit(_frame())
    new = pd.DataFrame({"text": ["zebra cat", None, np.nan, "unknownword"], "x": [0.0, 1, 2, 3]})
    out = tf.transform(new)
    row0 = out.iloc[0].drop("x")
    assert row0["text__tfidf__cat"] == pytest.approx(1.0)
    assert (out.drop(columns="x").iloc[1:] == 0).all().all()


def test_stop_words_and_multiple_columns():
    df = pd.DataFrame({"a": ["red apple", "green apple"], "b": ["big box", "small box"]})
    out = TfidfFeatures(["a", "b"], stop_words={"apple"}).fit_transform(df)
    assert "a__tfidf__apple" not in out.columns
    assert {"a__tfidf__red", "b__tfidf__big", "b__tfidf__box"} <= set(out.columns)
    np.testing.assert_allclose((out.filter(like="b__") ** 2).sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ngram_range": (2, 1)},
        {"ngram_range": (0, 1)},
        {"max_features": 0},
        {"min_df": 0},
        {"max_df": 1.5},
        {"norm": "max"},
    ],
)
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        TfidfFeatures("text", **kwargs)


def test_errors_on_fit():
    with pytest.raises(KeyError):
        TfidfFeatures("nope").fit(_frame())
    with pytest.raises(ValueError):
        TfidfFeatures("text", min_df=10).fit(_frame())
    with pytest.raises(RuntimeError):
        TfidfFeatures("text").transform(_frame())
