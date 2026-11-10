"""TF-IDF bag-of-n-grams features for free-text columns.

``TfidfFeatures`` turns one or more text columns into dense numeric columns
``{col}__tfidf__{term}`` that slot straight into the rest of the toolkit. The
weighting follows the usual (scikit-learn compatible) definition:

* ``tf(t, d)`` -- raw count of term ``t`` in document ``d``, or
  ``1 + log(tf)`` when ``sublinear_tf=True``;
* ``idf(t) = ln((1 + n) / (1 + df(t))) + 1`` with ``smooth_idf=True``
  (default), or ``ln(n / df(t)) + 1`` otherwise;
* each row vector is then L2- (or L1-) normalised.

The vocabulary, document frequencies and IDF weights are learned on ``fit``
only, so ``transform`` on new rows never leaks test-set statistics; unseen
terms are ignored. Missing values are treated as empty documents.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .base import Transformer

__all__ = ["TfidfFeatures"]

_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def _check_df(value, name: str):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int >= 1 or a float in (0, 1]")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name} must be an int >= 1 or a float in (0, 1]")
        return value
    if isinstance(value, float) and 0.0 < value <= 1.0:
        return value
    raise ValueError(f"{name} must be an int >= 1 or a float in (0, 1]")


class TfidfFeatures(Transformer):
    """Learn a TF-IDF vocabulary per text column and emit dense feature columns.

    Parameters
    ----------
    columns:
        Text column name(s).
    ngram_range:
        ``(min_n, max_n)`` word n-gram sizes.
    max_features:
        Keep only the ``max_features`` most frequent terms (corpus counts) per
        column; ``None`` keeps all.
    min_df, max_df:
        Drop terms appearing in fewer than ``min_df`` / more than ``max_df``
        documents (ints are counts, floats are fractions).
    lowercase:
        Lower-case text before tokenising.
    stop_words:
        Optional iterable of tokens to drop before building n-grams.
    token_pattern:
        Regex that defines a token (default: 2+ word characters).
    use_idf, smooth_idf, sublinear_tf:
        IDF weighting options (see module docstring).
    norm:
        ``"l2"`` (default), ``"l1"`` or ``None``.
    drop_original:
        Drop the source text columns from the output (default ``True``).

    Attributes
    ----------
    vocabulary_:
        ``{column: [term, ...]}`` sorted alphabetically.
    idf_:
        ``{column: np.ndarray}`` aligned with ``vocabulary_``.
    """

    def __init__(
        self,
        columns: str | Sequence[str],
        ngram_range: tuple[int, int] = (1, 1),
        max_features: int | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        lowercase: bool = True,
        stop_words: Iterable[str] | None = None,
        token_pattern: str = _DEFAULT_TOKEN_PATTERN,
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        norm: str | None = "l2",
        drop_original: bool = True,
    ):
        self.columns = [columns] if isinstance(columns, str) else list(columns)
        if not self.columns:
            raise ValueError("columns must not be empty")
        if (
            len(ngram_range) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in ngram_range)
            or not 1 <= ngram_range[0] <= ngram_range[1]
        ):
            raise ValueError("ngram_range must be (min_n, max_n) with 1 <= min_n <= max_n")
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        if max_features is not None and (
            isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer or None")
        self.max_features = max_features
        self.min_df = _check_df(min_df, "min_df")
        self.max_df = _check_df(max_df, "max_df")
        if norm not in ("l1", "l2", None):
            raise ValueError("norm must be 'l1', 'l2' or None")
        self.norm = norm
        self.lowercase = lowercase
        self.stop_words = None if stop_words is None else frozenset(stop_words)
        self.token_pattern = token_pattern
        self._regex = re.compile(token_pattern)
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.drop_original = drop_original

    # ------------------------------------------------------------ analysis
    def _analyze(self, doc) -> list[str]:
        if doc is None or (isinstance(doc, float) and math.isnan(doc)) or doc is pd.NA:
            return []
        text = str(doc)
        if self.lowercase:
            text = text.lower()
        tokens = self._regex.findall(text)
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        lo, hi = self.ngram_range
        if lo == 1 and hi == 1:
            return tokens
        grams: list[str] = []
        for n in range(lo, hi + 1):
            grams.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        return grams

    def _counts(self, series: pd.Series) -> list[Counter]:
        return [Counter(self._analyze(doc)) for doc in series.tolist()]

    # ----------------------------------------------------------------- fit
    def _fit(self, X: pd.DataFrame, y=None) -> None:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")
        n = len(X)
        if n == 0:
            raise ValueError("cannot fit TfidfFeatures on an empty DataFrame")
        min_count = self.min_df if isinstance(self.min_df, int) else math.ceil(self.min_df * n)
        max_count = self.max_df if isinstance(self.max_df, int) else math.floor(self.max_df * n)
        self.vocabulary_: dict[str, list[str]] = {}
        self.idf_: dict[str, np.ndarray] = {}
        for col in self.columns:
            counts = self._counts(X[col])
            df: Counter = Counter()
            total: Counter = Counter()
            for c in counts:
                df.update(c.keys())
                total.update(c)
            terms = [t for t, d in df.items() if min_count <= d <= max_count]
            if self.max_features is not None and len(terms) > self.max_features:
                terms.sort(key=lambda t: (-total[t], t))
                terms = terms[: self.max_features]
            terms.sort()
            if not terms:
                raise ValueError(f"empty vocabulary for column {col!r}; relax min_df/max_df or stop_words")
            dfs = np.array([df[t] for t in terms], dtype=float)
            if self.smooth_idf:
                idf = np.log((1.0 + n) / (1.0 + dfs)) + 1.0
            else:
                idf = np.log(n / dfs) + 1.0
            self.vocabulary_[col] = terms
            self.idf_[col] = idf

    # ----------------------------------------------------------- transform
    def _matrix(self, series: pd.Series, col: str) -> np.ndarray:
        terms = self.vocabulary_[col]
        index = {t: i for i, t in enumerate(terms)}
        M = np.zeros((len(series), len(terms)), dtype=float)
        for r, c in enumerate(self._counts(series)):
            for term, cnt in c.items():
                j = index.get(term)
                if j is not None:
                    M[r, j] = cnt
        if self.sublinear_tf:
            nz = M > 0
            M[nz] = 1.0 + np.log(M[nz])
        if self.use_idf:
            M *= self.idf_[col]
        if self.norm == "l2":
            norms = np.sqrt((M * M).sum(axis=1, keepdims=True))
        elif self.norm == "l1":
            norms = np.abs(M).sum(axis=1, keepdims=True)
        else:
            norms = None
        if norms is not None:
            M = np.divide(M, norms, out=np.zeros_like(M), where=norms > 0)
        return M

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")
        out = X.drop(columns=self.columns) if self.drop_original else X.copy()
        blocks = []
        for col in self.columns:
            M = self._matrix(X[col], col)
            names = [f"{col}__tfidf__{t.replace(' ', '_')}" for t in self.vocabulary_[col]]
            blocks.append(pd.DataFrame(M, columns=names, index=X.index))
        return pd.concat([out, *blocks], axis=1)
