"""Explicit kernel feature maps: random Fourier features and Nystroem.

Kernel methods capture non-linear structure but scale quadratically in the
number of rows. An explicit feature map ``z(x)`` with
``z(x) . z(y) ~= k(x, y)`` lets a *linear* model (ridge, logistic
regression, linear SVM) approximate the corresponding kernel machine in
linear time.

* :class:`RandomFourierFeatures` (Rahimi & Recht, NIPS 2007): by Bochner's
  theorem a shift-invariant kernel is the Fourier transform of a
  probability measure, so ``z(x) = sqrt(2 / D) cos(W x + b)`` with
  ``W ~ p(w)`` and ``b ~ U(0, 2 pi)`` is an unbiased estimate. ``p`` is
  Gaussian ``N(0, 2 gamma I)`` for the RBF kernel
  ``exp(-gamma ||x - y||^2)`` and Cauchy (scale ``gamma``) for the
  Laplacian kernel ``exp(-gamma ||x - y||_1)``.
* :class:`NystroemFeatures` (Williams & Seeger, NIPS 2001): pick ``m``
  landmark rows ``L``, and map ``z(x) = k(x, L) K_LL^{-1/2}``; then
  ``z(x) . z(y)`` is the Nystroem low-rank approximation of the kernel. It
  is data-dependent and usually more accurate than random features for
  the same number of components; it also supports non-shift-invariant
  kernels (polynomial).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Transformer

_RFF_KERNELS = ("rbf", "laplacian")
_NYSTROEM_KERNELS = ("rbf", "laplacian", "polynomial", "linear")


def _validate_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_choice(value, allowed, name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(repr(a) for a in allowed)}")
    return value


def _numeric_block(X: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"columns not found: {missing}")
    block = X[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(block).all():
        raise ValueError("kernel features need finite numeric values; impute missing values first")
    return block


def _resolve_gamma(gamma, n_features: int) -> float:
    if gamma is None:
        return 1.0 / n_features
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float, np.floating)) or not gamma > 0:
        raise ValueError("gamma must be a positive number or None")
    return float(gamma)


def kernel_matrix(A, B, kernel: str = "rbf", *, gamma: float = 1.0, degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    """Exact kernel matrix ``k(A_i, B_j)`` for the kernels supported here."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if kernel == "rbf":
        sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
        return np.exp(-gamma * np.maximum(sq, 0.0))
    if kernel == "laplacian":
        return np.exp(-gamma * np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2))
    if kernel == "polynomial":
        return (gamma * A @ B.T + coef0) ** degree
    if kernel == "linear":
        return A @ B.T
    raise ValueError(f"unknown kernel {kernel!r}")


class _KernelFeatureBase(Transformer):
    def __init__(self, columns, n_components: int, prefix: str, keep_original: bool):
        self.columns = list(columns)
        if not self.columns:
            raise ValueError("columns must name at least one column")
        self.n_components = _validate_positive_int(n_components, "n_components")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        self.prefix = prefix
        self.keep_original = bool(keep_original)

    def _feature_names(self) -> list[str]:
        return [f"{self.prefix}_{i}" for i in range(self.n_features_out_)]

    def _map(self, block: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        block = _numeric_block(X, self.columns)
        feats = pd.DataFrame(self._map(block), index=X.index, columns=self._feature_names())
        rest = X if self.keep_original else X.drop(columns=self.columns)
        clash = [c for c in feats.columns if c in rest.columns]
        if clash:
            raise ValueError(f"output columns already exist: {clash}; choose another prefix")
        return pd.concat([rest, feats], axis=1)

    def transform_array(self, X: pd.DataFrame) -> np.ndarray:
        """Just the kernel features as an array (no pass-through columns)."""
        self._check_fitted()
        return self._map(_numeric_block(X, self.columns))


class RandomFourierFeatures(_KernelFeatureBase):
    """Random Fourier features approximating an RBF or Laplacian kernel.

    Parameters
    ----------
    columns:
        Numeric columns that form the kernel input; they are replaced by
        ``n_components`` columns ``{prefix}_0 ..`` (other columns pass through).
    n_components:
        Number of random features ``D``; approximation error is ``O(1 / sqrt(D))``.
    kernel:
        ``"rbf"`` (``exp(-gamma ||x-y||^2)``) or ``"laplacian"`` (``exp(-gamma ||x-y||_1)``).
    gamma:
        Kernel bandwidth; ``None`` uses ``1 / n_columns``.
    random_state:
        Seed for the random projection.
    """

    def __init__(
        self,
        columns,
        n_components: int = 100,
        kernel: str = "rbf",
        gamma: float | None = None,
        random_state: int | None = None,
        prefix: str = "rff",
        keep_original: bool = False,
    ):
        super().__init__(columns, n_components, prefix, keep_original)
        self.kernel = _validate_choice(kernel, _RFF_KERNELS, "kernel")
        self.gamma = gamma
        self.random_state = random_state

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        block = _numeric_block(X, self.columns)
        d = block.shape[1]
        self.gamma_ = _resolve_gamma(self.gamma, d)
        rng = np.random.default_rng(self.random_state)
        if self.kernel == "rbf":
            self.weights_ = rng.normal(scale=np.sqrt(2.0 * self.gamma_), size=(d, self.n_components))
        else:
            self.weights_ = self.gamma_ * rng.standard_cauchy(size=(d, self.n_components))
        self.offsets_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        self.n_features_out_ = self.n_components

    def _map(self, block: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / self.n_components) * np.cos(block @ self.weights_ + self.offsets_)


class NystroemFeatures(_KernelFeatureBase):
    """Nystroem low-rank kernel feature map from ``n_components`` landmark rows.

    ``kernel`` is ``"rbf"``, ``"laplacian"``, ``"polynomial"``
    (``(gamma x.y + coef0) ** degree``) or ``"linear"``. Landmarks are drawn
    uniformly without replacement from the training rows (capped at the
    number of rows). Eigenvalues of ``K_LL`` below ``1e-12 * max`` are
    dropped, so the output can have fewer than ``n_components`` columns
    (``n_features_out_``).
    """

    def __init__(
        self,
        columns,
        n_components: int = 100,
        kernel: str = "rbf",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        random_state: int | None = None,
        prefix: str = "nys",
        keep_original: bool = False,
    ):
        super().__init__(columns, n_components, prefix, keep_original)
        self.kernel = _validate_choice(kernel, _NYSTROEM_KERNELS, "kernel")
        self.gamma = gamma
        self.degree = _validate_positive_int(degree, "degree")
        self.coef0 = float(coef0)
        self.random_state = random_state

    def _k(self, A, B):
        return kernel_matrix(A, B, self.kernel, gamma=self.gamma_, degree=self.degree, coef0=self.coef0)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        block = _numeric_block(X, self.columns)
        n, d = block.shape
        self.gamma_ = _resolve_gamma(self.gamma, d)
        rng = np.random.default_rng(self.random_state)
        m = min(self.n_components, n)
        idx = np.sort(rng.choice(n, size=m, replace=False))
        self.landmark_indices_ = idx
        self.landmarks_ = block[idx].copy()
        K = self._k(self.landmarks_, self.landmarks_)
        vals, vecs = np.linalg.eigh(0.5 * (K + K.T))
        keep = vals > 1e-12 * max(vals.max(), 1e-300)
        vals, vecs = vals[keep][::-1], vecs[:, keep][:, ::-1]
        self.normalization_ = vecs / np.sqrt(vals)
        self.n_features_out_ = int(vals.size)

    def _map(self, block: np.ndarray) -> np.ndarray:
        return self._k(block, self.landmarks_) @ self.normalization_


__all__ = ["NystroemFeatures", "RandomFourierFeatures", "kernel_matrix"]
