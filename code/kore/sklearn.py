"""sklearn-style estimator adapters around the KORE selectors.

Three thin wrappers exposing the closed-form plug-in selectors with the
``BaseEstimator`` / ``RegressorMixin`` contract: ``KOREAdditiveRegressor``
(additive family), ``KOREPairwiseRegressor`` (sparse pairwise family),
and ``KORERegressor`` (auto-selects structure by leave-one-out
comparison via :func:`auto_kore_intrinsic`).

Each estimator stores the fitted ``model_``, the integer resolution
``G_star_``, and the pilot constants ``A_hat_`` and ``tau_hat_`` after
``fit``. The supported input contract is a dense ``(n, d)`` float array
``X`` and a one-dimensional ``(n,)`` float array ``y``; sparse matrices
and one-dimensional ``X`` are rejected with a clear ``ValueError``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for kore.sklearn; install with "
        "`uv sync` or `uv add scikit-learn`."
    ) from exc

from .intrinsic import (
    auto_kore_intrinsic,
    fit_pairwise_sparse,
    kore_sande_pairwise,
    pred_pairwise_sparse,
)
from .lib import (
    fit_additive,
    kore_sande_additive,
    pred_additive,
)


Pair = Tuple[int, int]


def _check_fit_inputs(X, y):
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError(
            f"KORE estimators require 2D X with shape (n, d); got ndim={X.ndim}."
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]}."
        )
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must be numeric; sparse matrices are not supported.")
    return np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64)


def _check_predict_inputs(X, n_features_in):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            f"KORE estimators require 2D X for predict; got ndim={X.ndim}."
        )
    if X.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X.shape[1]} features but estimator was fit with "
            f"{n_features_in} features."
        )
    return np.ascontiguousarray(X, dtype=np.float64)


class KOREAdditiveRegressor(BaseEstimator, RegressorMixin):
    """Closed-form spline-resolution selector for the additive family."""

    def __init__(self, degree: int = 3, max_G: int = 20, ridge: float = 1e-8,
                 radius: int = 3):
        self.degree = degree
        self.max_G = max_G
        self.ridge = ridge
        self.radius = radius

    def fit(self, X, y):
        X, y = _check_fit_inputs(X, y)
        res = kore_sande_additive(X, y, degree=self.degree, ridge=self.ridge,
                                  max_G=self.max_G, radius=self.radius)
        self.model_ = res["model"]
        self.G_star_ = int(res["G_star"])
        self.A_hat_ = float(res["A_hat"])
        self.tau_hat_ = float(res["tau_hat"])
        self.G_dagger_ = float(res["G_dagger"])
        self.fits_ = int(res["fits"])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = _check_predict_inputs(X, self.n_features_in_)
        return pred_additive(self.model_, X)


class KOREPairwiseRegressor(BaseEstimator, RegressorMixin):
    """Closed-form spline-resolution selector for the sparse pairwise family."""

    def __init__(self, pairs: Optional[Sequence[Pair]] = None, degree: int = 3,
                 max_G: int = 8, ridge: float = 1e-8, radius: int = 1):
        self.pairs = pairs
        self.degree = degree
        self.max_G = max_G
        self.ridge = ridge
        self.radius = radius

    def fit(self, X, y):
        X, y = _check_fit_inputs(X, y)
        res = kore_sande_pairwise(X, y, pairs=self.pairs, degree=self.degree,
                                  ridge=self.ridge, max_G=self.max_G,
                                  radius=self.radius)
        if res.get("skipped"):
            raise ValueError(
                "Pairwise family infeasible on this dataset: "
                f"{res.get('skip_reason', 'p_pair(1) >= 0.45 * n')}. "
                "Use KORERegressor for automatic structure selection."
            )
        self.model_ = res["model"]
        self.G_star_ = int(res["G_star"])
        self.A_hat_ = float(res["A_hat"])
        self.tau_hat_ = float(res["tau_hat"])
        self.G_dagger_ = float(res["G_dagger"])
        self.fits_ = int(res["fits"])
        self.pairs_ = list(res["pairs"])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = _check_predict_inputs(X, self.n_features_in_)
        return pred_pairwise_sparse(self.model_, X)


class KORERegressor(BaseEstimator, RegressorMixin):
    """Auto-structure plug-in selector: fits both additive and sparse pairwise
    families and returns the LOO winner. Recommended default for users."""

    def __init__(self, pairs: Optional[Sequence[Pair]] = None, degree: int = 3,
                 max_G_add: int = 20, max_G_pair: int = 8, ridge: float = 1e-8):
        self.pairs = pairs
        self.degree = degree
        self.max_G_add = max_G_add
        self.max_G_pair = max_G_pair
        self.ridge = ridge

    def fit(self, X, y):
        X, y = _check_fit_inputs(X, y)
        res = auto_kore_intrinsic(X, y, pairs=self.pairs, degree=self.degree,
                                  ridge=self.ridge,
                                  max_G_add=self.max_G_add,
                                  max_G_pair=self.max_G_pair)
        self.model_ = res["model"]
        self.G_star_ = int(res["G_star"])
        self.structure_ = str(res["structure"])
        self.fits_ = int(res["fits_total"])
        self.candidates_ = res["candidates"]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = _check_predict_inputs(X, self.n_features_in_)
        if self.structure_ == "additive":
            return pred_additive(self.model_, X)
        return pred_pairwise_sparse(self.model_, X)
