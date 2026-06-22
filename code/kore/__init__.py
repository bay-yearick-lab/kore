"""Public API for KORE: closed-form spline resolution selection.

This package provides a search-free hyperparameter selector for spline
regression. Instead of cross-validating over a grid of resolutions, KORE
fits at two pilot resolutions, solves a leverage-calibrated 2x2 system
for the bias and noise/variance scales, and obtains the optimal
resolution analytically.

Quick start
-----------
>>> from kore import auto_kore_intrinsic, make_dataset, make_additive_target
>>> target = make_additive_target(d=10)
>>> X_train, y_train, X_test, y_test, sigma = make_dataset(
...     target, n_train=1200, n_test=500, d=10, noise_frac=0.1, seed=42)
>>> result = auto_kore_intrinsic(X_train, y_train)
>>> result["structure"], result["G_star"]
"""

from .lib import (
    closed_form_g_star,
    cv_spline_1d,
    fit_additive,
    fit_spline_1d,
    kore_sande_additive,
    loo_press,
    mse,
    pilot_solve,
    pred_additive,
    pred_spline_1d,
    rmse,
)
from .intrinsic import (
    auto_kore_discover,
    auto_kore_intrinsic,
    criteria_auto_intrinsic,
    cv_auto_intrinsic,
    cv_pairwise_sparse,
    fit_pairwise_sparse,
    gcv_auto_intrinsic,
    kore_diagnostic,
    kore_sande_pairwise,
    make_additive_target,
    make_dataset,
    make_franke,
    make_friedman1,
    make_friedman2,
    make_misspecified_pairwise,
    make_nonsmooth_target,
    make_sparse_additive_highdim,
    make_sparse_pairwise_target,
    make_threeway_target,
    pred_pairwise_sparse,
    refine_kore_additive,
    screen_pairs_residual,
)
from .sklearn import (
    KOREAdditiveRegressor,
    KOREPairwiseRegressor,
    KORERegressor,
)

__all__ = [
    "auto_kore_discover",
    "auto_kore_intrinsic",
    "closed_form_g_star",
    "criteria_auto_intrinsic",
    "cv_auto_intrinsic",
    "cv_pairwise_sparse",
    "cv_spline_1d",
    "fit_additive",
    "fit_pairwise_sparse",
    "fit_spline_1d",
    "gcv_auto_intrinsic",
    "KOREAdditiveRegressor",
    "kore_diagnostic",
    "KOREPairwiseRegressor",
    "KORERegressor",
    "kore_sande_additive",
    "kore_sande_pairwise",
    "loo_press",
    "make_additive_target",
    "make_dataset",
    "make_franke",
    "make_friedman1",
    "make_friedman2",
    "make_misspecified_pairwise",
    "make_nonsmooth_target",
    "make_sparse_additive_highdim",
    "make_sparse_pairwise_target",
    "make_threeway_target",
    "mse",
    "pilot_solve",
    "pred_additive",
    "pred_pairwise_sparse",
    "pred_spline_1d",
    "refine_kore_additive",
    "rmse",
    "screen_pairs_residual",
]
