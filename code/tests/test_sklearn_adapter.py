"""Unit tests for the sklearn-style estimator adapters in ``kore.sklearn``.

Exercises the ``BaseEstimator`` / ``RegressorMixin`` contract: ``fit`` /
``predict`` / ``score``, ``get_params`` / ``set_params`` round-tripping,
the ``sklearn.base.clone`` path, fitted-attribute presence, and the
input-validation rejections for 1D ``X``, mismatched ``(n)``, non-numeric
dtypes, and predict-time feature-count mismatches.
"""
from __future__ import annotations

import numpy as np
import pytest

from sklearn.base import clone

from kore import (
    KOREAdditiveRegressor,
    KOREPairwiseRegressor,
    KORERegressor,
    auto_kore_intrinsic,
    kore_sande_pairwise,
    make_additive_target,
    make_dataset,
    make_sparse_pairwise_target,
)
from kore.intrinsic import _pairwise_effective_max_g, pairwise_basis_dim


def _additive_xy(d: int = 4, n_train: int = 300, n_test: int = 150, seed: int = 0):
    target = make_additive_target(d=d, seed=seed)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(
        target, n_train=n_train, n_test=n_test, d=d, noise_frac=0.1, seed=seed)
    return X_tr, y_tr, X_te, y_te


def _pairwise_xy(d: int = 4, n_train: int = 400, n_test: int = 200, seed: int = 0):
    target, _pairs = make_sparse_pairwise_target(d=d, seed=seed)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(
        target, n_train=n_train, n_test=n_test, d=d, noise_frac=0.1, seed=seed)
    return X_tr, y_tr, X_te, y_te


@pytest.mark.parametrize("cls,kwargs,xy_fn,r2_min", [
    (KOREAdditiveRegressor, {"max_G": 16}, _additive_xy, 0.5),
    (KOREPairwiseRegressor, {"max_G": 6}, _pairwise_xy, 0.3),
    (KORERegressor, {"max_G_add": 16, "max_G_pair": 6}, _additive_xy, 0.5),
])
def test_fit_predict_score_contract(cls, kwargs, xy_fn, r2_min):
    X_tr, y_tr, X_te, y_te = xy_fn()
    est = cls(**kwargs)
    out = est.fit(X_tr, y_tr)
    assert out is est, "fit must return self per the sklearn contract"

    yhat = est.predict(X_te)
    assert yhat.shape == y_te.shape
    assert np.all(np.isfinite(yhat))

    r2 = est.score(X_te, y_te)
    assert np.isfinite(r2)
    assert r2 > r2_min, f"{cls.__name__} R^2 too low: {r2:.3f} < {r2_min}"

    assert isinstance(est.G_star_, int) and est.G_star_ >= 1
    assert hasattr(est, "n_features_in_") and est.n_features_in_ == X_tr.shape[1]


def test_kore_regressor_auto_selects_pairwise_when_appropriate():
    X_tr, y_tr, X_te, y_te = _pairwise_xy()
    est = KORERegressor(max_G_add=16, max_G_pair=6).fit(X_tr, y_tr)
    assert est.structure_ in {"additive", "pairwise"}
    r2 = est.score(X_te, y_te)
    assert r2 > 0.3, f"pairwise target R^2 too low: {r2:.3f}"


@pytest.mark.parametrize("cls", [KOREAdditiveRegressor, KOREPairwiseRegressor, KORERegressor])
def test_get_set_params_roundtrip(cls):
    est = cls()
    params = est.get_params()
    assert isinstance(params, dict) and params

    twin = cls(**params)
    assert twin.get_params() == params

    first_key = next(iter(params))
    current = params[first_key]
    new_value = current + 1 if isinstance(current, (int, float)) and not isinstance(current, bool) else current
    est.set_params(**{first_key: new_value})
    assert est.get_params()[first_key] == new_value


@pytest.mark.parametrize("cls", [KOREAdditiveRegressor, KOREPairwiseRegressor, KORERegressor])
def test_clone_strips_fitted_attributes(cls):
    X_tr, y_tr, _, _ = _additive_xy()
    est = cls().fit(X_tr, y_tr)
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()
    assert not hasattr(cloned, "model_")
    assert not hasattr(cloned, "G_star_")
    assert not hasattr(cloned, "n_features_in_")


@pytest.mark.parametrize("cls", [KOREAdditiveRegressor, KOREPairwiseRegressor, KORERegressor])
def test_fit_rejects_1d_X(cls):
    X = np.linspace(0.0, 1.0, 50)
    y = np.zeros(50)
    with pytest.raises(ValueError, match="2D X"):
        cls().fit(X, y)


@pytest.mark.parametrize("cls", [KOREAdditiveRegressor, KOREPairwiseRegressor, KORERegressor])
def test_fit_rejects_mismatched_shapes(cls):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=39)
    with pytest.raises(ValueError, match="rows"):
        cls().fit(X, y)


@pytest.mark.parametrize("cls", [KOREAdditiveRegressor, KOREPairwiseRegressor, KORERegressor])
def test_fit_rejects_non_numeric_X(cls):
    X = np.array([["a", "b"], ["c", "d"], ["e", "f"]], dtype=object)
    y = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="numeric"):
        cls().fit(X, y)


@pytest.mark.parametrize("cls", [KOREAdditiveRegressor, KOREPairwiseRegressor, KORERegressor])
def test_predict_rejects_wrong_feature_count(cls):
    X_tr, y_tr, _, _ = _additive_xy(d=4)
    est = cls().fit(X_tr, y_tr)
    bad = np.zeros((10, 5))
    with pytest.raises(ValueError, match="features"):
        est.predict(bad)


@pytest.mark.parametrize("cls", [KOREAdditiveRegressor, KOREPairwiseRegressor, KORERegressor])
def test_predict_rejects_1d_X(cls):
    X_tr, y_tr, _, _ = _additive_xy(d=4)
    est = cls().fit(X_tr, y_tr)
    with pytest.raises(ValueError, match="2D X"):
        est.predict(np.zeros(4))


def test_additive_pilot_constants_populated():
    X_tr, y_tr, _, _ = _additive_xy()
    est = KOREAdditiveRegressor().fit(X_tr, y_tr)
    assert np.isfinite(est.A_hat_) and est.A_hat_ >= 0.0
    assert np.isfinite(est.tau_hat_) and est.tau_hat_ >= 0.0
    assert np.isfinite(est.G_dagger_) and est.G_dagger_ > 0.0
    assert est.fits_ >= 2


def test_pairwise_effective_max_g_returns_zero_when_infeasible():
    # p_pair(1) = 1 + 190*3 + 17955*9 = 162166 >> 0.45 * 2000 = 900.
    assert _pairwise_effective_max_g(2000, 190, 17955, 3, 8) == 0
    assert _pairwise_effective_max_g(2000, 190, 17955, 3, 0) == 0
    # p_pair(1) = 1 + 10*3 + 45*9 = 436 < 0.45 * 2000 = 900; should return >= 1.
    assert _pairwise_effective_max_g(2000, 10, 45, 3, 8) >= 1


def test_kore_sande_pairwise_skips_high_d_without_allocating():
    import tracemalloc
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 190))
    y = rng.standard_normal(2000)
    tracemalloc.start()
    res = kore_sande_pairwise(X, y, max_G=8)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert res["skipped"] is True
    assert res["loo"] == float("inf")
    assert res["fits"] == 0
    assert res["model"] is None
    assert peak < 50_000_000, f"skip path allocated {peak / 1e6:.1f} MB"


def test_auto_kore_falls_back_to_additive_when_pairwise_skipped():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 190))
    y = X[:, 0] + 0.3 * rng.standard_normal(2000)
    choice = auto_kore_intrinsic(X, y, max_G_add=20, max_G_pair=8)
    assert choice["structure"] == "additive"
    assert choice["candidates"]["pairwise"].get("skipped") is True
    assert choice["candidates"]["pairwise"]["fits"] == 0
    assert np.isfinite(choice["loo"])


def test_kore_pairwise_regressor_raises_when_infeasible():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 190))
    y = rng.standard_normal(2000)
    with pytest.raises(ValueError, match="infeasible"):
        KOREPairwiseRegressor(max_G=8).fit(X, y)
