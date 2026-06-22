"""Baseline methods for the KORE real-world benchmark.

Twenty-one methods organised by family (linear, KORE-spline,
tree-based, kernel, neighbours, neural). Hyperparameter search spaces
for the tree-based, kernel, neighbours, and neural baselines are lifted
verbatim from Grinsztajn, Oyallon, Varoquaux 2022 NeurIPS Appendix B
("Why do tree-based models still outperform deep learning on tabular
data?", https://arxiv.org/abs/2207.08815). Each search space carries
the corresponding paper-table reference in its source comment so the
provenance is auditable. Linear baselines use the sklearn ``*CV``
variants with their canonical alpha grids. Spline-family baselines
delegate to KORE library functions.

Every method exposes the same uniform interface:

    fit(X_train, y_train, seed) -> FitOutput

where ``FitOutput`` carries the prediction callable, the number of
model fits the method actually performed (for compute accounting), the
wall-clock fit time, and the best hyperparameter dictionary (for
methods that tune).
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

from .intrinsic import (
    auto_kore_intrinsic,
    cv_auto_intrinsic,
    criteria_auto_intrinsic,
    gcv_auto_intrinsic,
    fit_pairwise_sparse,
    pred_pairwise_sparse,
)
from .lib import fit_additive, pred_additive

# Catch ``Exception`` (not just ``ImportError``) on these optional
# imports because several of them load native libraries at import time
# (libomp on macOS, libgomp on Linux). When the native dependency is
# missing the import raises a runtime error rather than ImportError,
# and we still want the experiment driver to skip the method
# gracefully rather than crash the whole run.
try:
    import optuna

    # ERROR-level (rather than WARNING) suppresses the per-trial "Trial X
    # failed with parameters ... because of <Exception>" log line. With the
    # ``_silenced_objective`` wrapper below, every in-trial error becomes
    # ``optuna.TrialPruned`` and is no longer FAILED, so this suppression
    # is defence in depth for any noise emitted from Optuna's pruner or
    # bookkeeping code.
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    HAS_OPTUNA = True
except Exception:  # noqa: BLE001
    HAS_OPTUNA = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except Exception:  # noqa: BLE001
    HAS_XGBOOST = False

try:
    import lightgbm as lgbm

    HAS_LIGHTGBM = True
except Exception:  # noqa: BLE001
    HAS_LIGHTGBM = False

try:
    import catboost as cb

    HAS_CATBOOST = True
except Exception:  # noqa: BLE001
    HAS_CATBOOST = False

try:
    from pygam import LinearGAM, s as gam_spline

    HAS_PYGAM = True
except Exception:  # noqa: BLE001
    HAS_PYGAM = False


# ----------------------------------------------------------------------
# Common scaffolding
# ----------------------------------------------------------------------


@dataclass
class FitOutput:
    """Uniform return type from every baseline ``fit()``."""

    predict: Callable[[np.ndarray], np.ndarray]
    n_fits: int
    fit_time_s: float
    best_hp: dict = field(default_factory=dict)
    family: str = ""
    # True when zero Optuna trials completed within the wall-clock
    # budget and the final estimator was fit with library defaults.
    # Cells with this flag are still scored normally (per AutoGluon's
    # "always produce a prediction" guarantee, Erickson et al. 2020),
    # but the flag is propagated to the per-cell row so the aggregator
    # can surface how often each method's tuning ran out of time.
    used_defaults: bool = False


# Default Optuna budget. Inside the Grinsztajn 2022 range of 50 to 400
# trials per dataset; chosen to keep the full 36 x 21 x 5 grid tractable
# overnight on a 16-core laptop. Override via the driver if compute
# permits more trials. The trial count is the upper bound; the wall
# clock budget below is the lower bound, and ``study.optimize`` stops at
# whichever comes first.
DEFAULT_OPTUNA_TRIALS = 50
DEFAULT_INNER_FOLDS = 3
EARLY_STOP = 50

# Minimum number of completed Optuna trials required before
# ``study.best_params`` is accepted as the tuned result. AMLB
# (Gijsbers et al. 2024 JMLR) uses ``study.best_params`` regardless of
# trial count: any completed trial is a legitimate point in the search
# space, and discarding it would silently reward methods whose defaults
# happen to be competitive over methods whose first trial is already an
# improvement. The threshold is therefore one. ``_run_optuna`` falls
# through to the library-defaults branch (``used_defaults=True`` on the
# per-cell row) only when zero trials completed within the budget.
MIN_OPTUNA_TRIALS = 1

# Per-method search budget in seconds, passed to ``study.optimize`` as
# ``timeout``. This is the AMLB convention (Gijsbers et al. 2024 JMLR),
# which sets ``max_runtime_seconds`` per task and aborts at exactly
# ``2 x max_runtime_seconds`` if the framework misbehaves. The cell
# driver's SIGALRM backstop is set to ``2 x OPTUNA_TIMEOUT_S`` to match
# that contract; see ``experiments._CELL_TIMEOUT_S``.
OPTUNA_TIMEOUT_S = 180.0


def _booster_njobs() -> int:
    """Threads per booster fit (xgboost / lightgbm / catboost).

    Default ``1`` reproduces the inter-trial parallelism contract used
    throughout the codebase: the driver runs many cells concurrently via
    loky, and each cell single-threads its booster so the workers do
    not oversubscribe the CPU. Override with ``KORE_BOOSTER_NJOBS`` for
    the intra-fit parallelism probe: e.g. ``KORE_BOOSTER_NJOBS=4`` paired
    with ``KORE_NJOBS_HEAVY=20`` keeps total HEAVY-stage CPU usage at 80
    cores but cuts joblib coordination overhead 4x and lets XGBoost /
    LightGBM / CatBoost use their own native thread pool. Values <= 0
    fall back to ``1``."""
    raw = os.environ.get("KORE_BOOSTER_NJOBS")
    if raw is None:
        return 1
    try:
        val = int(raw)
    except ValueError:
        return 1
    return val if val >= 1 else 1


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class _IgnoreSigint:
    """Drop SIGINT for the duration of a ``with`` block.

    The boosting libraries (CatBoost especially, also XGBoost and
    LightGBM) call ``PyErr_CheckSignals`` from inside their C++ training
    loop and raise an empty-message ``KeyboardInterrupt`` whenever any
    Python signal is pending. On Databricks the notebook reaper
    delivers periodic SIGINTs to worker processes; without this guard
    every booster trial dies almost immediately. SIGALRM (which the
    cell driver uses for the 240 s wall-clock timeout) is unaffected,
    and the joblib parent process retains its own SIGINT handling for
    real user interrupts."""

    def __enter__(self):
        import signal as _signal
        self._signal = _signal
        self._prev = None
        if hasattr(_signal, "SIGINT"):
            try:
                self._prev = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
            except (ValueError, OSError):
                self._prev = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._prev is not None:
            try:
                self._signal.signal(self._signal.SIGINT, self._prev)
            except (ValueError, OSError):
                pass
        return False


def _inner_cv_score(
    estimator_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int = DEFAULT_INNER_FOLDS,
    trial: Any = None,
) -> tuple[float, int]:
    """Return ``(mean_cv_rmse, n_fits)`` over a K-fold CV of a fresh estimator each fold.

    When ``trial`` is provided, the running mean RMSE is reported after
    each fold via ``trial.report`` and the trial is pruned via
    ``trial.should_prune`` whenever Optuna's pruner declares it
    hopeless. This is the standard MedianPruner integration."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    fits = 0
    for step, (tr_idx, va_idx) in enumerate(kf.split(X)):
        est = estimator_factory()
        try:
            est.fit(X[tr_idx], y[tr_idx])
        except BaseException as exc:
            if trial is not None:
                raise optuna.TrialPruned() from exc
            raise
        fits += 1
        pred = est.predict(X[va_idx])
        scores.append(_rmse(y[va_idx], pred))
        if trial is not None:
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), fits


def _inner_cv_score_es(
    estimator_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    fit_with_eval: Callable[[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None],
    folds: int = DEFAULT_INNER_FOLDS,
    eval_frac: float = 0.2,
    trial: Any = None,
) -> tuple[float, int]:
    """Inner-CV scorer for early-stopping boosters.

    Carves a fraction (default 20%) out of each training fold to serve
    as the early-stopping eval set. ``fit_with_eval(est, X_fit, y_fit,
    X_es, y_es)`` is method-specific because XGBoost, LightGBM, and
    CatBoost expose early stopping through different APIs. The outer
    fold's validation slice (``va_idx``) is never seen by either the
    fit or the early-stopping eval, so the reported RMSE is honest."""
    rng_eval = np.random.default_rng(seed + 12345)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    fits = 0
    for step, (tr_idx, va_idx) in enumerate(kf.split(X)):
        perm = rng_eval.permutation(len(tr_idx))
        n_es = max(1, int(eval_frac * len(perm)))
        es_idx = tr_idx[perm[:n_es]]
        fit_idx = tr_idx[perm[n_es:]]
        est = estimator_factory()
        try:
            fit_with_eval(est, X[fit_idx], y[fit_idx], X[es_idx], y[es_idx])
        except BaseException as exc:
            # CatBoost (and to a lesser extent XGBoost / LightGBM) call
            # PyErr_CheckSignals from their C++ training loop and raise
            # an empty-message KeyboardInterrupt whenever any Python
            # signal is pending. On Databricks the notebook reaper
            # delivers periodic SIGINTs to worker processes, which
            # would otherwise abort the whole study. Convert any such
            # interruption into a pruned trial so the study survives.
            if trial is not None:
                raise optuna.TrialPruned() from exc
            raise
        fits += 1
        pred = est.predict(X[va_idx])
        scores.append(_rmse(y[va_idx], pred))
        if trial is not None:
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), fits


def _scale_arrays(X_tr: np.ndarray, X_te: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """Standardise features. The kernel and neural baselines are
    sensitive to feature scaling; the tree baselines are not, but
    standardising uniformly costs nothing and keeps the inner-CV state
    deterministic across methods."""
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te) if X_te is not None else None
    return Xs_tr, Xs_te, scaler


# ----------------------------------------------------------------------
# Linear family (4)
# ----------------------------------------------------------------------


def fit_linear(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
    """Plain ordinary least squares. One fit, no tuning."""
    t0 = time.perf_counter()
    Xs, _, scaler = _scale_arrays(X)
    model = LinearRegression()
    model.fit(Xs, y)
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X_new))

    return FitOutput(predict=predict, n_fits=1, fit_time_s=elapsed, family="linear")


def fit_ridge_cv(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
    """Ridge with internal generalized CV over a 13-point alpha grid."""
    t0 = time.perf_counter()
    Xs, _, scaler = _scale_arrays(X)
    alphas = np.logspace(-3, 3, 13)
    model = RidgeCV(alphas=alphas, scoring="neg_root_mean_squared_error", cv=DEFAULT_INNER_FOLDS)
    model.fit(Xs, y)
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X_new))

    return FitOutput(
        predict=predict,
        n_fits=len(alphas) * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp={"alpha": float(model.alpha_)},
        family="linear",
    )


def fit_lasso_cv(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
    """Lasso with sklearn's path-based CV over 100 alphas."""
    t0 = time.perf_counter()
    Xs, _, scaler = _scale_arrays(X)
    model = LassoCV(n_alphas=100, cv=DEFAULT_INNER_FOLDS, random_state=seed, max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xs, y)
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X_new))

    return FitOutput(
        predict=predict,
        n_fits=100 * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp={"alpha": float(model.alpha_)},
        family="linear",
    )


def fit_elasticnet_cv(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
    """ElasticNet with path-based CV over 5 l1_ratio x 100 alphas."""
    t0 = time.perf_counter()
    Xs, _, scaler = _scale_arrays(X)
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        n_alphas=100,
        cv=DEFAULT_INNER_FOLDS,
        random_state=seed,
        max_iter=5000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xs, y)
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X_new))

    return FitOutput(
        predict=predict,
        n_fits=5 * 100 * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp={"alpha": float(model.alpha_), "l1_ratio": float(model.l1_ratio_)},
        family="linear",
    )


# ----------------------------------------------------------------------
# KORE spline family (7)
# ----------------------------------------------------------------------


def _scale_to_unit_box(X_tr: np.ndarray, X_te: np.ndarray | None = None):
    """KORE expects features in roughly [0, 1] for the spline basis
    construction in ``lib.py``. Min-max scale per column on the train
    fold, clip the test fold to [0, 1] to keep the spline well
    defined at evaluation time."""
    lo = X_tr.min(axis=0)
    hi = X_tr.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Xs_tr = (X_tr - lo) / span
    Xs_te = None
    if X_te is not None:
        Xs_te = np.clip((X_te - lo) / span, 0.0, 1.0)
    return Xs_tr, Xs_te, (lo, span)


def _kore_predict_factory(choice: dict, lo: np.ndarray, span: np.ndarray):
    structure = choice["structure"]
    model = choice["model"]

    def predict(X_new: np.ndarray) -> np.ndarray:
        Xn = np.clip((X_new - lo) / span, 0.0, 1.0)
        if structure == "additive":
            return pred_additive(model, Xn)
        return pred_pairwise_sparse(model, Xn)

    return predict


_KORE_GRID_ADD = list(range(2, 21))
_KORE_GRID_PAIR = list(range(2, 9))


def fit_kore(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
    """KORE: the closed-form plug-in selector from this paper."""
    t0 = time.perf_counter()
    Xs, _, (lo, span) = _scale_to_unit_box(X)
    choice = auto_kore_intrinsic(Xs, y, max_G_add=20, max_G_pair=8)
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=_kore_predict_factory(choice, lo, span),
        n_fits=int(choice.get("fits_total", 0)),
        fit_time_s=elapsed,
        best_hp={
            "structure": choice["structure"],
            "G_star": int(choice["G_star"]),
        },
        family="spline",
    )


def fit_cv_spline(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
    """Exhaustive K-fold CV over the resolution grid for both
    structure families."""
    t0 = time.perf_counter()
    Xs, _, (lo, span) = _scale_to_unit_box(X)
    choice = cv_auto_intrinsic(
        Xs, y,
        grid_add=_KORE_GRID_ADD,
        grid_pair=_KORE_GRID_PAIR,
        K=DEFAULT_INNER_FOLDS,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=_kore_predict_factory(choice, lo, span),
        n_fits=int(choice.get("fits", 0)),
        fit_time_s=elapsed,
        best_hp={"structure": choice["structure"], "G_star": int(choice["G_star"])},
        family="spline",
    )


def fit_gcv_spline(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
    t0 = time.perf_counter()
    Xs, _, (lo, span) = _scale_to_unit_box(X)
    choice = gcv_auto_intrinsic(Xs, y, grid_add=_KORE_GRID_ADD, grid_pair=_KORE_GRID_PAIR)
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=_kore_predict_factory(choice, lo, span),
        n_fits=int(choice.get("fits_total", 0)),
        fit_time_s=elapsed,
        best_hp={"structure": choice["structure"], "G_star": int(choice["G_star"])},
        family="spline",
    )


def _criteria_factory(criterion: str):
    def fit(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
        t0 = time.perf_counter()
        Xs, _, (lo, span) = _scale_to_unit_box(X)
        chosen_all = criteria_auto_intrinsic(
            Xs, y,
            grid_add=_KORE_GRID_ADD,
            grid_pair=_KORE_GRID_PAIR,
            criterion_names=("gcv", "aic", "bic", "cp"),
        )
        choice = chosen_all[criterion]
        elapsed = time.perf_counter() - t0
        return FitOutput(
            predict=_kore_predict_factory(choice, lo, span),
            n_fits=int(choice.get("fits_total", 0)),
            fit_time_s=elapsed,
            best_hp={
                "structure": choice["structure"],
                "G_star": int(choice["G_star"]),
                "criterion": criterion,
            },
            family="spline",
        )

    fit.__name__ = f"fit_{criterion}_spline"
    return fit


fit_aic_spline = _criteria_factory("aic")
fit_bic_spline = _criteria_factory("bic")
fit_cp_spline = _criteria_factory("cp")


def fit_pygam(X: np.ndarray, y: np.ndarray, seed: int) -> FitOutput:
    """Independent third-party GAM (pyGAM, Daniel Serven 2018) with
    its own GCV-internal lambda selection. Included as a sanity check
    so reviewers see KORE is not the only GAM in the comparison."""
    if not HAS_PYGAM:
        raise RuntimeError("pyGAM not installed; method skipped.")
    t0 = time.perf_counter()
    Xs, _, scaler = _scale_arrays(X)
    n_features = Xs.shape[1]
    if n_features == 0:
        raise RuntimeError("pyGAM: empty feature set.")
    terms = gam_spline(0)
    for j in range(1, n_features):
        terms = terms + gam_spline(j)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LinearGAM(terms).gridsearch(
            Xs, y, lam=np.logspace(-3, 3, 11), progress=False
        )
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return model.predict(scaler.transform(X_new))

    return FitOutput(
        predict=predict,
        n_fits=11,
        fit_time_s=elapsed,
        best_hp={"lam_grid": "logspace(-3,3,11)"},
        family="spline",
    )


# ----------------------------------------------------------------------
# Tree-based family (6). Search spaces from Grinsztajn et al. 2022
# NeurIPS Appendix B (https://arxiv.org/abs/2207.08815). Specific
# tables annotated next to each block. Bayesian search is via Optuna's
# TPE sampler with median pruning. Inner CV: 3-fold.
# ----------------------------------------------------------------------


def _optuna_sampler(seed: int):
    if not HAS_OPTUNA:
        raise RuntimeError("optuna not installed; tunable methods skipped.")
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=10)


def _optuna_pruner():
    return optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=0, interval_steps=1
    )


def _silenced_objective(objective: Callable[[Any], float]) -> Callable[[Any], float]:
    """Convert any in-trial exception into ``optuna.TrialPruned``.

    The cell driver in ``experiments._real_data_one_cell`` arms a
    one-shot ``SIGALRM`` for the hard wall-clock backstop. The handler
    raises ``TimeoutError`` in whichever Python frame happens to be
    executing when the timer fires: ``est.fit`` (covered by the inner
    converters in ``_inner_cv_score`` / ``_inner_cv_score_es``),
    ``est.predict``, ``trial.report``, ``trial.suggest_*``, or any
    point in Optuna's bookkeeping between iterations. Errors raised
    outside the inner-CV ``fit`` block escape to ``study.optimize``,
    which marks the trial FAILED and emits a WARN log per occurrence.

    Wrapping the entire objective converts any escaped error
    (``TimeoutError``, ``KeyboardInterrupt`` from the booster C++
    loop, anything else) into ``optuna.TrialPruned``. Optuna records
    the trial as PRUNED, never FAILED, and emits no log line; the
    study's ``best_trial`` lookup ignores PRUNED trials and surfaces
    the best COMPLETE one. This is the AMLB framework-wrapper
    convention (Gijsbers et al. 2024 JMLR) for trials that don't
    finish."""

    def wrapped(trial):
        try:
            return objective(trial)
        except optuna.TrialPruned:
            raise
        except BaseException as exc:  # noqa: BLE001
            raise optuna.TrialPruned() from exc

    return wrapped


def _run_optuna(
    objective: Callable[[Any], float],
    seed: int,
    n_trials: int,
    timeout: Optional[float] = None,
) -> tuple[dict, int]:
    """Run an Optuna study and return ``(best_params, n_trials_completed)``.

    The trial-count cap is the upper bound; the wall-clock ``timeout``
    (defaults to ``OPTUNA_TIMEOUT_S``) is the lower bound. ``study.optimize``
    stops at whichever comes first. This matches the AMLB protocol
    (Gijsbers et al. 2024 JMLR), which sets a soft per-task budget and
    relies on the framework to stop cleanly.

    Following AMLB, ``study.best_params`` is returned whenever at least
    one trial completed, regardless of trial count: any completed trial
    is a legitimate point in the search space and is at least as good
    as the library defaults under the same evaluation. Only when zero
    trials completed within the budget does the function return
    ``({}, 0)``; the caller then fits with library defaults (AutoGluon's
    "always produce a prediction" protocol, Erickson et al. 2020
    AutoML)."""
    sampler = _optuna_sampler(seed)
    pruner = _optuna_pruner()
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner
    )
    timeout = OPTUNA_TIMEOUT_S if timeout is None else float(timeout)
    # ``_silenced_objective`` converts any escaped in-trial exception
    # (TimeoutError from the cell-level SIGALRM, KeyboardInterrupt from
    # a booster's C++ loop, anything else) into ``optuna.TrialPruned``.
    # That keeps every trial in either COMPLETE or PRUNED state, never
    # FAILED, so Optuna emits no per-trial WARN log line and
    # ``study.best_trial`` remains derived from completed trials only.
    safe_objective = _silenced_objective(objective)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            study.optimize(
                safe_objective, n_trials=n_trials, timeout=timeout,
                show_progress_bar=False, catch=(),
            )
        except BaseException:
            # ``_silenced_objective`` already converts in-trial errors
            # to TrialPruned, so anything reaching this handler comes
            # from Optuna's own bookkeeping (a SIGALRM that landed
            # between trials, a worker-level interrupt). The in-flight
            # study state is preserved on ``study`` itself, so fall
            # through and decide based on completed trials below.
            pass
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < MIN_OPTUNA_TRIALS:
        # Zero completed trials: the budget was exhausted before any
        # trial finished. Signal a defaults fallback to the caller; the
        # wall-clock cost of any pruned/in-flight trials is still
        # captured by each fit function's ``fit_time_s`` field.
        return {}, 0
    return study.best_params, len(completed)


def fit_random_forest(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """Random Forest. Search space: Grinsztajn et al. 2022 Appendix B Table A.4.

    Ranges:
        max_depth :: int [2, 25] (None substituted by 25)
        n_estimators :: 250 (fixed per Table A.4)
        max_features :: float [0.1, 1.0]
        min_samples_split :: int [2, 30]
        min_samples_leaf :: int [1, 30]
    """

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 25),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        }
        params["n_estimators"] = 250
        score, _ = _inner_cv_score(
            lambda: RandomForestRegressor(**params, random_state=seed, n_jobs=1),
            X, y, seed, trial=trial,
        )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = RandomForestRegressor(**best, n_estimators=250, random_state=seed, n_jobs=1)
    final.fit(X, y)
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=lambda X_new: final.predict(X_new),
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="tree",
        used_defaults=(trials == 0),
    )


def fit_extra_trees(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """ExtraTrees. Same search space as Random Forest (Grinsztajn et al.
    2022 Appendix B Table A.4)."""

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 25),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        }
        params["n_estimators"] = 250
        score, _ = _inner_cv_score(
            lambda: ExtraTreesRegressor(**params, random_state=seed, n_jobs=1),
            X, y, seed, trial=trial,
        )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = ExtraTreesRegressor(**best, n_estimators=250, random_state=seed, n_jobs=1)
    final.fit(X, y)
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=lambda X_new: final.predict(X_new),
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="tree",
        used_defaults=(trials == 0),
    )


def fit_hist_gbm(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """HistGradientBoosting (sklearn). Search space: Grinsztajn et al.
    2022 Appendix B Table A.6.

    Ranges:
        max_iter :: 1000 (fixed)
        learning_rate :: log [0.01, 1.0]
        max_depth :: int [2, 25]
        max_leaf_nodes :: log_int [5, 50]
        min_samples_leaf :: log_int [1, 100]
        l2_regularization :: log [exp(-16), exp(2)]
    """

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 25),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 50, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100, log=True),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", math.exp(-16), math.exp(2), log=True
            ),
        }
        params["max_iter"] = 1000
        score, _ = _inner_cv_score(
            lambda: HistGradientBoostingRegressor(**params, random_state=seed, early_stopping=True),
            X, y, seed, trial=trial,
        )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = HistGradientBoostingRegressor(
        **best, max_iter=1000, random_state=seed, early_stopping=True
    )
    final.fit(X, y)
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=lambda X_new: final.predict(X_new),
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="tree",
        used_defaults=(trials == 0),
    )


def fit_xgboost(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """XGBoost. Search space: Grinsztajn et al. 2022 Appendix B Table A.5
    ("GBT" entry, mapping to XGBoost ranges).

    Ranges:
        n_estimators :: 1000 (fixed; early stopping at 50 rounds)
        max_depth :: int [1, 11]
        learning_rate :: log [exp(-7), 1.0]
        subsample :: uniform [0.5, 1.0]
        colsample_bytree :: uniform [0.5, 1.0]
        colsample_bylevel :: uniform [0.5, 1.0]
        min_child_weight :: log [exp(-16), exp(5)]
        reg_alpha :: log [exp(-16), exp(2)]
        reg_lambda :: log [exp(-16), exp(2)]
        gamma :: log [exp(-16), exp(2)]
    """
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost not installed; method skipped.")

    fixed = dict(
        n_estimators=1000,
        tree_method="hist",
        verbosity=0,
        n_jobs=_booster_njobs(),
        random_state=seed,
        early_stopping_rounds=EARLY_STOP,
    )

    def fit_with_eval(est, X_fit, y_fit, X_es, y_es):
        with _IgnoreSigint():
            est.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 11),
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-7), 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", math.exp(-16), math.exp(5), log=True
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", math.exp(-16), math.exp(2), log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", math.exp(-16), math.exp(2), log=True),
            "gamma": trial.suggest_float("gamma", math.exp(-16), math.exp(2), log=True),
        }
        params.update(fixed)
        score, _ = _inner_cv_score_es(
            lambda: xgb.XGBRegressor(**params),
            X, y, seed, fit_with_eval, trial=trial,
        )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = xgb.XGBRegressor(**best, **fixed)
    rng_eval = np.random.default_rng(seed + 12345)
    perm = rng_eval.permutation(len(y))
    n_es = max(1, int(0.2 * len(perm)))
    es_idx, fit_idx = perm[:n_es], perm[n_es:]
    with _IgnoreSigint():
        final.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[es_idx], y[es_idx])], verbose=False)
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=lambda X_new: final.predict(X_new),
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="tree",
        used_defaults=(trials == 0),
    )


def fit_lightgbm(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """LightGBM. Search space: Grinsztajn et al. 2022 Appendix B Table A.5
    (paper presents GBT-style ranges; LightGBM uses the same family.)

    Ranges (LightGBM-specific naming applied):
        n_estimators :: 1000 (fixed)
        num_leaves :: log_int [5, 50]
        max_depth :: int [-1, 25]  (-1 = unbounded)
        learning_rate :: log [exp(-7), 1.0]
        min_child_samples :: log_int [1, 100]
        subsample :: uniform [0.5, 1.0]
        colsample_bytree :: uniform [0.5, 1.0]
        reg_alpha :: log [exp(-16), exp(2)]
        reg_lambda :: log [exp(-16), exp(2)]
    """
    if not HAS_LIGHTGBM:
        raise RuntimeError("lightgbm not installed; method skipped.")

    fixed = dict(n_estimators=1000, verbose=-1, n_jobs=_booster_njobs(), random_state=seed)
    es_callbacks = [
        lgbm.early_stopping(EARLY_STOP, verbose=False),
        lgbm.log_evaluation(0),
    ]

    def fit_with_eval(est, X_fit, y_fit, X_es, y_es):
        with _IgnoreSigint():
            est.fit(X_fit, y_fit,
                    eval_set=[(X_es, y_es)], callbacks=es_callbacks)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 5, 50, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 25),
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-7), 1.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", math.exp(-16), math.exp(2), log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", math.exp(-16), math.exp(2), log=True),
        }
        params.update(fixed)
        score, _ = _inner_cv_score_es(
            lambda: lgbm.LGBMRegressor(**params),
            X, y, seed, fit_with_eval, trial=trial,
        )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = lgbm.LGBMRegressor(**best, **fixed)
    rng_eval = np.random.default_rng(seed + 12345)
    perm = rng_eval.permutation(len(y))
    n_es = max(1, int(0.2 * len(perm)))
    es_idx, fit_idx = perm[:n_es], perm[n_es:]
    with _IgnoreSigint():
        final.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[es_idx], y[es_idx])], callbacks=es_callbacks)
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=lambda X_new: final.predict(X_new),
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="tree",
        used_defaults=(trials == 0),
    )


def fit_catboost(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """CatBoost. Search space adapted from Prokhorenkova et al. 2018
    NeurIPS (CatBoost paper) and Grinsztajn et al. 2022 GBT ranges.

    Ranges:
        iterations :: 2000 (early-stopped at 50 rounds)
        depth :: int [3, 10] (CatBoost recommended)
        learning_rate :: log [exp(-7), 1.0]
        l2_leaf_reg :: log [1, 30]
        random_strength :: uniform [0, 10]
        bagging_temperature :: uniform [0, 5]
    """
    if not HAS_CATBOOST:
        raise RuntimeError("catboost not installed; method skipped.")

    fixed = dict(iterations=2000, verbose=False, random_seed=seed,
                 thread_count=_booster_njobs(), early_stopping_rounds=EARLY_STOP)

    def fit_with_eval(est, X_fit, y_fit, X_es, y_es):
        with _IgnoreSigint():
            est.fit(X_fit, y_fit, eval_set=(X_es, y_es), verbose=False)

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-7), 1.0, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        }
        params.update(fixed)
        score, _ = _inner_cv_score_es(
            lambda: cb.CatBoostRegressor(**params),
            X, y, seed, fit_with_eval, trial=trial,
        )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = cb.CatBoostRegressor(**best, **fixed)
    rng_eval = np.random.default_rng(seed + 12345)
    perm = rng_eval.permutation(len(y))
    n_es = max(1, int(0.2 * len(perm)))
    es_idx, fit_idx = perm[:n_es], perm[n_es:]
    with _IgnoreSigint():
        final.fit(X[fit_idx], y[fit_idx],
                  eval_set=(X[es_idx], y[es_idx]), verbose=False)
    elapsed = time.perf_counter() - t0
    return FitOutput(
        predict=lambda X_new: np.asarray(final.predict(X_new)).ravel(),
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="tree",
        used_defaults=(trials == 0),
    )


# ----------------------------------------------------------------------
# Kernel family (2). Search spaces from Grinsztajn et al. 2022
# Appendix B Table A.7 ("SVM" entry) for SVR-RBF; KernelRidge mirrors
# the same kernel hyperparameters with a different penalty parameter.
# ----------------------------------------------------------------------


def fit_svr_rbf(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """SVR with RBF kernel. Search space: Grinsztajn et al. 2022
    Appendix B Table A.7.

    Ranges:
        C :: log [exp(-10), exp(10)]
        gamma :: log [exp(-10), exp(10)]
        epsilon :: log [exp(-10), 1.0]

    Practical caps applied on top of the Grinsztajn search space:
    ``max_iter=200_000`` and ``cache_size=500`` MB. libsvm's solver has
    no built-in iteration cap, so without ``max_iter`` a single trial
    in the high-C / small-gamma corner of the search space can run for
    minutes inside one fold and burn the entire per-cell budget. The
    cap matches common practice in subsequent tabular benchmarks
    (Borisov et al. 2024) and only triggers in the pathological
    region; well-conditioned trials converge in << 200k iterations."""
    Xs, _, scaler = _scale_arrays(X)
    svr_extra = dict(max_iter=200_000, cache_size=500)

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", math.exp(-10), math.exp(10), log=True),
            "gamma": trial.suggest_float("gamma", math.exp(-10), math.exp(10), log=True),
            "epsilon": trial.suggest_float("epsilon", math.exp(-10), 1.0, log=True),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score, _ = _inner_cv_score(
                lambda: SVR(kernel="rbf", **params, **svr_extra),
                Xs, y, seed, trial=trial,
            )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = SVR(kernel="rbf", **best, **svr_extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(Xs, y)
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return final.predict(scaler.transform(X_new))

    return FitOutput(
        predict=predict,
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="kernel",
        used_defaults=(trials == 0),
    )


def fit_kernel_ridge_rbf(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """KernelRidge with RBF kernel. Search space mirrors SVR-RBF: same
    gamma range, with the regularisation strength on alpha (Pedregosa
    et al. 2011 sklearn defaults).

    Ranges:
        alpha :: log [exp(-10), exp(2)]
        gamma :: log [exp(-10), exp(10)]
    """
    Xs, _, scaler = _scale_arrays(X)

    def objective(trial):
        params = {
            "alpha": trial.suggest_float("alpha", math.exp(-10), math.exp(2), log=True),
            "gamma": trial.suggest_float("gamma", math.exp(-10), math.exp(10), log=True),
        }
        score, _ = _inner_cv_score(
            lambda: KernelRidge(kernel="rbf", **params), Xs, y, seed,
            trial=trial,
        )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = KernelRidge(kernel="rbf", **best)
    final.fit(Xs, y)
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return final.predict(scaler.transform(X_new))

    return FitOutput(
        predict=predict,
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="kernel",
        used_defaults=(trials == 0),
    )


# ----------------------------------------------------------------------
# Neighbours (1). Standard log-int sweep over k.
# ----------------------------------------------------------------------


def fit_knn(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS // 2) -> FitOutput:
    """KNN. Sweep over n_neighbors, weights, p (Pedregosa et al. 2011).

    Ranges:
        n_neighbors :: log_int [1, 50]
        weights :: {"uniform", "distance"}
        p :: {1, 2}
    """
    Xs, _, scaler = _scale_arrays(X)

    def objective(trial):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 50, log=True),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
        score, _ = _inner_cv_score(
            lambda: KNeighborsRegressor(**params, n_jobs=1), Xs, y, seed,
            trial=trial,
        )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    final = KNeighborsRegressor(**best, n_jobs=1)
    final.fit(Xs, y)
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return final.predict(scaler.transform(X_new))

    return FitOutput(
        predict=predict,
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=dict(best),
        family="neighbors",
        used_defaults=(trials == 0),
    )


# ----------------------------------------------------------------------
# Neural (1). MLP via sklearn. Search space from Grinsztajn et al.
# 2022 Appendix B Table A.8 (MLP entry).
# ----------------------------------------------------------------------


def fit_mlp(X: np.ndarray, y: np.ndarray, seed: int, n_trials: int = DEFAULT_OPTUNA_TRIALS) -> FitOutput:
    """sklearn MLPRegressor. Search space: Grinsztajn et al. 2022
    Appendix B Table A.8 (MLP entry).

    Ranges:
        hidden_layer_sizes :: choice {(64,), (128,), (256,),
                                       (64, 64), (128, 128), (256, 256)}
        alpha :: log [exp(-10), exp(-2)]
        learning_rate_init :: log [exp(-10), exp(-2)]
        activation :: {"relu", "tanh"}
        max_iter :: 500
    """
    Xs, _, scaler = _scale_arrays(X)

    def objective(trial):
        layer_choice = trial.suggest_categorical(
            "hidden_layers",
            ["64", "128", "256", "64-64", "128-128", "256-256"],
        )
        layers = tuple(int(t) for t in layer_choice.split("-"))
        params = {
            "hidden_layer_sizes": layers,
            "alpha": trial.suggest_float("alpha", math.exp(-10), math.exp(-2), log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", math.exp(-10), math.exp(-2), log=True
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "max_iter": 500,
            "random_state": seed,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score, _ = _inner_cv_score(
                lambda: MLPRegressor(**params, early_stopping=True),
                Xs, y, seed, trial=trial,
            )
        return score

    t0 = time.perf_counter()
    best, trials = _run_optuna(objective, seed, n_trials)
    if trials == 0:
        # Zero trials completed inside the wall-clock budget: fall back to
        # sklearn's MLPRegressor defaults (single hidden layer of 100).
        layer_choice = ""
        final = MLPRegressor(
            max_iter=500, random_state=seed, early_stopping=True,
        )
    else:
        layer_choice = best.pop("hidden_layers")
        layers = tuple(int(t) for t in layer_choice.split("-"))
        final = MLPRegressor(
            **best,
            hidden_layer_sizes=layers,
            max_iter=500,
            random_state=seed,
            early_stopping=True,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(Xs, y)
    elapsed = time.perf_counter() - t0

    def predict(X_new: np.ndarray) -> np.ndarray:
        return final.predict(scaler.transform(X_new))

    return FitOutput(
        predict=predict,
        n_fits=trials * DEFAULT_INNER_FOLDS + 1,
        fit_time_s=elapsed,
        best_hp=({"hidden_layers": layer_choice, **best} if trials else {}),
        family="neural",
        used_defaults=(trials == 0),
    )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


METHODS: dict[str, Callable[..., FitOutput]] = {
    "linear": fit_linear,
    "ridge_cv": fit_ridge_cv,
    "lasso_cv": fit_lasso_cv,
    "elasticnet_cv": fit_elasticnet_cv,
    "kore": fit_kore,
    "cv_spline": fit_cv_spline,
    "gcv_spline": fit_gcv_spline,
    "cp_spline": fit_cp_spline,
    "aic_spline": fit_aic_spline,
    "bic_spline": fit_bic_spline,
    "pygam": fit_pygam,
    "random_forest": fit_random_forest,
    "extra_trees": fit_extra_trees,
    "hist_gbm": fit_hist_gbm,
    "xgboost": fit_xgboost,
    "lightgbm": fit_lightgbm,
    "catboost": fit_catboost,
    "svr_rbf": fit_svr_rbf,
    "kernel_ridge_rbf": fit_kernel_ridge_rbf,
    "knn": fit_knn,
    "mlp": fit_mlp,
}


METHOD_FAMILY: dict[str, str] = {
    "linear": "linear",
    "ridge_cv": "linear",
    "lasso_cv": "linear",
    "elasticnet_cv": "linear",
    "kore": "spline",
    "cv_spline": "spline",
    "gcv_spline": "spline",
    "cp_spline": "spline",
    "aic_spline": "spline",
    "bic_spline": "spline",
    "pygam": "spline",
    "random_forest": "tree",
    "extra_trees": "tree",
    "hist_gbm": "tree",
    "xgboost": "tree",
    "lightgbm": "tree",
    "catboost": "tree",
    "svr_rbf": "kernel",
    "kernel_ridge_rbf": "kernel",
    "knn": "neighbors",
    "mlp": "neural",
}


def available_methods() -> list[str]:
    """Return the methods whose optional dependencies are present."""
    available: list[str] = []
    for name in METHODS:
        if name == "xgboost" and not HAS_XGBOOST:
            continue
        if name == "lightgbm" and not HAS_LIGHTGBM:
            continue
        if name == "catboost" and not HAS_CATBOOST:
            continue
        if name == "pygam" and not HAS_PYGAM:
            continue
        if name in {
            "random_forest", "extra_trees", "hist_gbm",
            "xgboost", "lightgbm", "catboost",
            "svr_rbf", "kernel_ridge_rbf", "knn", "mlp",
        } and not HAS_OPTUNA:
            continue
        available.append(name)
    return available
