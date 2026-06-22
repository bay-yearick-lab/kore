# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]
# ///
"""
Comprehensive Experiment Suite for Interaction-Aware KORE
=========================================================

Reproducibility
---------------
  MASTER_SEED = 2026  -- every data seed is derived deterministically from this.
  All per-task seeds are pure functions of (MASTER_SEED, experiment, config).

Parallelism
-----------
  All experiments except *scaling* are embarrassingly parallel across
  (config, seed) pairs via joblib, using ``min(n_tasks, os.cpu_count())``
  workers to avoid spawning idle processes. Robustness is parallelised at
  the per-scenario level (40 tasks) rather than per-(d, seed) (10 tasks).
  Scaling stays sequential because it measures wall-clock time. The
  real-data driver in ``run_real_data`` schedules cells with
  longest-processing-time-first ordering and dispatches them in three
  sequential ``Parallel(backend="loky", return_as="generator_unordered")``
  stages bucketed by method weight (heavy >= 4.0, medium in [2.0, 4.0),
  light < 2.0). Per-stage ``n_jobs`` are gated by ``KORE_NJOBS_HEAVY``,
  ``KORE_NJOBS_MEDIUM``, and ``KORE_NJOBS_LIGHT``, each capped at the
  local worker cap. Each stage catches ``TerminatedWorkerError`` so a
  single SIGKILL'd worker does not abort the remaining stages.
  Per-cell peak RSS is sampled at 0.5 s and written to
  ``results/real_data_memory.csv``. Every worker
  enforces a hard one-thread BLAS cap via
  ``threadpoolctl.threadpool_limits(1)`` so cluster-level
  ``OMP_NUM_THREADS`` settings cannot oversubscribe the box.

Usage
-----
  uv run python -m kore.experiments all          # everything
  uv run python -m kore.experiments frontier     # one experiment
  uv run python -m kore.experiments law frontier # two experiments
"""

import faulthandler
faulthandler.enable()

import gc
import logging, os, signal, socket, sys, threading, time, warnings
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _sample_peak_rss(proc, stop_event, peak_holder, interval_s=0.5,
                      advisory_mb: float = float("inf"),
                      advisory_flag=None):
    """Daemon-thread RSS sampler with soft per-cell cap.

    Samples ``proc.memory_info().rss`` every ``interval_s`` seconds and
    updates ``peak_holder[0]``. When ``advisory_mb`` is finite and the
    worker's RSS exceeds it, sets ``advisory_flag[0] = True``, prints a
    single ``[cell-rss-advisory]`` line to stderr, and calls
    ``_thread.interrupt_main()`` to raise ``KeyboardInterrupt`` on the
    cell's main thread. The cell's ``except BaseException`` branch
    catches the interrupt, falls back to the constant-predictor row
    with ``used_constant_predictor=True`` and ``error="rss advisory
    exceeded ..."``, and the worker continues to its next cell. In-
    process signalling is used (rather than ``SIGTERM`` to the worker
    process) so loky never observes a worker death; sending ``SIGTERM``
    here would raise ``TerminatedWorkerError`` and sentinel every cell
    queued behind the offender. The cgroup OOM killer and the
    cell-level ``SIGALRM`` (``_CELL_TIMEOUT_S``) remain the hard
    backstops for runaways that escape the soft cap (e.g. RSS that
    grows entirely inside a long-running C call that does not check
    for pending signals).
    """
    import _thread
    advised = False
    try:
        while not stop_event.is_set():
            try:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                if rss_mb > peak_holder[0]:
                    peak_holder[0] = rss_mb
                if not advised and rss_mb > advisory_mb:
                    advised = True
                    if advisory_flag is not None:
                        advisory_flag[0] = True
                    try:
                        sys.stderr.write(
                            f"[cell-rss-advisory] pid={proc.pid} "
                            f"rss_mb={rss_mb:.0f} advisory_mb={advisory_mb:.0f}"
                            f" interrupting cell\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
                    try:
                        _thread.interrupt_main()
                    except Exception:
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            stop_event.wait(interval_s)
    except Exception:
        return

warnings.filterwarnings("ignore")
# Some host stdouts (Databricks DatabricksOutStream, captured pytest
# streams, ...) do not implement TextIOWrapper.reconfigure. Best-effort.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

# On Databricks (and any host that preinstalls MLflow autologging), the
# autolog hook races sklearn's submodule imports and raises a noisy
# "cannot import name 'LeaveOneOut' from partially initialized module
# 'sklearn.model_selection'" warning. The hook adds per-fit overhead
# the benchmark does not need. Disable it before sklearn is pulled in
# transitively by .intrinsic / .lib below. No-op if mlflow is absent.
try:
    import mlflow as _mlflow
    _mlflow.autolog(disable=True, silent=True)
    try:
        import mlflow.sklearn as _mlflow_sk
        _mlflow_sk.autolog(disable=True, silent=True)
    except Exception:
        pass
    del _mlflow
except Exception:
    pass

from .intrinsic import (
    auto_kore_intrinsic, auto_kore_discover, cv_auto_intrinsic,
    criteria_additive, criteria_auto_intrinsic,
    gcv_auto_intrinsic, refine_kore_additive, fit_pairwise_sparse,
    pred_pairwise_sparse, screen_pairs_residual,
    make_additive_target, make_sparse_pairwise_target, make_dataset,
    make_threeway_target, make_nonsmooth_target, make_misspecified_pairwise,
    make_friedman1, make_friedman2, make_franke, make_sparse_additive_highdim,
    gcv_additive, _ensure_2d,
)
from .lib import (
    fit_additive, pred_additive, fit_spline_1d, pred_spline_1d,
    cv_spline_1d, rmse, mse,
    closed_form_g_star,
)

# ── reproducibility ──────────────────────────────────────────────────
MASTER_SEED = 2026

# ── paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Honor ``KORE_RESULTS_DIR`` so a non-editable install on a read-only
# checkout (Databricks Workspace Files, site-packages) can redirect CSV
# checkpoints to a writable Volume or DBFS path without monkey-patching.
_RESULTS_DIR_OVERRIDE = os.environ.get("KORE_RESULTS_DIR")
OUT = (
    Path(_RESULTS_DIR_OVERRIDE).expanduser()
    if _RESULTS_DIR_OVERRIDE
    else PROJECT_ROOT / "results"
)
OUT.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────
GRID_ADD  = list(range(1, 21))
GRID_PAIR = list(range(1, 11))
MAX_G_ADD = 20
MAX_G_PAIR = 10
CV_FOLDS  = 3
N_JOBS    = int(os.environ.get("KORE_NJOBS", os.cpu_count() or 1))
# Per-worker memory budget (GiB) used to derive the loky cap from the
# pod's available memory. 4 GiB accommodates the heaviest baseline
# cells (XGBoost / CatBoost / LightGBM with 1000 trees on n=2500, SVR
# RBF kernel matrices, Optuna study state) plus the Python interpreter
# and ML library imports (~500-800 MiB resident at steady state).
# Override per-run with the ``KORE_PER_WORKER_GB`` env var (e.g.
# ``KORE_PER_WORKER_GB=2.0`` for lighter rosters) without editing source.
_PER_WORKER_GB = 4.0
# Per-cell RSS soft cap (MiB). When a cell's peak RSS exceeds this
# value the sampler thread sets ``rss_advisory_exceeded=True`` on the
# row, prints one ``[cell-rss-advisory]`` line, and calls
# ``_thread.interrupt_main()`` to raise ``KeyboardInterrupt`` on the
# cell's main thread. The cell catches the interrupt via its
# ``except BaseException`` branch, falls back to the constant-predictor
# row, and the worker continues to its next cell. The signal stays
# in-process so loky never sees a worker death and never raises
# ``TerminatedWorkerError``; a prior revision sent ``SIGTERM`` to the
# worker and triggered exactly that cascade, sentinelling every cell
# queued behind the offender. Default 8 GiB; override via
# ``KORE_CELL_RSS_ADVISORY_MB`` (legacy ``KORE_CELL_RSS_CAP_MB`` still
# honored).
_CELL_RSS_ADVISORY_MB = 8000.0
# Fraction of CPU affinity to use. 0.9 leaves headroom for the OS,
# Spark driver process, log shipping, and joblib's own coordinator
# thread; on a 360-vCPU Databricks driver this gives 324 workers if
# memory is not the binding term.
_LOCAL_WORKER_CPU_FRAC = 0.9
# Cgroup memory limits this large mean "unlimited"; ignore.
_CGROUP_UNLIMITED_BYTES = 1 << 62


def _per_worker_gb():
    """Return the per-worker memory budget honoring the env override."""
    raw = os.environ.get("KORE_PER_WORKER_GB")
    if raw is None:
        return _PER_WORKER_GB
    try:
        val = float(raw)
    except ValueError:
        return _PER_WORKER_GB
    return val if val > 0 else _PER_WORKER_GB


def _cell_rss_advisory_mb() -> float:
    """Per-cell RSS advisory threshold (MiB).

    Reads ``KORE_CELL_RSS_ADVISORY_MB`` first, then the legacy
    ``KORE_CELL_RSS_CAP_MB`` for backward compatibility. Returns the
    module default (8000 MiB) when neither is set or both are
    malformed. The threshold is advisory only; see ``_sample_peak_rss``.
    """
    for key in ("KORE_CELL_RSS_ADVISORY_MB", "KORE_CELL_RSS_CAP_MB"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if val > 0:
            return val
    return _CELL_RSS_ADVISORY_MB


def _max_workers_env():
    """Return the user's hard ceiling from ``KORE_MAX_WORKERS``, if any."""
    raw = os.environ.get("KORE_MAX_WORKERS")
    if raw is None:
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return val if val >= 1 else None


def _read_int(path):
    """Read a single integer from a /sys or /proc file. Returns ``None`` on error."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _cgroup_memory_gb():
    """Return ``(limit_gb, free_gb)`` from the pod's cgroup, or ``(None, None)``.

    Reads cgroup v2 (``/sys/fs/cgroup/memory.max`` and ``memory.current``)
    first, falling back to cgroup v1
    (``/sys/fs/cgroup/memory/memory.limit_in_bytes`` and ``memory.usage_in_bytes``).
    Returns ``(None, None)`` if the cgroup is uncapped (limit > 4 EiB) or
    if neither cgroup hierarchy is mounted (e.g. macOS, non-containerized
    Linux). The "free" term is ``limit - current`` so the driver process
    itself, log shippers, and the Spark JVM are excluded from the cap.
    """
    # cgroup v2
    raw = None
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
    except (FileNotFoundError, OSError):
        raw = None
    if raw is not None:
        if raw == "max":
            return None, None
        try:
            limit = int(raw)
        except ValueError:
            limit = None
        if limit is not None and limit < _CGROUP_UNLIMITED_BYTES:
            current = _read_int("/sys/fs/cgroup/memory.current") or 0
            free = max(0, limit - current)
            return limit / (1024 ** 3), free / (1024 ** 3)
    # cgroup v1
    limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None and 0 < limit < _CGROUP_UNLIMITED_BYTES:
        current = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes") or 0
        free = max(0, limit - current)
        return limit / (1024 ** 3), free / (1024 ** 3)
    return None, None


def _meminfo_available_gb():
    """Return ``MemAvailable`` from ``/proc/meminfo`` in GiB, or ``None``.

    On Kubernetes pods this reports the *node's* free memory, not the
    pod's cgroup limit, so it is only used as a fallback when the cgroup
    hierarchy is uncapped or unreadable.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _available_memory_gb():
    """Return the pod-aware available memory in GiB, or ``None`` if unknown.

    Prefers the cgroup-derived ``limit - current`` so a containerized
    Databricks driver pod reports its real allocation rather than the
    underlying Kubernetes node's free memory. Falls back to
    ``MemAvailable`` when the cgroup is uncapped or unreadable, and
    returns ``None`` outside Linux (the caller then drops the memory
    term and uses the CPU fraction alone).
    """
    _limit_gb, free_gb = _cgroup_memory_gb()
    if free_gb is not None:
        return free_gb
    return _meminfo_available_gb()


def _local_worker_cap():
    """Return the loky worker count for the real_data driver.

    Sized as ``min(_LOCAL_WORKER_CPU_FRAC * cpu_affinity,
    available_memory_gb / per_worker_gb)``, then clamped by the
    optional ``KORE_MAX_WORKERS`` env-var ceiling. Both terms are
    cgroup-aware on Linux so a containerized Databricks driver pod
    reports its real allocation, not the host's. There is no fixed
    integer ceiling beyond the user's env-var override. On macOS the
    cgroup hierarchy is absent and the CPU fraction is the only
    binding constraint.
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing
        cores = multiprocessing.cpu_count()
    cpu_cap = max(1, int(cores * _LOCAL_WORKER_CPU_FRAC))
    mem_gb = _available_memory_gb()
    cap = cpu_cap
    if mem_gb is not None:
        mem_cap = max(1, int(mem_gb / _per_worker_gb()))
        cap = min(cap, mem_cap)
    user_cap = _max_workers_env()
    if user_cap is not None:
        cap = min(cap, user_cap)
    return max(1, cap)


def _local_worker_cap_explain():
    """Return a one-line diagnostic for the run banner.

    Surfaces every term that fed into the cap so the user can see
    whether they are CPU-bound, memory-bound (cgroup or MemInfo), or
    capped by ``KORE_MAX_WORKERS``, and tune accordingly.
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing
        cores = multiprocessing.cpu_count()
    cpu_cap = max(1, int(cores * _LOCAL_WORKER_CPU_FRAC))
    per_gb = _per_worker_gb()
    cgroup_limit_gb, cgroup_free_gb = _cgroup_memory_gb()
    meminfo_gb = _meminfo_available_gb()
    user_cap = _max_workers_env()

    if cgroup_free_gb is not None:
        mem_gb = cgroup_free_gb
        mem_src = f"cgroup={cgroup_limit_gb:.0f} GiB free={cgroup_free_gb:.0f} GiB"
    elif meminfo_gb is not None:
        mem_gb = meminfo_gb
        mem_src = f"meminfo={meminfo_gb:.0f} GiB (no cgroup cap)"
    else:
        mem_gb = None
        mem_src = "memory unknown"

    parts = [f"cpu={cpu_cap}/{cores}"]
    if mem_gb is None:
        parts.append(mem_src)
        cap = cpu_cap
        bound = "cpu"
    else:
        mem_cap = max(1, int(mem_gb / per_gb))
        parts.append(f"mem={mem_cap} ({mem_src} / {per_gb:.1f} GiB-per-worker)")
        cap = min(cpu_cap, mem_cap)
        bound = "cpu" if cpu_cap <= mem_cap else "memory"
    if user_cap is not None:
        if user_cap < cap:
            cap = user_cap
            bound = "KORE_MAX_WORKERS"
        parts.append(f"KORE_MAX_WORKERS={user_cap}")
    parts.append(f"-> {cap} ({bound}-bound)")
    return " ".join(parts)


def _staged_njobs_from_env():
    """Resolve per-stage worker counts for ``_dispatch_real_data_staged``.

    Defaults (heavy=16, medium=64, light=cap) trade memory headroom for
    throughput as per-cell footprint shrinks. Each stage is capped by
    ``_local_worker_cap()`` so the env-var overrides cannot exceed the
    pod's CPU + memory budget. Returns ``(heavy, medium, light)``.
    """
    cap = _local_worker_cap()
    heavy = int(os.environ.get("KORE_NJOBS_HEAVY", min(cap, 16)))
    medium = int(os.environ.get("KORE_NJOBS_MEDIUM", min(cap, 64)))
    light = int(os.environ.get("KORE_NJOBS_LIGHT", cap))
    return (
        max(1, min(heavy, cap)),
        max(1, min(medium, cap)),
        max(1, min(light, cap)),
    )


def _predict(choice, X):
    if choice["structure"] == "additive":
        return pred_additive(choice["model"], X)
    return pred_pairwise_sparse(choice["model"], X)


def _strhash(s):
    """Deterministic string hash (Python's hash() is randomised since 3.3)."""
    h = 5381
    for c in s:
        h = ((h << 5) + h + ord(c)) & 0x7FFFFFFF
    return h


def _dseed(*parts):
    """Deterministic data-seed from MASTER_SEED and arbitrary integer parts."""
    h = MASTER_SEED
    for p in parts:
        h = h * 1000003 + int(p)
    return h % (2**31)


def _smoothness_order(degree: int) -> int:
    """Bias exponent order for degree-k splines under classical approximation."""
    return degree + 1


# =====================================================================
# Experiment 1 -- Law Collapse  (3 seeds, parallel)
# =====================================================================

def _law_one(family, d, density, seed):
    if family == "additive":
        func = make_additive_target(d, seed=seed + 10)
        pairs = [(j, j + 1) for j in range(0, d - 1, 2)]
        n_eff = d
    else:
        func, pairs = make_sparse_pairwise_target(d, seed=seed + 20)
        n_eff = len(pairs)

    n_train = density * n_eff
    ds = _dseed(1, family == "pairwise", d, density, seed)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(
        func, n_train, 2000, d, noise_frac=0.03, seed=ds)

    t0 = time.perf_counter()
    kore = auto_kore_intrinsic(X_tr, y_tr, pairs=pairs,
                               max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
    wall = time.perf_counter() - t0

    return {"family": family, "d": d, "density": density,
            "n_eff": n_eff, "n_train": n_train, "seed": seed,
            "structure": kore["structure"], "G": kore["G_star"],
            "fits": kore["fits_total"], "time": wall,
            "rmse": rmse(y_te, _predict(kore, X_te))}


def run_law_collapse():
    print("=" * 78)
    print("EXP 1  Law-collapse  (5 seeds, dense grid)")
    print("=" * 78)
    tasks = []
    ADD_DENS  = [30, 45, 60, 90, 120, 180, 240, 360, 480, 720]
    PAIR_DENS = [60, 90, 120, 180, 240, 360, 480, 720]
    for d in [10, 20, 40, 80]:
        for rho in ADD_DENS:
            if rho * d > 60000:
                continue
            for s in range(5):
                tasks.append(("additive", d, rho, s))
    for d in [10, 20, 40, 80]:
        n_pairs = d // 2
        for rho in PAIR_DENS:
            if rho * n_pairs > 60000:
                continue
            for s in range(5):
                tasks.append(("pairwise", d, rho, s))

    rows = Parallel(n_jobs=min(len(tasks), N_JOBS), verbose=5)(
        delayed(_law_one)(*t) for t in tasks)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "law_collapse.csv", index=False)
    print(f"  -> law_collapse.csv  ({len(df)} rows)\n")
    return df


# =====================================================================
# Experiment 2 -- Frontier  (5 seeds, 3 methods, parallel)
# =====================================================================

def _frontier_one(family, d, density, seed, degree=3):
    q = _smoothness_order(degree)
    if family == "additive":
        func = make_additive_target(d, seed=1)
        pairs = [(j, j + 1) for j in range(0, d - 1, 2)]
        n_train = density * d
    else:
        func, pairs = make_sparse_pairwise_target(d, seed=2)
        n_train = density * len(pairs)

    ds = _dseed(2, family == "pairwise", d, density, seed)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(
        func, n_train, 2000, d, noise_frac=0.03, seed=ds)

    t0 = time.perf_counter()
    kore = auto_kore_intrinsic(X_tr, y_tr, pairs=pairs,
                               q=q, degree=degree,
                               max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
    t_kore = time.perf_counter() - t0
    kore_rmse = rmse(y_te, _predict(kore, X_te))

    t0 = time.perf_counter()
    cv = cv_auto_intrinsic(X_tr, y_tr, grid_add=GRID_ADD, grid_pair=GRID_PAIR,
                           pairs=pairs, degree=degree, K=CV_FOLDS, seed=seed)
    t_cv = time.perf_counter() - t0
    cv_rmse = rmse(y_te, _predict(cv, X_te))

    t0 = time.perf_counter()
    crit = criteria_auto_intrinsic(
        X_tr, y_tr, grid_add=GRID_ADD, grid_pair=GRID_PAIR, pairs=pairs,
        degree=degree, criterion_names=("gcv", "aic", "bic", "cp")
    )
    t_crit = time.perf_counter() - t0
    gcv = crit["gcv"]
    aic = crit["aic"]
    bic = crit["bic"]
    cp = crit["cp"]
    gcv_rmse = rmse(y_te, _predict(gcv, X_te))
    aic_rmse = rmse(y_te, _predict(aic, X_te))
    bic_rmse = rmse(y_te, _predict(bic, X_te))
    cp_rmse = rmse(y_te, _predict(cp, X_te))

    return {
        "family": family, "d": d, "density": density,
        "n_train": n_train, "seed": seed,
        "kore_structure": kore["structure"], "kore_G": kore["G_star"],
        "kore_fits": kore["fits_total"], "kore_time": t_kore,
        "kore_rmse": kore_rmse,
        "cv_structure": cv["structure"], "cv_G": cv["G_star"],
        "cv_fits": cv["fits"], "cv_time": t_cv, "cv_rmse": cv_rmse,
        "gcv_structure": gcv["structure"], "gcv_G": gcv["G_star"],
        "gcv_fits": gcv["fits_total"], "gcv_time": t_crit,
        "gcv_rmse": gcv_rmse,
        "aic_structure": aic["structure"], "aic_G": aic["G_star"],
        "aic_fits": aic["fits_total"], "aic_time": t_crit, "aic_rmse": aic_rmse,
        "bic_structure": bic["structure"], "bic_G": bic["G_star"],
        "bic_fits": bic["fits_total"], "bic_time": t_crit, "bic_rmse": bic_rmse,
        "cp_structure": cp["structure"], "cp_G": cp["G_star"],
        "cp_fits": cp["fits_total"], "cp_time": t_crit, "cp_rmse": cp_rmse,
        "structure_correct_kore": int(kore["structure"] == family),
        "structure_correct_cv":   int(cv["structure"]   == family),
        "structure_correct_gcv":  int(gcv["structure"]  == family),
        "structure_correct_aic":  int(aic["structure"]  == family),
        "structure_correct_bic":  int(bic["structure"]  == family),
        "structure_correct_cp":   int(cp["structure"]   == family),
    }


def run_frontier():
    print("=" * 78)
    print("EXP 2  Frontier  (5 seeds, 3 methods)")
    print("=" * 78)
    configs = [("additive",10,120),("additive",20,120),("additive",40,120),
               ("pairwise",10,240),("pairwise",20,240),("pairwise",40,240)]
    tasks = [(f, d, rho, s) for f, d, rho in configs for s in range(5)]

    rows = Parallel(n_jobs=min(len(tasks), N_JOBS), verbose=5)(
        delayed(_frontier_one)(*t) for t in tasks)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "frontier_replicates.csv", index=False)

    summary = []
    for (fam, d, dens), g in df.groupby(["family","d","density"], sort=False):
        summary.append({
            "family": fam, "d": d, "density": dens,
            "n_train": int(g["n_train"].iloc[0]),
            "kore_structure": g["kore_structure"].mode().iloc[0],
            "kore_G_mean": g["kore_G"].mean(),
            "kore_rmse_mean": g["kore_rmse"].mean(),
            "kore_rmse_std": g["kore_rmse"].std(ddof=1),
            "kore_fits_mean": g["kore_fits"].mean(),
            "kore_time_mean": g["kore_time"].mean(),
            "cv_structure": g["cv_structure"].mode().iloc[0],
            "cv_G_mean": g["cv_G"].mean(),
            "cv_rmse_mean": g["cv_rmse"].mean(),
            "cv_rmse_std": g["cv_rmse"].std(ddof=1),
            "cv_fits_mean": g["cv_fits"].mean(),
            "cv_time_mean": g["cv_time"].mean(),
            "gcv_structure": g["gcv_structure"].mode().iloc[0],
            "gcv_G_mean": g["gcv_G"].mean(),
            "gcv_rmse_mean": g["gcv_rmse"].mean(),
            "gcv_rmse_std": g["gcv_rmse"].std(ddof=1),
            "gcv_fits_mean": g["gcv_fits"].mean(),
            "gcv_time_mean": g["gcv_time"].mean(),
            "aic_structure": g["aic_structure"].mode().iloc[0],
            "aic_G_mean": g["aic_G"].mean(),
            "aic_rmse_mean": g["aic_rmse"].mean(),
            "aic_rmse_std": g["aic_rmse"].std(ddof=1),
            "aic_fits_mean": g["aic_fits"].mean(),
            "aic_time_mean": g["aic_time"].mean(),
            "bic_structure": g["bic_structure"].mode().iloc[0],
            "bic_G_mean": g["bic_G"].mean(),
            "bic_rmse_mean": g["bic_rmse"].mean(),
            "bic_rmse_std": g["bic_rmse"].std(ddof=1),
            "bic_fits_mean": g["bic_fits"].mean(),
            "bic_time_mean": g["bic_time"].mean(),
            "cp_structure": g["cp_structure"].mode().iloc[0],
            "cp_G_mean": g["cp_G"].mean(),
            "cp_rmse_mean": g["cp_rmse"].mean(),
            "cp_rmse_std": g["cp_rmse"].std(ddof=1),
            "cp_fits_mean": g["cp_fits"].mean(),
            "cp_time_mean": g["cp_time"].mean(),
            "rmse_ratio_kore_cv": g["kore_rmse"].mean()/g["cv_rmse"].mean(),
            "rmse_ratio_gcv_cv":  g["gcv_rmse"].mean()/g["cv_rmse"].mean(),
            "rmse_ratio_aic_cv":  g["aic_rmse"].mean()/g["cv_rmse"].mean(),
            "rmse_ratio_bic_cv":  g["bic_rmse"].mean()/g["cv_rmse"].mean(),
            "rmse_ratio_cp_cv":   g["cp_rmse"].mean()/g["cv_rmse"].mean(),
            "fit_speedup_cv_kore": g["cv_fits"].mean()/g["kore_fits"].mean(),
            "fit_speedup_cv_gcv":  g["cv_fits"].mean()/g["gcv_fits"].mean(),
            "fit_speedup_cv_aic":  g["cv_fits"].mean()/g["aic_fits"].mean(),
            "fit_speedup_cv_bic":  g["cv_fits"].mean()/g["bic_fits"].mean(),
            "fit_speedup_cv_cp":   g["cv_fits"].mean()/g["cp_fits"].mean(),
        })
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "frontier_summary.csv", index=False)
    method_rows = []
    for method in ["kore", "gcv", "aic", "bic", "cp"]:
        method_rows.append({
            "method": method.upper() if method != "cp" else "Cp",
            "gm_rmse_ratio_vs_cv": float(np.exp(np.log(sm[f"rmse_ratio_{method}_cv"]).mean())),
            "gm_fit_speedup_vs_cv": float(np.exp(np.log(sm[f"fit_speedup_cv_{method}"]).mean())),
        })
    pd.DataFrame(method_rows).to_csv(OUT / "frontier_method_summary.csv", index=False)
    print(f"  -> frontier_replicates.csv  ({len(df)} rows)")
    print(f"  -> frontier_summary.csv     ({len(summary)} rows)\n")
    return df


# =====================================================================
# Experiment 3 -- Benchmarks  (5 seeds, parallel)
# =====================================================================

def _build_benchmarks():
    def _eq(name, d, n, func, ranges, structure, pairs=None):
        return {"name": name, "d": d, "n": n, "func": func,
                "ranges": ranges, "structure": structure,
                "known_pairs": pairs or []}
    eqs = [
        _eq("Nguyen-1",1,500,
            lambda X: X[:,0]**3+X[:,0]**2+X[:,0], [(-1,1)],"additive"),
        _eq("Nguyen-4",1,500,
            lambda X: X[:,0]**6+X[:,0]**5+X[:,0]**4+X[:,0]**3+X[:,0]**2+X[:,0],
            [(-1,1)],"additive"),
        _eq("Nguyen-5",1,500,
            lambda X: np.sin(X[:,0]**2)*np.cos(X[:,0])-1,
            [(-1,1)],"additive"),
        _eq("Nguyen-7",1,500,
            lambda X: np.log(X[:,0]+1)+np.log(X[:,0]**2+1),
            [(0,2)],"additive"),
        _eq("Nguyen-9 (2D add)",2,1000,
            lambda X: np.sin(X[:,0])+np.sin(X[:,1]**2),
            [(0,1),(0,1)],"additive"),
        _eq("Nguyen-10 (2D int)",2,1000,
            lambda X: 2*np.sin(X[:,0])*np.cos(X[:,1]),
            [(0,1),(0,1)],"pairwise",[(0,1)]),
        _eq("Oscillator",1,500,
            lambda X: np.exp(-2.0*X[:,0])*np.cos(8*np.pi*X[:,0]),
            [(0,1)],"additive"),
    ]
    f,d,p = make_franke();       eqs.append(_eq("Franke (2D)",d,1000,f,[(0,1)]*d,"pairwise",p))
    f,d,p = make_friedman1();    eqs.append(_eq("Friedman-1 (5D)",d,2000,f,[(0,1)]*d,"pairwise",p))
    f,d,p = make_friedman2();    eqs.append(_eq("Friedman-2 (4D)",d,2000,f,[(0.1,1)]*d,"pairwise",p))
    f,act  = make_sparse_additive_highdim(20,5,seed=42)
    eqs.append(_eq("SparseAdd-20D",20,3000,f,[(0,1)]*20,"additive"))
    f,pa   = make_sparse_pairwise_target(10,seed=7)
    eqs.append(_eq("SparsePair-10D",10,2000,f,[(0,1)]*10,"pairwise",pa))
    return eqs

BENCHMARKS = _build_benchmarks()
MAIN_BENCHMARKS = [
    "Nguyen-1", "Nguyen-4", "Nguyen-5", "Nguyen-7", "Nguyen-9 (2D add)",
    "Nguyen-10 (2D int)", "Friedman-1 (5D)", "SparseAdd-20D", "SparsePair-10D",
]


def _bench_one(eq, seed, degree=3):
    q = _smoothness_order(degree)
    d, n_train = eq["d"], eq["n"]
    pairs = eq["known_pairs"] or [(j,j+1) for j in range(0,d-1,2)]
    if d == 1: pairs = []

    ds = _dseed(3, _strhash(eq["name"]), seed)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(
        eq["func"], n_train, 3000, d, noise_frac=0.01, seed=ds,
        ranges=eq["ranges"])

    # KORE oracle
    if d == 1:
        kr = refine_kore_additive(X_tr, y_tr, q=q, degree=degree, max_G=MAX_G_ADD)
        kr["fits_total"] = kr["fits"]
        kp = pred_additive(kr["model"], X_te)
    else:
        kr = auto_kore_intrinsic(X_tr, y_tr, pairs=pairs,
                                 q=q, degree=degree,
                                 max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
        kp = _predict(kr, X_te)

    # KORE discovered
    if d >= 2:
        dr = auto_kore_discover(X_tr, y_tr, q=q, degree=degree, max_G_add=MAX_G_ADD,
                                max_G_pair=MAX_G_PAIR, frac_threshold=0.01)
        dp = _predict(dr, X_te)
        disc_rmse = rmse(y_te, dp); n_disc = dr.get("n_discovered", 0)
    else:
        disc_rmse = rmse(y_te, kp); n_disc = 0

    # CV
    if d == 1:
        cv = cv_spline_1d(X_tr[:,0], y_tr, GRID_ADD, degree=degree, K=CV_FOLDS, seed=seed)
        cp = pred_spline_1d(cv["model"], X_te[:,0])
    else:
        cv = cv_auto_intrinsic(X_tr, y_tr, grid_add=GRID_ADD,
                               grid_pair=GRID_PAIR, pairs=pairs,
                               degree=degree, K=CV_FOLDS, seed=seed)
        cp = _predict(cv, X_te)

    # Classical full-grid criteria
    if d == 1:
        crit = criteria_additive(
            X_tr, y_tr, GRID_ADD, degree=degree,
            criterion_names=("gcv", "aic", "bic", "cp")
        )
    else:
        crit = criteria_auto_intrinsic(
            X_tr, y_tr, grid_add=GRID_ADD, grid_pair=GRID_PAIR, pairs=pairs,
            degree=degree, criterion_names=("gcv", "aic", "bic", "cp")
        )
    gv = crit["gcv"]
    av = crit["aic"]
    bv = crit["bic"]
    mp = crit["cp"]
    gp = pred_additive(gv["model"], X_te) if d == 1 else _predict(gv, X_te)
    ap = pred_additive(av["model"], X_te) if d == 1 else _predict(av, X_te)
    bp = pred_additive(bv["model"], X_te) if d == 1 else _predict(bv, X_te)
    mpred = pred_additive(mp["model"], X_te) if d == 1 else _predict(mp, X_te)

    return {"equation": eq["name"], "d": d, "n_train": n_train,
            "true_structure": eq["structure"], "seed": seed,
            "kore_rmse": rmse(y_te,kp),
            "kore_fits": kr.get("fits_total",kr["fits"]),
            "kore_G": kr["G_star"],
            "disc_rmse": disc_rmse, "n_discovered": n_disc,
            "cv_rmse": rmse(y_te,cp), "cv_fits": cv["fits"],
            "cv_G": cv["G_star"],
            "gcv_rmse": rmse(y_te,gp),
            "gcv_fits": gv.get("fits_total",gv["fits"]),
            "aic_rmse": rmse(y_te, ap), "aic_fits": av.get("fits_total", av["fits"]),
            "bic_rmse": rmse(y_te, bp), "bic_fits": bv.get("fits_total", bv["fits"]),
            "cp_rmse": rmse(y_te, mpred), "cp_fits": mp.get("fits_total", mp["fits"])}


def run_benchmark():
    print("=" * 78)
    print("EXP 3  Benchmarks  (5 seeds)")
    print("=" * 78)
    tasks = [(eq, s) for eq in BENCHMARKS for s in range(5)]
    rows = Parallel(n_jobs=min(len(tasks), N_JOBS), verbose=5)(
        delayed(_bench_one)(*t) for t in tasks)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "benchmark.csv", index=False)
    sm = (df.groupby("equation", sort=False)
          .agg(d=("d","first"), n_train=("n_train","first"),
               kore_rmse_mean=("kore_rmse","mean"),
               kore_rmse_std=("kore_rmse","std"),
               kore_fits=("kore_fits","mean"),
               disc_rmse_mean=("disc_rmse","mean"),
               cv_rmse_mean=("cv_rmse","mean"),
               cv_rmse_std=("cv_rmse","std"),
               cv_fits=("cv_fits","mean"),
               gcv_rmse_mean=("gcv_rmse","mean"),
               gcv_fits=("gcv_fits","mean"),
               aic_rmse_mean=("aic_rmse","mean"),
               aic_fits=("aic_fits","mean"),
               bic_rmse_mean=("bic_rmse","mean"),
               bic_fits=("bic_fits","mean"),
               cp_rmse_mean=("cp_rmse","mean"),
               cp_fits=("cp_fits","mean")).reset_index())
    sm["ratio_kore_cv"] = sm["kore_rmse_mean"] / sm["cv_rmse_mean"]
    sm["ratio_gcv_cv"] = sm["gcv_rmse_mean"] / sm["cv_rmse_mean"]
    sm["ratio_aic_cv"] = sm["aic_rmse_mean"] / sm["cv_rmse_mean"]
    sm["ratio_bic_cv"] = sm["bic_rmse_mean"] / sm["cv_rmse_mean"]
    sm["ratio_cp_cv"] = sm["cp_rmse_mean"] / sm["cv_rmse_mean"]
    sm["fit_speedup"]   = sm["cv_fits"] / sm["kore_fits"]
    sm["fit_speedup_gcv"] = sm["cv_fits"] / sm["gcv_fits"]
    sm["fit_speedup_aic"] = sm["cv_fits"] / sm["aic_fits"]
    sm["fit_speedup_bic"] = sm["cv_fits"] / sm["bic_fits"]
    sm["fit_speedup_cp"] = sm["cv_fits"] / sm["cp_fits"]
    sm.to_csv(OUT / "benchmark_summary.csv", index=False)
    method_specs = [
        ("KORE", "ratio_kore_cv", "fit_speedup"),
        ("GCV", "ratio_gcv_cv", "fit_speedup_gcv"),
        ("AIC", "ratio_aic_cv", "fit_speedup_aic"),
        ("BIC", "ratio_bic_cv", "fit_speedup_bic"),
        ("Cp", "ratio_cp_cv", "fit_speedup_cp"),
    ]
    method_rows = []
    for method, ratio_col, speed_col in method_specs:
        method_rows.append({
            "method": method,
            "gm_rmse_ratio_vs_cv": float(np.exp(np.log(sm[ratio_col]).mean())),
            "gm_fit_speedup_vs_cv": float(np.exp(np.log(sm[speed_col]).mean())),
        })
    pd.DataFrame(method_rows).to_csv(OUT / "benchmark_method_summary.csv", index=False)

    sm_main = sm[sm["equation"].isin(MAIN_BENCHMARKS)].copy()
    method_rows_main = []
    for method, ratio_col, speed_col in method_specs:
        method_rows_main.append({
            "method": method,
            "gm_rmse_ratio_vs_cv": float(np.exp(np.log(sm_main[ratio_col]).mean())),
            "gm_fit_speedup_vs_cv": float(np.exp(np.log(sm_main[speed_col]).mean())),
        })
    pd.DataFrame(method_rows_main).to_csv(OUT / "benchmark_method_summary_main.csv", index=False)
    print(f"  -> benchmark.csv          ({len(df)} rows)")
    print(f"  -> benchmark_summary.csv  ({len(sm)} rows)\n")
    return df


# =====================================================================
# Experiment 4 -- Discovery  (5 seeds, parallel)
# =====================================================================

def _disc_one(d, n_per_pair, seed):
    """One (d, density, seed) -- tests all thresholds internally."""
    func, true_pairs = make_sparse_pairwise_target(d, seed=seed + 30)
    n_eff = len(true_pairs)
    n_train = n_per_pair * n_eff
    ds = _dseed(4, d, n_per_pair, seed)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(
        func, n_train, 2000, d, noise_frac=0.03, seed=ds)

    true_set = set(true_pairs)
    n_true   = len(true_set)

    # oracle once
    ko = auto_kore_intrinsic(X_tr, y_tr, pairs=true_pairs,
                             max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
    oracle_rmse = rmse(y_te, _predict(ko, X_te))

    out = []
    for thr in [0.005, 0.01, 0.02, 0.05]:
        disc = screen_pairs_residual(X_tr, y_tr, frac_threshold=thr,
                                     max_pairs=d)
        disc_set = set(disc)
        tp = len(disc_set & true_set)
        fp = len(disc_set - true_set)
        fn = len(true_set - disc_set)
        pr = tp / max(tp+fp, 1)
        rc = tp / max(n_true, 1)
        f1 = 2*pr*rc / max(pr+rc, 1e-10)

        if disc:
            kd = auto_kore_intrinsic(X_tr, y_tr, pairs=disc,
                                     max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
            dp = _predict(kd, X_te)
        else:
            kd = refine_kore_additive(X_tr, y_tr, max_G=MAX_G_ADD)
            dp = pred_additive(kd["model"], X_te)

        out.append({"d": d, "n_per_pair": n_per_pair, "seed": seed,
                    "threshold": thr, "n_true": n_true,
                    "n_discovered": len(disc),
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": pr, "recall": rc, "f1": f1,
                    "disc_rmse": rmse(y_te, dp),
                    "oracle_rmse": oracle_rmse,
                    "rmse_ratio": rmse(y_te, dp) / max(oracle_rmse, 1e-10)})
    return out


def run_discovery():
    print("=" * 78)
    print("EXP 4  Discovery  (5 seeds)")
    print("=" * 78)
    tasks = [(d, npp, s)
             for d in [10,20,40]
             for npp in [120,240,480]
             for s in range(5)]
    nested = Parallel(n_jobs=min(len(tasks), N_JOBS), verbose=5)(
        delayed(_disc_one)(*t) for t in tasks)
    rows = [r for block in nested for r in block]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "discovery.csv", index=False)
    print(f"  -> discovery.csv  ({len(df)} rows)\n")
    return df


# =====================================================================
# Experiment 5 -- Robustness  (5 seeds, parallel)
# =====================================================================

def _robust_scenario(scenario, d, seed):
    """One robustness scenario for one (d, seed) -- fully independent."""
    wrong_pairs = [(j, j+1) for j in range(0, d-1, 2)]
    n_train = 300 * d

    if scenario == "3-way":
        f3, _ = make_threeway_target(d, seed=seed+50)
        X,y,Xt,yt,_ = make_dataset(f3, n_train, 2000, d, 0.03,
                                    seed=_dseed(5,0,d,seed))
        kr = auto_kore_intrinsic(X, y, pairs=wrong_pairs,
                                 max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
        cv = cv_auto_intrinsic(X, y, grid_add=GRID_ADD, grid_pair=GRID_PAIR,
                               pairs=wrong_pairs, K=CV_FOLDS, seed=seed)
        return {"scenario":"3-way interactions","d":d,"seed":seed,
                "n_train":n_train,
                "kore_rmse":rmse(yt,_predict(kr,Xt)),
                "kore_G":kr["G_star"],"kore_structure":kr["structure"],
                "cv_rmse":rmse(yt,_predict(cv,Xt)),
                "cv_G":cv["G_star"],"cv_structure":cv["structure"]}

    elif scenario == "non-smooth":
        fns = make_nonsmooth_target(d, seed=seed+60)
        X,y,Xt,yt,_ = make_dataset(fns, n_train, 2000, d, 0.03,
                                    seed=_dseed(5,1,d,seed))
        kr = auto_kore_intrinsic(X, y, pairs=wrong_pairs,
                                 max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
        cv = cv_auto_intrinsic(X, y, grid_add=GRID_ADD, grid_pair=GRID_PAIR,
                               pairs=wrong_pairs, K=CV_FOLDS, seed=seed)
        return {"scenario":"non-smooth","d":d,"seed":seed,
                "n_train":n_train,
                "kore_rmse":rmse(yt,_predict(kr,Xt)),
                "kore_G":kr["G_star"],"kore_structure":kr["structure"],
                "cv_rmse":rmse(yt,_predict(cv,Xt)),
                "cv_G":cv["G_star"],"cv_structure":cv["structure"]}

    elif scenario == "misspecified":
        fms, tp, wp = make_misspecified_pairwise(d, seed=seed+70)
        n_ms = 240 * len(tp)
        X,y,Xt,yt,_ = make_dataset(fms, n_ms, 2000, d, 0.03,
                                    seed=_dseed(5,2,d,seed))
        kw = auto_kore_intrinsic(X, y, pairs=wp,
                                 max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
        ko = auto_kore_intrinsic(X, y, pairs=tp,
                                 max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
        kd = auto_kore_discover(X, y, max_G_add=MAX_G_ADD,
                                max_G_pair=MAX_G_PAIR, frac_threshold=0.01)
        return {"scenario":"misspecified graph","d":d,"seed":seed,
                "n_train":n_ms,
                "kore_rmse":rmse(yt,_predict(kw,Xt)),
                "kore_G":kw["G_star"],"kore_structure":kw["structure"],
                "cv_rmse":rmse(yt,_predict(ko,Xt)),
                "cv_G":ko["G_star"],"cv_structure":ko["structure"],
                "disc_rmse":rmse(yt,_predict(kd,Xt))}

    else:  # control
        fok, pok = make_sparse_pairwise_target(d, seed=seed+80)
        n_ok = 240 * len(pok)
        X,y,Xt,yt,_ = make_dataset(fok, n_ok, 2000, d, 0.03,
                                    seed=_dseed(5,3,d,seed))
        kr = auto_kore_intrinsic(X, y, pairs=pok,
                                 max_G_add=MAX_G_ADD, max_G_pair=MAX_G_PAIR)
        cv = cv_auto_intrinsic(X, y, grid_add=GRID_ADD, grid_pair=GRID_PAIR,
                               pairs=pok, K=CV_FOLDS, seed=seed)
        return {"scenario":"correct (control)","d":d,"seed":seed,
                "n_train":n_ok,
                "kore_rmse":rmse(yt,_predict(kr,Xt)),
                "kore_G":kr["G_star"],"kore_structure":kr["structure"],
                "cv_rmse":rmse(yt,_predict(cv,Xt)),
                "cv_G":cv["G_star"],"cv_structure":cv["structure"]}


def run_robustness():
    print("=" * 78)
    print("EXP 5  Robustness  (5 seeds, per-scenario parallel)")
    print("=" * 78)
    tasks = [(sc, d, s)
             for sc in ["3-way", "non-smooth", "misspecified", "control"]
             for d in [12, 24]
             for s in range(5)]
    rows = Parallel(n_jobs=min(len(tasks), N_JOBS), verbose=5)(
        delayed(_robust_scenario)(*t) for t in tasks)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "robustness.csv", index=False)
    print(f"  -> robustness.csv  ({len(df)} rows)\n")
    return df


# =====================================================================
# Experiment 6 -- Scaling  (sequential -- measures wall-clock)
# =====================================================================

def run_scaling():
    print("=" * 78)
    print("EXP 6  Scaling  (sequential -- timing)")
    print("=" * 78)
    rows = []
    for family in ["additive", "pairwise"]:
        for d in [10, 20, 40, 80]:
            for seed in range(3):
                if family == "additive":
                    func = make_additive_target(d, seed=seed)
                    pairs = [(j,j+1) for j in range(0,d-1,2)]
                    n_train = 120 * d
                else:
                    func, pairs = make_sparse_pairwise_target(d, seed=seed)
                    n_train = 240 * len(pairs)

                ds = _dseed(6, family == "pairwise", d, seed)
                X,y,Xt,yt,_ = make_dataset(func, n_train, 1000, d,
                                            noise_frac=0.03, seed=ds)
                t0 = time.perf_counter()
                kr = auto_kore_intrinsic(X, y, pairs=pairs,
                                         max_G_add=MAX_G_ADD,
                                         max_G_pair=MAX_G_PAIR)
                tk = time.perf_counter() - t0

                t0 = time.perf_counter()
                cv = cv_auto_intrinsic(X, y, grid_add=GRID_ADD,
                                       grid_pair=GRID_PAIR,
                                       pairs=pairs, K=CV_FOLDS, seed=seed)
                tc = time.perf_counter() - t0

                rows.append({"family":family,"d":d,"n_train":n_train,
                             "seed":seed,
                             "kore_time":tk,"kore_fits":kr["fits_total"],
                             "kore_rmse":rmse(yt,_predict(kr,Xt)),
                             "cv_time":tc,"cv_fits":cv["fits"],
                             "cv_rmse":rmse(yt,_predict(cv,Xt)),
                             "time_speedup":tc/max(tk,1e-6)})
                print(f"  {family:<8} d={d:>3} seed={seed}"
                      f"  KORE {tk:.2f}s  CV {tc:.2f}s"
                      f"  speedup={rows[-1]['time_speedup']:.1f}x")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scaling.csv", index=False)
    print(f"  -> scaling.csv  ({len(df)} rows)\n")
    return df


# =====================================================================
# Experiment 7 -- Noise Sweep / Applicability  (5 seeds, parallel)
# =====================================================================

def _null_loo_terms(y):
    """Per-sample LOO squared errors for the intercept-only model."""
    n = len(y)
    y_bar = float(np.mean(y))
    h = 1.0 / n
    return ((y - y_bar) / (1.0 - h)) ** 2


def _model_loo_terms(X, y, model, ridge=1e-8):
    """Per-sample LOO squared errors for an additive model (PRESS identity)."""
    from scipy.linalg import solve_triangular
    blocks = [spl.transform(X[:, [j]]) for j, spl in enumerate(model["spls"])]
    B = np.hstack(blocks)
    p = B.shape[1]
    BTB = B.T @ B + ridge * np.eye(p)
    L = np.linalg.cholesky(BTB)
    Z = solve_triangular(L, B.T, lower=True)
    h_diag = np.sum(Z ** 2, axis=0)
    residuals = y - B @ model["coef"]
    return (residuals / (1.0 - h_diag)) ** 2


def _noise_one(d, n_train, noise_frac, seed):
    func = make_additive_target(d, seed=seed)
    ds = _dseed(7, d, int(noise_frac * 1000), seed)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(
        func, n_train, 3000, d, noise_frac=noise_frac, seed=ds)

    kr = refine_kore_additive(X_tr, y_tr, max_G=MAX_G_ADD, radius=3)
    kore_rmse = rmse(y_te, pred_additive(kr["model"], X_te))

    null_terms = _null_loo_terms(y_tr)
    model_terms = _model_loo_terms(X_tr, y_tr, kr["model"])

    z = 1.96
    n = len(y_tr)
    diff = null_terms - model_terms
    signal_gain = float(np.mean(diff))
    signal_se = float(np.std(diff, ddof=1) / np.sqrt(n))
    signal_score = signal_gain / max(z * signal_se, 1e-12)

    G_star = kr["G_star"]
    lo_loo = kr["loo"]
    bracketed = True
    for dg in [-1, 1]:
        g_nb = G_star + dg
        if g_nb < 1 or g_nb > MAX_G_ADD:
            continue
        m_nb = fit_additive(X_tr, y_tr, g_nb)
        nb_loo = float(np.mean(_model_loo_terms(X_tr, y_tr, m_nb)))
        if nb_loo < lo_loo:
            bracketed = False

    null_rmse = rmse(y_te, np.full(len(y_te), np.mean(y_tr)))
    use_kore = signal_score > 1.0 and bracketed

    return {
        "seed": seed, "d": d, "n_train": n_train,
        "noise_frac": noise_frac,
        "signal_score": signal_score, "bracketed": int(bracketed),
        "use_kore": int(use_kore),
        "kore_G": G_star, "kore_fits": kr["fits"],
        "kore_rmse": kore_rmse, "null_rmse": null_rmse,
        "ratio_kore_null": kore_rmse / max(null_rmse, 1e-12),
    }


def run_noise_sweep():
    print("=" * 78)
    print("EXP 7  Noise sweep / applicability  (5 seeds)")
    print("=" * 78)
    NOISE_GRID = [0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60, 2.00]
    d, n_train = 10, 200
    tasks = [(d, n_train, nf, s)
             for nf in NOISE_GRID for s in range(5)]

    rows = Parallel(n_jobs=min(len(tasks), N_JOBS), verbose=5)(
        delayed(_noise_one)(*t) for t in tasks)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_sweep.csv", index=False)

    summary = (df.groupby("noise_frac")
               .agg(use_rate=("use_kore", "mean"),
                    signal_score=("signal_score", "mean"),
                    bracketed_rate=("bracketed", "mean"),
                    kore_fits=("kore_fits", "mean"),
                    ratio_kore_null=("ratio_kore_null", "mean"))
               .reset_index())
    summary.to_csv(OUT / "noise_sweep_summary.csv", index=False)
    print(f"  -> noise_sweep.csv          ({len(df)} rows)")
    print(f"  -> noise_sweep_summary.csv  ({len(summary)} rows)\n")
    return df


# =====================================================================
# Experiment 8 -- Degree ablation  (5 seeds, parallel)
# =====================================================================

def _degree_one(degree, density, seed, d=20):
    """One degree-ablation cell.

    Fits the closed-form additive plug-in at spline degree ``degree`` and
    effective density ``density = n / d`` in the interior-optimum regime
    (10% training noise, where the two-pilot plug-in tracks the population
    optimizer; cf. the plug-in consistency experiment), and records the
    continuous plug-in resolution ``G_dagger`` whose scaling the resolution
    law predicts. The grid is wide enough (``max_G = 40``) that the plug-in
    is never clipped on the swept density range.
    """
    q = _smoothness_order(degree)
    func = make_additive_target(d, seed=seed + 10)
    n_train = density * d
    ds = _dseed(8, degree, d, density, seed)
    X_tr, y_tr, X_te, y_te, _ = make_dataset(
        func, n_train, 2000, d, noise_frac=0.10, seed=ds
    )
    kore = refine_kore_additive(
        X_tr, y_tr, q=q, degree=degree, max_G=40, radius=3
    )
    return {
        "degree": degree,
        "smoothness_order": q,
        "d": d,
        "density": density,
        "n_train": n_train,
        "seed": seed,
        "G_dagger": float(kore["G_dagger"]),
        "G_star": kore["G_star"],
        "fits": kore["fits"],
    }


def run_degree_ablation():
    print("=" * 78)
    print("EXP 8  Degree ablation: plug-in resolution exponent  (5 seeds)")
    print("=" * 78)
    densities = [30, 60, 120, 240, 480, 960, 1920]
    tasks = [(degree, density, seed) for degree in [2, 3, 5]
             for density in densities for seed in range(5)]
    rows = Parallel(n_jobs=min(len(tasks), N_JOBS), verbose=5)(
        delayed(_degree_one)(*t) for t in tasks
    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "degree_ablation.csv", index=False)

    summary_rows = []
    for degree, g in df.groupby("degree", sort=True):
        agg = g.groupby("density", sort=True).agg(
            G_dagger_mean=("G_dagger", "mean"),
        ).reset_index()
        obs = float(np.polyfit(np.log(agg["density"]),
                               np.log(agg["G_dagger_mean"]), 1)[0])
        beta = int(g["smoothness_order"].iloc[0])
        theory = 1.0 / (2 * beta + 1)
        summary_rows.append({
            "degree": degree,
            "smoothness_order": beta,
            "predicted_g_exponent": theory,
            "observed_g_exponent": obs,
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "degree_ablation_summary.csv", index=False)
    print(f"  -> degree_ablation.csv          ({len(df)} rows)")
    print(f"  -> degree_ablation_summary.csv  ({len(summary_rows)} rows)\n")
    return df


# =====================================================================
# Experiment 9 -- Search-free plug-in consistency  (Theorem 2 verification)
# =====================================================================

def _plugin_one(d: int, n_train: int, seed: int, degree: int = 3,
                noise_frac: float = 0.10):
    """One (n, seed) pair: record A_hat, tau_hat, G_dagger, plus the
    excess proxy risk relative to a reference (oracle) resolution.

    Returns the per-run record. Aggregation across seeds and reference
    extraction live in :func:`run_plugin_consistency`.
    """
    func = make_additive_target(d, seed=seed)
    ds = _dseed(9, d, n_train, seed)
    X_tr, y_tr, X_te, y_te, sigma = make_dataset(
        func, n_train, 3000, d, noise_frac=noise_frac, seed=ds,
    )
    res = refine_kore_additive(X_tr, y_tr, q=degree + 1, degree=degree,
                               max_G=20, radius=3)
    return {
        "d": d,
        "n_train": int(n_train),
        "seed": int(seed),
        "noise_frac": float(noise_frac),
        "sigma": float(sigma),
        "tau_true": float(sigma) ** 2,
        "A_hat": float(res["A_hat"]),
        "tau_hat": float(res["tau_hat"]),
        "G_dagger": float(res["G_dagger"]),
        "G_star": int(res["G_star"]),
        "fits": int(res["fits"]),
        "rmse": float(rmse(y_te, pred_additive(res["model"], X_te))),
    }


def run_plugin_consistency():
    """Empirical verification of the search-free plug-in guarantee
    (Theorem 2 in the paper).

    Runs the additive plug-in selector at d=20 across a geometric
    sample-size ladder, compares the plug-in constants to a high-n
    anchor, and reports the ratios A_hat / A_anchor and tau_hat / tau_true
    plus the rounded-plug-in vs population-optimum gap.
    """
    print("=" * 78)
    print("EXP 9  Plug-in consistency  (additive d=20, 20 seeds, n in [300, 19200])")
    print("=" * 78)
    d = 20
    degree = 3
    beta = degree + 1
    n_grid = [300, 600, 1200, 2400, 4800, 9600, 19200]
    seeds = list(range(20))
    tasks = [(d, n, s, degree) for n in n_grid for s in seeds]

    rows = Parallel(n_jobs=min(len(tasks), N_JOBS), verbose=5)(
        delayed(_plugin_one)(*t) for t in tasks
    )
    df = pd.DataFrame(rows)

    n_anchor = max(n_grid)
    anchor = df[df["n_train"] == n_anchor]
    A_anchor = float(anchor["A_hat"].median())

    def p_add(G):
        return d * (G + degree) - (d - 1)

    def dp_add(_G):
        return float(d)

    def proxy_risk(A, tau, G, n):
        return A * G ** (-2 * beta) + tau * p_add(G) / n

    bullet_records = []
    for n in n_grid:
        sub = df[df["n_train"] == n]
        tau_target = float(sub["tau_true"].median())
        G_bullet = closed_form_g_star(
            A_anchor, tau_target, beta, n, p_add, dp_add,
            G_min=1.0, G_max=200.0,
        )
        bullet_records.append({"n_train": n, "G_bullet": float(G_bullet),
                               "A_anchor": A_anchor, "tau_target": tau_target})
    bullet_df = pd.DataFrame(bullet_records)
    df = df.merge(bullet_df, on="n_train", how="left")
    df["A_ratio"] = df["A_hat"] / df["A_anchor"]
    df["tau_ratio"] = df["tau_hat"] / df["tau_target"]
    df["delta_proxy"] = [
        proxy_risk(row["A_anchor"], row["tau_target"], row["G_dagger"], row["n_train"])
        - proxy_risk(row["A_anchor"], row["tau_target"], row["G_bullet"], row["n_train"])
        for _, row in df.iterrows()
    ]
    df["delta_proxy"] = np.maximum(df["delta_proxy"].values, 0.0)
    df.to_csv(OUT / "plugin_consistency.csv", index=False)

    summary = (df.groupby("n_train", sort=True)
               .agg(A_ratio_median=("A_ratio", "median"),
                    A_ratio_q25=("A_ratio", lambda s: float(s.quantile(0.25))),
                    A_ratio_q75=("A_ratio", lambda s: float(s.quantile(0.75))),
                    tau_ratio_median=("tau_ratio", "median"),
                    tau_ratio_q25=("tau_ratio", lambda s: float(s.quantile(0.25))),
                    tau_ratio_q75=("tau_ratio", lambda s: float(s.quantile(0.75))),
                    G_dagger_mean=("G_dagger", "mean"),
                    G_dagger_std=("G_dagger", "std"),
                    G_bullet=("G_bullet", "first"),
                    delta_proxy_mean=("delta_proxy", "mean"),
                    delta_proxy_std=("delta_proxy", "std"))
               .reset_index())
    summary.to_csv(OUT / "plugin_consistency_summary.csv", index=False)
    print(f"  -> plugin_consistency.csv          ({len(df)} rows)")
    print(f"  -> plugin_consistency_summary.csv  ({len(summary)} rows)\n")
    return df


# =====================================================================
# Main
# =====================================================================

# =====================================================================
# Experiment 10  Real-world benchmark
# =====================================================================


# Hard per-cell wall-clock backstop. Aligned with the AutoML Benchmark
# protocol (Gijsbers et al. 2024 JMLR), which kills any task that exceeds
# 2 x its soft search budget. The soft budget lives in baselines.py as
# ``OPTUNA_TIMEOUT_S`` (180 s); this backstop is set to exactly twice
# that value so the SIGALRM only fires for genuine hangs (a native-code
# fold that ignores the optuna timeout, or a pathological predict()
# call), not for tuning that simply ran out of search time.
_CELL_TIMEOUT_S = 360

# Per-method dispatch weight for longest-processing-time-first (LPT)
# scheduling. The unordered ``Parallel`` generator pulls cells in the
# order they appear in the iterable, so heavy cells must come first to
# keep all workers busy until the very end of the run. Weights are
# coarse rank-orderings of typical fit cost, not calibrated runtimes.
_METHOD_WEIGHT = {
    "catboost": 10.0,
    "xgboost": 5.0,
    "lightgbm": 4.0,
    "hist_gbm": 4.0,
    "mlp": 3.0,
    "svr_rbf": 3.0,
    "kernel_ridge_rbf": 3.0,
    "random_forest": 2.0,
    "extra_trees": 2.0,
    "pygam": 2.0,
    "knn": 1.0,
    "kore": 0.5,
    "cv_spline": 0.5,
    "gcv_spline": 0.5,
    "cp_spline": 0.5,
    "aic_spline": 0.5,
    "bic_spline": 0.5,
    "linear": 0.1,
    "ridge_cv": 0.1,
    "lasso_cv": 0.1,
    "elasticnet_cv": 0.1,
}

# Methods whose per-cell cost is dominated by d (distance matrices,
# kernel matrices, full-grid spline design). At ``d >= _HIGH_D_THRESHOLD``
# their footprint and runtime move them out of their nominal bucket;
# without this promotion, high-d KNN cells leak into the LIGHT stage and
# can OOM the 324-wide thread pool. The threshold sits between the
# CTR23 smooth-low-d cap of 30 and the lowest non-smooth-low-d dataset
# in the registry (``superconductivity: d=81``).
_HIGH_D_THRESHOLD = 64
_HIGH_D_PROMOTION = {
    "knn": 3.0,              # light (1.0) -> medium-top
    "svr_rbf": 4.5,          # medium (3.0) -> heavy
    "kernel_ridge_rbf": 4.5, # medium (3.0) -> heavy
}


def _method_weight(method_name: str, d: int = 0) -> float:
    """Dispatch weight for ``method_name`` on a cell of feature-dim ``d``.

    Equal to ``_METHOD_WEIGHT[method_name]`` for the default case; for
    methods in ``_HIGH_D_PROMOTION`` and ``d >= _HIGH_D_THRESHOLD``, the
    weight is replaced by the (heavier) promotion entry. The promotion
    affects only the staged bucketing in ``_dispatch_real_data_staged``;
    every cell still runs the same fit code path."""
    base = _METHOD_WEIGHT.get(method_name, 1.0)
    if d >= _HIGH_D_THRESHOLD and method_name in _HIGH_D_PROMOTION:
        return _HIGH_D_PROMOTION[method_name]
    return base


def _cell_cost(method_name: str, n: int, d: int) -> float:
    """Heuristic per-cell cost. Boosters and kernel methods scale
    superlinearly in n; tree methods scale ~ n d log n; linear methods
    are essentially constant. The exact form does not matter much for
    LPT; ordering by ``method_weight * (n / 1000) * sqrt(d)`` is enough
    to push the heaviest cells to the front of the dispatch queue."""
    w = _method_weight(method_name, d)
    return w * max(n, 1) / 1000.0 * float(np.sqrt(max(d, 1)))


def _set_worker_thread_cap(limit: int = 1) -> None:
    """Apply ``threadpoolctl.threadpool_limits(limit)`` in-process.

    Called once per cell from inside the loky worker. ``threadpool_limits``
    is idempotent so the per-cell cost is negligible (a dict lookup),
    and the limit persists across cells within the same worker because
    the underlying registration is stored on the threadpool objects."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limit)
    except Exception:  # noqa: BLE001
        pass


class _ProgressReporter:
    """Unified progress feedback for ``run_real_data``.

    Detects environment in this order:

    1. ipywidgets-capable Jupyter / Databricks notebook -> tqdm.notebook;
    2. attached TTY (``sys.stdout.isatty()``) -> tqdm.tqdm on stdout;
    3. otherwise (Databricks job cluster, redirected stdout) -> a
       periodic stdout line ``"[real_data] N/T cells (P%) | rate=R cells/s
       | elapsed E min | ETA F min"``.

    Exposes ``update(n=1)``, ``write(msg)``, ``close()`` matching the
    tqdm API so callers do not branch.
    """

    def __init__(self, total, initial, desc):
        self.total = int(total)
        self.n = int(initial)
        self.desc = desc
        self._t0 = time.perf_counter()
        self._last_print_t = self._t0
        self._last_print_n = self.n
        self._period_s = max(5.0, self.total / 400.0)
        self._period_n = max(1, self.total // 100)
        self._bar = self._make_tqdm()
        self._is_periodic = self._bar is None

    def _make_tqdm(self):
        if self._notebook_capable():
            try:
                from tqdm.notebook import tqdm as _tqdm
                return _tqdm(total=self.total, initial=self.n,
                             desc=self.desc, unit="cell",
                             smoothing=0.05, mininterval=0.5)
            except Exception:  # noqa: BLE001
                pass
        if sys.stdout.isatty():
            try:
                from tqdm import tqdm as _tqdm
                return _tqdm(total=self.total, initial=self.n,
                             desc=self.desc, unit="cell", file=sys.stdout,
                             smoothing=0.05, mininterval=0.5,
                             dynamic_ncols=True)
            except Exception:  # noqa: BLE001
                pass
        return None

    @staticmethod
    def _notebook_capable():
        try:
            from IPython import get_ipython
        except Exception:  # noqa: BLE001
            return False
        ip = get_ipython()
        if ip is None or ip.__class__.__name__ != "ZMQInteractiveShell":
            return False
        try:
            import ipywidgets  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def update(self, n=1):
        self.n += int(n)
        if self._bar is not None:
            self._bar.update(n)
            return
        now = time.perf_counter()
        if (now - self._last_print_t >= self._period_s
                or self.n - self._last_print_n >= self._period_n
                or self.n >= self.total):
            self._emit_periodic(now)

    def _emit_periodic(self, now):
        elapsed = now - self._t0
        rate = (self.n - self._last_print_n) / max(1e-9, now - self._last_print_t)
        pct = 100.0 * self.n / max(1, self.total)
        remaining = max(0, self.total - self.n)
        eta_s = remaining / rate if rate > 0 else float("inf")
        eta_str = f"{eta_s / 60:.1f} min" if eta_s != float("inf") else "n/a"
        print(f"[real_data] {self.n}/{self.total} cells ({pct:.1f}%) | "
              f"rate={rate:.2f} cells/s | elapsed {elapsed / 60:.1f} min | "
              f"ETA {eta_str}", flush=True)
        self._last_print_t = now
        self._last_print_n = self.n

    def write(self, msg):
        if self._bar is not None:
            self._bar.write(msg)
        else:
            print(msg, flush=True)

    def close(self):
        if self._bar is not None:
            self._bar.close()
        elif self.n < self.total:
            self._emit_periodic(time.perf_counter())


def _real_data_one_cell(spec, X, y, method_name, seed):
    """Run one (dataset, method, seed) cell.

    Returns a dict with the per-cell record. The fit honors a soft
    search budget inside ``_run_optuna`` (``OPTUNA_TIMEOUT_S``, default
    180 s); methods that exhaust this budget without a single completed
    trial fall back to library defaults and set ``used_defaults=True``
    on the row. The hard ``_CELL_TIMEOUT_S`` SIGALRM (default 360 s,
    matching the AMLB protocol's ``2 x max_runtime_seconds`` rule)
    only fires for genuine hangs; in that case the cell falls back to
    a constant predictor (training-set mean), records real metrics
    against that prediction, and sets ``used_constant_predictor=True``
    so the row is auditable. This is the AMLB + AutoGluon "always
    produce a prediction" guarantee (Gijsbers et al. 2024 JMLR;
    Erickson et al. 2020 AutoML); no NaN predictions are ever emitted.
    """
    import signal

    from .baselines import METHODS, METHOD_FAMILY

    # Hard thread cap via threadpoolctl. The module-level
    # ``os.environ.setdefault("OMP_NUM_THREADS", "1")`` block is a no-op
    # when the host (Databricks notebooks, in particular) pre-exports
    # ``OMP_NUM_THREADS=8`` or similar in the worker init script. Without
    # this in-worker cap, every loky worker silently inherits the host
    # thread count and 128 workers x 8 BLAS threads oversubscribes the
    # box. ``threadpool_limits`` rebinds the cap process-locally so the
    # protection holds regardless of the inherited environment.
    # Construction immediately activates the limit; the reference is
    # kept on the worker so subsequent cells in the same worker inherit
    # it (joblib reuses workers across delayed calls).
    _set_worker_thread_cap()

    # Per-cell memory instrumentation. Capture host/pid up front, start a
    # 0.5 s daemon-thread RSS sampler, and read end RSS at each return
    # site. ``peak_holder`` is shared with the sampler thread; reading
    # ``peak_holder[0]`` at return time gives the running peak. The
    # ``finally`` block stops the sampler regardless of fit outcome.
    host = socket.gethostname()
    pid = os.getpid()
    proc = psutil.Process(pid) if _PSUTIL_AVAILABLE else None
    rss_start_mb = (proc.memory_info().rss / 1024 / 1024) if proc else float("nan")
    peak_holder = [rss_start_mb if proc else 0.0]
    advisory_flag = [False]
    stop_event = threading.Event()
    sampler = None
    if proc is not None:
        sampler = threading.Thread(
            target=_sample_peak_rss,
            args=(proc, stop_event, peak_holder),
            kwargs={
                "advisory_mb": _cell_rss_advisory_mb(),
                "advisory_flag": advisory_flag,
            },
            daemon=True,
        )
        sampler.start()

    def _rss_now_mb():
        if proc is None:
            return float("nan")
        try:
            return proc.memory_info().rss / 1024 / 1024
        except Exception:
            return float("nan")

    rng = np.random.default_rng(seed * 1000 + 7)
    idx = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    base = {
        "dataset": spec.name,
        "method": method_name,
        "family": METHOD_FAMILY.get(method_name, ""),
        "seed": int(seed),
        "n_total": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "d": int(X.shape[1]),
        "host": host,
        "pid": pid,
        "rss_start_mb": rss_start_mb,
    }

    timed_out = [False]

    def _timeout_handler(signum, frame):  # noqa: ARG001
        timed_out[0] = True
        raise TimeoutError(f"cell timed out after {_CELL_TIMEOUT_S}s")

    # SIGALRM only fires on the main thread of the main interpreter, so
    # the hard cell-level timeout is skipped when the cell runs inside a
    # joblib threading-backend worker (the LIGHT stage). LIGHT methods
    # are pure-Python / numpy / sklearn fits with no known hang modes;
    # the soft Optuna budget in baselines._run_optuna remains in force.
    have_alarm = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    prev_handler = None
    cell_t0 = time.perf_counter()
    if have_alarm:
        prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_CELL_TIMEOUT_S)
    try:
        fit_func = METHODS[method_name]
        out = fit_func(X_tr, y_tr, int(seed))
        y_pred = np.asarray(out.predict(X_te), dtype=np.float64).ravel()
        # Four metrics per cell, following the AMLB + Grinsztajn 2022
        # union: RMSE (primary loss), R^2 (scale-free, AMLB standard),
        # MAE (heavy-tailed-robust, Grinsztajn 2022), NRMSE (RMSE
        # normalised by the *training-set* standard deviation, TabPFN
        # convention; using the training std avoids any test leakage
        # in the normalisation). R^2 is undefined when the test target
        # is constant; NRMSE is undefined when the training target is
        # constant; both report NaN in those degenerate cases.
        err = y_te - y_pred
        rmse_test = float(np.sqrt(np.mean(err ** 2)))
        mae_test = float(np.mean(np.abs(err)))
        var_te = float(np.var(y_te))
        r2_test = float("nan") if var_te == 0.0 else 1.0 - float(np.mean(err ** 2)) / var_te
        std_tr = float(np.std(y_tr))
        nrmse_test = float("nan") if std_tr == 0.0 else rmse_test / std_tr
        return {
            **base,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "r2_test": r2_test,
            "nrmse_test": nrmse_test,
            "n_fits": int(out.n_fits),
            "fit_time_s": float(out.fit_time_s),
            "best_hp": str(out.best_hp),
            "used_defaults": bool(getattr(out, "used_defaults", False)),
            "used_constant_predictor": False,
            "error": "",
            "rss_peak_mb": peak_holder[0],
            "rss_end_mb": _rss_now_mb(),
            "rss_advisory_exceeded": bool(advisory_flag[0]),
        }
    except BaseException as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - cell_t0
        is_timeout = (
            isinstance(exc, TimeoutError)
            or timed_out[0]
            or (have_alarm and isinstance(exc, KeyboardInterrupt)
                and elapsed >= _CELL_TIMEOUT_S * 0.9)
        )
        if is_timeout:
            msg = f"cell timed out after {_CELL_TIMEOUT_S}s"
        elif isinstance(exc, KeyboardInterrupt) and advisory_flag[0]:
            # The daemon RSS sampler raised KeyboardInterrupt on the
            # cell's main thread because the per-cell RSS soft cap was
            # exceeded. Surface the peak and the threshold so the audit
            # trail in real_data.csv is self-explanatory.
            msg = (f"rss advisory exceeded: peak={peak_holder[0]:.0f}MiB "
                   f"advisory={_cell_rss_advisory_mb():.0f}MiB t={elapsed:.1f}s")
        elif isinstance(exc, KeyboardInterrupt):
            # CatBoost (and other native-code libraries) translate any
            # pending Python signal into an empty-message
            # KeyboardInterrupt from inside their C++ training loop.
            # On Databricks the notebook reaper periodically delivers
            # SIGINTs to worker processes; if we propagated those, the
            # entire run would abort. Treat any worker-level KBI as a
            # cell failure and fall through to the constant-predictor
            # fallback below; the joblib parent process retains its
            # own SIGINT handling for real user interrupts.
            msg = f"interrupted by signal at t={elapsed:.1f}s"
        else:
            msg = str(exc)[:200]
        # Constant-predictor fallback. The training-set mean is the
        # optimal constant predictor under squared loss, so its R^2 is
        # 0 on a held-out sample drawn from the same distribution and
        # negative when there is any meaningful test/train shift. This
        # is the AMLB + AutoGluon "always produce a prediction"
        # guarantee (Gijsbers et al. 2024 JMLR; Erickson et al. 2020):
        # no cell is allowed to emit NaN, and the constant baseline is
        # the canonical floor against which any tuned method should
        # compare favourably. ``used_constant_predictor=True`` flags
        # the row so the aggregator can surface the fraction of cells
        # that hit this fallback per (method, dataset).
        y_const = float(np.mean(y_tr))
        y_pred = np.full_like(y_te, y_const, dtype=np.float64)
        err = y_te - y_pred
        rmse_test = float(np.sqrt(np.mean(err ** 2)))
        mae_test = float(np.mean(np.abs(err)))
        var_te = float(np.var(y_te))
        r2_test = float("nan") if var_te == 0.0 else 1.0 - float(np.mean(err ** 2)) / var_te
        std_tr = float(np.std(y_tr))
        nrmse_test = float("nan") if std_tr == 0.0 else rmse_test / std_tr
        return {
            **base,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "r2_test": r2_test,
            "nrmse_test": nrmse_test,
            "n_fits": 0,
            "fit_time_s": float(elapsed),
            "best_hp": "",
            "used_defaults": False,
            "used_constant_predictor": True,
            "error": msg,
            "rss_peak_mb": peak_holder[0],
            "rss_end_mb": _rss_now_mb(),
            "rss_advisory_exceeded": bool(advisory_flag[0]),
        }
    finally:
        stop_event.set()
        if sampler is not None:
            sampler.join(timeout=2.0)
        if have_alarm:
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)
        # Drop the fitted estimator, Optuna study, and CV buffers
        # before this loky worker picks up its next cell. With
        # ``max_tasks_per_child`` set per stage this is belt-and-
        # suspenders; without it, it is the primary RSS bound.
        gc.collect()


def _sentinel_row_for_cell(cell, error_msg: str) -> dict:
    """Build a constant-predictor row for a cell that never returned.

    When a stage's worker pool is torn down by ``TerminatedWorkerError``
    (or any other generator-level failure), the cells that were
    dispatched but did not produce a row are otherwise invisible: the
    aggregator sees a stage as "never ran" instead of "ran and was
    killed". This helper re-creates the same train/test split that
    ``_real_data_one_cell`` would have used and computes the
    constant-predictor (train mean) metrics so every (dataset, method,
    seed) attempted by the driver appears in ``real_data.csv``.

    ``error_msg`` records why the cell was sentineled (e.g.
    ``stage_terminated_worker``, ``stage_keyboard_interrupt``) and the
    row carries ``used_constant_predictor=True`` so downstream filters
    can exclude it from ranking just like a worker-level fallback.
    """
    from .baselines import METHOD_FAMILY
    spec = cell["spec"]
    X = cell["X"]
    y = cell["y"]
    method_name = cell["method"]
    seed = int(cell["seed"])
    rng = np.random.default_rng(seed * 1000 + 7)
    idx = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    y_tr, y_te = y[train_idx], y[test_idx]
    y_const = float(np.mean(y_tr))
    err = y_te - y_const
    rmse_test = float(np.sqrt(np.mean(err ** 2)))
    mae_test = float(np.mean(np.abs(err)))
    var_te = float(np.var(y_te))
    r2_test = float("nan") if var_te == 0.0 else 1.0 - float(np.mean(err ** 2)) / var_te
    std_tr = float(np.std(y_tr))
    nrmse_test = float("nan") if std_tr == 0.0 else rmse_test / std_tr
    return {
        "dataset": spec.name,
        "method": method_name,
        "family": METHOD_FAMILY.get(method_name, ""),
        "seed": seed,
        "n_total": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "d": int(X.shape[1]),
        "host": "",
        "pid": -1,
        "rss_start_mb": float("nan"),
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test,
        "nrmse_test": nrmse_test,
        "n_fits": 0,
        "fit_time_s": 0.0,
        "best_hp": "",
        "used_defaults": False,
        "used_constant_predictor": True,
        "error": error_msg,
        "rss_peak_mb": float("nan"),
        "rss_end_mb": float("nan"),
        "rss_advisory_exceeded": False,
    }


def _driver_rss_mb() -> float:
    """Driver process RSS in MiB; NaN if psutil unavailable."""
    if not _PSUTIL_AVAILABLE:
        return float("nan")
    try:
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return float("nan")


def _cgroup_free_mb() -> float:
    """Cgroup-free memory in MiB, or NaN if the cgroup is uncapped."""
    _limit_gb, free_gb = _cgroup_memory_gb()
    if free_gb is None:
        return float("nan")
    return float(free_gb) * 1024.0


def _cnl(r2: np.ndarray,
         r2_ref: np.ndarray,
         t: np.ndarray,
         alpha: float = 1.0) -> np.ndarray:
    """Per-cell Compute-Normalized Lift (CNL) over the linear baseline.

    Defined as

        CNL_alpha = max(0, max(0, R^2) - max(0, R^2_ref)) / (1 + t)^alpha

    with ``t`` measured in seconds and ``r2_ref`` the matched per-cell
    R^2 of the reference forecast (here ordinary least squares, the
    operational no-effort baseline for tabular regression). The
    numerator is a Murphy 1988 skill score taken against an arbitrary
    reference forecast; the operational reference is OLS rather than
    the climatology constant because a method that does not strictly
    out-predict OLS adds no practitioner value. The metric satisfies
    four axioms: (i) no-skill predictors (R^2 <= 0) score zero;
    (ii) methods that do not beat OLS score zero, so a free near-OLS
    predictor cannot ride OLS's compute advantage to a high score;
    (iii) the denominator is bounded below by 1, so reporting a
    near-zero wall time cannot inflate the score; (iv) the metric is
    dimensionless, scale-free in y, and bounded in [0, 1].
    Headline weight is ``alpha = 1``; sensitivity over
    ``alpha in {0, 0.25, 0.5, 1, 2}`` is reported in the
    joint-significance table.
    """
    skill = np.maximum(0.0, r2.astype(float))
    skill_ref = np.maximum(0.0, r2_ref.astype(float))
    lift = np.maximum(0.0, skill - skill_ref)
    t_safe = np.maximum(0.0, t.astype(float))
    return lift / np.power(1.0 + t_safe, float(alpha))


def _ols_r2_lookup(df: "pd.DataFrame") -> "pd.Series":
    """Per-(dataset, seed) OLS R^2 keyed for paired CNL evaluation.

    Returns the median (across replicates within a (dataset, seed))
    R^2 of the ``linear`` method, indexed by (dataset, seed). The
    median collapses the rare case of duplicated cells; in the
    canonical sweep there is exactly one ``linear`` row per cell and
    the median is the identity.
    """
    if "linear" not in df["method"].values:
        return pd.Series(dtype=float, name="r2_ols")
    ols = (df[df["method"] == "linear"]
           .groupby(["dataset", "seed"])["r2_test"].median()
           .rename("r2_ols"))
    return ols


def _joint_significance(df: "pd.DataFrame",
                         alphas: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
                         min_obs: int = 10) -> "pd.DataFrame":
    """Paired Wilcoxon test on the Compute-Normalized Lift (CNL).

    For each (method, alpha) pair, computes per-cell

        delta = CNL_alpha(kore) - CNL_alpha(competitor)
              = max(0, max(0, R^2_k) - max(0, R^2_ols)) / (1 + t_k)^alpha
                - max(0, max(0, R^2_m) - max(0, R^2_ols)) / (1 + t_m)^alpha

    paired across (dataset, seed) with the OLS reference cell taken
    from the same (dataset, seed) row. Reports the median, the raw
    paired-Wilcoxon p-value of ``delta`` against zero, the
    Holm-Bonferroni adjusted p-value across methods within each
    ``alpha``, and the direction (``kore_better`` when the median is
    positive at p_holm < 0.05, ``kore_worse`` when negative,
    ``tied`` otherwise). Returns one row per (method, alpha). The
    OLS row itself has CNL identically zero by construction and is
    skipped.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return pd.DataFrame(columns=[
            "method", "alpha", "n_obs", "median_delta",
            "wilcoxon_p_raw", "wilcoxon_p_holm", "direction",
        ])
    if "kore" not in df["method"].values:
        return pd.DataFrame(columns=[
            "method", "alpha", "n_obs", "median_delta",
            "wilcoxon_p_raw", "wilcoxon_p_holm", "direction",
        ])
    ols = _ols_r2_lookup(df)
    if ols.empty:
        return pd.DataFrame(columns=[
            "method", "alpha", "n_obs", "median_delta",
            "wilcoxon_p_raw", "wilcoxon_p_holm", "direction",
        ])
    kore = df[df["method"] == "kore"].set_index(["dataset", "seed"])[
        ["r2_test", "fit_time_s"]
    ]
    other_methods = [m for m in df["method"].unique() if m != "kore"]
    rows = []
    for alpha in alphas:
        m_names: list[str] = []
        p_raw: list[float] = []
        med: list[float] = []
        n_obs_list: list[int] = []
        for m in other_methods:
            other = df[df["method"] == m].set_index(["dataset", "seed"])[
                ["r2_test", "fit_time_s"]
            ]
            common = kore.index.intersection(other.index).intersection(ols.index)
            if len(common) < min_obs:
                continue
            kr = kore.loc[common, "r2_test"].to_numpy()
            kt = kore.loc[common, "fit_time_s"].to_numpy()
            orr = other.loc[common, "r2_test"].to_numpy()
            ot = other.loc[common, "fit_time_s"].to_numpy()
            ref = ols.loc[common].to_numpy()
            mask = (np.isfinite(kr) & np.isfinite(orr)
                    & np.isfinite(kt) & np.isfinite(ot)
                    & np.isfinite(ref))
            if mask.sum() < min_obs:
                continue
            cnl_k = _cnl(kr[mask], ref[mask], kt[mask], alpha)
            cnl_m = _cnl(orr[mask], ref[mask], ot[mask], alpha)
            delta = cnl_k - cnl_m
            try:
                _stat, pval = wilcoxon(delta, zero_method="wilcox")
            except ValueError:
                pval = float("nan")
            m_names.append(m)
            p_raw.append(float(pval) if pval == pval else float("nan"))
            med.append(float(np.median(delta)))
            n_obs_list.append(int(mask.sum()))
        if not p_raw:
            continue
        p_arr = np.asarray(p_raw)
        holm = _holm_bonferroni(np.where(np.isfinite(p_arr), p_arr, 1.0))
        for name, p, h, mu, n_obs in zip(m_names, p_raw, holm, med, n_obs_list):
            if h < 0.05 and mu > 0.0:
                direction = "kore_better"
            elif h < 0.05 and mu < 0.0:
                direction = "kore_worse"
            else:
                direction = "tied"
            rows.append({
                "method": name,
                "alpha": float(alpha),
                "n_obs": int(n_obs),
                "median_delta": float(mu),
                "wilcoxon_p_raw": float(p),
                "wilcoxon_p_holm": float(h),
                "direction": direction,
            })
    return pd.DataFrame(rows)


def _threshold_sensitivity(df: "pd.DataFrame",
                            max_d_grid: Sequence[int] = (20, 30, 40, 50),
                            ) -> "pd.DataFrame":
    """Post-hoc sensitivity sweep for the smooth-low-d cutoff ``max_d``.

    For each ``max_d`` in ``max_d_grid`` the function restricts the
    per-cell table to datasets with ``d <= max_d``, recomputes the
    geometric-mean RMSE ratio of every method against KORE, and reports
    KORE's mean Friedman rank on the headline Compute-Normalized Lift
    score (``mean_rank_cnl``, lower is better, since ranks are taken on
    ``-CNL``) alongside the raw-RMSE rank (``mean_rank``) for
    transparency. The output has one row per (max_d, method) and is the
    source for the threshold sensitivity table in the appendix.
    """
    if "kore" not in df["method"].values:
        return pd.DataFrame()
    work = df.copy()
    if "r2_ols" not in work.columns:
        ols = _ols_r2_lookup(work)
        if ols.empty:
            return pd.DataFrame()
        work = work.join(ols, on=["dataset", "seed"])
        work["r2_ols"] = work["r2_ols"].fillna(0.0)
    work["cnl"] = _cnl(work["r2_test"].to_numpy(),
                        work["r2_ols"].to_numpy(),
                        work["fit_time_s"].to_numpy(), alpha=1.0)
    rows = []
    for max_d in max_d_grid:
        sub = work[work["d"] <= int(max_d)].copy()
        if sub.empty:
            continue
        med = (sub.groupby(["dataset", "method"])
                  .agg(rmse_med=("rmse_test", "median"),
                       cnl_med=("cnl", "median"))
                  .reset_index())
        pivot = med.pivot_table(index="dataset", columns="method",
                                 values="rmse_med")
        cnl_pivot = med.pivot_table(index="dataset", columns="method",
                                     values="cnl_med")
        if "kore" not in pivot.columns:
            continue
        ref = pivot["kore"]
        complete_rmse = pivot.dropna(axis=1, how="any")
        ranks_rmse = (complete_rmse.rank(axis=1, method="average").mean(axis=0)
                      if not complete_rmse.empty else None)
        # Higher CNL is better, so rank ``-CNL`` (lower mean rank = larger
        # OLS-relative lift per unit compute on the typical dataset).
        complete_cnl = cnl_pivot.dropna(axis=1, how="any")
        ranks_cnl = ((-complete_cnl).rank(axis=1, method="average").mean(axis=0)
                      if not complete_cnl.empty else None)
        for m in pivot.columns:
            ratio = (pivot[m] / ref).dropna()
            ratio = ratio[ratio > 0]
            if ratio.empty:
                continue
            gm = float(np.exp(np.log(ratio).mean()))
            rank = (float(ranks_rmse[m])
                    if (ranks_rmse is not None and m in ranks_rmse.index)
                    else float("nan"))
            rank_cnl = (float(ranks_cnl[m])
                         if (ranks_cnl is not None and m in ranks_cnl.index)
                         else float("nan"))
            rows.append({
                "max_d": int(max_d),
                "method": m,
                "n_datasets": int((pivot[m].notna() & ref.notna()).sum()),
                "gm_rmse_ratio_vs_kore": gm,
                "mean_rank": rank,
                "mean_rank_cnl": rank_cnl,
            })
    return pd.DataFrame(rows)


def _failure_modes(df: "pd.DataFrame",
                    classical_methods: Sequence[str] = ("gcv_spline", "aic_spline", "bic_spline", "cp_spline"),
                    booster_methods: Sequence[str] = ("xgboost", "lightgbm", "catboost", "hist_gbm"),
                    top_k: int = 5) -> "pd.DataFrame":
    """Identify the datasets where KORE loses most to the best classical
    spline selector and where it loses most to the best tuned booster.

    For each dataset, compute the median test RMSE per method (across
    seeds), then form ``log_ratio_vs_classical = log(rmse_kore /
    min(rmse_gcv, rmse_aic, rmse_bic, rmse_cp))`` and the analogous
    booster ratio. Positive log-ratio means KORE is worse than the
    competitor on that dataset. Returns the union of the top-``k``
    datasets by each ratio with a short structural-explanation tag
    attached for the failure-mode appendix table.
    """
    if "kore" not in df["method"].values:
        return pd.DataFrame()
    med = (df.groupby(["method", "dataset"])
             .agg(rmse_med=("rmse_test", "median"),
                  n_train=("n_train", "median"),
                  d=("d", "median"))
             .reset_index())
    med = med[(med["rmse_med"] > 0) & np.isfinite(med["rmse_med"])]
    pivot = med.pivot_table(index="dataset", columns="method", values="rmse_med")
    n_train_map = med.groupby("dataset")["n_train"].first()
    d_map = med.groupby("dataset")["d"].first()
    if "kore" not in pivot.columns:
        return pd.DataFrame()
    classical_present = [m for m in classical_methods if m in pivot.columns]
    booster_present = [m for m in booster_methods if m in pivot.columns]
    rows = []
    for ds in pivot.index:
        kore_r = pivot.at[ds, "kore"]
        if not np.isfinite(kore_r) or kore_r <= 0:
            continue
        cls_vals = pivot.loc[ds, classical_present].dropna()
        cls_vals = cls_vals[cls_vals > 0]
        bst_vals = pivot.loc[ds, booster_present].dropna()
        bst_vals = bst_vals[bst_vals > 0]
        cls_best = float(cls_vals.min()) if not cls_vals.empty else float("nan")
        bst_best = float(bst_vals.min()) if not bst_vals.empty else float("nan")
        log_cls = float(np.log(kore_r / cls_best)) if cls_best > 0 else float("nan")
        log_bst = float(np.log(kore_r / bst_best)) if bst_best > 0 else float("nan")
        rows.append({
            "dataset": ds,
            "n_train": float(n_train_map.get(ds, np.nan)),
            "d": float(d_map.get(ds, np.nan)),
            "kore_rmse": float(kore_r),
            "classical_best_rmse": cls_best,
            "booster_best_rmse": bst_best,
            "log_ratio_vs_classical": log_cls,
            "log_ratio_vs_booster": log_bst,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    cls_top = table.sort_values("log_ratio_vs_classical", ascending=False).head(top_k)
    bst_top = table.sort_values("log_ratio_vs_booster", ascending=False).head(top_k)
    sel = pd.concat([cls_top, bst_top]).drop_duplicates(subset=["dataset"])
    sel["high_d"] = sel["d"] > 20.0
    sel["large_loss_vs_classical"] = sel["log_ratio_vs_classical"] > 0.05
    sel["large_loss_vs_booster"] = sel["log_ratio_vs_booster"] > 0.05
    def _tag(row) -> str:
        if row["d"] >= 25:
            return "high post-one-hot dimension"
        if row["log_ratio_vs_booster"] > 0.20:
            return "non-additive structure favors trees"
        if row["log_ratio_vs_classical"] > 0.05:
            return "low signal-to-noise"
        return "near-tie"
    sel["structural_explanation"] = sel.apply(_tag, axis=1)
    sel = sel.sort_values("log_ratio_vs_booster", ascending=False).reset_index(drop=True)
    return sel


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Step-down Holm correction. Input ``p_values`` is unordered; the
    output keeps the input order."""
    order = np.argsort(p_values)
    n = len(p_values)
    holm = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        adj = p_values[i] * (n - rank)
        running = max(running, adj)
        holm[i] = min(1.0, running)
    return holm


def _friedman_nemenyi(metric_pivot: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman omnibus + Nemenyi post-hoc on a (datasets x methods) table.

    Implements the Demsar 2006 JMLR protocol for comparing multiple
    classifiers (here, regressors) over multiple datasets:

      1. Per row (dataset), rank methods 1..k from best to worst by the
         metric. Ties receive average ranks.
      2. Friedman chi-square statistic on the rank table tests the null
         "all methods are equally good across the N datasets".
      3. Nemenyi post-hoc declares two methods significantly different
         when their mean ranks differ by more than the critical
         difference ``CD = q_alpha * sqrt(k(k+1) / (6N))``, where
         ``q_alpha`` is the (1 - alpha) quantile of the studentized
         range distribution at infinite degrees of freedom.

    Rows with any NaN entry are dropped before ranking so every
    surviving row contributes a complete ranking. Lower metric is
    assumed to be better (RMSE / MAE / NRMSE convention); for R^2 the
    caller should pass ``-r2`` instead.
    """
    from scipy.stats import friedmanchisquare, studentized_range

    pivot = metric_pivot.dropna(axis=0, how="any")
    n_datasets, k_methods = pivot.shape
    if n_datasets < 2 or k_methods < 2:
        return {
            "n_datasets": int(n_datasets),
            "k_methods": int(k_methods),
            "friedman_stat": float("nan"),
            "friedman_p": float("nan"),
            "cd_alpha": float(alpha),
            "cd": float("nan"),
            "mean_ranks": {},
        }
    ranks = pivot.rank(axis=1, method="average", ascending=True)
    mean_ranks = ranks.mean(axis=0)
    columns = [pivot[c].to_numpy() for c in pivot.columns]
    try:
        stat, p_val = friedmanchisquare(*columns)
    except ValueError:
        stat, p_val = float("nan"), float("nan")
    q_alpha = float(studentized_range.ppf(1.0 - alpha, k_methods, np.inf))
    cd = q_alpha * float(np.sqrt(k_methods * (k_methods + 1) / (6.0 * n_datasets)))
    return {
        "n_datasets": int(n_datasets),
        "k_methods": int(k_methods),
        "friedman_stat": float(stat),
        "friedman_p": float(p_val),
        "cd_alpha": float(alpha),
        "cd": float(cd),
        "mean_ranks": {str(m): float(r) for m, r in mean_ranks.items()},
    }


def _resolve_real_data_backend(_=None):
    """Backwards-compatible stub.

    The Spark dispatch path was removed in favor of a staged loky
    dispatch (``_dispatch_real_data_staged``). The signature is preserved
    so any out-of-tree caller does not break.
    """
    return ("loky", None, "loky (driver-only, staged)")


def _max_tasks_per_child(stage: str) -> int:
    """Per-stage loky worker recycle threshold (cells per worker).

    Loky workers are reused indefinitely by default; with Optuna study
    state and sklearn estimator residue, per-worker RSS grows by tens
    to hundreds of MB per cell. Recycling at fixed cell counts caps
    the per-worker steady-state RSS without measurable fork overhead
    on the 21 x 36 x 5 sweep (~236 forks across ~324 workers).
    """
    defaults = {"heavy": 8, "medium": 16, "light": 32}
    env_keys = {
        "heavy":  "KORE_MAX_TASKS_HEAVY",
        "medium": "KORE_MAX_TASKS_MEDIUM",
        "light":  "KORE_MAX_TASKS_LIGHT",
    }
    raw = os.environ.get(env_keys[stage], "").strip()
    if not raw:
        return defaults[stage]
    try:
        v = int(raw)
        return v if v > 0 else defaults[stage]
    except ValueError:
        return defaults[stage]


def _dispatch_real_data_staged(
    cells, fn, *, n_jobs_heavy, n_jobs_medium, n_jobs_light,
    on_result, logger, on_stage_event=None,
):
    """Run cells in three sequential stages bucketed by method weight.

    Bucket boundaries: heavy >= 4.0, medium in [2.0, 4.0), light < 2.0;
    weight is read from ``_method_weight``, which promotes high-d KNN /
    SVR / KernelRidge cells (``d >= _HIGH_D_THRESHOLD``) out of their
    nominal bucket to reflect the larger distance/kernel matrices.
    Each stage is a fresh ``Parallel(return_as='generator_unordered')``;
    a ``TerminatedWorkerError`` raised inside one bucket is logged and
    the next bucket still runs. The descending ``n_jobs`` profile gives
    the memory-bound heavy stage the most headroom and the cheap light
    stage the most concurrency.

    Cells dispatched into a stage that does not return a row before the
    generator aborts (typically because of ``TerminatedWorkerError``)
    are emitted as constant-predictor sentinel rows via ``on_result``
    so every attempted (dataset, method, seed) appears in
    ``real_data.csv``. The per-stage diagnostics (cells, n_jobs, n
    completed, n sentineled, wall time, driver/cgroup RSS at start/end,
    last exception class and message) are forwarded to
    ``on_stage_event`` when provided so the caller can persist a
    ``real_data_stage_events.csv`` audit trail.
    """
    from joblib import Parallel, delayed
    from joblib.externals.loky.process_executor import TerminatedWorkerError

    def weight(cell):
        d = cell["X"].shape[1] if hasattr(cell["X"], "shape") else 0
        return _method_weight(cell["method"], d)

    buckets = {
        "heavy":  ([c for c in cells if weight(c) >= 4.0], n_jobs_heavy),
        "medium": ([c for c in cells if 2.0 <= weight(c) < 4.0], n_jobs_medium),
        "light":  ([c for c in cells if weight(c) < 2.0], n_jobs_light),
    }
    # Backend per stage. All three stages run on ``loky`` (process-based).
    # The threading backend was previously selectable via
    # ``KORE_LIGHT_BACKEND=threading`` for short LIGHT stages, but it
    # retains every Optuna study, sklearn estimator, and CV buffer in
    # the driver heap for the duration of the stage and reliably OOM-
    # kills 200+ concurrency LIGHT runs on the 21 x 36 x 5 sweep. The
    # opt-in has been removed; ``KORE_LIGHT_BACKEND`` is no longer read.
    stage_backend = {"heavy": "loky", "medium": "loky", "light": "loky"}
    for stage, (stage_cells, n_jobs) in buckets.items():
        if not stage_cells:
            continue
        backend = stage_backend[stage]
        max_tasks = _max_tasks_per_child(stage)
        # Pre-stage diagnostics: driver RSS, cgroup free memory, wall
        # clock. Captured before the loky pool is forked so a later
        # death point can be diagnosed without re-running.
        wall_start = time.perf_counter()
        ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")
        driver_rss_start = _driver_rss_mb()
        cgroup_free_start = _cgroup_free_mb()
        completed_keys: set = set()
        dispatched_keys = {_cell_key(c) for c in stage_cells}
        exc_class = ""
        exc_msg = ""
        logger.info(
            "[real_data][stage=%s] cells=%d n_jobs=%d backend=%s "
            "max_tasks_per_child=%d driver_rss_mb=%.0f cgroup_free_mb=%.0f",
            stage, len(stage_cells), n_jobs, backend, max_tasks,
            driver_rss_start, cgroup_free_start,
        )
        # Inject the dispatched key into each returned row so the
        # driver can track completion even when the row doesn't carry
        # ``dataset`` / ``seed`` (test stubs, future wrappers). The key
        # is stripped before forwarding through ``on_result`` so it
        # never reaches ``real_data.csv``.
        def _fn_wrap(cell, _key=None):  # closure over fn
            row = fn(cell)
            if isinstance(row, dict):
                row["__dispatch_key"] = _cell_key(cell)
            return row

        try:
            gen = Parallel(
                n_jobs=n_jobs,
                backend=backend,
                return_as="generator_unordered",
                batch_size=1,
                max_tasks_per_child=max_tasks,
            )(delayed(_fn_wrap)(c) for c in stage_cells)
            for row in gen:
                try:
                    if isinstance(row, dict):
                        key = row.pop("__dispatch_key", None)
                        if key is not None:
                            completed_keys.add(key)
                    on_result(row)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[real_data][stage=%s] on_result failed: %s",
                        stage, exc,
                    )
        except TerminatedWorkerError as exc:
            exc_class = "TerminatedWorkerError"
            exc_msg = str(exc)[:500]
            logger.error(
                "[real_data][stage=%s] TerminatedWorkerError: %s "
                "(continuing with next stage; sentineling %d missing cells)",
                stage, exc, len(dispatched_keys) - len(completed_keys),
            )
        except KeyboardInterrupt:
            exc_class = "KeyboardInterrupt"
            logger.error("[real_data][stage=%s] KeyboardInterrupt", stage)
            _emit_sentinel_rows(
                stage_cells, completed_keys, on_result,
                f"stage_keyboard_interrupt:{stage}", logger, stage,
            )
            _post_stage_event(
                on_stage_event, stage, len(stage_cells), n_jobs, backend,
                max_tasks, len(completed_keys),
                len(dispatched_keys) - len(completed_keys),
                wall_start, ts_start, driver_rss_start, cgroup_free_start,
                "KeyboardInterrupt", "",
            )
            raise
        except BaseException as exc:  # noqa: BLE001
            exc_class = type(exc).__name__
            exc_msg = str(exc)[:500]
            logger.error(
                "[real_data][stage=%s] %s: %s (continuing with next stage)",
                stage, exc_class, exc_msg,
            )
        # Emit sentinel rows for every cell that was dispatched into the
        # stage but did not return. Without this the next run looks like
        # the stage never executed; with it, the per-cell audit trail
        # records exactly which (dataset, method, seed) cells the
        # stage-level failure swallowed.
        _emit_sentinel_rows(
            stage_cells, completed_keys, on_result,
            f"stage_{exc_class.lower()}" if exc_class else "stage_unreturned",
            logger, stage,
        )
        _post_stage_event(
            on_stage_event, stage, len(stage_cells), n_jobs, backend,
            max_tasks, len(completed_keys),
            len(dispatched_keys) - len(completed_keys),
            wall_start, ts_start, driver_rss_start, cgroup_free_start,
            exc_class, exc_msg,
        )
        gc.collect()  # release per-stage Parallel/future/generator state


def _cell_key(c):
    """Stable ``(dataset, method, seed)`` key for a real-data cell.

    Tolerant of test stubs whose ``spec`` is ``None``: falls back to
    ``c.get('_name')`` or ``str(id(c))`` so the dispatched/completed
    key sets still match in unit tests that monkeypatch
    ``joblib.Parallel`` and pass minimal cell dictionaries.
    """
    spec = c.get("spec")
    if spec is not None and hasattr(spec, "name"):
        ds = spec.name
    else:
        ds = c.get("_name", str(id(c)))
    return (ds, c["method"], int(c["seed"]))


def _emit_sentinel_rows(stage_cells, completed_keys, on_result, error_msg,
                         logger, stage):
    """Push a constant-predictor row through ``on_result`` for each
    cell in ``stage_cells`` whose ``(dataset, method, seed)`` key is
    not in ``completed_keys``. Errors are logged and swallowed so a
    single bad sentinel does not abort the audit trail. Cells whose
    ``spec`` is ``None`` (test stubs that monkeypatch Parallel) are
    skipped silently; the sentinel path is only meaningful for real
    cells with the X / y / spec needed to compute the train-mean
    prediction."""
    n_emitted = 0
    for c in stage_cells:
        key = _cell_key(c)
        if key in completed_keys:
            continue
        if c.get("spec") is None or c.get("X") is None or c.get("y") is None:
            continue
        try:
            row = _sentinel_row_for_cell(c, error_msg)
            on_result(row)
            n_emitted += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[real_data][stage=%s] sentinel emit failed for %s: %s",
                stage, key, exc,
            )
    if n_emitted:
        logger.info(
            "[real_data][stage=%s] emitted %d sentinel rows (error=%s)",
            stage, n_emitted, error_msg,
        )


def _post_stage_event(on_stage_event, stage, n_dispatched, n_jobs, backend,
                       max_tasks, n_completed, n_sentineled, wall_start,
                       ts_start, driver_rss_start, cgroup_free_start,
                       exc_class, exc_msg):
    """Forward a stage-summary event to the optional ``on_stage_event``
    callback. Captures post-stage driver/cgroup RSS so the event row
    records both endpoints."""
    if on_stage_event is None:
        return
    try:
        on_stage_event({
            "stage": stage,
            "n_cells_dispatched": int(n_dispatched),
            "n_jobs": int(n_jobs),
            "backend": backend,
            "max_tasks_per_child": int(max_tasks),
            "n_completed": int(n_completed),
            "n_sentineled": int(n_sentineled),
            "wall_time_s": float(time.perf_counter() - wall_start),
            "ts_start": ts_start,
            "ts_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "driver_rss_start_mb": float(driver_rss_start),
            "driver_rss_end_mb": float(_driver_rss_mb()),
            "cgroup_free_start_mb": float(cgroup_free_start),
            "cgroup_free_end_mb": float(_cgroup_free_mb()),
            "exception_class": exc_class,
            "exception_message": exc_msg,
        })
    except Exception:
        pass


def run_real_data(*, n_seeds: int = 5, max_n: int = 10_000,
                  methods=None, n_trials=None, n_jobs=None,
                  backend=None, progress: bool = True):
    """Run the 36-dataset / 21-method real-world benchmark.

    Parameters
    ----------
    n_seeds :
        Outer 80/20 split repetitions. 5 matches the TabPFN protocol.
    max_n :
        Per-dataset row cap to keep per-cell compute bounded; datasets
        with ``n > max_n`` are uniformly subsampled.
    methods :
        Optional restriction to a subset of method names.
    n_trials :
        Optuna trial budget per tunable method (default: keep the
        baselines module's default of 50).
    n_jobs :
        Ignored. Per-stage worker counts are resolved by
        ``_staged_njobs_from_env()`` from the env vars
        ``KORE_NJOBS_HEAVY``, ``KORE_NJOBS_MEDIUM``, ``KORE_NJOBS_LIGHT``
        (each capped at ``_local_worker_cap()``: ``min(0.9 * cpu_affinity,
        cgroup_free_gb / KORE_PER_WORKER_GB, KORE_MAX_WORKERS)``).
    backend :
        Deprecated and ignored. The driver dispatches via three sequential
        ``joblib.Parallel(backend='loky')`` stages bucketed by method
        weight (heavy >= 4.0, medium in [2.0, 4.0), light < 2.0), with a
        per-stage ``TerminatedWorkerError`` guard.
    progress :
        When True (default), dispatch progress through ``_ProgressReporter``:
        a tqdm widget bar inside an ipywidgets-capable notebook, a tqdm
        terminal bar when stdout is a TTY, and a periodic stdout line
        ``[real_data] N/T cells (P%) | rate=... | elapsed ... | ETA ...``
        otherwise (Databricks job-cluster logs, redirected stdout).
    """
    if backend is not None:
        warnings.warn(
            "run_real_data(backend=...) is deprecated and ignored; "
            "the driver always dispatches via the staged loky path.",
            DeprecationWarning,
            stacklevel=2,
        )
    if n_jobs is not None:
        warnings.warn(
            "run_real_data(n_jobs=...) is deprecated and ignored; set "
            "KORE_NJOBS_HEAVY / KORE_NJOBS_MEDIUM / KORE_NJOBS_LIGHT instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    from .baselines import (
        METHODS,
        METHOD_FAMILY,
        DEFAULT_OPTUNA_TRIALS,
        available_methods,
    )
    from .datasets import DATASETS, fetch_one, is_smooth_lowd

    print("=" * 78)
    print("EXP 10  Real-world benchmark  "
          "(36-dataset registry, up to 21 methods, Grinsztajn-cited grids)")
    print("=" * 78)

    if n_trials is None:
        n_trials = DEFAULT_OPTUNA_TRIALS

    avail = set(available_methods())
    if methods is None:
        methods = [m for m in METHODS if m in avail]
    else:
        methods = [m for m in methods if m in avail]
    print(f"Methods: {len(methods)} available  ({', '.join(methods)})")

    rng = np.random.default_rng(MASTER_SEED)
    cached = []
    subset_names = set()
    for spec in DATASETS:
        try:
            X, y = fetch_one(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {spec.name}: fetch failed ({exc})")
            continue
        if X.shape[0] > max_n:
            idx = rng.choice(X.shape[0], size=max_n, replace=False)
            idx.sort()
            X, y = X[idx], y[idx]
        in_subset = bool(is_smooth_lowd(X, y))
        if in_subset:
            subset_names.add(spec.name)
        cached.append((spec, X, y))
        print(f"  {spec.name}: n={X.shape[0]}, d={X.shape[1]}, "
              f"smooth_lowd={in_subset}")

    if not cached:
        raise RuntimeError("No datasets fetched; check network/cache.")

    cells = [
        {"spec": spec, "X": X, "y": y, "method": method, "seed": seed}
        for (spec, X, y) in cached
        for method in methods
        for seed in range(n_seeds)
    ]
    # Longest-processing-time-first dispatch order within each weight
    # bucket. The unordered generator pulls cells in iterator order, so
    # heavy cells (CatBoost, XGBoost, SVR) must come first to keep all
    # workers busy until the end of each stage. Within a tie, fall back
    # to the natural enumeration order so the dispatch is deterministic.
    cells.sort(
        key=lambda c: (-_cell_cost(c["method"], c["X"].shape[0], c["X"].shape[1]),
                       c["spec"].name, c["method"], int(c["seed"]))
    )
    print(f"\nTotal cells: {len(cells)}  "
          f"(datasets={len(cached)} x methods={len(methods)} x seeds={n_seeds})")

    njobs_h, njobs_m, njobs_l = _staged_njobs_from_env()
    print(f"Workers (staged loky): heavy={njobs_h} medium={njobs_m} "
          f"light={njobs_l}  BLAS threads=1  scheduling=LPT  "
          f"return_as=generator_unordered")
    print(f"  loky cap: {_local_worker_cap_explain()}")
    print(f"  per-cell RSS advisory: {_cell_rss_advisory_mb():.0f} MiB "
          f"(soft cap; cell aborts via KeyboardInterrupt, worker continues)")
    print(f"  driver RSS at start: {_driver_rss_mb():.0f} MiB  "
          f"cgroup free: {_cgroup_free_mb():.0f} MiB")
    print()

    checkpoint_path = OUT / "real_data.csv"
    memory_path = OUT / "real_data_memory.csv"
    stage_events_path = OUT / "real_data_stage_events.csv"
    rows: list = []
    stage_events: list = []
    if checkpoint_path.exists():
        try:
            prev = pd.read_csv(checkpoint_path)
            mask = prev["dataset"].notna()
            done_keys = set(
                zip(
                    prev.loc[mask, "dataset"].astype(str).to_numpy(),
                    prev.loc[mask, "method"].astype(str).to_numpy(),
                    prev.loc[mask, "seed"].astype(int).to_numpy(),
                )
            )
            cells = [
                c for c in cells
                if (c["spec"].name, c["method"], int(c["seed"])) not in done_keys
            ]
            rows = prev.to_dict("records")
            print(f"Resuming from checkpoint: {len(rows)} cells already done, "
                  f"{len(cells)} remaining.")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not resume from checkpoint ({exc}); starting fresh.")
            rows = []

    bar = _ProgressReporter(
        total=len(rows) + len(cells),
        initial=len(rows),
        desc="real_data cells",
    ) if progress else None

    # Staged dispatch. Cells are bucketed by ``_METHOD_WEIGHT`` into
    # heavy / medium / light stages; each stage is a fresh
    # ``Parallel(backend='loky', return_as='generator_unordered')`` with
    # its own ``n_jobs`` so the memory-bound heavy stage gets the most
    # headroom and the cheap light stage saturates the pool. A
    # ``TerminatedWorkerError`` raised inside one bucket is logged and
    # the next bucket still runs. Per-cell peak RSS is captured by the
    # instrumentation in ``_real_data_one_cell`` and written to
    # ``real_data_memory.csv``.
    checkpoint_every = max(50, min(max(njobs_h, njobs_m, njobs_l), 50))
    n_pending = len(cells)
    rss_rows: list = []
    if n_pending:
        wall_t0 = time.perf_counter()
        progress_state = {"since_ckpt": 0, "completed": 0}

        def _run_cell(cell):
            return _real_data_one_cell(
                cell["spec"], cell["X"], cell["y"],
                cell["method"], cell["seed"],
            )

        def _on_result(row):
            rows.append(row)
            rss_rows.append({
                "method":       row.get("method"),
                "dataset":      row.get("dataset"),
                "seed":         row.get("seed"),
                "host":         row.get("host"),
                "pid":          row.get("pid"),
                "rss_start_mb": row.get("rss_start_mb"),
                "rss_peak_mb":  row.get("rss_peak_mb"),
                "rss_end_mb":   row.get("rss_end_mb"),
                "rss_advisory_exceeded": row.get("rss_advisory_exceeded", False),
                "fit_time_s":   row.get("fit_time_s"),
                "error":        row.get("error", ""),
            })
            progress_state["since_ckpt"] += 1
            progress_state["completed"] += 1
            if bar is not None:
                bar.update(1)
            if progress_state["since_ckpt"] >= checkpoint_every:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                pd.DataFrame(rss_rows).to_csv(memory_path, index=False)
                if stage_events:
                    pd.DataFrame(stage_events).to_csv(stage_events_path, index=False)
                wall = time.perf_counter() - wall_t0
                msg = (f"  checkpoint: {progress_state['completed']}/{n_pending} cells, "
                       f"wall={wall / 60:.1f} min")
                if bar is not None:
                    bar.write(msg)
                else:
                    print(msg, flush=True)
                progress_state["since_ckpt"] = 0

        def _on_stage_event(event):
            stage_events.append(event)
            # Flush each stage's event row to disk immediately so a
            # later subprocess crash (after the stage but before the
            # final write below) still leaves the audit trail behind.
            try:
                pd.DataFrame(stage_events).to_csv(stage_events_path, index=False)
            except Exception:
                pass

        _dispatch_real_data_staged(
            cells=cells,
            fn=_run_cell,
            n_jobs_heavy=njobs_h,
            n_jobs_medium=njobs_m,
            n_jobs_light=njobs_l,
            on_result=_on_result,
            logger=logging.getLogger("kore.real_data"),
            on_stage_event=_on_stage_event,
        )
        if bar is not None:
            bar.close()
    pd.DataFrame(rss_rows).to_csv(memory_path, index=False)
    if stage_events:
        pd.DataFrame(stage_events).to_csv(stage_events_path, index=False)

    df = pd.DataFrame(rows)
    # Backfill any columns missing from a pre-upgrade checkpoint so the
    # downstream aggregation never KeyError's on a legacy schema.
    for col, default in (("mae_test", float("nan")),
                         ("r2_test", float("nan")),
                         ("nrmse_test", float("nan")),
                         ("used_defaults", False),
                         ("used_constant_predictor", False)):
        if col not in df.columns:
            df[col] = default
    df.to_csv(checkpoint_path, index=False)

    # With the constant-predictor fallback in ``_real_data_one_cell``,
    # every row carries valid metrics regardless of error state, so the
    # aggregation and downstream pivots run on the full frame. The
    # ``error`` column survives for auditing and is summarised via
    # ``used_constant_predictor_frac``. ``valid`` is retained as an
    # alias for the downstream Pareto and paired-test blocks.
    n_fallback = int(df["used_constant_predictor"].sum()) if len(df) else 0
    if n_fallback:
        print(f"\n{n_fallback} cells fell back to the constant predictor "
              f"(see used_constant_predictor column for the audit trail).")
    if df.empty:
        # All cells were SIGKILL'd before any could complete (e.g. macOS
        # OOM on a high-d kore pairwise structure search). The staged
        # dispatch caught the TerminatedWorkerError per stage, but there
        # is nothing to aggregate; emit empty companion CSVs and return
        # rather than crash the downstream groupby.
        print("\nNo cells completed; downstream aggregation skipped. "
              "See real_data_memory.csv for per-cell RSS audit.")
        return
    _aggregate_real_data_results(df, methods, OUT)


def _bootstrap_ranks(per_method_per_dataset_ranks: "pd.DataFrame",
                      n_boot: int = 1000,
                      seed: int = 2026) -> "pd.DataFrame":
    """Bootstrap mean-rank confidence intervals over the dataset axis.

    Resamples the row (dataset) axis of the per-dataset rank table with
    replacement ``n_boot`` times. For each bootstrap sample, recomputes
    the mean rank per method. Reports the bootstrap mean and the 2.5 /
    97.5 percentiles. Returns columns ``method, mean_rank, ci_lo_2p5,
    ci_hi_97p5, n_boot``, sorted ascending by ``mean_rank``.
    """
    rng = np.random.default_rng(seed)
    R = per_method_per_dataset_ranks.to_numpy(dtype=float)
    n_rows, n_methods = R.shape
    means = np.empty((n_boot, n_methods), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        means[b] = R[idx].mean(axis=0)
    out = pd.DataFrame({
        "method": list(per_method_per_dataset_ranks.columns),
        "mean_rank": means.mean(axis=0),
        "ci_lo_2p5": np.percentile(means, 2.5, axis=0),
        "ci_hi_97p5": np.percentile(means, 97.5, axis=0),
        "n_boot": int(n_boot),
    })
    return out.sort_values("mean_rank").reset_index(drop=True)


def _dataset_metadata_table() -> "pd.DataFrame":
    """One row per registered dataset with the schema fields used in the
    paper's per-dataset table.

    Returns columns ``dataset, n, d_raw_numeric, d_raw_categorical,
    d_onehot``. Missing fields become ``np.nan`` rather than raising.
    """
    from .datasets import DATASETS, dataset_schema

    rows = []
    for spec in DATASETS:
        try:
            sch = dataset_schema(spec)
        except Exception:
            rows.append({
                "dataset": getattr(spec, "name", None),
                "n": np.nan,
                "d_raw_numeric": np.nan,
                "d_raw_categorical": np.nan,
                "d_onehot": np.nan,
            })
            continue
        rows.append({
            "dataset": sch.get("name", np.nan),
            "n": sch.get("n", np.nan),
            "d_raw_numeric": sch.get("d_raw_numeric", np.nan),
            "d_raw_categorical": sch.get("d_raw_categorical", np.nan),
            "d_onehot": sch.get("d_onehot", np.nan),
        })
    return pd.DataFrame(rows)


def _per_method_failure_fractions(real_data_df: "pd.DataFrame") -> "pd.DataFrame":
    """Per-method fraction of cells that fell back to library defaults
    or to the constant predictor.

    Returns columns ``method, used_defaults_frac,
    used_constant_predictor_frac, n_cells``, sorted ascending by method.
    """
    g = real_data_df.groupby("method")
    out = pd.DataFrame({
        "method": list(g.groups.keys()),
        "used_defaults_frac": [
            float(real_data_df.loc[g.groups[m], "used_defaults"]
                  .astype(float).mean())
            for m in g.groups
        ],
        "used_constant_predictor_frac": [
            float(real_data_df.loc[g.groups[m], "used_constant_predictor"]
                  .astype(float).mean())
            for m in g.groups
        ],
        "n_cells": [int(len(g.groups[m])) for m in g.groups],
    })
    return out.sort_values("method").reset_index(drop=True)


def _aggregate_real_data_results(df: "pd.DataFrame",
                                  methods: Sequence[str],
                                  OUT: Path) -> None:
    """Aggregate the per-cell ``real_data.csv`` frame into the headline
    summary, Pareto, significance, Friedman/Nemenyi, and diagnostic CSVs.

    The function is also callable offline: any saved ``real_data.csv``
    can be re-aggregated by loading it into a frame and passing the
    sorted method list as ``methods``.

    Headline scoring is the Compute-Normalized Lift (CNL) over the
    linear baseline at ``alpha = 1``,

        cnl = max(0, max(0, r2_test) - max(0, r2_ols)) / (1 + fit_time_s)^alpha

    a bounded ``[0, 1]`` skill-over-OLS score (Murphy 1988 with the
    operational reference forecast set to OLS) discounted by per-cell
    wall time. A method that does not strictly out-predict the OLS
    baseline scores zero, removing the trivial ``copy OLS in zero
    time`` attack from contention; a method that reports a near-zero
    wall time cannot inflate its score because the denominator is
    bounded below by 1. Higher CNL is better. The OLS row itself has
    CNL identically zero by construction and is ranked at the bottom
    of the panel as the operational floor. Raw RMSE/MAE/R^2/NRMSE
    columns are still written to ``real_data_summary.csv`` for
    transparency, and ``_failure_modes`` operates on raw RMSE so the
    structural-failure diagnostic is not entangled with the compute
    weight.
    """
    from .baselines import METHOD_FAMILY

    # Per-cell Compute-Normalized Lift over OLS at the headline alpha=1.
    # The OLS reference R^2 is paired on (dataset, seed); cells that
    # have no matching OLS row score zero (which only triggers if the
    # linear method failed for that cell, an event already flagged via
    # ``used_constant_predictor``).
    df = df.copy()
    ols = _ols_r2_lookup(df)
    df = df.join(ols, on=["dataset", "seed"])
    df["r2_ols"] = df["r2_ols"].fillna(0.0)
    df["cnl"] = _cnl(df["r2_test"].to_numpy(),
                      df["r2_ols"].to_numpy(),
                      df["fit_time_s"].to_numpy(), alpha=1.0)
    valid = df

    # Per (method, dataset) aggregation. Median + IQR per metric is the
    # robust summary recommended by Demsar 2006 JMLR for cross-dataset
    # comparisons (mean is too sensitive to a single hard dataset). The
    # four base metrics correspond to the AMLB + Grinsztajn 2022 union:
    # RMSE, R^2, MAE, NRMSE. ``cnl_*`` is the headline Compute-Normalized
    # Lift over OLS at alpha = 1.
    def _q25(s): return float(s.quantile(0.25))
    def _q75(s): return float(s.quantile(0.75))
    summary = (df
        .groupby(["method", "dataset"], sort=False)
        .agg(rmse_median=("rmse_test", "median"),
             rmse_iqr_lo=("rmse_test", _q25),
             rmse_iqr_hi=("rmse_test", _q75),
             r2_median=("r2_test", "median"),
             r2_iqr_lo=("r2_test", _q25),
             r2_iqr_hi=("r2_test", _q75),
             mae_median=("mae_test", "median"),
             mae_iqr_lo=("mae_test", _q25),
             mae_iqr_hi=("mae_test", _q75),
             nrmse_median=("nrmse_test", "median"),
             nrmse_iqr_lo=("nrmse_test", _q25),
             nrmse_iqr_hi=("nrmse_test", _q75),
             cnl_median=("cnl", "median"),
             cnl_iqr_lo=("cnl", _q25),
             cnl_iqr_hi=("cnl", _q75),
             fit_time_median=("fit_time_s", "median"),
             fit_time_total=("fit_time_s", "sum"),
             n_fits_median=("n_fits", "median"),
             used_defaults_frac=("used_defaults", "mean"),
             used_constant_predictor_frac=("used_constant_predictor", "mean"),
             family=("family", "first"),
             n_seeds=("seed", "count"))
        .reset_index())
    summary.to_csv(OUT / "real_data_summary.csv", index=False)

    # Smooth-low-d subset definition matches the live driver: any dataset
    # whose recorded post-one-hot dimension is at most 30. Derived here
    # from the per-cell frame so the helper is self-contained and can
    # also be invoked offline from a saved real_data.csv.
    if "d" in valid.columns:
        subset_names = set(
            valid.loc[valid["d"].fillna(999).astype(int) <= 30, "dataset"].unique()
        )
    else:
        subset_names = set(valid["dataset"].unique())

    pivot = summary.pivot_table(index="dataset", columns="method",
                                values="rmse_median")
    fittime_pivot = summary.pivot_table(index="dataset", columns="method",
                                         values="fit_time_total")

    def _pareto(pivot_local, fittime_local, df_local):
        ref_col = "kore" if "kore" in pivot_local.columns else pivot_local.columns[0]
        ref = pivot_local[ref_col]
        rows_out = []
        for m in pivot_local.columns:
            ratio = (pivot_local[m] / ref).dropna()
            ratio = ratio[ratio > 0]
            if ratio.empty:
                continue
            gm = float(np.exp(np.log(ratio).mean()))
            total_time = float(df_local.loc[df_local["method"] == m,
                                            "fit_time_s"].sum())
            total_fits = int(df_local.loc[df_local["method"] == m,
                                           "n_fits"].sum())
            won = int((ratio < 1.0).sum())
            rows_out.append({
                "method": m,
                "family": METHOD_FAMILY.get(m, ""),
                "ref_method": ref_col,
                "gm_rmse_ratio": gm,
                "total_fit_time_s": total_time,
                "total_n_fits": total_fits,
                "n_datasets_won": won,
                "n_datasets_total": len(ratio),
            })
        return pd.DataFrame(rows_out)

    pareto_full = _pareto(pivot, fittime_pivot, valid)
    pareto_full.to_csv(OUT / "real_data_pareto.csv", index=False)

    subset_pivot = pivot.loc[pivot.index.isin(subset_names)]
    subset_fit = fittime_pivot.loc[fittime_pivot.index.isin(subset_names)]
    subset_df = valid[valid["dataset"].isin(subset_names)]
    pareto_subset = _pareto(subset_pivot, subset_fit, subset_df)
    pareto_subset.to_csv(OUT / "real_data_subset.csv", index=False)

    # Headline significance test: paired Wilcoxon signed-rank on the
    # per-cell Compute-Normalized Lift (CNL) at alpha = 1,
    #     delta = CNL(kore) - CNL(competitor)
    # against zero, paired across (dataset, seed) with the OLS R^2
    # from the same (dataset, seed) row used as the skill reference.
    # A positive median delta means KORE has a higher CNL (extracts
    # more OLS-relative lift per unit compute) on the typical cell;
    # a negative median delta means the competitor does. The
    # ``direction`` column flags methods whose Holm-corrected p-value
    # against zero falls below 0.05. The companion file
    # ``real_data_joint_significance.csv`` reports the same statistic
    # over a sensitivity sweep ``alpha in {0, 0.25, 0.5, 1, 2}``;
    # this file is the headline ``alpha = 1`` slice with the median
    # delta included in-line so the figure layer does not have to
    # recompute it.
    sig_rows = []
    try:
        from scipy.stats import wilcoxon

        if "kore" in valid["method"].values and not ols.empty:
            kore_idx = (valid[valid["method"] == "kore"]
                        .set_index(["dataset", "seed"]))[["r2_test", "fit_time_s"]]
            method_names = []
            p_raw = []
            med_delta = []
            n_obs_list = []
            for m in methods:
                if m == "kore":
                    continue
                if m not in valid["method"].values:
                    continue
                other = (valid[valid["method"] == m]
                         .set_index(["dataset", "seed"]))[["r2_test", "fit_time_s"]]
                common = (kore_idx.index
                          .intersection(other.index)
                          .intersection(ols.index))
                if len(common) < 10:
                    continue
                kr = kore_idx.loc[common, "r2_test"].to_numpy()
                kt = kore_idx.loc[common, "fit_time_s"].to_numpy()
                orr = other.loc[common, "r2_test"].to_numpy()
                ot = other.loc[common, "fit_time_s"].to_numpy()
                ref = ols.loc[common].to_numpy()
                mask = (np.isfinite(kr) & np.isfinite(orr)
                        & np.isfinite(kt) & np.isfinite(ot)
                        & np.isfinite(ref))
                if mask.sum() < 10:
                    continue
                cnl_k = _cnl(kr[mask], ref[mask], kt[mask], alpha=1.0)
                cnl_m = _cnl(orr[mask], ref[mask], ot[mask], alpha=1.0)
                delta = cnl_k - cnl_m
                try:
                    _stat, pval = wilcoxon(delta, zero_method="wilcox")
                except ValueError:
                    pval = np.nan
                method_names.append(m)
                p_raw.append(float(pval) if pval == pval else np.nan)
                med_delta.append(float(np.median(delta)))
                n_obs_list.append(int(mask.sum()))
            if p_raw:
                p_arr = np.array(p_raw)
                holm = _holm_bonferroni(np.where(np.isfinite(p_arr), p_arr, 1.0))
                for m, p, h, mu, n_obs in zip(
                        method_names, p_raw, holm, med_delta, n_obs_list):
                    if h < 0.05 and mu > 0.0:
                        direction = "kore_better"
                    elif h < 0.05 and mu < 0.0:
                        direction = "kore_worse"
                    else:
                        direction = "tied"
                    sig_rows.append({
                        "method": m,
                        "n_obs": int(n_obs),
                        "median_delta": float(mu),
                        "wilcoxon_p_raw": float(p),
                        "wilcoxon_p_holm": float(h),
                        "direction": direction,
                    })
    except ImportError:
        print("scipy.stats unavailable; skipping significance table.")

    pd.DataFrame(sig_rows).to_csv(OUT / "real_data_significance.csv", index=False)

    # Compute-Normalized Lift significance over a sensitivity sweep
    # ``alpha in {0, 0.25, 0.5, 1, 2}``; the alpha=0 row reduces to a
    # paired test on raw lift over OLS (no compute penalty) and the
    # alpha=1 row matches the headline test above. The full sweep
    # drives the sensitivity panel of ``fig_real_data_joint_significance``.
    joint_sig = _joint_significance(valid)
    joint_sig.to_csv(OUT / "real_data_joint_significance.csv", index=False)

    # Failure-mode table: per-dataset RMSE log-ratios of KORE vs the best
    # classical spline selector and vs the best tuned booster, filtered
    # to the union of the worst-five datasets in each comparison and
    # tagged with a short structural explanation. Operates on raw RMSE
    # so the diagnostic is not entangled with the compute weight.
    failure_modes = _failure_modes(valid)
    failure_modes.to_csv(OUT / "real_data_failure_modes.csv", index=False)

    # Threshold sensitivity: how the headline KORE statistics shift when
    # the smooth-low-d cutoff ``max_d`` is swept over {20, 30, 40, 50}.
    # Reports both raw-RMSE and CNL ranks so a reviewer can audit each
    # axis independently of the compute weight.
    threshold_sens = _threshold_sensitivity(valid)
    threshold_sens.to_csv(OUT / "real_data_threshold_sensitivity.csv", index=False)

    # Friedman omnibus + Nemenyi post-hoc per Demsar 2006 JMLR on the
    # Compute-Normalized Lift. Per (dataset, method) the score is the
    # median per-cell ``cnl = max(0, max(0, r2) - max(0, r2_ols)) /
    # (1 + fit_time_s)``; higher CNL is better, so the rank is taken on
    # ``-cnl_median`` and a low mean rank means the method extracts
    # the most OLS-relative lift per unit compute on the typical cell.
    # OLS itself has CNL identically zero by construction and lands at
    # the floor of the ranking. Methods missing any dataset are dropped
    # from the rank table; the dropped list is recorded for
    # transparency. The CD value lets the figure layer draw the
    # canonical critical-difference diagram without re-running scipy.
    import json as _json
    rank_csv_path = OUT / "real_data_ranks.csv"
    fried_json_path = OUT / "real_data_friedman.json"
    try:
        score_pivot = summary.pivot_table(
            index="dataset", columns="method", values="cnl_median"
        )
        complete_methods = [m for m in score_pivot.columns
                            if score_pivot[m].notna().all()]
        dropped = [m for m in score_pivot.columns if m not in complete_methods]
        # Higher CNL is better; pass ``-CNL`` so low rank = high score.
        fried = _friedman_nemenyi(-score_pivot[complete_methods])
        fried["dropped_methods"] = dropped
        fried["score_metric"] = (
            "cnl_median (max(0, max(0, r2_test) - max(0, r2_ols)) "
            "/ (1 + fit_time_s), alpha=1)"
        )
        with open(fried_json_path, "w") as fh:
            _json.dump(fried, fh, indent=2, sort_keys=True)
        rank_rows = [
            {"method": m,
             "family": METHOD_FAMILY.get(m, ""),
             "mean_rank": fried["mean_ranks"].get(m, float("nan")),
             "n_datasets": fried["n_datasets"],
             "k_methods": fried["k_methods"]}
            for m in complete_methods
        ]
        pd.DataFrame(rank_rows).sort_values("mean_rank").to_csv(
            rank_csv_path, index=False
        )
    except ImportError:
        print("scipy.stats unavailable; skipping Friedman/Nemenyi.")

    # Bootstrap confidence intervals over the dataset axis for the
    # Friedman per-method mean ranks. Reuses the per-dataset rank table
    # built above when available; otherwise reconstructs it from the
    # cnl_median pivot following the same ``-cnl`` convention so a low
    # rank still means higher OLS-relative lift per unit compute.
    try:
        try:
            rank_pivot = (-score_pivot[complete_methods]).rank(
                axis=1, method="average"
            )
        except NameError:
            cnl_pivot = summary.pivot_table(
                index="dataset", columns="method", values="cnl_median"
            )
            cm = [m for m in cnl_pivot.columns if cnl_pivot[m].notna().all()]
            rank_pivot = (-cnl_pivot[cm]).rank(axis=1, method="average")
        boot_ranks = _bootstrap_ranks(rank_pivot, n_boot=1000, seed=2026)
        boot_ranks.to_csv(OUT / "real_data_bootstrap_ranks.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Bootstrap rank CIs skipped: {exc}")

    try:
        meta = _dataset_metadata_table()
        meta.to_csv(OUT / "dataset_metadata.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Dataset metadata table skipped: {exc}")

    try:
        fail_frac = _per_method_failure_fractions(df)
        fail_frac.to_csv(OUT / "method_failure_fractions.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Method failure fractions skipped: {exc}")

    print("\nReal-world benchmark complete.")
    print(f"  per-cell rows : {len(df)}")
    print(f"  per-method-per-dataset rows : {len(summary)}")
    if len(pareto_full):
        print(f"  Pareto rows : {len(pareto_full)}")
    if len(pareto_subset):
        print(f"  subset rows : {len(pareto_subset)}")
    if rank_csv_path.exists():
        print(f"  Friedman/Nemenyi (CNL) : "
              f"{rank_csv_path.name}, {fried_json_path.name}")


def run_real_data_chunked(*, n_seeds: int = 5, max_n: int = 2500,
                          chunks: list[list[str]] | None = None,
                          python_executable: str | None = None,
                          per_chunk_env: dict[str, str] | None = None,
                          log_dir: str | None = None,
                          final_aggregate: bool = True):
    """Run the real-data sweep as serial subprocess-per-chunk.

    Each chunk is a list of method names. The orchestrator launches one
    Python subprocess per chunk, invoking
    ``run_real_data(methods=chunk, n_seeds=n_seeds, max_n=max_n,
    progress=False)`` inside that subprocess. Chunks run serially; the
    next subprocess starts only after the previous one has exited and
    the OS has reclaimed its RSS. The shared ``real_data.csv``
    checkpoint dedups across chunks via the existing
    ``(dataset, method, seed)`` resume logic. A chunk's non-zero exit
    code is logged and the orchestrator continues to the next chunk.

    After every chunk completes, the orchestrator launches one final
    aggregation subprocess that invokes ``run_real_data(methods=None,
    ...)``. The resume logic drops ``n_pending`` to zero (every cell is
    already in ``real_data.csv``) and only the aggregation block runs.
    This rebuilds ``real_data_significance.csv`` and
    ``real_data_joint_significance.csv`` over the full method set; per-
    chunk subprocesses only wrote those two tables for their own
    methods. The other aggregation outputs (summary, Pareto, Friedman,
    ranks, failure modes, threshold sensitivity) are already complete
    after the last chunk because they iterate over every method present
    in the CSV.

    Parameters
    ----------
    n_seeds, max_n :
        Forwarded to each subprocess's ``run_real_data`` call.
    chunks :
        Optional override of the default chunk plan. Each inner list is
        a set of method names that share one subprocess. Default plan:
        one chunk per heavy / medium-weight method, one bundled chunk
        for all light-weight methods.
    python_executable :
        Defaults to ``sys.executable`` (the venv Python that has
        ``kore`` importable). Override only for cross-environment
        dispatch.
    per_chunk_env :
        Optional env-var dict merged into each subprocess's environment
        on top of the parent environment. Defaults to the per-stage
        worker-count + per-worker-GB profile used for the real-data sweep.
    log_dir :
        Directory for per-chunk stdout/stderr logs. Defaults to
        ``OUT / "chunk_logs"``.
    final_aggregate :
        When True (default), launch one final ``run_real_data(methods=
        None)`` subprocess after every chunk has exited so the two
        Wilcoxon tables are rebuilt over every available method. Set to
        False when chaining multiple ``run_real_data_chunked`` calls
        (rebuild aggregation only once at the end of the outer driver).
    """
    import subprocess
    default_chunks = [
        ["catboost"], ["xgboost"], ["lightgbm"], ["hist_gbm"],
        ["mlp"], ["svr_rbf"], ["kernel_ridge_rbf"],
        ["random_forest"], ["extra_trees"], ["pygam"],
        ["knn", "kore", "cv_spline", "gcv_spline", "cp_spline",
         "aic_spline", "bic_spline", "linear", "ridge_cv",
         "lasso_cv", "elasticnet_cv"],
    ]
    chunks = chunks if chunks is not None else default_chunks
    py = python_executable or sys.executable
    env_overrides = per_chunk_env if per_chunk_env is not None else {
        "KORE_NJOBS_HEAVY":   "80",
        "KORE_NJOBS_MEDIUM":  "200",
        "KORE_NJOBS_LIGHT":   "200",
        "KORE_PER_WORKER_GB": "6",
        "KORE_BOOSTER_NJOBS": "4",
    }
    log_root = Path(log_dir) if log_dir else (OUT / "chunk_logs")
    log_root.mkdir(parents=True, exist_ok=True)

    def _run_subprocess(cmd, log_path, prefix):
        env = os.environ.copy()
        env.update(env_overrides)
        env.pop("KORE_LIGHT_BACKEND", None)
        env.setdefault("PYTHONUNBUFFERED", "1")
        print(f"{prefix}: starting (log: {log_path})", flush=True)
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1, text=True,
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(f"{prefix} | {line}"
                                     if line.endswith("\n")
                                     else f"{prefix} | {line}\n")
                    sys.stdout.flush()
                return proc.wait()
            finally:
                if proc.stdout is not None:
                    proc.stdout.close()

    print("=" * 78)
    print(f"Real-data subprocess-chunked sweep: {len(chunks)} chunks, "
          f"n_seeds={n_seeds}, max_n={max_n}")
    print("=" * 78)
    failures: list[tuple[int, list[str], int]] = []
    t_start = time.perf_counter()
    for idx, methods in enumerate(chunks, start=1):
        tag = methods[0] if len(methods) == 1 else f"bundle_{len(methods)}"
        log_path = log_root / f"chunk_{idx:02d}_{tag}.log"
        cmd = [py, "-u", "-c",
               "from kore.experiments import run_real_data; "
               f"run_real_data(methods={methods!r}, n_seeds={int(n_seeds)}, "
               f"max_n={int(max_n)}, progress=True)"]
        t0 = time.perf_counter()
        prefix = f"  [{idx:02d}/{len(chunks)}] {tag}"
        rc = _run_subprocess(cmd, log_path, prefix)
        dt = time.perf_counter() - t0
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"{prefix}: {status} in {dt / 60:.2f} min", flush=True)
        if rc != 0:
            failures.append((idx, list(methods), int(rc)))
    total_min = (time.perf_counter() - t_start) / 60
    print(f"\nAll chunks done in {total_min:.1f} min. "
          f"Failed: {len(failures)}/{len(chunks)}")
    for idx, methods, rc in failures:
        print(f"  [{idx:02d}] {methods} rc={rc}")

    agg_rc: int | None = None
    if final_aggregate:
        agg_log = log_root / "chunk_FINAL_aggregate.log"
        agg_cmd = [py, "-u", "-c",
                   "from kore.experiments import run_real_data; "
                   f"run_real_data(n_seeds={int(n_seeds)}, "
                   f"max_n={int(max_n)}, progress=False)"]
        prefix = "  [FINAL] aggregate"
        t_agg = time.perf_counter()
        agg_rc = _run_subprocess(agg_cmd, agg_log, prefix)
        dt_agg = time.perf_counter() - t_agg
        status = "OK" if agg_rc == 0 else f"FAIL rc={agg_rc}"
        print(f"{prefix}: {status} in {dt_agg / 60:.2f} min", flush=True)
    return {"chunks": len(chunks), "failed": failures,
            "elapsed_min": total_min,
            "final_aggregate_rc": agg_rc}


# =====================================================================
EXPERIMENTS = {
    "law":          run_law_collapse,
    "frontier":     run_frontier,
    "benchmark":    run_benchmark,
    "discovery":    run_discovery,
    "robustness":   run_robustness,
    "scaling":      run_scaling,
    "noise":        run_noise_sweep,
    "degree":       run_degree_ablation,
    "consistency":  run_plugin_consistency,
    "real_data":    run_real_data,
    "real_data_chunked": run_real_data_chunked,
}

if __name__ == "__main__":
    modes = sys.argv[1:] or ["all"]
    if "all" in modes:
        modes = list(EXPERIMENTS.keys())

    print(f"\n{'='*78}")
    print(f"  KORE -- Comprehensive Experiments")
    print(f"  MASTER_SEED = {MASTER_SEED}    N_JOBS = {N_JOBS}")
    print(f"  Running: {', '.join(modes)}")
    print(f"{'='*78}\n")

    t0 = time.perf_counter()
    for mode in modes:
        if mode not in EXPERIMENTS:
            print(f"Unknown: {mode}.  Choose from {', '.join(EXPERIMENTS)}, all")
            sys.exit(1)
        EXPERIMENTS[mode]()

    elapsed = time.perf_counter() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"Total wall-clock: {m}m {s}s  ({elapsed:.1f}s)")
