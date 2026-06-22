"""Unit tests for the per-cell RSS instrumentation in ``_real_data_one_cell``.

Validates that every cell record carries ``host``, ``pid``, ``rss_start_mb``,
``rss_peak_mb``, and ``rss_end_mb``, that the daemon sampler thread is
started and stopped exactly once per cell, that a failing cell still
records the RSS keys via the constant-predictor fallback, and that the
function degrades gracefully when ``psutil`` is unavailable.
"""
from __future__ import annotations

import math
import threading
from unittest.mock import patch

import numpy as np
import pytest

from kore import experiments as exp


class _StubSpec:
    name = "stub_dataset"


def _xy(n: int = 60, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X.sum(axis=1) + rng.normal(scale=0.1, size=n)
    return X, y


def _stub_methods(monkeypatch, *, fail: bool = False, slow: bool = False):
    """Replace the baselines registry so the cell does not import sklearn.

    ``slow=True`` makes the stub fit sleep in short interruptible chunks
    so the daemon RSS sampler has time to sample at least once and
    deliver an in-process signal before the fit returns.
    """
    class _Out:
        n_fits = 1
        fit_time_s = 0.001
        best_hp = {}
        used_defaults = False

        def __init__(self, mean):
            self._mean = mean

        def predict(self, X):
            return np.full(X.shape[0], self._mean, dtype=float)

    def _good_fit(X, y, seed):
        return _Out(float(np.mean(y)))

    def _slow_fit(X, y, seed):  # noqa: ARG001
        import time
        for _ in range(60):
            time.sleep(0.05)
        return _Out(float(np.mean(y)))

    def _bad_fit(X, y, seed):  # noqa: ARG001
        raise RuntimeError("synthetic failure")

    if fail:
        fit = _bad_fit
    elif slow:
        fit = _slow_fit
    else:
        fit = _good_fit
    methods = {"stub_method": fit}
    family = {"stub_method": "stub_family"}
    from kore import baselines
    monkeypatch.setattr(baselines, "METHODS", methods, raising=True)
    monkeypatch.setattr(baselines, "METHOD_FAMILY", family, raising=True)


def test_one_cell_returns_rss_keys(monkeypatch):
    _stub_methods(monkeypatch)
    X, y = _xy()
    row = exp._real_data_one_cell(_StubSpec(), X, y, "stub_method", 0)

    assert isinstance(row["pid"], int) and row["pid"] > 0
    assert isinstance(row["host"], str) and row["host"]
    for key in ("rss_start_mb", "rss_peak_mb", "rss_end_mb"):
        val = row[key]
        if exp._PSUTIL_AVAILABLE:
            assert isinstance(val, float) and math.isfinite(val) and val >= 0.0
        else:
            assert math.isnan(val)
    assert row["used_constant_predictor"] is False
    assert row["error"] == ""


def test_rss_sampler_starts_and_stops(monkeypatch):
    if not exp._PSUTIL_AVAILABLE:
        pytest.skip("psutil unavailable; sampler thread is not constructed")
    _stub_methods(monkeypatch)
    X, y = _xy()

    # Record only the (Thread, Event) pair that ``_real_data_one_cell``
    # constructs for the RSS sampler; spy on those instance methods so
    # unrelated Thread/Event objects created elsewhere by joblib, psutil,
    # or the interpreter's own machinery do not inflate the counts.
    real_thread_cls = threading.Thread
    real_event_cls = threading.Event
    instances = {"thread": None, "event": None}
    starts = {"n": 0}
    sets = {"n": 0}

    class _SpyThread(real_thread_cls):
        def __init__(self, *a, target=None, **kw):
            super().__init__(*a, target=target, **kw)
            if target is exp._sample_peak_rss:
                instances["thread"] = self

        def start(self):
            if instances["thread"] is self:
                starts["n"] += 1
            super().start()

    class _SpyEvent(real_event_cls):
        def __init__(self):
            super().__init__()
            if instances["event"] is None:
                instances["event"] = self

        def set(self):
            if instances["event"] is self:
                sets["n"] += 1
            super().set()

    monkeypatch.setattr(threading, "Thread", _SpyThread, raising=True)
    monkeypatch.setattr(threading, "Event", _SpyEvent, raising=True)

    row = exp._real_data_one_cell(_StubSpec(), X, y, "stub_method", 0)
    assert starts["n"] == 1, "sampler thread was not started exactly once"
    assert sets["n"] == 1, "sampler stop_event was not set exactly once"
    assert row["error"] == ""


def test_failing_cell_still_records_rss(monkeypatch):
    _stub_methods(monkeypatch, fail=True)
    X, y = _xy()
    row = exp._real_data_one_cell(_StubSpec(), X, y, "stub_method", 0)

    assert row["used_constant_predictor"] is True
    assert "synthetic failure" in row["error"]
    for key in ("rss_start_mb", "rss_peak_mb", "rss_end_mb"):
        val = row[key]
        if exp._PSUTIL_AVAILABLE:
            assert isinstance(val, float) and math.isfinite(val) and val >= 0.0
        else:
            assert math.isnan(val)
    assert isinstance(row["pid"], int) and row["pid"] > 0
    # Constant predictor reduces to mean(y_train); RMSE is finite and the
    # row must carry every standard metric so the aggregator never KeyErrors.
    for key in ("rmse_test", "mae_test"):
        assert math.isfinite(row[key])


def test_psutil_unavailable_graceful_degradation(monkeypatch):
    _stub_methods(monkeypatch)
    monkeypatch.setattr(exp, "_PSUTIL_AVAILABLE", False, raising=True)
    X, y = _xy()
    row = exp._real_data_one_cell(_StubSpec(), X, y, "stub_method", 0)

    assert math.isnan(row["rss_start_mb"])
    assert math.isnan(row["rss_end_mb"])
    # ``peak_holder`` is initialized to 0.0 when psutil is unavailable;
    # the function must still return a finite (or NaN) numeric value
    # without raising. The exact value (0.0 or NaN) is an implementation
    # detail of the sampler-disabled path.
    assert isinstance(row["rss_peak_mb"], float)
    assert row["error"] == ""


def test_advisory_threshold_aborts_cell_without_killing_worker(monkeypatch):
    """Crossing the advisory threshold aborts the cell, not the worker.

    A prior revision sent ``SIGTERM`` to the worker process when its
    RSS exceeded the per-cell cap; that converted transient peaks
    (e.g. inside ``kore_intrinsic`` pairwise structure search) into
    loky ``TerminatedWorkerError`` events and took down whole stages.
    The current implementation uses ``_thread.interrupt_main()`` to
    raise ``KeyboardInterrupt`` on the cell's main thread instead, so
    the worker process is never signalled, loky never sees a worker
    death, and the next cell on the same worker runs normally. This
    test forces the advisory to fire by setting the threshold to 1
    MiB, then asserts (a) the cell falls back to the constant
    predictor with an ``rss advisory exceeded`` error, (b) the
    advisory flag is set on the row, and (c) ``os.kill`` was never
    invoked.
    """
    if not exp._PSUTIL_AVAILABLE:
        pytest.skip("psutil unavailable; sampler thread is not constructed")
    _stub_methods(monkeypatch, slow=True)
    monkeypatch.setenv("KORE_CELL_RSS_ADVISORY_MB", "1")
    kills = {"n": 0}
    real_kill = exp.os.kill
    def _spy_kill(pid, sig):  # noqa: ARG001
        kills["n"] += 1
        return real_kill(pid, sig)
    monkeypatch.setattr(exp.os, "kill", _spy_kill, raising=True)
    X, y = _xy()
    row = exp._real_data_one_cell(_StubSpec(), X, y, "stub_method", 0)
    assert kills["n"] == 0, "advisory threshold must not signal the worker process"
    assert row["used_constant_predictor"] is True
    assert row.get("rss_advisory_exceeded") is True
    assert row["error"].startswith("rss advisory exceeded"), row["error"]


def test_advisory_threshold_default_not_exceeded(monkeypatch):
    """At the default 8 GiB advisory, a normal stub cell does not trip."""
    if not exp._PSUTIL_AVAILABLE:
        pytest.skip("psutil unavailable; sampler thread is not constructed")
    _stub_methods(monkeypatch)
    monkeypatch.delenv("KORE_CELL_RSS_ADVISORY_MB", raising=False)
    monkeypatch.delenv("KORE_CELL_RSS_CAP_MB", raising=False)
    X, y = _xy()
    row = exp._real_data_one_cell(_StubSpec(), X, y, "stub_method", 0)
    assert row.get("rss_advisory_exceeded") is False
