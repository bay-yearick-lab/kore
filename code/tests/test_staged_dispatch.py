"""Unit tests for the staged loky dispatch in ``kore.experiments``.

Validates the heavy/medium/light bucketing in ``_dispatch_real_data_staged``,
the per-stage ``TerminatedWorkerError`` guard, the env-var resolution in
``_staged_njobs_from_env``, the ``_local_worker_cap`` clipping, and the
empty-bucket short-circuit. ``joblib.Parallel`` is patched so the tests
do not actually fork workers.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from kore import experiments as exp


def _make_cell(method: str, name: str = "ds") -> dict:
    return {"method": method, "spec": None, "X": None, "y": None,
            "seed": 0, "_name": f"{method}-{name}"}


class _FakeParallel:
    """Stand-in for ``joblib.Parallel`` that records cells per stage."""

    captured: list = []

    def __init__(self, *_, n_jobs=1, max_tasks_per_child=None, **__):
        self.n_jobs = n_jobs
        self.max_tasks_per_child = max_tasks_per_child

    def __call__(self, gen):
        # Materialize the generator (it yields ``delayed(fn)(c)`` thunks
        # whose first positional argument is the cell). The ``_FakeParallel``
        # implementation just runs ``fn(c)`` inline and records the cell.
        rows = []
        cells_this_stage = []
        for thunk in gen:
            fn, args, _kwargs = thunk
            cell = args[0]
            cells_this_stage.append(cell)
            rows.append(fn(cell))
        _FakeParallel.captured.append({
            "n_jobs": self.n_jobs,
            "max_tasks_per_child": self.max_tasks_per_child,
            "cells": cells_this_stage,
        })
        return iter(rows)


@pytest.fixture(autouse=True)
def _reset_capture():
    _FakeParallel.captured = []
    yield
    _FakeParallel.captured = []


def _patch_joblib(monkeypatch):
    import joblib
    monkeypatch.setattr(joblib, "Parallel", _FakeParallel, raising=True)


def test_partition_by_weight(monkeypatch, caplog):
    _patch_joblib(monkeypatch)
    monkeypatch.setattr(
        exp, "_METHOD_WEIGHT", {"a": 0.5, "b": 2.5, "c": 5.0}, raising=True
    )
    cells = [_make_cell(m, str(i)) for m in ("a", "b", "c") for i in range(2)]
    collected: list = []
    with caplog.at_level(logging.INFO, logger="kore.test"):
        exp._dispatch_real_data_staged(
            cells=cells,
            fn=lambda c: {"method": c["method"], "row": c["_name"]},
            n_jobs_heavy=1, n_jobs_medium=1, n_jobs_light=1,
            on_result=collected.append,
            logger=logging.getLogger("kore.test"),
        )
    assert len(_FakeParallel.captured) == 3
    methods_per_stage = [
        sorted(c["method"] for c in stage["cells"])
        for stage in _FakeParallel.captured
    ]
    assert methods_per_stage == [["c", "c"], ["b", "b"], ["a", "a"]]
    assert sorted(r["method"] for r in collected) == ["a", "a", "b", "b", "c", "c"]


def test_terminated_worker_in_one_bucket_does_not_abort_others(monkeypatch, caplog):
    monkeypatch.setattr(
        exp, "_METHOD_WEIGHT", {"a": 0.5, "b": 2.5, "c": 5.0}, raising=True
    )
    from joblib.externals.loky.process_executor import TerminatedWorkerError

    call_count = {"n": 0}

    class _RaisingParallel(_FakeParallel):
        def __call__(self, gen):
            call_count["n"] += 1
            if call_count["n"] == 1:  # the heavy bucket
                _FakeParallel.captured.append({"n_jobs": self.n_jobs, "cells": []})
                raise TerminatedWorkerError("heavy worker SIGKILL'd")
            return super().__call__(gen)

    import joblib
    monkeypatch.setattr(joblib, "Parallel", _RaisingParallel, raising=True)
    cells = [_make_cell(m, str(i)) for m in ("a", "b", "c") for i in range(2)]
    collected: list = []
    with caplog.at_level(logging.ERROR, logger="kore.test"):
        exp._dispatch_real_data_staged(
            cells=cells, fn=lambda c: {"method": c["method"]},
            n_jobs_heavy=1, n_jobs_medium=1, n_jobs_light=1,
            on_result=collected.append,
            logger=logging.getLogger("kore.test"),
        )
    assert call_count["n"] == 3
    assert sorted(r["method"] for r in collected) == ["a", "a", "b", "b"]
    assert any("TerminatedWorkerError" in rec.getMessage() for rec in caplog.records)


def test_env_vars_honored(monkeypatch):
    monkeypatch.setattr(exp, "_local_worker_cap", lambda: 64, raising=True)
    monkeypatch.setenv("KORE_NJOBS_HEAVY", "2")
    monkeypatch.setenv("KORE_NJOBS_MEDIUM", "8")
    monkeypatch.setenv("KORE_NJOBS_LIGHT", "32")
    assert exp._staged_njobs_from_env() == (2, 8, 32)


def test_njobs_capped_at_local_worker_cap(monkeypatch):
    monkeypatch.setattr(exp, "_local_worker_cap", lambda: 4, raising=True)
    monkeypatch.delenv("KORE_NJOBS_HEAVY", raising=False)
    monkeypatch.delenv("KORE_NJOBS_MEDIUM", raising=False)
    monkeypatch.setenv("KORE_NJOBS_LIGHT", "999")
    assert exp._staged_njobs_from_env() == (4, 4, 4)


def test_empty_bucket_skipped(monkeypatch, caplog):
    _patch_joblib(monkeypatch)
    monkeypatch.setattr(
        exp, "_METHOD_WEIGHT", {"a": 0.5, "b": 2.5, "c": 5.0}, raising=True
    )
    cells = [_make_cell("a", str(i)) for i in range(3)]  # light only
    with caplog.at_level(logging.INFO, logger="kore.test"):
        exp._dispatch_real_data_staged(
            cells=cells, fn=lambda c: {"method": c["method"]},
            n_jobs_heavy=1, n_jobs_medium=1, n_jobs_light=1,
            on_result=lambda _: None,
            logger=logging.getLogger("kore.test"),
        )
    assert len(_FakeParallel.captured) == 1
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("stage=light" in m for m in msgs)
    assert not any("stage=heavy" in m for m in msgs)
    assert not any("stage=medium" in m for m in msgs)


def test_threading_env_var_ignored(monkeypatch, caplog):
    """KORE_LIGHT_BACKEND is no longer read; LIGHT stage must use loky."""
    _patch_joblib(monkeypatch)
    monkeypatch.setenv("KORE_LIGHT_BACKEND", "threading")
    monkeypatch.setattr(
        exp, "_METHOD_WEIGHT", {"a": 0.5, "b": 2.5, "c": 5.0}, raising=True
    )
    cells = [_make_cell(m, str(i)) for m in ("a", "b", "c") for i in range(1)]
    with caplog.at_level(logging.INFO, logger="kore.test"):
        exp._dispatch_real_data_staged(
            cells=cells,
            fn=lambda c: {"method": c["method"]},
            n_jobs_heavy=1, n_jobs_medium=1, n_jobs_light=1,
            on_result=lambda _: None,
            logger=logging.getLogger("kore.test"),
        )
    msgs = [rec.getMessage() for rec in caplog.records]
    light_msgs = [m for m in msgs if "stage=light" in m]
    assert light_msgs, "expected a stage=light log line"
    assert all("backend=loky" in m for m in light_msgs)
    assert not any("backend=threading" in m for m in msgs)


def test_stage_event_callback_called_per_stage(monkeypatch):
    """``on_stage_event`` must fire once per non-empty bucket."""
    _patch_joblib(monkeypatch)
    monkeypatch.setattr(
        exp, "_METHOD_WEIGHT", {"a": 0.5, "b": 2.5, "c": 5.0}, raising=True
    )
    cells = [_make_cell(m, str(i)) for m in ("a", "b", "c") for i in range(2)]
    events: list = []
    exp._dispatch_real_data_staged(
        cells=cells,
        fn=lambda c: {"method": c["method"]},
        n_jobs_heavy=1, n_jobs_medium=1, n_jobs_light=1,
        on_result=lambda _: None,
        logger=logging.getLogger("kore.test"),
        on_stage_event=events.append,
    )
    stages = [e["stage"] for e in events]
    assert stages == ["heavy", "medium", "light"]
    for e in events:
        assert e["n_cells_dispatched"] == 2
        assert e["n_completed"] == 2
        assert e["n_sentineled"] == 0
        assert e["exception_class"] == ""
        assert e["wall_time_s"] >= 0.0
        assert e["backend"] == "loky"


def test_sentinel_rows_emitted_on_terminated_worker(monkeypatch):
    """When a stage raises ``TerminatedWorkerError`` before any row
    returns, every dispatched cell with a real spec must show up as a
    constant-predictor sentinel row through ``on_result`` so the
    aggregator never sees a stage as 'never ran'."""
    from joblib.externals.loky.process_executor import TerminatedWorkerError
    monkeypatch.setattr(
        exp, "_METHOD_WEIGHT", {"a": 5.0}, raising=True
    )

    class _AlwaysRaises(_FakeParallel):
        def __call__(self, gen):
            _FakeParallel.captured.append({"n_jobs": self.n_jobs, "cells": []})
            raise TerminatedWorkerError("forced")

    import joblib
    monkeypatch.setattr(joblib, "Parallel", _AlwaysRaises, raising=True)

    import numpy as np
    class _Spec:
        def __init__(self, name): self.name = name
    cells = []
    for i in range(3):
        rng = np.random.default_rng(i)
        X = rng.normal(size=(40, 2))
        y = X.sum(axis=1) + rng.normal(scale=0.1, size=40)
        cells.append({
            "spec": _Spec(f"ds_{i}"), "X": X, "y": y,
            "method": "a", "seed": i,
        })

    # Stub METHOD_FAMILY for the sentinel-row family lookup.
    from kore import baselines
    monkeypatch.setattr(baselines, "METHOD_FAMILY", {"a": "fam"}, raising=True)

    collected: list = []
    events: list = []
    exp._dispatch_real_data_staged(
        cells=cells,
        fn=lambda c: {"method": c["method"]},
        n_jobs_heavy=1, n_jobs_medium=1, n_jobs_light=1,
        on_result=collected.append,
        logger=logging.getLogger("kore.test"),
        on_stage_event=events.append,
    )
    assert len(collected) == 3, "expected one sentinel per dispatched cell"
    for row in collected:
        assert row["used_constant_predictor"] is True
        assert row["error"].startswith("stage_terminatedworkererror")
        assert row["method"] == "a"
        assert row["family"] == "fam"
        assert row["dataset"].startswith("ds_")
    assert events and events[0]["exception_class"] == "TerminatedWorkerError"
    assert events[0]["n_sentineled"] == 3
    assert events[0]["n_completed"] == 0


def test_max_tasks_per_child_propagated(monkeypatch):
    """Defaults: heavy=8, medium=16, light=32; env overrides honored."""
    _patch_joblib(monkeypatch)
    monkeypatch.setattr(
        exp, "_METHOD_WEIGHT", {"a": 0.5, "b": 2.5, "c": 5.0}, raising=True
    )
    monkeypatch.delenv("KORE_MAX_TASKS_HEAVY",  raising=False)
    monkeypatch.delenv("KORE_MAX_TASKS_MEDIUM", raising=False)
    monkeypatch.delenv("KORE_MAX_TASKS_LIGHT",  raising=False)
    cells = [_make_cell(m, str(i)) for m in ("a", "b", "c") for i in range(1)]
    exp._dispatch_real_data_staged(
        cells=cells, fn=lambda c: {"method": c["method"]},
        n_jobs_heavy=1, n_jobs_medium=1, n_jobs_light=1,
        on_result=lambda _: None,
        logger=logging.getLogger("kore.test"),
    )
    # Stage order in the dispatcher is heavy, medium, light.
    captured = _FakeParallel.captured
    assert [c["max_tasks_per_child"] for c in captured] == [8, 16, 32]

    # Env overrides take effect.
    _FakeParallel.captured = []
    monkeypatch.setenv("KORE_MAX_TASKS_HEAVY",  "3")
    monkeypatch.setenv("KORE_MAX_TASKS_MEDIUM", "5")
    monkeypatch.setenv("KORE_MAX_TASKS_LIGHT",  "7")
    exp._dispatch_real_data_staged(
        cells=cells, fn=lambda c: {"method": c["method"]},
        n_jobs_heavy=1, n_jobs_medium=1, n_jobs_light=1,
        on_result=lambda _: None,
        logger=logging.getLogger("kore.test"),
    )
    assert [c["max_tasks_per_child"] for c in _FakeParallel.captured] == [3, 5, 7]
