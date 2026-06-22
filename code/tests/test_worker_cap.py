"""Unit tests for the loky worker-cap logic in ``kore.experiments``.

Validates the cgroup-aware memory reader (v2 and v1), the env-var
overrides (``KORE_PER_WORKER_GB`` and ``KORE_MAX_WORKERS``), and the
banner diagnostic string. Mocks every filesystem and OS call so the
tests are deterministic on any host (macOS dev, Linux CI, Databricks).
"""
from __future__ import annotations

import pytest

from kore import experiments as exp


_GiB = 1024 ** 3


# ── env-var overrides ─────────────────────────────────────────────────
def test_per_worker_gb_default_when_unset(monkeypatch):
    monkeypatch.delenv("KORE_PER_WORKER_GB", raising=False)
    assert exp._per_worker_gb() == exp._PER_WORKER_GB


@pytest.mark.parametrize("raw,expected", [("1.0", 1.0), ("8", 8.0), ("0.5", 0.5)])
def test_per_worker_gb_honors_env(monkeypatch, raw, expected):
    monkeypatch.setenv("KORE_PER_WORKER_GB", raw)
    assert exp._per_worker_gb() == expected


@pytest.mark.parametrize("bad", ["", "garbage", "-2.0", "0"])
def test_per_worker_gb_rejects_invalid_env(monkeypatch, bad):
    monkeypatch.setenv("KORE_PER_WORKER_GB", bad)
    assert exp._per_worker_gb() == exp._PER_WORKER_GB


def test_max_workers_env_default_is_none(monkeypatch):
    monkeypatch.delenv("KORE_MAX_WORKERS", raising=False)
    assert exp._max_workers_env() is None


@pytest.mark.parametrize("raw,expected", [("1", 1), ("32", 32), ("324", 324)])
def test_max_workers_env_honors_env(monkeypatch, raw, expected):
    monkeypatch.setenv("KORE_MAX_WORKERS", raw)
    assert exp._max_workers_env() == expected


@pytest.mark.parametrize("bad", ["", "garbage", "-4", "0"])
def test_max_workers_env_rejects_invalid(monkeypatch, bad):
    monkeypatch.setenv("KORE_MAX_WORKERS", bad)
    assert exp._max_workers_env() is None


# ── cgroup readers ────────────────────────────────────────────────────
def _patch_files(monkeypatch, mapping):
    """Patch ``open`` so paths in ``mapping`` return the given text."""
    real_open = open

    def fake_open(path, *args, **kwargs):
        path_s = str(path)
        if path_s in mapping:
            content = mapping[path_s]
            if content is FileNotFoundError:
                raise FileNotFoundError(path_s)
            from io import StringIO
            return StringIO(content)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)


def test_cgroup_v2_capped_returns_limit_and_free(monkeypatch):
    _patch_files(monkeypatch, {
        "/sys/fs/cgroup/memory.max":     str(64 * _GiB),
        "/sys/fs/cgroup/memory.current": str(8 * _GiB),
    })
    limit_gb, free_gb = exp._cgroup_memory_gb()
    assert limit_gb == pytest.approx(64.0)
    assert free_gb == pytest.approx(56.0)


def test_cgroup_v2_unlimited_max_returns_none(monkeypatch):
    _patch_files(monkeypatch, {"/sys/fs/cgroup/memory.max": "max"})
    assert exp._cgroup_memory_gb() == (None, None)


def test_cgroup_v2_unlimited_huge_value_returns_none(monkeypatch):
    _patch_files(monkeypatch, {
        "/sys/fs/cgroup/memory.max": str(1 << 63),  # > _CGROUP_UNLIMITED_BYTES
    })
    assert exp._cgroup_memory_gb() == (None, None)


def test_cgroup_v1_capped_returns_limit_and_free(monkeypatch):
    _patch_files(monkeypatch, {
        "/sys/fs/cgroup/memory.max": FileNotFoundError,
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(128 * _GiB),
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": str(16 * _GiB),
    })
    limit_gb, free_gb = exp._cgroup_memory_gb()
    assert limit_gb == pytest.approx(128.0)
    assert free_gb == pytest.approx(112.0)


def test_cgroup_absent_returns_none(monkeypatch):
    _patch_files(monkeypatch, {
        "/sys/fs/cgroup/memory.max": FileNotFoundError,
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": FileNotFoundError,
    })
    assert exp._cgroup_memory_gb() == (None, None)


# ── _available_memory_gb prefers cgroup ───────────────────────────────
def test_available_memory_prefers_cgroup_over_meminfo(monkeypatch):
    monkeypatch.setattr(exp, "_cgroup_memory_gb", lambda: (64.0, 56.0))
    monkeypatch.setattr(exp, "_meminfo_available_gb", lambda: 2754.0)
    assert exp._available_memory_gb() == 56.0


def test_available_memory_falls_back_to_meminfo_when_cgroup_uncapped(monkeypatch):
    monkeypatch.setattr(exp, "_cgroup_memory_gb", lambda: (None, None))
    monkeypatch.setattr(exp, "_meminfo_available_gb", lambda: 700.0)
    assert exp._available_memory_gb() == 700.0


# ── _local_worker_cap composition ─────────────────────────────────────
@pytest.fixture
def _cap_env(monkeypatch):
    """Force a known CPU affinity, no env overrides."""
    monkeypatch.delenv("KORE_PER_WORKER_GB", raising=False)
    monkeypatch.delenv("KORE_MAX_WORKERS", raising=False)
    monkeypatch.setattr("os.sched_getaffinity",
                        lambda _pid: set(range(360)), raising=False)


def test_cap_cgroup_memory_bound_on_databricks_pod(_cap_env, monkeypatch):
    """360 vCPU pod with a real 64 GiB cgroup limit: memory binds at 16 workers."""
    monkeypatch.setattr(exp, "_cgroup_memory_gb", lambda: (64.0, 64.0))
    monkeypatch.setattr(exp, "_meminfo_available_gb", lambda: 2754.0)
    assert exp._local_worker_cap() == 16


def test_cap_cgroup_uncapped_falls_back_to_cpu(_cap_env, monkeypatch):
    monkeypatch.setattr(exp, "_cgroup_memory_gb", lambda: (None, None))
    monkeypatch.setattr(exp, "_meminfo_available_gb", lambda: 2754.0)
    assert exp._local_worker_cap() == 324  # 0.9 * 360, mem term 688 > cpu


def test_cap_user_ceiling_dominates(_cap_env, monkeypatch):
    monkeypatch.setattr(exp, "_cgroup_memory_gb", lambda: (None, None))
    monkeypatch.setattr(exp, "_meminfo_available_gb", lambda: 2754.0)
    monkeypatch.setenv("KORE_MAX_WORKERS", "32")
    assert exp._local_worker_cap() == 32


def test_cap_explain_surfaces_cgroup(_cap_env, monkeypatch):
    monkeypatch.setattr(exp, "_cgroup_memory_gb", lambda: (64.0, 56.0))
    monkeypatch.setattr(exp, "_meminfo_available_gb", lambda: 2754.0)
    out = exp._local_worker_cap_explain()
    assert "cgroup=64 GiB" in out
    assert "cpu=324/360" in out
    assert "memory-bound" in out
