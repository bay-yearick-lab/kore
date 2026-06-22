"""Unit tests for the subprocess-per-chunk orchestrator
``kore.experiments.run_real_data_chunked``. The tests mock
``subprocess.Popen`` to avoid actually forking and to verify the chunk
plan, the per-subprocess env-var profile, the sequential dispatch
order, the failure-collection behavior, and the live stdout-tee."""
from __future__ import annotations

import io
import sys

import pytest

from kore import experiments as exp


def _patch_subprocess(monkeypatch, returncodes: list[int],
                      stdout_lines_per_call: list[list[str]] | None = None):
    """Patch ``subprocess.Popen`` with a fake whose ``stdout`` yields the
    given per-call lines and whose ``wait()`` returns the given rc."""
    calls: list[dict] = []
    iter_rcs = iter(returncodes)
    iter_stdouts = iter(stdout_lines_per_call
                        if stdout_lines_per_call is not None
                        else [[] for _ in returncodes])

    class _FakePopen:
        def __init__(self, cmd, *, env, stdout, stderr,
                     bufsize=-1, text=False):
            calls.append({"cmd": list(cmd), "env": dict(env)})
            self._rc = next(iter_rcs)
            self.stdout = io.StringIO("".join(next(iter_stdouts)))

        def wait(self):
            return self._rc

    import subprocess
    monkeypatch.setattr(subprocess, "Popen", _FakePopen, raising=True)
    return calls


def test_default_chunks_have_11_entries(monkeypatch, tmp_path):
    calls = _patch_subprocess(monkeypatch, [0] * 11)
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    result = exp.run_real_data_chunked(n_seeds=2, max_n=400,
                                       final_aggregate=False)
    assert len(calls) == 11
    assert result["failed"] == []
    assert result["chunks"] == 11
    assert result["final_aggregate_rc"] is None


def test_chunk_env_contains_njobs_and_per_worker_gb(monkeypatch, tmp_path):
    calls = _patch_subprocess(monkeypatch, [0])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    exp.run_real_data_chunked(
        n_seeds=2, max_n=400, chunks=[["catboost"]],
        final_aggregate=False,
    )
    env = calls[0]["env"]
    assert env["KORE_NJOBS_HEAVY"]   == "80"
    assert env["KORE_NJOBS_MEDIUM"]  == "200"
    assert env["KORE_NJOBS_LIGHT"]   == "200"
    assert env["KORE_PER_WORKER_GB"] == "6"
    assert env["KORE_BOOSTER_NJOBS"] == "4"
    assert "KORE_LIGHT_BACKEND" not in env


def test_chunk_command_invokes_run_real_data_with_methods(monkeypatch, tmp_path):
    calls = _patch_subprocess(monkeypatch, [0])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    exp.run_real_data_chunked(
        n_seeds=3, max_n=1234, chunks=[["catboost", "xgboost"]],
        final_aggregate=False,
    )
    cmd = calls[0]["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1] == "-u"
    assert cmd[2] == "-c"
    py_src = cmd[3]
    assert "from kore.experiments import run_real_data" in py_src
    assert "methods=['catboost', 'xgboost']" in py_src
    assert "n_seeds=3" in py_src
    assert "max_n=1234" in py_src
    assert "progress=True" in py_src


def test_failed_chunk_does_not_abort_remaining(monkeypatch, tmp_path):
    calls = _patch_subprocess(monkeypatch, [0, 1, 0])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    result = exp.run_real_data_chunked(
        n_seeds=1, max_n=100,
        chunks=[["catboost"], ["xgboost"], ["kore"]],
        final_aggregate=False,
    )
    assert len(calls) == 3
    assert len(result["failed"]) == 1
    failed_idx, failed_methods, failed_rc = result["failed"][0]
    assert failed_idx == 2
    assert failed_methods == ["xgboost"]
    assert failed_rc == 1


def test_serial_dispatch_order(monkeypatch, tmp_path):
    calls = _patch_subprocess(monkeypatch, [0, 0, 0])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    exp.run_real_data_chunked(
        n_seeds=1, max_n=100,
        chunks=[["a"], ["b"], ["c"]],
        final_aggregate=False,
    )
    ordered_methods = [
        c["cmd"][3].split("methods=")[1].split(",")[0].strip("[ ]'\"")
        for c in calls
    ]
    assert ordered_methods == ["a", "b", "c"]


def test_subprocess_stdout_is_teed_to_log_and_parent(monkeypatch, tmp_path, capsys):
    child_lines = ["  checkpoint: 5/180 cells, wall=0.4 min\n",
                   "  checkpoint: 10/180 cells, wall=0.8 min\n"]
    _patch_subprocess(monkeypatch, [0], stdout_lines_per_call=[child_lines])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    exp.run_real_data_chunked(
        n_seeds=1, max_n=100, chunks=[["catboost"]],
        final_aggregate=False,
    )
    log_path = tmp_path / "chunk_logs" / "chunk_01_catboost.log"
    log_text = log_path.read_text()
    for line in child_lines:
        assert line in log_text
    captured = capsys.readouterr().out
    for line in child_lines:
        assert line.rstrip("\n") in captured
    assert "[01/1] catboost | " in captured


def test_log_files_created_per_chunk(monkeypatch, tmp_path):
    _patch_subprocess(monkeypatch, [0, 0])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    exp.run_real_data_chunked(
        n_seeds=1, max_n=100, chunks=[["catboost"], ["xgboost"]],
        final_aggregate=False,
    )
    log_dir = tmp_path / "chunk_logs"
    assert log_dir.exists()
    logs = sorted(log_dir.glob("chunk_*.log"))
    assert len(logs) == 2
    assert "catboost" in logs[0].name
    assert "xgboost" in logs[1].name


def test_per_chunk_env_override_honored(monkeypatch, tmp_path):
    calls = _patch_subprocess(monkeypatch, [0])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    exp.run_real_data_chunked(
        n_seeds=1, max_n=100, chunks=[["catboost"]],
        per_chunk_env={"KORE_NJOBS_HEAVY": "40",
                       "KORE_PER_WORKER_GB": "12"},
        final_aggregate=False,
    )
    env = calls[0]["env"]
    assert env["KORE_NJOBS_HEAVY"]   == "40"
    assert env["KORE_PER_WORKER_GB"] == "12"


def test_final_aggregate_launches_methods_none_subprocess(monkeypatch, tmp_path):
    """With ``final_aggregate=True`` (default), one extra subprocess
    runs ``run_real_data`` without a ``methods=`` argument so the
    Wilcoxon tables are rebuilt over every available method."""
    calls = _patch_subprocess(monkeypatch, [0, 0])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    result = exp.run_real_data_chunked(
        n_seeds=2, max_n=400, chunks=[["catboost"]],
    )
    assert len(calls) == 2
    chunk_src = calls[0]["cmd"][3]
    final_src = calls[1]["cmd"][3]
    assert "methods=['catboost']" in chunk_src
    assert "methods=" not in final_src
    assert "from kore.experiments import run_real_data" in final_src
    assert "progress=False" in final_src
    assert result["final_aggregate_rc"] == 0
    final_log = tmp_path / "chunk_logs" / "chunk_FINAL_aggregate.log"
    assert final_log.exists()


def test_final_aggregate_runs_even_when_a_chunk_failed(monkeypatch, tmp_path):
    """A non-zero chunk rc must not skip the final aggregation pass.
    The CSV still contains every successful cell from prior chunks and
    the user wants the best-possible tables given that data."""
    calls = _patch_subprocess(monkeypatch, [1, 0])
    monkeypatch.setattr(exp, "OUT", tmp_path, raising=True)
    result = exp.run_real_data_chunked(
        n_seeds=1, max_n=100, chunks=[["catboost"]],
    )
    assert len(calls) == 2
    assert len(result["failed"]) == 1
    assert result["final_aggregate_rc"] == 0
