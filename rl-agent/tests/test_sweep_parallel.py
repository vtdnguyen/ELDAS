"""C9 — tests for the parallel sweep (job planning, port pinning, safe merge).

The sweep is the multi-hour job, so its failure modes are expensive: a lost
results file at hour 4, or two runs silently sharing one simulation. These tests
pin the parts that are checkable without actually training.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.sweep_budget import _init_worker, _job_list, preflight_gateways


class TestJobPlanning:
    def test_every_budget_seed_pair_is_scheduled_once(self):
        jobs = _job_list([0.02, 0.04], [42, 43, 44])
        assert len(jobs) == 6
        assert len(set(jobs)) == 6

    def test_pairs_cover_the_full_grid(self):
        jobs = set(_job_list([0.02, 0.04], [42, 43]))
        assert jobs == {(0.02, 42), (0.02, 43), (0.04, 42), (0.04, 43)}


class TestPortPinning:
    def test_worker_pins_its_port_from_the_queue(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_PORT", raising=False)
        q = mp.Queue()
        q.put(25335)
        _init_worker(q)
        assert os.environ["GATEWAY_PORT"] == "25335"

    def test_each_worker_takes_a_distinct_port(self, monkeypatch):
        # Two live workers sharing a gateway would drive one simulation with two
        # policies — silently invalid results, not a crash.
        q = mp.Queue()
        for p in (25333, 25334):
            q.put(p)

        seen = []
        for _ in range(2):
            monkeypatch.delenv("GATEWAY_PORT", raising=False)
            _init_worker(q)
            seen.append(os.environ["GATEWAY_PORT"])
        assert sorted(seen) == ["25333", "25334"]


class TestPreflight:
    def test_unreachable_gateway_fails_fast_with_instructions(self):
        # Port 1 is never a gateway. The message must say how to fix it —
        # this check exists to save a multi-hour run from a 3-minute death.
        with pytest.raises(SystemExit) as e:
            preflight_gateways(1, 2, host="127.0.0.1")
        msg = str(e.value)
        assert "PREFLIGHT FAILED" in msg
        assert "NUM_GATEWAYS=2" in msg

    def test_reachable_port_passes(self):
        import socket

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            preflight_gateways(port, 1, host="127.0.0.1")  # must not raise
        finally:
            srv.close()


def _merge_worker(args):
    """Top-level so it is picklable by the spawn/fork pool."""
    path, key, row = args
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from train_cmdp import _merge_result_json

    _merge_result_json(Path(path), key, row)


class TestLockedMerge:
    def test_merge_writes_a_row(self, tmp_path):
        from train_cmdp import _merge_result_json

        p = tmp_path / "baseline_results.json"
        _merge_result_json(p, "cmdp-d0.02", {"scheduler": "cmdp-d0.02", "seed": 42})
        assert json.loads(p.read_text())["cmdp-d0.02"]["seed"] == 42

    def test_merge_preserves_other_keys(self, tmp_path):
        from train_cmdp import _merge_result_json

        p = tmp_path / "baseline_results.json"
        p.write_text(json.dumps({"bestfit": {"scheduler": "bestfit"}}))
        _merge_result_json(p, "cmdp-d0.02", {"scheduler": "cmdp-d0.02"})
        data = json.loads(p.read_text())
        assert set(data) == {"bestfit", "cmdp-d0.02"}

    def test_corrupt_existing_file_does_not_lose_the_new_row(self, tmp_path):
        # A finished training run must still record its result even if the
        # convenience file was mangled; points.jsonl is the source of truth.
        from train_cmdp import _merge_result_json

        p = tmp_path / "baseline_results.json"
        p.write_text("{not json")
        _merge_result_json(p, "cmdp-d0.04", {"scheduler": "cmdp-d0.04"})
        assert json.loads(p.read_text())["cmdp-d0.04"]["scheduler"] == "cmdp-d0.04"

    @pytest.mark.skipif(sys.platform == "win32", reason="fcntl lock is POSIX-only")
    def test_concurrent_merges_do_not_lose_rows(self, tmp_path):
        # THE regression this lock exists for: unsynchronised read-modify-write
        # drops whichever writer read first, silently losing budgets from the
        # results file at the end of a long sweep.
        p = tmp_path / "baseline_results.json"
        rows = [(str(p), f"cmdp-d{i}", {"scheduler": f"cmdp-d{i}", "seed": i})
                for i in range(12)]

        ctx = mp.get_context("fork")
        with ctx.Pool(6) as pool:
            pool.map(_merge_worker, rows)

        data = json.loads(p.read_text())
        assert len(data) == 12, f"lost rows under concurrency: got {sorted(data)}"
        assert all(f"cmdp-d{i}" in data for i in range(12))
