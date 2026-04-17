"""Unit tests for baseline_eval.py (T4.4).

Uses MockGateway to simulate a full baseline evaluation run.
"""

import sys
import os
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mock_gateway import MockGateway, NUM_HOSTS, NUM_TASKS
import environment
import baseline_eval
from tracker import ExperimentTracker


@pytest.fixture
def mock_gw():
    return MockGateway()


@pytest.fixture
def env(mock_gw):
    with patch.object(environment, "_connect_gateway", return_value=mock_gw):
        e = environment.CloudSimEnv(scenario="HIGH", seed=42)
        yield e
        e.close()


# ── evaluate_baseline ─────────────────────────────────────────────────────

class TestEvaluateBaseline:
    def test_k8s_baseline(self, env):
        result = baseline_eval.evaluate_baseline(
            env, "k8s", scenario="HIGH", seed=42
        )
        assert result["scheduler"] == "k8s"
        assert result["scenario"] == "HIGH"
        assert result["seed"] == 42
        assert result["steps"] == NUM_TASKS
        assert isinstance(result["total_energy_reward"], float)
        assert isinstance(result["total_sla_reward"], float)
        assert result["total_energy_reward"] < 0  # penalties are negative
        assert result["total_sla_reward"] <= 0

    def test_random_baseline(self, env):
        result = baseline_eval.evaluate_baseline(
            env, "random", scenario="HIGH", seed=42
        )
        assert result["scheduler"] == "random"
        assert result["steps"] == NUM_TASKS

    def test_mean_rewards_computed(self, env):
        result = baseline_eval.evaluate_baseline(
            env, "k8s", scenario="HIGH", seed=42
        )
        assert result["mean_energy_reward"] == pytest.approx(
            result["total_energy_reward"] / NUM_TASKS
        )
        assert result["mean_sla_reward"] == pytest.approx(
            result["total_sla_reward"] / NUM_TASKS
        )


# ── run_all_baselines ─────────────────────────────────────────────────────

class TestRunAllBaselines:
    def test_runs_both_policies(self, mock_gw):
        with patch.object(environment, "_connect_gateway", return_value=mock_gw):
            results = baseline_eval.run_all_baselines(
                scenario="HIGH", seed=42,
                gateway_host="localhost", gateway_port=25333,
            )
        assert "k8s" in results
        assert "random" in results

    def test_results_have_required_fields(self, mock_gw):
        with patch.object(environment, "_connect_gateway", return_value=mock_gw):
            results = baseline_eval.run_all_baselines(scenario="HIGH", seed=42)

        for name, r in results.items():
            assert "scheduler" in r
            assert "steps" in r
            assert "total_energy_reward" in r
            assert "total_sla_reward" in r
            assert "total_energy_kwh" in r

    def test_with_tracker(self, mock_gw, capsys):
        t = ExperimentTracker(enabled=False)
        with patch.object(environment, "_connect_gateway", return_value=mock_gw):
            results = baseline_eval.run_all_baselines(
                scenario="HIGH", seed=42, tracker=t,
            )
        t.finish()

        captured = capsys.readouterr()
        assert "k8s" in captured.out
        assert "random" in captured.out
        assert "Scheduler Comparison" in captured.out


# ── save_results ──────────────────────────────────────────────────────────

class TestSaveResults:
    def test_saves_json(self, tmp_path):
        results = {
            "k8s": {
                "scheduler": "k8s",
                "steps": 100,
                "total_energy_reward": -500.0,
                "total_sla_reward": -10.0,
                "total_energy_kwh": np.float64(0.123),
            },
            "random": {
                "scheduler": "random",
                "steps": 100,
                "total_energy_reward": -600.0,
                "total_sla_reward": -15.0,
                "total_energy_kwh": 0.456,
            },
        }
        baseline_eval.save_results(results, str(tmp_path))

        import json
        out_file = tmp_path / "baseline_results.json"
        assert out_file.exists()

        data = json.loads(out_file.read_text())
        assert "k8s" in data
        assert "random" in data
        assert data["k8s"]["steps"] == 100
        # numpy float should have been converted
        assert isinstance(data["k8s"]["total_energy_kwh"], float)


# ── Integration: full pipeline ────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end(self, mock_gw, tmp_path):
        """Simulate the full baseline_eval workflow."""
        with patch.object(environment, "_connect_gateway", return_value=mock_gw):
            results = baseline_eval.run_all_baselines(
                scenario="HIGH", seed=42,
            )
        baseline_eval.save_results(results, str(tmp_path))

        import json
        data = json.loads((tmp_path / "baseline_results.json").read_text())

        # Both baselines ran
        assert len(data) == 2
        # Each ran the expected number of steps
        for name in ["k8s", "random"]:
            assert data[name]["steps"] == NUM_TASKS
            assert data[name]["total_energy_reward"] < 0
