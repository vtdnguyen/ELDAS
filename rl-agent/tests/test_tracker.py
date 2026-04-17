"""Unit tests for tracker.py (T4.4).

Tests run with WandB disabled (no API key) to avoid external dependencies.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tracker


# ── ExperimentTracker (disabled mode) ─────────────────────────────────────

class TestTrackerDisabled:
    def test_init_disabled(self):
        t = tracker.ExperimentTracker(enabled=False)
        assert not t.enabled

    def test_log_step(self):
        t = tracker.ExperimentTracker(enabled=False)
        r = np.array([-100.0, -2.0], dtype=np.float32)
        # Should not raise
        t.log_step(step=1, reward_vec=r, scalar_reward=-80.4)

    def test_log_step_with_extra(self):
        t = tracker.ExperimentTracker(enabled=False)
        r = np.array([-50.0, -1.0], dtype=np.float32)
        t.log_step(step=1, reward_vec=r, extra={"custom/metric": 0.5})

    def test_log_episode(self, capsys):
        t = tracker.ExperimentTracker(enabled=False)
        info = {
            "episode": {
                "length": 100,
                "total_energy_reward": -5000.0,
                "total_sla_reward": -50.0,
                "mean_energy_reward": -50.0,
                "mean_sla_reward": -0.5,
                "total_energy_kwh": 0.1234,
            }
        }
        t.log_episode(episode=1, info=info, scheduler="morl")
        captured = capsys.readouterr()
        assert "Episode 1" in captured.out
        assert "morl" in captured.out
        assert "0.1234" in captured.out

    def test_log_comparison(self, capsys):
        t = tracker.ExperimentTracker(enabled=False)
        results = {
            "k8s": {"energy_kwh": 0.15, "sla_violations": 3},
            "random": {"energy_kwh": 0.20, "sla_violations": 7},
        }
        t.log_comparison(results)
        captured = capsys.readouterr()
        assert "Scheduler Comparison" in captured.out
        assert "k8s" in captured.out
        assert "random" in captured.out

    def test_save_model(self, capsys):
        t = tracker.ExperimentTracker(enabled=False)
        t.save_model("/tmp/model.zip", name="test-model")
        captured = capsys.readouterr()
        assert "WandB disabled" in captured.out

    def test_finish(self):
        t = tracker.ExperimentTracker(enabled=False)
        t.finish()  # should not raise

    def test_config_stored(self):
        t = tracker.ExperimentTracker(
            enabled=False,
            config={"lr": 0.001, "scenario": "HIGH"},
        )
        assert t._config["lr"] == 0.001


# ── Auto-detection ────────────────────────────────────────────────────────

class TestAutoDetect:
    def test_no_api_key_means_disabled(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        assert not tracker._should_use_wandb()

    def test_empty_api_key_means_disabled(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "")
        assert not tracker._should_use_wandb()

    def test_whitespace_api_key_means_disabled(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "   ")
        assert not tracker._should_use_wandb()


# ── Integration: log a full episode lifecycle ─────────────────────────────

class TestTrackerIntegration:
    def test_full_lifecycle(self, capsys):
        t = tracker.ExperimentTracker(
            project="test",
            run_name="test-run",
            config={"scenario": "HIGH"},
            tags=["test"],
            enabled=False,
        )

        # Log some steps
        for step in range(5):
            r = np.array([-(step + 1) * 10.0, -(step + 1) * 0.5], dtype=np.float32)
            t.log_step(step=step, reward_vec=r, scalar_reward=float(r.sum()))

        # Log episode
        info = {
            "episode": {
                "length": 5,
                "total_energy_reward": -150.0,
                "total_sla_reward": -7.5,
                "mean_energy_reward": -30.0,
                "mean_sla_reward": -1.5,
                "total_energy_kwh": 0.05,
            }
        }
        t.log_episode(episode=1, info=info, scheduler="morl")

        # Comparison
        t.log_comparison({
            "morl": {"energy_kwh": 0.05, "r_sla": -7.5},
            "k8s": {"energy_kwh": 0.08, "r_sla": -3.0},
        })

        t.finish()

        captured = capsys.readouterr()
        assert "Episode 1" in captured.out
        assert "Scheduler Comparison" in captured.out
