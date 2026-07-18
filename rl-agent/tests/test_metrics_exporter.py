"""Unit tests for metrics_exporter.py (G1.6).

The exporter must NEVER break the RL loop (CLAUDE.md Lưu ý #16): when monitoring
is off or prometheus_client is unavailable it degrades to a silent no-op.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import metrics_exporter
from metrics_exporter import CMDPMetricsExporter


class TestDisabled:
    def test_explicit_disabled_is_noop(self):
        exp = CMDPMetricsExporter(enabled=False)
        assert exp.enabled is False
        # update() must not raise even though nothing is wired up.
        exp.update(lambda_=0.3, episode_cost_sla=0.05, episode_reward_energy=-1.0)

    def test_default_off_without_monitoring_env(self, monkeypatch):
        monkeypatch.delenv("MONITORING_ENABLED", raising=False)
        exp = CMDPMetricsExporter(enabled=None)
        assert exp.enabled is False

    def test_unavailable_prometheus_forces_noop(self, monkeypatch):
        # Even if explicitly enabled, missing prometheus_client → disabled.
        monkeypatch.setattr(metrics_exporter, "_prometheus_available", lambda: False)
        exp = CMDPMetricsExporter(enabled=True)
        assert exp.enabled is False
        exp.update(lambda_=1.0, episode_cost_sla=0.1, episode_reward_energy=-5.0)


class TestEnvDetection:
    @pytest.mark.parametrize("val,expected", [
        ("true", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("", False),
    ])
    def test_monitoring_requested(self, monkeypatch, val, expected):
        monkeypatch.setenv("MONITORING_ENABLED", val)
        assert metrics_exporter._monitoring_requested(None) is expected

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MONITORING_ENABLED", "false")
        assert metrics_exporter._monitoring_requested(True) is True


class TestEnabled:
    """Only runs if prometheus_client is actually installed."""

    def test_gauges_set_when_enabled(self):
        pytest.importorskip("prometheus_client")
        from prometheus_client import REGISTRY
        # Use a non-default port to avoid clashes if 8000 is taken.
        exp = CMDPMetricsExporter(port=0, enabled=True)
        if not exp.enabled:  # start_http_server may fail on some CI sandboxes
            pytest.skip("exporter could not bind a port in this sandbox")
        exp.update(lambda_=0.42, episode_cost_sla=0.07,
                   episode_reward_energy=-1234.0, episode_cost_sla_raw=4.2e7,
                   budget_d=0.05, pid_integral=0.4)
        assert REGISTRY.get_sample_value("eldas_lambda") == pytest.approx(0.42)
        assert REGISTRY.get_sample_value("eldas_episode_cost_sla") == pytest.approx(0.07)
        assert REGISTRY.get_sample_value("eldas_budget_d") == pytest.approx(0.05)
